"""
Spectral extension of frozen pretrained Graphormer (no retraining).

Adds a SAN-style Laplacian positional encoding as an additive bias to the
attention matrix:

    bias_Laplacian(i, j) = alpha * <U_k[i, :], W U_k[j, :]>

where U_k are the k smallest non-trivial Laplacian eigenvectors of each
molecular graph and W is a diagonal weight matrix (uniform / inverse-eigen /
heat-kernel). The hook is attached on top of GraphAttnBias.forward, so the
new term is added AFTER Graphormer's own (spatial, edge, virtual-distance)
attention biases — they coexist additively.

Configurations evaluated (all on the same loaded checkpoint, no retraining):

    baseline           — Graphormer untouched, no Laplacian
    laplacian only     — every Graphormer attn / centrality bias zeroed,
                         only Laplacian active
    laplacian + edge   — only Graphormer's multi-hop edge bias kept,
                         plus Laplacian
    all + laplacian    — full Graphormer, plus Laplacian additively

The "ablate Graphormer biases" half is implemented with the same forward
hooks used in ablate.py; the "add Laplacian bias" half is a forward hook on
the GraphAttnBias module that adds a per-graph (N+1, N+1) bias tensor
(broadcast across heads) to its output. Because both kinds of hooks live on
disjoint modules, they compose without interference.

Run:
    python extend_spectral.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000
    python extend_spectral.py --ckpt ... --alpha 0.5 --k 8 --weight uniform
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

# Make the package importable from any cwd, then pull in shared helpers.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from mini_graphormer.pretrain import load_pretrained  # noqa: E402
from ablate import collect_batches                    # noqa: E402


# ---------------------------------------------------------------------------
# Configuration dataclass — encodes which Graphormer biases are active and
# whether/how the Laplacian PE is added.
# ---------------------------------------------------------------------------


@dataclass
class Config:
    name: str
    use_spatial: bool = True            # Graphormer's b(phi(i,j)) shortest-path bias
    use_edge: bool = True               # Graphormer's c(i,j) multi-hop edge bias
    use_virtual_distance: bool = True   # learned [graph]-token-to-node bias
    use_centrality: bool = True         # in/out degree node-feature additions
    use_laplacian: bool = False         # our new spectral additive bias
    alpha: float = 1.0
    k: int = 8
    weight_mode: str = "uniform"        # "uniform" | "inv" | "heat"


# ---------------------------------------------------------------------------
# Hook implementations.
# ---------------------------------------------------------------------------


def _zero_hook(_module, _inp, output):
    return torch.zeros_like(output)


def _compute_laplacian_bias(
    spatial_pos: torch.Tensor,    # (B, N, N), +1-shifted; sp==2 means adjacent
    padding_mask: torch.Tensor,   # (B, N+1) bool, True = pad slot
    alpha: float,
    k: int,
    weight_mode: str,
) -> torch.Tensor:
    """Build the (B, N+1, N+1) additive Laplacian PE bias.

    Per-graph: extract the real (non-pad) sub-block, build L = D - A from
    the adjacency derived from spatial_pos, eigendecompose, drop zero
    eigenvalues (one per connected component), keep the next k, and form
    S = U_k diag(w) U_k^T. Place S into bias[1:1+N_real, 1:1+N_real].
    """
    B, N, _ = spatial_pos.shape
    bias = torch.zeros(B, N + 1, N + 1, dtype=torch.float32, device=spatial_pos.device)

    # adjacency: shifted spatial_pos == 2 means original distance == 1.
    adj_all = (spatial_pos == 2).float()
    real_mask = ~padding_mask[:, 1:]   # (B, N), True for real nodes

    for b in range(B):
        n_real = int(real_mask[b].sum().item())
        if n_real < 2:
            continue

        a = adj_all[b, :n_real, :n_real]
        d = a.sum(dim=1)
        L = torch.diag(d) - a

        try:
            eigvals, eigvecs = torch.linalg.eigh(L)
        except Exception:
            # eigendecomp failed (very rare); skip this graph.
            continue

        # Skip the trivial zero eigenvalue(s). One per connected component.
        n_zero = int((eigvals.abs() < 1e-5).sum().item())
        n_zero = max(n_zero, 1)
        k_use = min(k, n_real - n_zero)
        if k_use < 1:
            continue

        sel_vals = eigvals[n_zero:n_zero + k_use]
        sel_vecs = eigvecs[:, n_zero:n_zero + k_use]

        if weight_mode == "uniform":
            w = torch.ones_like(sel_vals)
        elif weight_mode == "inv":
            # Boost low-frequency (small-eigenvalue) modes for global structure.
            w = 1.0 / (sel_vals + 1e-3)
        elif weight_mode == "heat":
            # exp(-t * lambda) at t=1.
            w = torch.exp(-sel_vals)
        else:
            raise ValueError(f"unknown weight_mode: {weight_mode}")

        # S = U_k diag(w) U_k^T   shape (n_real, n_real)
        S = (sel_vecs * w.unsqueeze(0)) @ sel_vecs.T

        bias[b, 1:1 + n_real, 1:1 + n_real] = alpha * S

    return bias


class LaplacianHook:
    """Forward hook on GraphAttnBias that adds the Laplacian PE bias to the
    (B, H, T, T) attention bias output, broadcasting the (B, T, T) addend
    over all heads."""

    def __init__(self, alpha: float, k: int, weight_mode: str):
        self.alpha = alpha
        self.k = k
        self.weight_mode = weight_mode

    def __call__(self, module, inputs, output):
        batch = inputs[0]
        spatial_pos = batch["spatial_pos"]
        x = batch["x"]
        # padding_mask: graph-token slot at index 0 is never padded; real
        # nodes are pad iff every atom feature is 0 (since real x values are
        # >= 1 after the offset).
        pad_real = x[:, :, 0].eq(0)
        cls_pad = torch.zeros(
            pad_real.size(0), 1, dtype=pad_real.dtype, device=pad_real.device,
        )
        padding_mask = torch.cat([cls_pad, pad_real], dim=1)

        lap = _compute_laplacian_bias(
            spatial_pos, padding_mask, self.alpha, self.k, self.weight_mode,
        )
        # output is (B, H, T, T); the addend is (B, T, T) → (B, 1, T, T) → broadcast
        return output + lap.unsqueeze(1)


# ---------------------------------------------------------------------------
# Configuration application: register the right hooks for a given Config.
# ---------------------------------------------------------------------------


def apply_config(model, cfg: Config) -> list:
    """Register all hooks for `cfg` and return the list of removable handles."""
    handles = []
    enc = model.encoder.graph_encoder
    nf = enc.graph_node_feature
    ab = enc.graph_attn_bias

    # Zero out Graphormer biases that this config disables.
    if not cfg.use_spatial:
        handles.append(ab.spatial_pos_encoder.register_forward_hook(_zero_hook))
    if not cfg.use_edge:
        handles.append(ab.edge_encoder.register_forward_hook(_zero_hook))
    if not cfg.use_virtual_distance:
        handles.append(ab.graph_token_virtual_distance.register_forward_hook(_zero_hook))
    if not cfg.use_centrality:
        handles.append(nf.in_degree_encoder.register_forward_hook(_zero_hook))
        handles.append(nf.out_degree_encoder.register_forward_hook(_zero_hook))

    # Add Laplacian PE bias (additive, hooks the parent GraphAttnBias module
    # so it sees the post-aggregation bias tensor).
    if cfg.use_laplacian:
        handles.append(ab.register_forward_hook(
            LaplacianHook(cfg.alpha, cfg.k, cfg.weight_mode),
        ))

    return handles


# ---------------------------------------------------------------------------
# Per-config evaluation (reuses the pre-collated batches across configs).
# ---------------------------------------------------------------------------


def run_one(model, cfg: Config, batches: list[dict], device: torch.device) -> tuple[float, float]:
    handles = apply_config(model, cfg)
    try:
        abs_err_sum = 0.0
        n_seen = 0
        t0 = time.time()
        for batch in batches:
            b = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                pred = model.predict(b).squeeze(-1)
            y = b["y"].to(pred.dtype)
            abs_err_sum += (pred - y).abs().sum().item()
            n_seen += y.numel()
        elapsed = time.time() - t0
    finally:
        for h in handles:
            h.remove()
    return abs_err_sum / n_seen, elapsed


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="Local .pt path or PRETRAINED_URLS key.")
    parser.add_argument("--max-graphs", type=int, default=2000,
                        help="Validation subset size.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Laplacian PE strength. Scale relative to "
                             "Graphormer's own bias logits (~O(1)).")
    parser.add_argument("--k", type=int, default=8,
                        help="Number of non-trivial eigenvectors retained.")
    parser.add_argument("--weight", default="uniform",
                        choices=["uniform", "inv", "heat"],
                        help="Eigenvalue weighting in S = U_k diag(w) U_k^T.")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.ckpt} ...")
    model = load_pretrained(args.ckpt).to(device)
    print("Loaded.\n")

    print("Materialising validation batches (one-time cost) ...")
    batches, n_items = collect_batches(args.batch_size, args.max_graphs)
    print()

    laplacian_kw = dict(
        use_laplacian=True, alpha=args.alpha, k=args.k, weight_mode=args.weight,
    )
    configs = [
        Config(name="baseline"),
        Config(name="laplacian only",
               use_spatial=False, use_edge=False, use_virtual_distance=False,
               use_centrality=False, **laplacian_kw),
        Config(name="laplacian + edge",
               use_spatial=False, use_virtual_distance=False, use_centrality=False,
               **laplacian_kw),
        Config(name="all + laplacian", **laplacian_kw),
    ]

    print(f"Spectral (Laplacian PE) extension")
    print(f"  N = {n_items} graphs   alpha = {args.alpha}   k = {args.k}   "
          f"weight = {args.weight}\n")
    print(f"  {'configuration':22s} {'MAE':>10s} {'diff vs base':>14s}  {'time':>7s}")
    print(f"  {'-'*22} {'-'*10} {'-'*14}  {'-'*7}")

    baseline_mae: float | None = None
    for cfg in configs:
        mae, elapsed = run_one(model, cfg, batches, device)
        if cfg.name == "baseline":
            baseline_mae = mae
            delta_str = ""
        else:
            d = mae - baseline_mae
            pct = 100 * d / baseline_mae if baseline_mae else 0.0
            delta_str = f"{d:+.4f} ({pct:+.0f}%)"
        print(f"  {cfg.name:22s} {mae:10.6f} {delta_str:>14s}  {elapsed:>5.1f}s")


if __name__ == "__main__":
    main()
