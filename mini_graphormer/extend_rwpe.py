"""
Random-Walk Positional Encoding (RWPE) extension for frozen Graphormer.

Adds a random-walk-based pairwise structural bias to the attention matrix:

    bias_RWPE(i, j) = alpha * sum_{k=1}^{K} (A_norm^k)_{ij}

where A_norm = D^(-1/2) A D^(-1/2) is the symmetrically-normalised adjacency.
The k-th power gives the (normalised) probability mass of length-k walks
between i and j; summing over k = 1..K produces a multi-scale "soft
adjacency" — pairs reachable by short random walks share a positive bias
even if they are not directly connected.

Conceptual link: this is essentially the structural prior that Graph
Transformer Networks (Yun et al. 2019) learn to construct by combining
adjacency matrices. We bake the construction in by hand and inject it as
an additive bias on top of the frozen pretrained model.

Configurations evaluated (all on the same loaded checkpoint, no retraining):

    baseline           — Graphormer untouched, no RWPE
    rwpe only          — every Graphormer attn / centrality bias zeroed,
                         only RWPE active
    rwpe + edge        — only Graphormer's multi-hop edge bias kept,
                         plus RWPE
    all + rwpe         — full Graphormer, plus RWPE additively
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from mini_graphormer.pretrain import load_pretrained  # noqa: E402
from ablate import collect_batches                    # noqa: E402


# ---------------------------------------------------------------------------
# Configuration dataclass.
# ---------------------------------------------------------------------------


@dataclass
class Config:
    name: str
    use_spatial: bool = True
    use_edge: bool = True
    use_virtual_distance: bool = True
    use_centrality: bool = True
    use_rwpe: bool = False
    alpha: float = 0.5
    K: int = 4


# ---------------------------------------------------------------------------
# Hooks.
# ---------------------------------------------------------------------------


def _zero_hook(_module, _inp, output):
    return torch.zeros_like(output)


def _compute_rwpe_bias(
    spatial_pos: torch.Tensor,    # (B, N, N), +1-shifted; sp == 2 means adjacent
    padding_mask: torch.Tensor,   # (B, N+1) bool, True = pad slot
    alpha: float,
    K: int,
) -> torch.Tensor:
    """Build the (B, N+1, N+1) additive RWPE bias.

    Per-graph: derive A from spatial_pos (sp == 2 means adjacent), build
    A_norm = D^(-1/2) A D^(-1/2), accumulate S = sum_{k=1}^{K} A_norm^k, and
    place alpha * S into bias[1:1+N_real, 1:1+N_real].
    """
    B, N, _ = spatial_pos.shape
    bias = torch.zeros(B, N + 1, N + 1, dtype=torch.float32, device=spatial_pos.device)

    adj_all = (spatial_pos == 2).float()       # (B, N, N)
    real_mask = ~padding_mask[:, 1:]            # (B, N)

    for b in range(B):
        n_real = int(real_mask[b].sum().item())
        if n_real < 2:
            continue

        a = adj_all[b, :n_real, :n_real]       # (n, n) symmetric
        d = a.sum(dim=1)                       # (n,)
        # D^(-1/2); guard against degree-0 nodes (shouldn't happen for a
        # connected molecule, but be defensive — use 0 for isolated nodes).
        d_inv_sqrt = torch.where(d > 0, d.clamp(min=1e-6).rsqrt(), torch.zeros_like(d))
        # A_norm = D^(-1/2) A D^(-1/2); broadcast row-scale and column-scale.
        A_norm = d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)

        # Accumulate sum_{k=1}^{K} A_norm^k.
        S = torch.zeros_like(A_norm)
        Ak = torch.eye(n_real, device=a.device, dtype=A_norm.dtype)
        for _ in range(K):
            Ak = Ak @ A_norm
            S = S + Ak

        bias[b, 1:1 + n_real, 1:1 + n_real] = alpha * S

    return bias


class RWPEHook:
    """Forward hook on GraphAttnBias adding the RWPE bias broadcast over heads."""

    def __init__(self, alpha: float, K: int):
        self.alpha = alpha
        self.K = K

    def __call__(self, module, inputs, output):
        batch = inputs[0]
        spatial_pos = batch["spatial_pos"]
        x = batch["x"]
        pad_real = x[:, :, 0].eq(0)
        cls_pad = torch.zeros(
            pad_real.size(0), 1, dtype=pad_real.dtype, device=pad_real.device,
        )
        padding_mask = torch.cat([cls_pad, pad_real], dim=1)

        rw = _compute_rwpe_bias(spatial_pos, padding_mask, self.alpha, self.K)
        return output + rw.unsqueeze(1)   # broadcast over heads


# ---------------------------------------------------------------------------
# Configuration application.
# ---------------------------------------------------------------------------


def apply_config(model, cfg: Config) -> list:
    handles = []
    enc = model.encoder.graph_encoder
    nf = enc.graph_node_feature
    ab = enc.graph_attn_bias

    if not cfg.use_spatial:
        handles.append(ab.spatial_pos_encoder.register_forward_hook(_zero_hook))
    if not cfg.use_edge:
        handles.append(ab.edge_encoder.register_forward_hook(_zero_hook))
    if not cfg.use_virtual_distance:
        handles.append(ab.graph_token_virtual_distance.register_forward_hook(_zero_hook))
    if not cfg.use_centrality:
        handles.append(nf.in_degree_encoder.register_forward_hook(_zero_hook))
        handles.append(nf.out_degree_encoder.register_forward_hook(_zero_hook))

    if cfg.use_rwpe:
        handles.append(ab.register_forward_hook(RWPEHook(cfg.alpha, cfg.K)))

    return handles


# ---------------------------------------------------------------------------
# Per-config evaluation.
# ---------------------------------------------------------------------------


def run_one(model, cfg: Config, batches: list[dict], device: torch.device):
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
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max-graphs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="RWPE bias strength.")
    parser.add_argument("--K", type=int, default=4,
                        help="Max walk length (sum k=1..K of A_norm^k).")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.ckpt} ...")
    model = load_pretrained(args.ckpt).to(device)
    print("Loaded.\n")

    print("Materialising validation batches (one-time cost) ...")
    batches, n_items = collect_batches(args.batch_size, args.max_graphs)
    print()

    rwpe_kw = dict(use_rwpe=True, alpha=args.alpha, K=args.K)
    configs = [
        Config(name="baseline"),
        Config(name="rwpe only",
               use_spatial=False, use_edge=False, use_virtual_distance=False,
               use_centrality=False, **rwpe_kw),
        Config(name="rwpe + edge",
               use_spatial=False, use_virtual_distance=False, use_centrality=False,
               **rwpe_kw),
        Config(name="all + rwpe", **rwpe_kw),
    ]

    print(f"Random-Walk PE (RWPE) extension")
    print(f"  N = {n_items} graphs   alpha = {args.alpha}   K = {args.K}\n")
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
