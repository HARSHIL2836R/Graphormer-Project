"""
Ablation study on the PCQM4Mv2 validation set.

For each of the three encodings that Graphormer adds to the (B, H, T, T)
attention bias, we register a forward hook that replaces the encoding's
output with zeros, then re-run the same batches. Weights are NEVER modified
— the same model instance is reused across all ablations, so any difference
in MAE is attributable purely to the ablated bias term.

We also ablate the centrality encoding (which is added to the *node*
features rather than the attention matrix) for context, since Graphormer's
inductive bias triple is usually quoted as (centrality, spatial, edge).

Run:
    python ablate.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Same sys.path trick as validate.py / run_demo.py — make the package
# importable from any cwd, then import the streaming loader from
# validate.py (which lives next to this file).
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # parent of mini_graphormer/
sys.path.insert(0, str(HERE))          # mini_graphormer/ itself

from mini_graphormer import collate           # noqa: E402
from mini_graphormer.pretrain import load_pretrained  # noqa: E402
from validate import _build_pcqm4mv2_valid_loader     # noqa: E402


# ---------------------------------------------------------------------------
# Hooks. Each ablation = a list of submodules whose output is replaced with
# zeros. The original tensors are not touched, the hook can be removed.
# ---------------------------------------------------------------------------


def _zero_hook(_module, _inp, output):
    return torch.zeros_like(output)


def build_ablations(model) -> dict[str, list[torch.nn.Module]]:
    """Return {ablation_name -> [modules_to_zero]} for the loaded model."""
    enc = model.encoder.graph_encoder
    nf = enc.graph_node_feature
    ab = enc.graph_attn_bias
    return {
        "baseline":             [],
        # ---- biases added to the attention matrix ----
        "no_spatial_bias":      [ab.spatial_pos_encoder],
        "no_edge_bias":         [ab.edge_encoder],
        "no_virtual_distance":  [ab.graph_token_virtual_distance],
        "no_all_attn_biases":   [ab.spatial_pos_encoder, ab.edge_encoder,
                                 ab.graph_token_virtual_distance],
        # ---- centrality encoding (added to node features, not attention) ----
        "no_centrality":        [nf.in_degree_encoder, nf.out_degree_encoder],
    }


# ---------------------------------------------------------------------------
# Preprocess once, reuse across ablations.
# ---------------------------------------------------------------------------


def collect_batches(batch_size: int, max_graphs: int | None) -> tuple[list[dict], int]:
    """Stream validation items, run preprocess + collate once, return batches."""
    loader_fn, n_total = _build_pcqm4mv2_valid_loader(batch_size, max_graphs)
    batches = []
    n_items = 0
    t0 = time.time()
    for buf in loader_fn():
        batch = collate(
            buf,
            max_node=512,
            multi_hop_max_dist=5,
            spatial_pos_max=1024,
        )
        batches.append(batch)
        n_items += batch["y"].numel()
    print(f"  preprocessed {n_items} graphs into {len(batches)} batches "
          f"in {time.time()-t0:.1f}s")
    return batches, n_items


# ---------------------------------------------------------------------------
# One ablation = attach hooks, sweep the prebuilt batches, compute MAE.
# ---------------------------------------------------------------------------


def run_one(model, modules: list[torch.nn.Module], batches: list[dict],
            device: torch.device) -> tuple[float, float]:
    handles = [m.register_forward_hook(_zero_hook) for m in modules]
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
                        help="Validation subset size. 1000-5000 gives a tight "
                             "signal on CPU; full set (~73k) is the published "
                             "number but takes ~25 min.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.ckpt} ...")
    model = load_pretrained(args.ckpt).to(device)
    print("Loaded.\n")

    print("Materialising validation batches (one-time cost) ...")
    batches, n_items = collect_batches(args.batch_size, args.max_graphs)
    print()

    ablations = build_ablations(model)

    # Header.
    print(f"PCQM4Mv2 validation MAE  —  N = {n_items} graphs, "
          f"batch_size = {args.batch_size}\n")
    print(f"  {'ablation':22s} {'MAE':>10s} {'diff vs base':>14s}  {'time':>7s}")
    print(f"  {'-'*22} {'-'*10} {'-'*14}  {'-'*7}")

    baseline_mae: float | None = None
    for name, mods in ablations.items():
        mae, elapsed = run_one(model, mods, batches, device)
        if name == "baseline":
            baseline_mae = mae
            delta_str = ""
        else:
            d = mae - baseline_mae
            pct = 100 * d / baseline_mae if baseline_mae else 0.0
            delta_str = f"{d:+.4f} ({pct:+.0f}%)"
        print(f"  {name:22s} {mae:10.6f} {delta_str:>14s}  {elapsed:>5.1f}s")


if __name__ == "__main__":
    main()
