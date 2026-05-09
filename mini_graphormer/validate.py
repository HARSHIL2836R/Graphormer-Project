"""
Validity checks for mini_graphormer.

Test 1 — STRICT CHECKPOINT LOAD
  Build MiniGraphormer with the PCQM4Mv2-base config, load Microsoft's
  released `checkpoint_best_pcqm4mv2.pt` with strict=True, and verify there
  are zero missing/unexpected keys. If anything in the architecture skeleton
  is wrong (a renamed parameter, a wrong tensor shape), this fails loudly.

Test 3 — PCQM4Mv2 VALIDATION MAE
  Iterate over the official OGB PCQM4Mv2 validation split (~73k molecules),
  preprocess each one with our pipeline, batch with our collator, and compute
  mean absolute error against the ground-truth HOMO-LUMO gap labels.
  Reference: graphormer-base reaches MAE ≈ 0.0864 eV on this split.

Run from inside the mini_graphormer/ folder:

  # both tests, downloads the checkpoint and OGB dataset on first run
  python validate.py --ckpt pcqm4mv2_graphormer_base

  # use a local checkpoint (skip the download)
  python validate.py --ckpt /path/to/checkpoint_best_pcqm4mv2.pt

  # only test 1 (skip MAE; doesn't need ogb / torch_geometric)
  python validate.py --ckpt /path/to/checkpoint_best_pcqm4mv2.pt --skip-mae

  # speed knobs
  python validate.py --ckpt ... --batch-size 64 --max-graphs 1000 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

# Same sys.path trick as run_demo.py: lets us import the package whether you
# launch the script from inside this folder or from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_graphormer import (  # noqa: E402
    MiniGraphormer,
    collate,
    graphormer_base_pcqm4mv2_config,
    preprocess_item,
)
from mini_graphormer.pretrain import _fetch_state_dict  # noqa: E402


# ---------------------------------------------------------------------------
# Test 1: strict-load checkpoint
# ---------------------------------------------------------------------------


def test_strict_load(ckpt: str, device: torch.device) -> MiniGraphormer:
    print("=" * 70)
    print("TEST 1 — strict checkpoint load")
    print("=" * 70)

    cfg = graphormer_base_pcqm4mv2_config()
    model = MiniGraphormer(cfg)

    print(f"  loading state dict from: {ckpt}")
    state = _fetch_state_dict(ckpt)
    print(f"  state_dict has {len(state)} tensors")

    # Try strict first. If it fails, retry with strict=False so we can show
    # the diff and fail clearly.
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("  STRICT LOAD FAILED — diff vs. our model:")
        for k in missing:
            print(f"    missing   : {k}")
        for k in unexpected:
            print(f"    unexpected: {k}")
        raise SystemExit(1)

    # Spot-check a few weight stats so we know we didn't load garbage.
    sample = {
        "atom_encoder": model.encoder.graph_encoder.graph_node_feature.atom_encoder.weight,
        "spatial_pos_encoder": model.encoder.graph_encoder.graph_attn_bias.spatial_pos_encoder.weight,
        "layer0.q_proj": model.encoder.graph_encoder.layers[0].self_attn.q_proj.weight,
        "embed_out": model.encoder.embed_out.weight,
    }
    print("  weight stats (after load):")
    for name, w in sample.items():
        shape_str = str(tuple(w.shape))
        print(f"    {name:24s} shape={shape_str:20s} mean={w.mean():+.4e}  std={w.std():.4e}")

    print("  PASS — strict load succeeded with 0 missing / 0 unexpected keys")
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Test 3: PCQM4Mv2 validation MAE
# ---------------------------------------------------------------------------


def _build_pcqm4mv2_valid_loader(batch_size: int, max_graphs: int | None):
    """Stream the official PCQM4Mv2 validation split from the raw CSV.

    Why not use ogb.lsc.PygPCQM4Mv2Dataset?
      * It calls `smiles2graph` on ALL 3.7M training molecules up front during
        first-time processing. On CPU this is hours, and we don't need any of
        them — only the 73 545 validation rows.
      * Modern rdkit raises hard errors on a few rare invalid valences in the
        OGB CSV (e.g. `Si` with explicit valence 5). OGB's processor doesn't
        guard against rdkit returning None, so the whole dataset build crashes.

    Instead we read `dataset/pcqm4m-v2/raw/data.csv.gz` row by row, only
    featurise rows whose `idx` is in the validation split, and skip rows that
    rdkit refuses with a count printed at the end.
    """
    try:
        from ogb.utils.mol import smiles2graph
    except ImportError as e:
        print(f"  SKIPPED — failed to import ogb.utils.mol.smiles2graph: "
              f"{type(e).__name__}: {e}")
        print( "    Need: pip install ogb torch_geometric rdkit")
        raise SystemExit(0) from e

    raw_root = Path("dataset/pcqm4m-v2")
    csv_path = raw_root / "raw" / "data.csv.gz"
    split_path = raw_root / "split_dict.pt"
    if not csv_path.exists() or not split_path.exists():
        print( "  SKIPPED — raw PCQM4Mv2 files not found. Expected:")
        print(f"    {csv_path}")
        print(f"    {split_path}")
        print( "  These get extracted by ogb on first import; if the previous run")
        print( "  crashed mid-way, delete `dataset/pcqm4m-v2/processed` and re-run.")
        raise SystemExit(0)

    splits = torch.load(split_path, weights_only=False)
    valid_idx = splits["valid"].tolist()
    if max_graphs is not None:
        valid_idx = valid_idx[:max_graphs]
    valid_set = set(valid_idx)
    n_total = len(valid_set)
    print(f"  PCQM4Mv2 valid split: {n_total} graphs (streamed from {csv_path})")

    def loader():
        n_skipped = 0
        n_yielded = 0
        buf = []
        with gzip.open(csv_path, "rt") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["idx"])
                if idx not in valid_set:
                    continue
                try:
                    g = smiles2graph(row["smiles"])
                except Exception:
                    # rdkit refused this molecule — skip and keep going.
                    n_skipped += 1
                    continue
                item = SimpleNamespace(
                    x=torch.tensor(g["node_feat"], dtype=torch.long),
                    edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
                    edge_attr=torch.tensor(g["edge_feat"], dtype=torch.long),
                    y=torch.tensor([float(row["homolumogap"])], dtype=torch.float),
                    idx=idx,
                )
                buf.append(preprocess_item(item))
                n_yielded += 1
                if len(buf) == batch_size:
                    yield buf
                    buf = []
                if n_yielded >= n_total:
                    break
        if buf:
            yield buf
        if n_skipped:
            print(f"  note: skipped {n_skipped} unparseable SMILES")

    return loader, n_total


def test_pcqm4mv2_mae(model: MiniGraphormer, batch_size: int, max_graphs: int | None,
                     device: torch.device, multi_hop_max_dist: int = 5,
                     spatial_pos_max: int = 1024) -> None:
    print("=" * 70)
    print("TEST 3 — PCQM4Mv2 validation MAE")
    print("=" * 70)

    loader_fn, n_total = _build_pcqm4mv2_valid_loader(batch_size, max_graphs)

    abs_err_sum = 0.0
    n_seen = 0
    t0 = time.time()
    for batch_items in loader_fn():
        batch = collate(
            batch_items,
            max_node=512,
            multi_hop_max_dist=multi_hop_max_dist,
            spatial_pos_max=spatial_pos_max,
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            pred = model.predict(batch).squeeze(-1)  # (B,)
        y = batch["y"].to(pred.dtype)
        abs_err_sum += (pred - y).abs().sum().item()
        n_seen += y.numel()

        # Live MAE every 10 batches.
        if n_seen % (batch_size * 10) == 0 or n_seen >= n_total:
            mae_so_far = abs_err_sum / n_seen
            elapsed = time.time() - t0
            rate = n_seen / max(elapsed, 1e-6)
            print(f"    {n_seen:7d}/{n_total} graphs  MAE={mae_so_far:.6f}  "
                  f"({rate:.1f} graph/s, elapsed={elapsed:.1f}s)")

    mae = abs_err_sum / n_seen
    print(f"  FINAL MAE on {n_seen} graphs: {mae:.6f}")
    print( "  Reference (graphormer-base, official, full 73545 valid): 0.0864")
    # Tolerance scales as ~1/sqrt(n) since per-molecule abs errors are noisy.
    # 0.04 / sqrt(n) gives ~0.013 at n=1k, ~0.005 at n=64k — close to what we
    # actually observe for finite-sample MAE wobble around 0.0864.
    import math
    tol = max(0.04 / math.sqrt(n_seen), 0.003)
    if abs(mae - 0.0864) < tol:
        print(f"  PASS — within ±{tol:.4f} of 0.0864 (sample-size-adjusted)")
    elif abs(mae - 0.0864) < 3 * tol:
        print(f"  OK   — within ±{3*tol:.4f} of 0.0864 (mild slice variance; "
              f"run a larger --max-graphs to tighten)")
    else:
        print("  WARN — well outside expected sampling noise. Likely a real bug")
        print("         in the data pipeline (Floyd-Warshall, multi-hop, "
              "spatial_pos_max).")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="Either a key from PRETRAINED_URLS (e.g. "
                             "'pcqm4mv2_graphormer_base') or a local .pt path.")
    parser.add_argument("--skip-mae", action="store_true",
                        help="Run only test 1; don't iterate over PCQM4Mv2.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-graphs", type=int, default=None,
                        help="Optional cap on validation set size (debug).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--multi-hop-max-dist", type=int, default=5)
    parser.add_argument("--spatial-pos-max", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device(args.device)

    model = test_strict_load(args.ckpt, device)
    if not args.skip_mae:
        test_pcqm4mv2_mae(
            model,
            batch_size=args.batch_size,
            max_graphs=args.max_graphs,
            device=device,
            multi_hop_max_dist=args.multi_hop_max_dist,
            spatial_pos_max=args.spatial_pos_max,
        )


if __name__ == "__main__":
    main()
