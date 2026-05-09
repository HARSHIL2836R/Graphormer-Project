"""
Ring-aware bias extension for frozen Graphormer (PAGTN-flavored).

For molecular graphs, atoms in the same ring are structurally and
chemically related. We add a binary attention bias:

    bias_ring(i, j) = alpha * 1[atoms i and j share at least one ring]

extracted via rdkit's `GetSymmSSSR` (smallest set of smallest rings).
This is one of the path features emphasised by PAGTN (Chen et al. 2019)
and a natural test of whether ring topology — not directly visible to
Graphormer's shortest-path bias — carries complementary signal post-hoc.

Note: this extension needs ring identity, not just per-atom "is in some
ring" (which OGB already encodes in atom feature 8). Two atoms in different
fused rings would both pass the per-atom test but should NOT receive a
same-ring bias. So we re-stream the validation CSV with rdkit, extract a
per-graph (N, N) `same_ring` matrix, and pass it through to the attention
bias hook. We do NOT reuse `ablate.collect_batches` (it can't carry the
extra field).

Configurations evaluated:

    baseline           — Graphormer untouched, no ring bias
    ring only          — every Graphormer attn / centrality bias zeroed,
                         only ring bias active
    ring + edge        — only Graphormer's multi-hop edge bias kept,
                         plus ring bias
    all + ring         — full Graphormer, plus ring bias additively
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from mini_graphormer import collate, preprocess_item   # noqa: E402
from mini_graphormer.pretrain import load_pretrained   # noqa: E402


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
    use_ring: bool = False
    alpha: float = 1.0


# ---------------------------------------------------------------------------
# Custom data loader: stream the CSV, extract ring info via rdkit, pad and
# stack into batches that carry the extra `same_ring` (B, N, N) field.
# ---------------------------------------------------------------------------


def _build_ring_aware_batches(batch_size: int, max_graphs: int | None):
    """Stream PCQM4Mv2 valid split with per-graph ring-membership matrices.

    Reuses the same raw files as validate.py (`dataset/pcqm4m-v2/raw/data.csv.gz`
    and `dataset/pcqm4m-v2/split_dict.pt`), but additionally calls rdkit to
    enumerate rings per molecule.
    """
    try:
        from ogb.utils.mol import smiles2graph
        from rdkit import Chem
    except ImportError as e:
        print(f"  required package missing: {e}")
        print( "    Need: pip install ogb torch_geometric rdkit")
        raise SystemExit(0) from e

    raw_root = Path("dataset/pcqm4m-v2")
    csv_path = raw_root / "raw" / "data.csv.gz"
    split_path = raw_root / "split_dict.pt"
    if not csv_path.exists() or not split_path.exists():
        print(f"  raw PCQM4Mv2 files not found at {raw_root}")
        raise SystemExit(0)

    splits = torch.load(split_path, weights_only=False)
    valid_idx = splits["valid"].tolist()
    if max_graphs is not None:
        valid_idx = valid_idx[:max_graphs]
    valid_set = set(valid_idx)
    n_total = len(valid_set)
    print(f"  PCQM4Mv2 valid split: {n_total} graphs (streamed from {csv_path})")

    items: list[SimpleNamespace] = []
    n_skipped = 0
    t0 = time.time()
    with gzip.open(csv_path, "rt") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["idx"])
            if idx not in valid_set:
                continue
            smiles = row["smiles"]
            try:
                g = smiles2graph(smiles)
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    raise ValueError("rdkit could not parse")
                rings = Chem.GetSymmSSSR(mol)
                N = mol.GetNumAtoms()
                # Build (N, N) symmetric same-ring matrix. Atom i and j get
                # a 1 iff they appear in at least one common ring.
                same_ring = torch.zeros(N, N, dtype=torch.float32)
                for ring in rings:
                    atoms = list(ring)
                    for ii in atoms:
                        for jj in atoms:
                            if ii != jj:
                                same_ring[ii, jj] = 1.0
            except Exception:
                n_skipped += 1
                continue

            item = SimpleNamespace(
                x=torch.tensor(g["node_feat"], dtype=torch.long),
                edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
                edge_attr=torch.tensor(g["edge_feat"], dtype=torch.long),
                y=torch.tensor([float(row["homolumogap"])], dtype=torch.float),
                idx=idx,
                same_ring=same_ring,
            )
            items.append(preprocess_item(item))
            if len(items) >= n_total:
                break

    if n_skipped:
        print(f"  note: skipped {n_skipped} unparseable SMILES")

    # Batch + pad. Standard collate handles x/edge/spatial_pos; we additionally
    # pad and stack the same_ring matrices into the batch dict.
    batches = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        batch = collate(chunk, max_node=512, multi_hop_max_dist=5, spatial_pos_max=1024)

        max_n = batch["x"].size(1)
        sr = torch.zeros(len(chunk), max_n, max_n, dtype=torch.float32)
        for k, it in enumerate(chunk):
            n = it.same_ring.size(0)
            sr[k, :n, :n] = it.same_ring
        batch["same_ring"] = sr
        batches.append(batch)

    print(f"  preprocessed {len(items)} graphs into {len(batches)} batches "
          f"in {time.time()-t0:.1f}s")
    return batches, len(items)


# ---------------------------------------------------------------------------
# Hooks.
# ---------------------------------------------------------------------------


def _zero_hook(_module, _inp, output):
    return torch.zeros_like(output)


class RingHook:
    """Forward hook on GraphAttnBias adding the ring bias broadcast over heads.

    The bias is alpha * same_ring(i, j) placed into the (1:, 1:) sub-block
    of the (N+1, N+1) attention bias. The graph-token row/column gets 0.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha

    def __call__(self, module, inputs, output):
        batch = inputs[0]
        if "same_ring" not in batch:
            return output  # safety net; should never hit if loader is right
        sr = batch["same_ring"]              # (B, N, N)
        B, N, _ = sr.shape
        bias = torch.zeros(B, N + 1, N + 1, dtype=output.dtype, device=output.device)
        bias[:, 1:, 1:] = self.alpha * sr
        return output + bias.unsqueeze(1)    # broadcast over heads


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

    if cfg.use_ring:
        handles.append(ab.register_forward_hook(RingHook(cfg.alpha)))

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
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ring bias strength. Same-ring pairs get +alpha "
                             "added to attention logits.")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.ckpt} ...")
    model = load_pretrained(args.ckpt).to(device)
    print("Loaded.\n")

    print("Materialising ring-aware validation batches (one-time cost) ...")
    batches, n_items = _build_ring_aware_batches(args.batch_size, args.max_graphs)
    print()

    ring_kw = dict(use_ring=True, alpha=args.alpha)
    configs = [
        Config(name="baseline"),
        Config(name="ring only",
               use_spatial=False, use_edge=False, use_virtual_distance=False,
               use_centrality=False, **ring_kw),
        Config(name="ring + edge",
               use_spatial=False, use_virtual_distance=False, use_centrality=False,
               **ring_kw),
        Config(name="all + ring", **ring_kw),
    ]

    print(f"Ring-aware (PAGTN-flavored) extension")
    print(f"  N = {n_items} graphs   alpha = {args.alpha}\n")
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
