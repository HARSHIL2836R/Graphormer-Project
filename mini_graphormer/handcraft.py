"""
Per-molecule comparison of baseline vs each extension.

For each handcrafted molecule, we run the loaded Graphormer four times:
    baseline       — no extension
    + laplacian    — Laplacian PE bias only
    + rwpe         — Random-walk PE bias only
    + ring         — Same-ring binary bias only

The three extensions are NEVER combined inside a single forward pass — each
forward attaches at most one hook on `GraphAttnBias`, then removes it before
the next forward. This script's purpose is the per-molecule story for the
project report: show that the chemically-motivated extension for each
molecule shifts the prediction toward the experimental HOMO–LUMO gap, while
the other extensions either do nothing or shift it differently.

Usage:
    python handcraft.py --ckpt checkpoint_best_pcqm4mv2.pt
    python handcraft.py --ckpt ... --alpha-laplacian 0.5 --alpha-rwpe 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# Make the package importable from any cwd.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from mini_graphormer import collate, preprocess_item       # noqa: E402
from mini_graphormer.pretrain import load_pretrained       # noqa: E402
from extend_spectral import LaplacianHook                  # noqa: E402
from extend_rwpe import RWPEHook                           # noqa: E402
from extend_ring import RingHook                           # noqa: E402


# ---------------------------------------------------------------------------
# Handcrafted test set. Each molecule pairs with the extension we predict
# (a-priori, from graph-structure reasoning) should help most.
#
# Experimental gaps are B3LYP HOMO–LUMO gaps (eV) from public databases /
# literature; PCQM4Mv2 was trained on B3LYP-style targets. Numbers should
# be treated as ~±0.2 eV approximate references, not exact ground truth.
# ---------------------------------------------------------------------------


MOLECULES = [
    # (name,           SMILES,                              recommended,   exp_gap)
    ("naphthalene",    "c1ccc2ccccc2c1",                    "rwpe",        4.4),
    ("azulene",        "c1ccc2cccc-2cc1",                   "rwpe",        1.7),
    ("anthracene",     "c1ccc2cc3ccccc3cc2c1",              "laplacian",   3.3),
    ("phenanthrene",   "c1ccc2ccc3ccccc3c2c1",              "laplacian",   3.9),
    ("pyrene",         "c1cc2ccc3cccc4ccc(c1)c2c34",        "ring",        3.3),
]


# ---------------------------------------------------------------------------
# SMILES → preprocessed item, including the same_ring matrix needed by the
# RingHook (the Laplacian and RWPE hooks ignore it; including it is harmless).
# ---------------------------------------------------------------------------


def smiles_to_item(smiles: str, label: float) -> SimpleNamespace:
    from ogb.utils.mol import smiles2graph
    from rdkit import Chem

    g = smiles2graph(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"rdkit could not parse SMILES: {smiles}")
    rings = Chem.GetSymmSSSR(mol)

    N = mol.GetNumAtoms()
    same_ring = torch.zeros(N, N, dtype=torch.float32)
    for ring in rings:
        atoms = list(ring)
        for ii in atoms:
            for jj in atoms:
                if ii != jj:
                    same_ring[ii, jj] = 1.0

    item = SimpleNamespace(
        x=torch.tensor(g["node_feat"], dtype=torch.long),
        edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
        edge_attr=torch.tensor(g["edge_feat"], dtype=torch.long),
        y=torch.tensor([float(label)], dtype=torch.float),
        idx=0,
        same_ring=same_ring,
    )
    return preprocess_item(item)


def item_to_batch(item: SimpleNamespace) -> dict:
    """Wrap a single preprocessed item into a 1-element batch carrying same_ring."""
    batch = collate([item], max_node=512, multi_hop_max_dist=5, spatial_pos_max=1024)
    max_n = batch["x"].size(1)
    sr_padded = torch.zeros(1, max_n, max_n, dtype=torch.float32)
    n = item.same_ring.size(0)
    sr_padded[0, :n, :n] = item.same_ring
    batch["same_ring"] = sr_padded
    return batch


# ---------------------------------------------------------------------------
# Single forward pass with optional hook on GraphAttnBias.
# Each call attaches at most one hook and removes it before returning, so
# extensions can never accidentally compose.
# ---------------------------------------------------------------------------


def predict_with(model, batch: dict, hook_callable=None) -> float:
    enc = model.encoder.graph_encoder
    ab = enc.graph_attn_bias
    handle = ab.register_forward_hook(hook_callable) if hook_callable is not None else None
    try:
        with torch.no_grad():
            return model.predict(batch).item()
    finally:
        if handle is not None:
            handle.remove()


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True,
                        help="Local .pt path or PRETRAINED_URLS key.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alpha-laplacian", type=float, default=1.0)
    parser.add_argument("--k-laplacian", type=int, default=8)
    parser.add_argument("--weight-laplacian", default="uniform",
                        choices=["uniform", "inv", "heat"])
    parser.add_argument("--alpha-rwpe", type=float, default=0.5)
    parser.add_argument("--K-rwpe", type=int, default=4)
    parser.add_argument("--alpha-ring", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading model from {args.ckpt} ...")
    model = load_pretrained(args.ckpt).to(device)
    print("Loaded.\n")

    laplacian_hook = LaplacianHook(args.alpha_laplacian, args.k_laplacian, args.weight_laplacian)
    rwpe_hook = RWPEHook(args.alpha_rwpe, args.K_rwpe)
    ring_hook = RingHook(args.alpha_ring)

    print(f"  alphas: laplacian={args.alpha_laplacian}  rwpe={args.alpha_rwpe}  "
          f"ring={args.alpha_ring}")
    print()
    header = (f"  {'molecule':14s} {'exp.':>7s} {'baseline':>10s} "
              f"{'+laplac.':>10s} {'+rwpe':>10s} {'+ring':>10s}  "
              f"{'closest':>10s}  {'a-priori':>10s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for name, smiles, recommended, exp_gap in MOLECULES:
        try:
            item = smiles_to_item(smiles, exp_gap)
        except Exception as e:
            print(f"  {name:14s} SKIPPED ({e})")
            continue

        batch = item_to_batch(item)
        batch = {k: v.to(device) for k, v in batch.items()}

        pred_base = predict_with(model, batch, hook_callable=None)
        pred_lap = predict_with(model, batch, hook_callable=laplacian_hook)
        pred_rw = predict_with(model, batch, hook_callable=rwpe_hook)
        pred_ring = predict_with(model, batch, hook_callable=ring_hook)

        # Which configuration's prediction is closest to the experimental gap?
        candidates = [
            ("baseline", pred_base),
            ("laplacian", pred_lap),
            ("rwpe", pred_rw),
            ("ring", pred_ring),
        ]
        best_name, _ = min(candidates, key=lambda c: abs(c[1] - exp_gap))

        print(f"  {name:14s} {exp_gap:7.2f} {pred_base:10.4f} "
              f"{pred_lap:10.4f} {pred_rw:10.4f} {pred_ring:10.4f}  "
              f"{best_name:>10s}  {recommended:>10s}")

    print()
    print("  exp.       = literature B3LYP HOMO-LUMO gap (eV)")
    print("  baseline   = pretrained Graphormer, no extension")
    print("  +X         = same model + extension X applied (no other extensions active)")
    print("  closest    = which configuration's prediction is closest to exp.")
    print("  a-priori   = the extension we predicted should help most for this molecule")


if __name__ == "__main__":
    main()
