"""
End-to-end Graphormer demo, top to bottom, with notes on what each step does.

How to run, from inside the `mini_graphormer/` folder (where this file lives):

    python run_demo.py                                  # random init, no download
    python run_demo.py --ckpt pcqm4mv2_graphormer_base  # download the MS weights
    python run_demo.py --ckpt path/to/ckpt.pt           # use a local checkpoint

The script walks through the full pipeline:
    raw graph -> preprocess -> collate -> model -> prediction
and finally runs three ablations (zero-out spatial / centrality / edge biases)
to show how to probe the model for an ablation study.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

# This script lives inside the `mini_graphormer/` package, but the package's
# own files use relative imports (e.g. `from .model import ...`). To make
# `python run_demo.py` work no matter the cwd, we put the *parent* directory
# on sys.path and import `mini_graphormer` as a proper package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mini_graphormer import (  # noqa: E402  (import after sys.path tweak)
    MiniGraphormer,
    collate,
    convert_to_single_emb,
    graphormer_base_pcqm4mv2_config,
    load_pretrained,
    preprocess_item,
)


# ---------------------------------------------------------------------------
# Step 1 — fabricate a "molecule"
# ---------------------------------------------------------------------------
# Graphormer was trained on OGB's PCQM4Mv2, where every sample is a
# torch_geometric.data.Data object with:
#   .x          (N, 9)  long  — 9 integer atom features per node
#                              (atomic number, chirality, degree, ...)
#   .edge_index (2, E)  long  — COO connectivity, both directions
#   .edge_attr  (E, 3)  long  — 3 integer bond features per edge
#   .y          (1,)    float — HOMO–LUMO gap (the regression target)
#
# We don't need the real dataset to demo the pipeline — duck-typing with a
# SimpleNamespace is enough, because preprocess_item only touches those four
# attributes.
# ---------------------------------------------------------------------------


def make_fake_molecule(n_nodes: int = 8, seed: int = 42) -> SimpleNamespace:
    g = torch.Generator().manual_seed(seed)

    # 9 integer features per node, ranges roughly matching OGB atom features.
    x = torch.randint(0, 119, (n_nodes, 9), generator=g)

    # Build a connected graph: a chain 0-1-2-...-(n-1), bidirectional, plus a
    # few random extra "bonds" to make it non-trivial.
    src = torch.arange(n_nodes - 1)
    dst = torch.arange(1, n_nodes)
    extra = torch.randint(0, n_nodes, (n_nodes // 2, 2), generator=g)
    edge_src = torch.cat([src, dst, extra[:, 0]])  # forward + reverse + extras
    edge_dst = torch.cat([dst, src, extra[:, 1]])
    edge_index = torch.stack([edge_src, edge_dst], dim=0)

    # 3 integer features per edge.
    edge_attr = torch.randint(0, 5, (edge_index.size(1), 3), generator=g)

    # Placeholder regression target so collate() finds a `.y` attribute.
    y = torch.zeros(1)

    return SimpleNamespace(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ---------------------------------------------------------------------------
# Step 2 — preprocess: turn a raw graph into Graphormer's tensor inputs
# ---------------------------------------------------------------------------
# preprocess_item(item) attaches these new attributes IN PLACE:
#
#   item.x              (N, 9)  long  — features, but offset by
#                                       convert_to_single_emb so each column
#                                       lives in its own bucket of the shared
#                                       atom_encoder embedding table
#                                       (column i adds 1 + i*512).
#   item.attn_bias      (N+1, N+1) float — the additive attention mask. Starts
#                                       at zero; the +1 row/col is the
#                                       [graph]-token slot.
#   item.attn_edge_type (N, N, 3) long  — bond-feature ids on the (src,dst)
#                                       slot, also offset+1 (so 0 = "no edge").
#   item.spatial_pos    (N, N)    long  — Floyd–Warshall shortest-path lengths
#                                       between every pair of nodes. The
#                                       paper's `phi(i,j)`. Unreachable = 510.
#   item.in_degree      (N,)      long  — node degrees (used by the centrality
#   item.out_degree     (N,)      long    encoding; equal here, undirected).
#   item.edge_input     (N, N, max_dist, 3) long
#                                      — for each (i,j) pair, the bond features
#                                        of the edges along the shortest i->j
#                                        path. The "multi-hop edge encoding"
#                                        used to build the c_ij bias.
#
# Everything past edge_index/edge_attr is precomputed here once per graph,
# because Floyd–Warshall is O(N^3) and you don't want it inside the training
# loop. This is what algos.pyx in the original repo does in Cython; we do it
# in NumPy.
# ---------------------------------------------------------------------------


def show_preprocessing(item: SimpleNamespace) -> SimpleNamespace:
    raw_n_edges = item.edge_index.size(1)
    print(f"  raw molecule: {item.x.size(0)} nodes, {raw_n_edges} directed edges")

    # Demo the offset trick on the first row so the embedding-table layout is
    # obvious. Column i is shifted into [1 + i*512, (i+1)*512].
    print(f"  raw    x[0] = {item.x[0].tolist()}")
    print(f"  offset x[0] = {convert_to_single_emb(item.x[:1])[0].tolist()}")

    item = preprocess_item(item)

    # Sanity-check Floyd–Warshall by recomputing it directly. Diameter of the
    # graph = the max finite shortest-path distance.
    sp = item.spatial_pos.numpy()
    finite = sp[sp < 510]
    print(f"  shortest paths: min={finite.min()}, max(diameter)={finite.max()}")
    print(f"  multi-hop edge_input tensor shape = {tuple(item.edge_input.shape)}")
    return item


# ---------------------------------------------------------------------------
# Step 3 — collate: stack variable-size graphs into one padded batch
# ---------------------------------------------------------------------------
# collate(items) pads every per-graph tensor up to (max_n) or (max_n+1) and
# stacks them along a new batch axis. Two padding tricks worth knowing:
#
#  * Token ids in `x` and `spatial_pos` get +1 added before zero-padding, so
#    `0` becomes the unambiguous PAD id. The atom_encoder uses padding_idx=0
#    to make sure padded rows produce zero embeddings.
#
#  * `attn_bias` is padded with -inf (so real tokens can't attend to padding).
#    On top of that, every (i,j) with spatial_pos >= spatial_pos_max is also
#    set to -inf — the model only attends within a bounded receptive field.
# ---------------------------------------------------------------------------


def show_collation(items):
    batch = collate(items)
    print("  batch tensors:")
    for k, v in batch.items():
        print(f"    {k:18s} {tuple(v.shape)}  {v.dtype}")
    return batch


# ---------------------------------------------------------------------------
# Step 4 — build (or load) the model
# ---------------------------------------------------------------------------
# The released checkpoint `pcqm4mv2_graphormer_base` is:
#   12 transformer blocks, embed_dim=768, ffn_dim=768, 32 attention heads,
#   GeLU activation, post-LN, ~48M parameters.
#
# load_pretrained(name_or_path) does three things:
#   1. Build a MiniGraphormer with the matching config.
#   2. torch.hub-download (or open locally) the .pt file.
#   3. Strict-match the state_dict — every parameter in the model has the
#      exact same name as in Microsoft's checkpoint, so this just works.
# ---------------------------------------------------------------------------


def build_model(ckpt: str | None) -> MiniGraphormer:
    if ckpt is None:
        cfg = graphormer_base_pcqm4mv2_config()
        model = MiniGraphormer(cfg)
        msg = "random-init (no checkpoint)"
    else:
        model = load_pretrained(ckpt)
        msg = f"checkpoint='{ckpt}'"
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  built MiniGraphormer ({n_params:.1f}M params, {msg})")
    return model.eval()  # turn off all dropout for inference


# ---------------------------------------------------------------------------
# Step 5 — forward pass and graph-level prediction
# ---------------------------------------------------------------------------
# Internally:
#   GraphNodeFeature  -> sum atom features + add (in_degree, out_degree)
#                        embeddings ("centrality encoding") + prepend a
#                        learned [graph] token row.
#   GraphAttnBias     -> additive (B, H, N+1, N+1) bias built from
#                          spatial_pos_encoder(spatial_pos)         (b_phi)
#                        + multi-hop edge_encoder over edge_input    (c_ij)
#                        + a learned bias on the [graph] row/col.
#   12 transformer layers, each: self-attn(+bias) -> add+LN -> FFN -> add+LN.
#   Final LN + Linear head.
#
# model.predict(batch) returns the [graph]-token row: (B, num_classes).
# The full output (B, N+1, num_classes) is `model(batch)`.
# ---------------------------------------------------------------------------


def show_forward(model: MiniGraphormer, batch: dict) -> torch.Tensor:
    with torch.no_grad():
        full = model(batch)               # (B, N+1, num_classes)
        graph_pred = model.predict(batch) # (B, num_classes) — same as full[:, 0]
    print(f"  full output shape         : {tuple(full.shape)}")
    print(f"  graph-token prediction    : {graph_pred.squeeze().tolist()}")
    return graph_pred


# ---------------------------------------------------------------------------
# Step 6 — ablations via forward hooks
# ---------------------------------------------------------------------------
# The three Graphormer-specific inductive biases each live in a dedicated
# nn.Module, so we can disable any one of them by registering a hook that
# replaces its output with zeros. This is non-destructive: the weights aren't
# touched, the hook can be removed, and the same model instance can be reused
# across ablations.
#
#   - Spatial bias    : graph_attn_bias.spatial_pos_encoder
#   - Centrality enc. : graph_node_feature.in_degree_encoder
#                     + graph_node_feature.out_degree_encoder
#   - Edge bias (c_ij): graph_attn_bias.edge_encoder
# ---------------------------------------------------------------------------


def _zero_hook(_module, _inp, output):
    return torch.zeros_like(output)


def run_ablation(model: MiniGraphormer, batch: dict, name: str, modules):
    handles = [m.register_forward_hook(_zero_hook) for m in modules]
    try:
        with torch.no_grad():
            pred = model.predict(batch).squeeze().item()
    finally:
        # Always remove hooks, even if forward raised — otherwise they
        # silently corrupt every subsequent forward pass.
        for h in handles:
            h.remove()
    print(f"  {name:18s} -> {pred:+.6f}")
    return pred


def show_ablations(model: MiniGraphormer, batch: dict, baseline: float):
    enc = model.encoder.graph_encoder

    print(f"  baseline           -> {baseline:+.6f}")
    run_ablation(model, batch, "no_spatial",     [enc.graph_attn_bias.spatial_pos_encoder])
    run_ablation(model, batch, "no_centrality",  [enc.graph_node_feature.in_degree_encoder,
                                                  enc.graph_node_feature.out_degree_encoder])
    run_ablation(model, batch, "no_edge",        [enc.graph_attn_bias.edge_encoder])


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", default=None,
        help="Pretrained name (e.g. pcqm4mv2_graphormer_base) or local .pt path. "
             "If omitted, runs with random init.")
    parser.add_argument("--n-nodes", type=int, default=8,
                        help="Size of the synthetic molecule.")
    args = parser.parse_args()

    print("\n[1/6] Build a fake molecule -----------------------------------")
    raw = make_fake_molecule(n_nodes=args.n_nodes)

    print("\n[2/6] Preprocess (Floyd-Warshall, multi-hop edges, degrees) ---")
    item = show_preprocessing(raw)

    print("\n[3/6] Collate into a batch ------------------------------------")
    batch = show_collation([item])

    print("\n[4/6] Build / load the model ----------------------------------")
    model = build_model(args.ckpt)

    print("\n[5/6] Forward pass --------------------------------------------")
    baseline = show_forward(model, batch).squeeze().item()

    print("\n[6/6] Ablations via forward hooks -----------------------------")
    show_ablations(model, batch, baseline)

    print("\nDone.")


if __name__ == "__main__":
    main()
