"""End-to-end smoke test + ablation recipes.

Run from the project root:
    python -m mini_graphormer.example                         # synthetic graph, random init
    python -m mini_graphormer.example --ckpt pcqm4mv2_graphormer_base
    python -m mini_graphormer.example --ckpt /path/to/checkpoint_best_pcqm4mv2.pt
"""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch

from .data import collate, preprocess_item
from .model import MiniGraphormer, graphormer_base_pcqm4mv2_config
from .pretrain import load_pretrained


# ---------------------------------------------------------------------------
# A tiny synthetic "Data" object mimicking torch_geometric.data.Data so the
# example runs without any extra dependencies.
# ---------------------------------------------------------------------------


def synthetic_molecule(n_nodes: int = 8, n_atom_feats: int = 9, n_bond_feats: int = 3, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 119, (n_nodes, n_atom_feats), generator=g)

    # Random simple connected graph: chain + a few extra bonds.
    src = torch.arange(n_nodes - 1)
    dst = torch.arange(1, n_nodes)
    extra = torch.randint(0, n_nodes, (n_nodes // 2, 2), generator=g)
    edge_src = torch.cat([src, dst, extra[:, 0]])
    edge_dst = torch.cat([dst, src, extra[:, 1]])
    edge_index = torch.stack([edge_src, edge_dst], dim=0)

    edge_attr = torch.randint(0, 5, (edge_index.size(1), n_bond_feats), generator=g)
    y = torch.zeros(1)
    return SimpleNamespace(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# ---------------------------------------------------------------------------
# Ablation hooks. Each returns a list of removable handles.
# ---------------------------------------------------------------------------


def ablate_spatial_bias(model: MiniGraphormer):
    """Zero out the spatial-distance bias (b_phi in the paper)."""
    enc = model.encoder.graph_encoder.graph_attn_bias.spatial_pos_encoder

    def _zero(_module, _inp, out):
        return torch.zeros_like(out)

    return [enc.register_forward_hook(_zero)]


def ablate_centrality_encoding(model: MiniGraphormer):
    """Drop the in/out degree embeddings added to node features."""
    nf = model.encoder.graph_encoder.graph_node_feature
    handles = []

    def _zero(_module, _inp, out):
        return torch.zeros_like(out)

    handles.append(nf.in_degree_encoder.register_forward_hook(_zero))
    handles.append(nf.out_degree_encoder.register_forward_hook(_zero))
    return handles


def ablate_edge_encoding(model: MiniGraphormer):
    """Drop the multi-hop edge bias term."""
    enc = model.encoder.graph_encoder.graph_attn_bias.edge_encoder

    def _zero(_module, _inp, out):
        return torch.zeros_like(out)

    return [enc.register_forward_hook(_zero)]


ABLATIONS = {
    "none": lambda m: [],
    "no_spatial": ablate_spatial_bias,
    "no_centrality": ablate_centrality_encoding,
    "no_edge": ablate_edge_encoding,
}


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None,
                        help="Pretrained name (e.g. pcqm4mv2_graphormer_base) or local .pt path. "
                             "If omitted, runs with random init.")
    parser.add_argument("--ablation", default="none", choices=list(ABLATIONS))
    parser.add_argument("--n-nodes", type=int, default=8)
    args = parser.parse_args()

    if args.ckpt:
        model = load_pretrained(args.ckpt)
        print(f"[mini_graphormer] loaded {args.ckpt}")
    else:
        model = MiniGraphormer(graphormer_base_pcqm4mv2_config())
        print("[mini_graphormer] random init (no checkpoint)")

    item = preprocess_item(synthetic_molecule(n_nodes=args.n_nodes))
    batch = collate([item])

    handles = ABLATIONS[args.ablation](model)
    try:
        with torch.no_grad():
            pred = model.predict(batch)
        print(f"ablation={args.ablation:14s}  graph token output: {pred.squeeze().tolist()}")
    finally:
        for h in handles:
            h.remove()


if __name__ == "__main__":
    main()
