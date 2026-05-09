"""Pure-NumPy / PyTorch reimplementation of the data pipeline.

Replaces the Cython `algos.pyx` (Floyd-Warshall + multi-hop edge features) and
the fairseq collator with a minimal version that produces identical tensors.
"""

from __future__ import annotations

import sys
from typing import List, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Token-id offset (same convention as the reference: feature i in column i
# gets shifted by 1 + i*offset so we can use a single embedding table).
# ---------------------------------------------------------------------------

UNREACHABLE = 510  # reference uses 510 to mark "no path / too far".


def convert_to_single_emb(x: torch.Tensor, offset: int = 512) -> torch.Tensor:
    feat_num = x.size(1) if x.dim() > 1 else 1
    feat_offset = 1 + torch.arange(0, feat_num * offset, offset, dtype=torch.long)
    return x + feat_offset


# ---------------------------------------------------------------------------
# Floyd-Warshall + multi-hop edge feature extraction (pure NumPy).
# ---------------------------------------------------------------------------


def floyd_warshall(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """All-pairs shortest paths on an unweighted boolean/int adjacency matrix.

    Returns (dist, path) where path[i, j] is the highest-index intermediate
    vertex on the shortest i->j path (or -1 if direct, or 510 if unreachable).
    """
    n = adj.shape[0]
    M = adj.astype(np.int64, copy=True)
    path = -np.ones((n, n), dtype=np.int64)

    # Initialize: 0 on diagonal, UNREACHABLE for missing edges.
    np.fill_diagonal(M, 0)
    M[(M == 0) & (np.eye(n, dtype=bool) == 0)] = UNREACHABLE

    for k in range(n):
        new_dist = M[:, k:k + 1] + M[k:k + 1, :]
        better = new_dist < M
        path = np.where(better, k, path)
        M = np.where(better, new_dist, M)

    unreachable = M >= UNREACHABLE
    M[unreachable] = UNREACHABLE
    path[unreachable] = UNREACHABLE
    return M, path


def _get_all_edges(path: np.ndarray, i: int, j: int) -> List[int]:
    """Return the intermediate vertices on the shortest i->j path, in order."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))

    def rec(a: int, b: int) -> List[int]:
        k = int(path[a, b])
        if k == -1:
            return []
        return rec(a, k) + [k] + rec(k, b)

    return rec(i, j)


def gen_edge_input(max_dist: int, path: np.ndarray, edge_feat: np.ndarray) -> np.ndarray:
    """For every (i, j) pair, list the edge features along the shortest path.

    Output shape: (n, n, max_dist, F). Padded with -1 for shorter / absent paths.
    """
    n = path.shape[0]
    F = edge_feat.shape[-1]
    out = -np.ones((n, n, max_dist, F), dtype=np.int64)
    if n == 0:
        return out

    for i in range(n):
        for j in range(n):
            if i == j or path[i, j] == UNREACHABLE:
                continue
            seq = [i] + _get_all_edges(path, i, j) + [j]
            steps = min(len(seq) - 1, max_dist)
            for k in range(steps):
                out[i, j, k, :] = edge_feat[seq[k], seq[k + 1], :]
    return out


# ---------------------------------------------------------------------------
# Per-item preprocessing. Operates on a torch_geometric.data.Data-like object
# with `.x`, `.edge_index`, `.edge_attr` (and optionally `.y`).
# ---------------------------------------------------------------------------


def preprocess_item(item):
    """Attach all Graphormer fields to `item` in-place and return it."""
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True

    if edge_attr.dim() == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1

    sp_dist, sp_path = floyd_warshall(adj.numpy())
    max_dist = int(sp_dist.max()) if N > 0 else 0
    edge_input = gen_edge_input(max_dist, sp_path, attn_edge_type.numpy())

    item.x = x
    item.attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = torch.from_numpy(sp_dist).long()
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # undirected.
    item.edge_input = torch.from_numpy(edge_input).long()
    return item


# ---------------------------------------------------------------------------
# Collator (pads variable-size graphs into a single batched dict).
# ---------------------------------------------------------------------------


def _pad_1d(x: torch.Tensor, padlen: int) -> torch.Tensor:
    x = x + 1
    out = x.new_zeros([padlen], dtype=x.dtype)
    out[: x.size(0)] = x
    return out.unsqueeze(0)


def _pad_2d(x: torch.Tensor, padlen: int) -> torch.Tensor:
    x = x + 1
    out = x.new_zeros([padlen, x.size(1)], dtype=x.dtype)
    out[: x.size(0)] = x
    return out.unsqueeze(0)


def _pad_attn_bias(x: torch.Tensor, padlen: int) -> torch.Tensor:
    out = x.new_full([padlen, padlen], float("-inf"))
    n = x.size(0)
    out[:n, :n] = x
    out[n:, :n] = 0  # per the reference
    return out.unsqueeze(0)


def _pad_edge_type(x: torch.Tensor, padlen: int) -> torch.Tensor:
    out = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
    n = x.size(0)
    out[:n, :n] = x
    return out.unsqueeze(0)


def _pad_spatial(x: torch.Tensor, padlen: int) -> torch.Tensor:
    x = x + 1
    out = x.new_zeros([padlen, padlen], dtype=x.dtype)
    n = x.size(0)
    out[:n, :n] = x
    return out.unsqueeze(0)


def _pad_3d(x: torch.Tensor, p1: int, p2: int, p3: int) -> torch.Tensor:
    x = x + 1
    out = x.new_zeros([p1, p2, p3, x.size(-1)], dtype=x.dtype)
    a, b, c, _ = x.shape
    out[:a, :b, :c, :] = x
    return out.unsqueeze(0)


def collate(items: Sequence, max_node: int = 512, multi_hop_max_dist: int = 20, spatial_pos_max: int = 20) -> dict:
    items = [it for it in items if it is not None and it.x.size(0) <= max_node]
    if not items:
        raise ValueError("collate received no items below max_node")

    # Mask far-away nodes via attn_bias = -inf (reference behavior).
    for it in items:
        it.attn_bias[1:, 1:][it.spatial_pos >= spatial_pos_max] = float("-inf")

    max_n = max(it.x.size(0) for it in items)
    max_d = max(it.edge_input[:, :, :multi_hop_max_dist, :].size(-2) for it in items)

    batch = {
        "idx": torch.tensor([getattr(it, "idx", i) for i, it in enumerate(items)], dtype=torch.long),
        "x": torch.cat([_pad_2d(it.x, max_n) for it in items]),
        "attn_bias": torch.cat([_pad_attn_bias(it.attn_bias, max_n + 1) for it in items]),
        "attn_edge_type": torch.cat([_pad_edge_type(it.attn_edge_type, max_n) for it in items]),
        "spatial_pos": torch.cat([_pad_spatial(it.spatial_pos, max_n) for it in items]),
        "in_degree": torch.cat([_pad_1d(it.in_degree, max_n) for it in items]),
        "edge_input": torch.cat(
            [_pad_3d(it.edge_input[:, :, :multi_hop_max_dist, :], max_n, max_n, max_d) for it in items]
        ),
    }
    batch["out_degree"] = batch["in_degree"]  # undirected
    if all(hasattr(it, "y") and it.y is not None for it in items):
        batch["y"] = torch.cat([it.y.view(-1) for it in items])
    return batch
