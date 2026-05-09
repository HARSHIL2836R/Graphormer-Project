"""Minimal Graphormer reimplementation.

Mirrors the parameter naming of Microsoft's reference implementation so that
the official checkpoints load directly via `load_state_dict`. No fairseq, no
custom CUDA, no Cython.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class GraphormerConfig:
    # Vocabularies / token tables (must match preprocessing offsets).
    num_atoms: int = 512 * 9
    num_edges: int = 512 * 3
    num_in_degree: int = 512
    num_out_degree: int = 512
    num_spatial: int = 512
    num_edge_dis: int = 128
    edge_type: str = "multi_hop"
    multi_hop_max_dist: int = 5

    # Transformer.
    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 768
    encoder_attention_heads: int = 32
    activation_fn: str = "gelu"

    # Dropout (forward only — inference uses model.eval()).
    dropout: float = 0.0
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1

    # LayerNorm placement.
    encoder_normalize_before: bool = True   # input embedding LN
    pre_layernorm: bool = False             # pre-LN inside each block

    # Output head.
    num_classes: int = 1


def graphormer_base_pcqm4mv2_config(num_classes: int = 1) -> GraphormerConfig:
    """Hyperparameters used by the released `pcqm4mv2_graphormer_base` checkpoint."""
    return GraphormerConfig(num_classes=num_classes)


# ---------------------------------------------------------------------------
# Multi-head attention (matches MS naming: q_proj/k_proj/v_proj/out_proj).
# ---------------------------------------------------------------------------


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_dropout = dropout

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,                          # (T, B, C)
        attn_bias: Optional[torch.Tensor] = None, # (B, H, T, T)
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T) bool, True = pad
    ) -> torch.Tensor:
        T, B, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (T, B, C) -> (B*H, T, D)
        def split(t: torch.Tensor) -> torch.Tensor:
            return t.contiguous().view(T, B * H, D).transpose(0, 1)

        q, k, v = split(q), split(k), split(v)

        attn = torch.bmm(q, k.transpose(1, 2))  # (B*H, T, T)
        if attn_bias is not None:
            attn = attn + attn_bias.reshape(B * H, T, T)
        if key_padding_mask is not None:
            attn = attn.view(B, H, T, T).masked_fill(
                key_padding_mask.view(B, 1, 1, T).to(torch.bool),
                float("-inf"),
            ).view(B * H, T, T)

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)

        out = torch.bmm(attn, v)                       # (B*H, T, D)
        out = out.transpose(0, 1).contiguous().view(T, B, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Node feature & attention bias modules.
# ---------------------------------------------------------------------------


class GraphNodeFeature(nn.Module):
    def __init__(self, num_atoms: int, num_in_degree: int, num_out_degree: int, hidden_dim: int):
        super().__init__()
        # +1 for graph token slot kept aligned with the reference.
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch["x"]                # (B, N, F)
        in_deg = batch["in_degree"]   # (B, N)
        out_deg = batch["out_degree"] # (B, N)
        B = x.size(0)

        node_feature = self.atom_encoder(x).sum(dim=-2)  # (B, N, C)
        node_feature = node_feature + self.in_degree_encoder(in_deg) + self.out_degree_encoder(out_deg)

        graph_token = self.graph_token.weight.unsqueeze(0).expand(B, 1, -1)
        return torch.cat([graph_token, node_feature], dim=1)  # (B, N+1, C)


class GraphAttnBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        if edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

    def forward(self, batch: dict) -> torch.Tensor:
        attn_bias = batch["attn_bias"]            # (B, N+1, N+1)
        spatial_pos = batch["spatial_pos"]        # (B, N, N)
        x = batch["x"]
        edge_input = batch["edge_input"]          # (B, N, N, D, F)
        attn_edge_type = batch["attn_edge_type"]  # (B, N, N, F)

        B, N = x.size(0), x.size(1)
        H = self.num_heads

        # (B, H, N+1, N+1)
        bias = attn_bias.unsqueeze(1).repeat(1, H, 1, 1)

        spatial = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + spatial

        # virtual graph token distance (broadcast to row 0 and column 0).
        t = self.graph_token_virtual_distance.weight.view(1, H, 1)
        bias[:, :, 1:, 0] = bias[:, :, 1:, 0] + t
        bias[:, :, 0, :] = bias[:, :, 0, :] + t

        if self.edge_type == "multi_hop":
            sp = spatial_pos.clone()
            sp[sp == 0] = 1
            sp = torch.where(sp > 1, sp - 1, sp)
            if self.multi_hop_max_dist > 0:
                sp = sp.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]

            ein = self.edge_encoder(edge_input).mean(-2)        # (B, N, N, D, H)
            D = ein.size(-2)
            flat = ein.permute(3, 0, 1, 2, 4).reshape(D, -1, H) # (D, B*N*N, H)
            flat = torch.bmm(
                flat,
                self.edge_dis_encoder.weight.reshape(-1, H, H)[:D],
            )
            ein = flat.reshape(D, B, N, N, H).permute(1, 2, 3, 0, 4)
            ein = (ein.sum(-2) / sp.float().unsqueeze(-1)).permute(0, 3, 1, 2)
        else:
            ein = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        bias[:, :, 1:, 1:] = bias[:, :, 1:, 1:] + ein
        bias = bias + attn_bias.unsqueeze(1)  # mirror reference: re-add the raw mask.
        return bias


# ---------------------------------------------------------------------------
# Encoder block.
# ---------------------------------------------------------------------------


def _act(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"unsupported activation: {name}")


class GraphormerLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        activation_fn: str,
        pre_layernorm: bool,
    ):
        super().__init__()
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.pre_layernorm = pre_layernorm
        self.activation_fn = _act(activation_fn)

        self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, self_attn_bias=None, self_attn_padding_mask=None):
        # ---- self-attention sublayer ----
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, attn_bias=self_attn_bias, key_padding_mask=self_attn_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        # ---- FFN sublayer ----
        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# Graph encoder (stack + input embeddings).
# ---------------------------------------------------------------------------


class GraphormerGraphEncoder(nn.Module):
    def __init__(self, cfg: GraphormerConfig):
        super().__init__()
        self.cfg = cfg
        self.dropout = cfg.dropout

        self.graph_node_feature = GraphNodeFeature(
            num_atoms=cfg.num_atoms,
            num_in_degree=cfg.num_in_degree,
            num_out_degree=cfg.num_out_degree,
            hidden_dim=cfg.encoder_embed_dim,
        )
        self.graph_attn_bias = GraphAttnBias(
            num_heads=cfg.encoder_attention_heads,
            num_edges=cfg.num_edges,
            num_spatial=cfg.num_spatial,
            num_edge_dis=cfg.num_edge_dis,
            edge_type=cfg.edge_type,
            multi_hop_max_dist=cfg.multi_hop_max_dist,
        )

        if cfg.encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(cfg.encoder_embed_dim)
        else:
            self.emb_layer_norm = None

        if cfg.pre_layernorm:
            self.final_layer_norm = nn.LayerNorm(cfg.encoder_embed_dim)

        self.layers = nn.ModuleList(
            [
                GraphormerLayer(
                    embed_dim=cfg.encoder_embed_dim,
                    ffn_dim=cfg.encoder_ffn_embed_dim,
                    num_heads=cfg.encoder_attention_heads,
                    dropout=cfg.dropout,
                    attention_dropout=cfg.attention_dropout,
                    activation_dropout=cfg.activation_dropout,
                    activation_fn=cfg.activation_fn,
                    pre_layernorm=cfg.pre_layernorm,
                )
                for _ in range(cfg.encoder_layers)
            ]
        )

    def forward(
        self,
        batch: dict,
        return_inner_states: bool = False,
    ):
        x = batch["x"]
        B, N = x.size(0), x.size(1)

        # padding mask: a node is pad iff every feature is 0 (since real values
        # are >= 1 after the offset). Prepend a False slot for the graph token.
        padding_mask = x[:, :, 0].eq(0)  # (B, N)
        cls_pad = torch.zeros(B, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat([cls_pad, padding_mask], dim=1)  # (B, N+1)

        h = self.graph_node_feature(batch)             # (B, N+1, C)
        attn_bias = self.graph_attn_bias(batch)        # (B, H, N+1, N+1)

        if self.emb_layer_norm is not None:
            h = self.emb_layer_norm(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # (B, T, C) -> (T, B, C)
        h = h.transpose(0, 1)

        inner = [h] if return_inner_states else None
        for layer in self.layers:
            h = layer(h, self_attn_bias=attn_bias, self_attn_padding_mask=padding_mask)
            if return_inner_states:
                inner.append(h)

        graph_rep = h[0]  # (B, C) — the [graph] token.
        if return_inner_states:
            return inner, graph_rep
        return h, graph_rep


# ---------------------------------------------------------------------------
# Output head + top-level wrapper. Names mirror MS so the checkpoint loads.
# ---------------------------------------------------------------------------


class GraphormerEncoder(nn.Module):
    def __init__(self, cfg: GraphormerConfig):
        super().__init__()
        self.cfg = cfg
        self.graph_encoder = GraphormerGraphEncoder(cfg)
        self.activation_fn = _act(cfg.activation_fn)

        self.masked_lm_pooler = nn.Linear(cfg.encoder_embed_dim, cfg.encoder_embed_dim)
        self.lm_head_transform_weight = nn.Linear(cfg.encoder_embed_dim, cfg.encoder_embed_dim)
        self.layer_norm = nn.LayerNorm(cfg.encoder_embed_dim)
        self.embed_out = nn.Linear(cfg.encoder_embed_dim, cfg.num_classes, bias=False)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch: dict, return_inner_states: bool = False):
        if return_inner_states:
            inner, _ = self.graph_encoder(batch, return_inner_states=True)
            h = inner[-1]
        else:
            h, _ = self.graph_encoder(batch)
        h = h.transpose(0, 1)  # (B, N+1, C)

        h = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(h)))
        out = self.embed_out(h) + self.lm_output_learned_bias

        if return_inner_states:
            return out, inner
        return out


class MiniGraphormer(nn.Module):
    """Top-level wrapper. State-dict layout matches Microsoft's release.

    The graph-level prediction is `out[:, 0, :]` (the [graph] token).
    """

    def __init__(self, cfg: GraphormerConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = GraphormerEncoder(cfg)

    def forward(self, batch: dict, return_inner_states: bool = False):
        return self.encoder(batch, return_inner_states=return_inner_states)

    def predict(self, batch: dict) -> torch.Tensor:
        """Convenience: graph-level scalar/logits over the [graph] token."""
        out = self.encoder(batch)
        return out[:, 0, :]
