"""
Standalone Graphormer inference module.

Re-implements the Microsoft Graphormer encoder
(graphormer/modules/{graphormer_graph_encoder,graphormer_graph_encoder_layer,
graphormer_layers,multihead_attention}.py + graphormer/models/graphormer.py)
in pure PyTorch >= 2.0, with no Fairseq / DGL / torch-scatter dependency.

Three attention backends are exposed for the ablation:
    - "manual"      : explicit torch.matmul softmax path (numerical baseline,
                      mathematically identical to the original MultiheadAttention).
    - "sdpa"        : torch.nn.functional.scaled_dot_product_attention with the
                      structural bias passed as an additive attn_mask. PyTorch
                      auto-dispatches to flash / memory-efficient / math kernels.
    - "sdpa-flash"  : forces the FlashAttention-2 backend via the SDPA context
                      manager. Falls back to mem-efficient if FA-2 rejects the
                      bias shape (FA-2 supports additive bias only on Hopper +
                      recent PyTorch builds; Ampere typically routes to mem-eff).

State-dict loading remaps the official Fairseq checkpoint keys
(`encoder.graph_encoder.<...>` and `encoder.<head>`) onto this module.

KV caching is intentionally NOT implemented:
    Graphormer is an encoder-only model performing a single bidirectional pass
    over [CLS, v1..vN] with full attention. The structural bias depends on all
    node-pairs jointly. There is no autoregressive step over which K, V can be
    reused. KV caching is therefore mathematically and structurally inapplicable
    to graph property prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GraphormerConfig:
    num_atoms: int = 512 * 9
    num_edges: int = 512 * 3
    num_in_degree: int = 512
    num_out_degree: int = 512
    num_spatial: int = 512
    num_edge_dis: int = 128
    edge_type: str = "multi_hop"
    multi_hop_max_dist: int = 5

    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 768
    encoder_attention_heads: int = 32

    activation_fn: str = "gelu"
    encoder_normalize_before: bool = True
    pre_layernorm: bool = False

    num_classes: int = 1
    max_nodes: int = 128

    @classmethod
    def graphormer_base(cls) -> "GraphormerConfig":
        return cls()

    @classmethod
    def graphormer_slim(cls) -> "GraphormerConfig":
        return cls(
            encoder_layers=12,
            encoder_embed_dim=80,
            encoder_ffn_embed_dim=80,
            encoder_attention_heads=8,
        )

    @classmethod
    def graphormer_large(cls) -> "GraphormerConfig":
        return cls(
            encoder_layers=24,
            encoder_embed_dim=1024,
            encoder_ffn_embed_dim=1024,
            encoder_attention_heads=32,
        )


# ---------------------------------------------------------------------------
# Structural encodings
# ---------------------------------------------------------------------------
class GraphNodeFeature(nn.Module):
    """Sum of atom-feature embedding + in/out-degree centrality + CLS token."""

    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.atom_encoder = nn.Embedding(num_atoms + 1, hidden_dim, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(num_in_degree, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(num_out_degree, hidden_dim, padding_idx=0)
        self.graph_token = nn.Embedding(1, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        in_degree: torch.Tensor,
        out_degree: torch.Tensor,
    ) -> torch.Tensor:
        # x: [B, T, K] (K atom-feature slots per node, summed)
        n_graph = x.size(0)
        node_feat = (
            self.atom_encoder(x).sum(dim=-2)
            + self.in_degree_encoder(in_degree)
            + self.out_degree_encoder(out_degree)
        )
        cls = self.graph_token.weight.unsqueeze(0).expand(n_graph, -1, -1)
        return torch.cat([cls, node_feat], dim=1)  # [B, T+1, C]


class GraphAttnBias(nn.Module):
    """Spatial-pos + edge-feature additive attention bias, shape [B, H, T+1, T+1]."""

    def __init__(
        self,
        num_heads: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.edge_type = edge_type
        self.multi_hop_max_dist = multi_hop_max_dist

        self.edge_encoder = nn.Embedding(num_edges + 1, num_heads, padding_idx=0)
        if edge_type == "multi_hop":
            self.edge_dis_encoder = nn.Embedding(num_edge_dis * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

    def forward(
        self,
        attn_bias: torch.Tensor,         # [B, T+1, T+1] base bias (typically zeros)
        spatial_pos: torch.Tensor,       # [B, T, T]
        edge_input: torch.Tensor,        # [B, T, T, max_dist, K] (multi_hop)
        attn_edge_type: torch.Tensor,    # [B, T, T, K]
    ) -> torch.Tensor:
        n_graph, n_node = spatial_pos.size(0), spatial_pos.size(1)
        H = self.num_heads

        graph_attn_bias = attn_bias.clone().unsqueeze(1).repeat(1, H, 1, 1)

        # Spatial encoding (Eq. 5 in Graphormer paper).
        spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + spatial_pos_bias

        # CLS-token virtual distance.
        t = self.graph_token_virtual_distance.weight.view(1, H, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # Edge encoding.
        if self.edge_type == "multi_hop":
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1
            spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, : self.multi_hop_max_dist, :]
            edge_input = self.edge_encoder(edge_input).mean(-2)  # [B,T,T,D,H]
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, H)
            edge_input_flat = torch.bmm(
                edge_input_flat,
                self.edge_dis_encoder.weight.reshape(-1, H, H)[:max_dist, :, :],
            )
            edge_input = edge_input_flat.reshape(max_dist, n_graph, n_node, n_node, H)
            edge_input = edge_input.permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) / spatial_pos_.float().unsqueeze(-1))
            edge_input = edge_input.permute(0, 3, 1, 2)
        else:
            edge_input = self.edge_encoder(attn_edge_type).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:, :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)
        return graph_attn_bias  # [B, H, T+1, T+1]


# ---------------------------------------------------------------------------
# Multi-head attention with switchable backend
# ---------------------------------------------------------------------------
class MultiheadAttention(nn.Module):
    """Self-attention with additive structural bias and a switchable backend."""

    def __init__(self, embed_dim: int, num_heads: int, attn_backend: str = "manual") -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.attn_backend = attn_backend

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,                    # [B, T, C]
        attn_bias: torch.Tensor,            # [B, H, T, T] (already includes spatial+edge+CLS)
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T] bool, True = pad
    ) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(x).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, D).transpose(1, 2)

        # Fold key-padding mask into the additive bias (-inf on pad columns).
        bias = attn_bias
        if key_padding_mask is not None:
            kpm = key_padding_mask[:, None, None, :].to(torch.bool)  # [B,1,1,T]
            bias = bias.masked_fill(kpm, float("-inf"))

        if self.attn_backend == "manual":
            attn = torch.matmul(q * self.scaling, k.transpose(-1, -2))   # [B,H,T,T]
            attn = attn + bias
            attn = attn.softmax(dim=-1)
            out = torch.matmul(attn, v)                                   # [B,H,T,D]
        elif self.attn_backend == "sdpa":
            # SDPA scales internally; do NOT pre-scale q.
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
            )
        elif self.attn_backend == "sdpa-flash":
            try:
                with torch.nn.attention.sdpa_kernel(
                    [
                        torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                        torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
                    ]
                ):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
                    )
            except (RuntimeError, AttributeError):
                out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False
                )
        else:
            raise ValueError(f"unknown attn_backend: {self.attn_backend}")

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Encoder layer + stack
# ---------------------------------------------------------------------------
class GraphormerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ffn_embed_dim: int,
        num_heads: int,
        activation_fn: str = "gelu",
        pre_layernorm: bool = False,
        attn_backend: str = "manual",
    ) -> None:
        super().__init__()
        self.pre_layernorm = pre_layernorm
        self.self_attn = MultiheadAttention(embed_dim, num_heads, attn_backend=attn_backend)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.activation_fn = _get_activation(activation_fn)

    def forward(
        self,
        x: torch.Tensor,                 # [B, T, C]
        attn_bias: torch.Tensor,         # [B, H, T, T]
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        if self.pre_layernorm:
            x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, attn_bias=attn_bias, key_padding_mask=key_padding_mask)
        x = residual + x
        if not self.pre_layernorm:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.pre_layernorm:
            x = self.final_layer_norm(x)
        x = self.fc2(self.activation_fn(self.fc1(x)))
        x = residual + x
        if not self.pre_layernorm:
            x = self.final_layer_norm(x)
        return x


class GraphormerGraphEncoder(nn.Module):
    def __init__(self, cfg: GraphormerConfig, attn_backend: str = "manual") -> None:
        super().__init__()
        self.cfg = cfg
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
        self.emb_layer_norm = (
            nn.LayerNorm(cfg.encoder_embed_dim) if cfg.encoder_normalize_before else None
        )
        self.layers = nn.ModuleList(
            [
                GraphormerEncoderLayer(
                    embed_dim=cfg.encoder_embed_dim,
                    ffn_embed_dim=cfg.encoder_ffn_embed_dim,
                    num_heads=cfg.encoder_attention_heads,
                    activation_fn=cfg.activation_fn,
                    pre_layernorm=cfg.pre_layernorm,
                    attn_backend=attn_backend,
                )
                for _ in range(cfg.encoder_layers)
            ]
        )

    def forward(self, batched_data: dict) -> torch.Tensor:
        x = batched_data["x"]
        n_graph, n_node = x.size()[:2]

        # Padding mask matches Fairseq convention: a node whose first atom-feature
        # slot is 0 (padding_idx of atom_encoder) is treated as padding.
        padding_mask = x[:, :, 0].eq(0)                     # [B, T]
        cls_pad = torch.zeros(n_graph, 1, device=x.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat([cls_pad, padding_mask], dim=1)  # [B, T+1]

        h = self.graph_node_feature(x, batched_data["in_degree"], batched_data["out_degree"])

        attn_bias = self.graph_attn_bias(
            batched_data["attn_bias"],
            batched_data["spatial_pos"],
            batched_data["edge_input"],
            batched_data["attn_edge_type"],
        )

        if self.emb_layer_norm is not None:
            h = self.emb_layer_norm(h)

        for layer in self.layers:
            h = layer(h, attn_bias=attn_bias, key_padding_mask=padding_mask)

        return h  # [B, T+1, C]


class GraphormerForGraphPrediction(nn.Module):
    """Mirrors `GraphormerEncoder` from graphormer/models/graphormer.py."""

    def __init__(self, cfg: GraphormerConfig, attn_backend: str = "manual") -> None:
        super().__init__()
        self.cfg = cfg
        self.graph_encoder = GraphormerGraphEncoder(cfg, attn_backend=attn_backend)
        self.lm_head_transform_weight = nn.Linear(cfg.encoder_embed_dim, cfg.encoder_embed_dim)
        self.activation_fn = _get_activation(cfg.activation_fn)
        self.layer_norm = nn.LayerNorm(cfg.encoder_embed_dim)
        self.embed_out = nn.Linear(cfg.encoder_embed_dim, cfg.num_classes, bias=False)
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        # Present in checkpoints but unused at inference for graph regression.
        self.masked_lm_pooler = nn.Linear(cfg.encoder_embed_dim, cfg.encoder_embed_dim)

    def forward(self, batched_data: dict) -> torch.Tensor:
        h = self.graph_encoder(batched_data)                          # [B, T+1, C]
        h = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(h)))
        out = self.embed_out(h) + self.lm_output_learned_bias         # [B, T+1, num_classes]
        return out


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------
def _get_activation(name: str):
    name = name.lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    if name == "linear":
        return lambda x: x
    raise ValueError(f"unsupported activation_fn: {name}")


# ---------------------------------------------------------------------------
# Checkpoint loader: maps Microsoft / Fairseq state-dict keys onto this module
# ---------------------------------------------------------------------------
def load_pretrained_checkpoint(
    model: GraphormerForGraphPrediction,
    ckpt_path: str,
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """Load an official Graphormer .pt checkpoint into the standalone model.

    The Fairseq-saved state dict is wrapped under {"model": OrderedDict(...)}
    with keys prefixed `encoder.<...>`. After stripping the prefix we land
    directly on the names this module uses, except for the FFN sublayers which
    are re-mapped from the `nn.Sequential`-style layout used internally by
    quant_noise (a no-op when q_noise=0). The original code uses bare
    `nn.Linear` so keys already match.
    """
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = obj["model"] if isinstance(obj, dict) and "model" in obj else obj

    remapped = {}
    for k, v in raw.items():
        nk = k
        if nk.startswith("encoder."):
            nk = nk[len("encoder."):]
        # in_proj_{weight,bias} -> q/k/v split (legacy Fairseq nn.MultiheadAttention)
        if nk.endswith("self_attn.in_proj_weight"):
            base = nk[: -len("in_proj_weight")]
            dim = v.shape[0] // 3
            remapped[base + "q_proj.weight"] = v[:dim]
            remapped[base + "k_proj.weight"] = v[dim : 2 * dim]
            remapped[base + "v_proj.weight"] = v[2 * dim :]
            continue
        if nk.endswith("self_attn.in_proj_bias"):
            base = nk[: -len("in_proj_bias")]
            dim = v.shape[0] // 3
            remapped[base + "q_proj.bias"] = v[:dim]
            remapped[base + "k_proj.bias"] = v[dim : 2 * dim]
            remapped[base + "v_proj.bias"] = v[2 * dim :]
            continue
        remapped[nk] = v

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if verbose:
        print(f"[ckpt] loaded {len(remapped)} tensors from {ckpt_path}")
        if missing:
            print(f"[ckpt] missing keys ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[ckpt] unexpected keys ({len(unexpected)}): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    if strict and (missing or unexpected):
        raise RuntimeError(f"strict load failed: {len(missing)} missing, {len(unexpected)} unexpected")
    return {"missing": missing, "unexpected": unexpected}


# ---------------------------------------------------------------------------
# Sanity helper
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
