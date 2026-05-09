# mini_graphormer

A self-contained reimplementation of [Microsoft's Graphormer](../Graphormer) that:

- Has **no fairseq / Cython / OGB-as-import-time dependency** — only `torch` and `numpy`.
- **Loads the released checkpoints directly** (`load_state_dict` is exact-match) because every parameter is named identically to the reference.
- Exposes the spatial / centrality / edge bias modules as plain `nn.Module`s so an ablation is one `register_forward_hook` away.

Targeted at the `pcqm4mv2_graphormer_base` checkpoint (12 layers, 768 dim, 32 heads, post-LN, GeLU). The other released configs are also supported via `GraphormerConfig` overrides.

## Files

| file        | what it contains |
|-------------|------------------|
| `model.py`  | `MiniGraphormer`, `GraphormerEncoder`, `GraphormerGraphEncoder`, `GraphormerLayer`, `GraphNodeFeature`, `GraphAttnBias`, `MultiheadAttention`, `GraphormerConfig` |
| `data.py`   | `preprocess_item`, `collate`, `floyd_warshall`, `gen_edge_input`, `convert_to_single_emb` (replaces `algos.pyx` + the fairseq collator) |
| `pretrain.py` | `load_pretrained(name_or_path)` + checkpoint URL table |
| `example.py`  | runnable smoke test + four ablation recipes |

## Quickstart

```python
import torch
from mini_graphormer import load_pretrained, preprocess_item, collate

model = load_pretrained("pcqm4mv2_graphormer_base")  # downloads on first call

# `item` is a torch_geometric.data.Data (or any object with .x, .edge_index,
# .edge_attr, optionally .y). Plain SimpleNamespace works fine too.
batch = collate([preprocess_item(item)])
with torch.no_grad():
    pred = model.predict(batch)   # (B, num_classes)
```

`model.predict(batch)` returns the [graph]-token output. To peek at intermediate layer states for analysis:

```python
out, inner_states = model(batch, return_inner_states=True)
# inner_states[i] has shape (T, B, C); inner_states[0] is post-input-LN.
```

## Ablations

The three signature inductive biases of Graphormer live in dedicated submodules:

- **Spatial bias** (b_phi): `model.encoder.graph_encoder.graph_attn_bias.spatial_pos_encoder`
- **Centrality encoding**: `model.encoder.graph_encoder.graph_node_feature.{in,out}_degree_encoder`
- **Edge bias** (c_ij): `model.encoder.graph_encoder.graph_attn_bias.{edge_encoder, edge_dis_encoder}`

Disabling one is a forward hook. `example.py` ships four canned variants (`none`, `no_spatial`, `no_centrality`, `no_edge`):

```bash
python -m mini_graphormer.example --ckpt pcqm4mv2_graphormer_base --ablation no_spatial
```

For a sweep, use the `ABLATIONS` dict in `example.py` as a template. The hooks keep weights untouched, so you can flip them on and off across batches without reloading the checkpoint.

## What was dropped vs. the reference

- **fairseq registration / dataclass plumbing**: replaced with a flat `GraphormerConfig`.
- **Cython `algos.pyx`**: rewritten in NumPy. Floyd–Warshall is vectorised; multi-hop edge reconstruction is per-pair Python (fine for n ≤ ~128, which is the ablation regime).
- **Quant-noise**: dropped — q_noise=0 in every released checkpoint.
- **Layer-drop**: dropped — also unused at inference.
- **Output head modes for masked-LM / shared-embed**: only the regression head used by the released checkpoints is implemented. `embed_out`, `lm_output_learned_bias`, `lm_head_transform_weight`, `layer_norm`, `masked_lm_pooler` are kept so the state dict still matches.

## Sanity check

```bash
python -m mini_graphormer.example                                  # random init smoke test
python -m mini_graphormer.example --ckpt pcqm4mv2_graphormer_base  # downloads + runs
```

If `load_state_dict` reports any missing or unexpected keys, the corresponding ablation/architecture change is the cause — strict-mode loading is the contract this package is built around.
