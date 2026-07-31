# Graphormer: which attention biases actually matter?

An ablation and extension study on [Graphormer](https://github.com/microsoft/Graphormer), the graph transformer that injects structural information as additive biases on the attention logits. Two questions:

1. **Which of Graphormer's structural biases is the trained model actually relying on?** Answered by zeroing each one at inference through a forward hook, with no weight modification.
2. **Do additional structural encodings from the wider literature help on top?** Answered by injecting Laplacian PE, random-walk PE, and a ring-aware bias into the same attention logits across an alpha sweep.

The answer to the second question is **no**, at every alpha tested. That is reported here in full rather than buried, because a negative result measured cleanly is still a result.

All numbers come from Microsoft's released `pcqm4mv2_graphormer_base` checkpoint, loaded with strict parameter matching (0 missing, 0 unexpected keys), evaluated on the first 2,000 graphs of the official PCQM4Mv2 validation split in CSV order.

---

## 1. Which biases matter

Each ablation replaces one bias module's output with zeros via a forward hook. The same loaded model is reused throughout, so nothing is retrained and the comparison is exact.

| Ablation | MAE | vs baseline |
|---|---|---|
| baseline | 0.101972 | |
| no_virtual_distance | 0.101972 | +0% |
| no_centrality | 0.117845 | +16% |
| no_spatial_bias | 0.138399 | +36% |
| no_edge_bias | 0.447120 | **+338%** |
| no_all_attn_biases | 0.757560 | **+643%** |

**The edge bias carries the model.** Removing it alone costs more than four times what removing the spatial bias costs. Removing everything degrades MAE by a factor of seven, which puts a floor under how much of Graphormer's performance is structural encoding rather than transformer capacity.

**The virtual-node distance bias does nothing measurable** on this slice. Its ablation is identical to baseline to six decimal places, which suggests the trained model has learned to ignore it rather than that it is unused by construction.

## 2. Do extra structural encodings help?

Three extensions, each added as `alpha * B` to the attention logits, evaluated in four configurations: baseline, the extension alone with Graphormer's own biases zeroed, the extension plus only the edge bias, and the extension on top of the full model.

- **Laplacian PE** (SAN-flavoured): `U_k W U_k^T` from the k=8 smallest non-trivial Laplacian eigenvectors, W = I
- **Random-walk PE** (GTN-flavoured): `sum_{k=1..4} (D^-1/2 A D^-1/2)^k`, a multi-scale soft adjacency
- **Ring-aware bias** (PAGTN-flavoured): ring membership structure from RDKit

Best result per extension, across the full sweep, on top of the complete model:

| Extension | Best alpha | MAE | vs baseline |
|---|---|---|---|
| Laplacian PE | 0.5 | 0.101821 | -0.0002 |
| RWPE | 0.1 | 0.101895 | -0.0001 |
| Ring-aware | 0.05 | 0.101922 | -0.0000 |

Every improvement is smaller than 0.0002 MAE, which is within finite-sample variation on a 2,000-graph slice. **None of the three extensions helps.** Raising alpha only makes it worse: at alpha = 2.0 the ring bias costs +57% and RWPE at alpha = 1.8 costs +8%.

The substituted configurations are more informative than the additive ones. With Graphormer's biases zeroed, each extension alone lands at 0.70 to 0.81 MAE, close to the +643% full-ablation figure. Adding back only the edge bias recovers most of the gap, to roughly 0.15 to 0.17.

**Read together, the two experiments say the same thing.** What Graphormer has learned is largely carried by its edge encoding, and generic graph-structural priors, whether spectral, random-walk, or chemistry-specific, do not add information the trained model is missing. They can only perturb it.

Per-alpha tables for all fifteen sweeps are in [`mini_graphormer/README.md`](mini_graphormer/README.md).

## 3. Per-molecule probe

Five aromatic molecules with documented B3LYP HOMO-LUMO gaps, each run through baseline and the three extensions in isolation:

| Molecule | Experimental | Baseline | +Laplacian | +RWPE | +Ring |
|---|---|---|---|---|---|
| naphthalene | 4.40 | 4.8344 | 4.8303 | 4.8342 | 4.8351 |
| azulene | 1.70 | 3.3331 | 3.3357 | 3.3076 | 3.3286 |
| anthracene | 3.30 | 3.5862 | 3.5883 | 3.5793 | 3.5856 |
| phenanthrene | 3.90 | 4.7476 | 4.7516 | 4.7495 | 4.7479 |
| pyrene | 3.30 | 3.8424 | 3.8431 | 3.8345 | 3.8408 |

Predictions sit well off the experimental values, most starkly on azulene where the model predicts 3.33 against a measured 1.70. Azulene is the non-alternant outlier in this set, and the miss is larger than any extension moves the number. The extensions shift predictions in the fourth decimal place; the model's actual error is in the first.

## 4. Limitations

- **2,000-graph slice**, taken in CSV order from the validation split rather than sampled. Differences below roughly 0.001 MAE should not be read as signal.
- **Inference-time interventions only.** Nothing is retrained, so this measures what the trained model relies on, not what a model trained with these extensions could learn.
- **Single checkpoint**, single architecture. No seed variation.
- The per-molecule probe compares against experimental gaps, while the model was trained on DFT-computed gaps. Some of that offset is expected.
- Timings in the detailed tables were taken on shared hardware and vary widely between runs. They are not a benchmark.

## 5. Layout

| Path | Contents |
|---|---|
| [`mini_graphormer/`](mini_graphormer) | The implementation and the full results log. `model.py`, `data.py`, `ablate.py`, `extend_spectral.py`, `extend_rwpe.py`, `extend_ring.py`, `validate.py`, `handcraft.py` |
| [`mini_graphormer/SETUP.md`](mini_graphormer/SETUP.md) | Getting the checkpoint and the PCQM4Mv2 dataset |
| [`ablation/`](ablation) | Attention-backend benchmark harness and its JSON output |
| [`references.bib`](references.bib) | Graphormer, SAN, GRPE, GTN |

## Running

Follow [`SETUP.md`](mini_graphormer/SETUP.md) first: the checkpoint (~193 MB) and dataset (~88 MB) are not committed.

```bash
cd mini_graphormer
python validate.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 5000
python ablate.py   --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000
python extend_spectral.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 0.5
```

`validate.py --skip-mae` verifies the checkpoint loads with 0 missing and 0 unexpected keys before you spend time on a sweep.

## Attribution

`mini_graphormer` is an independent reimplementation. Parameter naming deliberately mirrors Microsoft's [Graphormer](https://github.com/microsoft/Graphormer) (MIT) so the released checkpoints load without remapping; no Microsoft source is copied. Evaluated on [PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/) (OGB-LSC). See [NOTICE](NOTICE).

Originated as a course project at IIT Bombay.
