============================================================================
GRAPHORMER ABLATION & EXTENSION EXPERIMENTS — RESULTS LOG
============================================================================
All runs use Microsoft's pretrained pcqm4mv2_graphormer_base checkpoint
(loaded with strict parameter matching: 0 missing / 0 unexpected keys).
"n" graphs are the first n entries of the official PCQM4Mv2 valid split,
streamed in CSV order from `dataset/pcqm4m-v2/raw/data.csv.gz`.


============================================================================
1. Validation MAE
============================================================================
Loads the pretrained checkpoint, streams the first --max-graphs molecules
of the PCQM4Mv2 validation split, runs forward through the model, and
reports mean absolute error against the ground-truth HOMO-LUMO gap labels.
Used to confirm mini_graphormer reproduces the published 0.0864 eV number
(within finite-sample variance on a non-shuffled CSV-ordered slice).

  $ python validate.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 5000


============================================================================
2. Ablation study
============================================================================
Disables one Graphormer-specific bias module at a time via a forward hook
that replaces the module's output with zeros. No weight modification — the
same loaded model is reused across all ablations. Reveals which structural
encodings the trained model actually relies on at inference.

  $ python ablate.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000

  ablation                      MAE      Δ vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   12.0s
  no_spatial_bias          0.138399 +0.0364 (+36%)    12.4s
  no_edge_bias             0.447120 +0.3451 (+338%)   12.8s
  no_virtual_distance      0.101972  +0.0000 (+0%)    13.5s
  no_centrality            0.117845 +0.0159 (+16%)    12.6s
  no_all_attn_biases       0.757560 +0.6556 (+643%)   13.0s


============================================================================
3. Extension: Laplacian PE (SAN-flavoured)
============================================================================
Adds α · U_k W U_k^T as an additive attention bias, where U_k are the k=8
smallest non-trivial Laplacian eigenvectors per molecule and W = I (uniform
weighting). Each run sweeps a different α; tables ordered by ascending α.

The four configurations (same for §4 and §5):
  baseline           — full Graphormer, no extension
  X only             — every Graphormer attn / centrality bias zeroed
  X + edge           — only Graphormer's edge bias kept + extension
  all + X            — full Graphormer + extension on top

  $ python extend_spectral.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 0.2

  N = 2000 graphs   alpha = 0.2   k = 8   weight = uniform

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   31.2s
  laplacian only           0.713425 +0.6115 (+600%)   32.8s
  laplacian + edge         0.169970 +0.0680 (+67%)    32.5s
  all + laplacian          0.101855  -0.0001 (-0%)    32.4s

  $ python extend_spectral.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 0.5

  N = 2000 graphs   alpha = 0.5   k = 8   weight = uniform

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   31.7s
  laplacian only           0.728484 +0.6265 (+614%)   32.6s
  laplacian + edge         0.182124 +0.0802 (+79%)    32.7s
  all + laplacian          0.101821  -0.0002 (-0%)    39.3s

  $ python extend_spectral.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 1.0

  N = 2000 graphs   alpha = 1.0   k = 8   weight = uniform

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   13.5s
  laplacian only           0.755055 +0.6531 (+640%)   13.5s
  laplacian + edge         0.212169 +0.1102 (+108%)   24.9s
  all + laplacian          0.101977  +0.0000 (+0%)    32.4s

  $ python extend_spectral.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 2.0

  N = 2000 graphs   alpha = 2.0   k = 8   weight = uniform

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   30.1s
  laplacian only           0.811686 +0.7097 (+696%)   31.3s
  laplacian + edge         0.294564 +0.1926 (+189%)   31.8s
  all + laplacian          0.104514  +0.0025 (+2%)    31.1s


============================================================================
4. Extension: Random-Walk PE (RWPE, GTN-flavoured)
============================================================================
Adds α · sum_{k=1..K} (D^{-1/2} A D^{-1/2})^k as an additive attention bias
— a multi-scale "soft adjacency" from random-walk powers. K = 4 throughout.
Same four-config evaluation as Laplacian. Tables ordered by ascending α.

  $ python extend_rwpe.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 0.05

  N = 2000 graphs   alpha = 0.05   K = 4

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   45.5s
  rwpe only                0.709562 +0.6076 (+596%)   47.0s
  rwpe + edge              0.164401 +0.0624 (+61%)    25.6s
  all + rwpe               0.101905  -0.0001 (-0%)    24.8s

  $ python extend_rwpe.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 0.1

  N = 2000 graphs   alpha = 0.1   K = 4

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   45.0s
  rwpe only                0.715822 +0.6138 (+602%)   48.0s
  rwpe + edge              0.165066 +0.0631 (+62%)    27.3s
  all + rwpe               0.101895  -0.0001 (-0%)    25.4s

  $ python extend_rwpe.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 0.2

  N = 2000 graphs   alpha = 0.2   K = 4

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   32.7s
  rwpe only                0.730403 +0.6284 (+616%)   43.2s
  rwpe + edge              0.166579 +0.0646 (+63%)    83.7s
  all + rwpe               0.102035  +0.0001 (+0%)    99.6s

  $ python extend_rwpe.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 0.5

  N = 2000 graphs   alpha = 0.5   K = 4

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   35.3s
  rwpe only                0.793685 +0.6917 (+678%)   45.4s
  rwpe + edge              0.172888 +0.0709 (+70%)    50.1s
  all + rwpe               0.102568  +0.0006 (+1%)    48.3s

  $ python extend_rwpe.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 0.8

  N = 2000 graphs   alpha = 0.8   K = 4

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   86.9s
  rwpe only                0.892205 +0.7902 (+775%)   93.5s
  rwpe + edge              0.183970 +0.0820 (+80%)    80.0s
  all + rwpe               0.103209  +0.0012 (+1%)    64.2s

  $ python extend_rwpe.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 1.8

  N = 2000 graphs   alpha = 1.8   K = 4

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   60.2s
  rwpe only                1.196780 +1.0948 (+1074%)  19.0s
  rwpe + edge              0.284413 +0.1824 (+179%)   17.0s
  all + rwpe               0.110219  +0.0082 (+8%)    17.6s


============================================================================
5. Extension: Ring-aware bias (PAGTN-flavoured)
============================================================================
Adds α · 1[i, j share a ring] as an additive attention bias, with rings
extracted via rdkit's smallest-set-of-smallest-rings (SSSR). Binary mask;
shows the steepest sensitivity to α of the three extensions. Same four-
config eval. Tables ordered by ascending α.

  $ python extend_ring.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 0.05

  N = 2000 graphs   alpha = 0.05

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   24.9s
  ring only                0.704122 +0.6022 (+591%)   32.9s
  ring + edge              0.161926 +0.0600 (+59%)    46.0s
  all + ring               0.101922  -0.0000 (-0%)    40.9s

  $ python extend_ring.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 0.2

  N = 2000 graphs   alpha = 0.2

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   23.4s
  ring only                0.705834 +0.6039 (+592%)   26.8s
  ring + edge              0.157889 +0.0559 (+55%)    39.7s
  all + ring               0.102196  +0.0002 (+0%)    46.0s

  $ python extend_ring.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 0.5

  N = 2000 graphs   alpha = 0.5

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   63.5s
  ring only                0.714197 +0.6122 (+600%)   87.7s
  ring + edge              0.154714 +0.0527 (+52%)    78.1s
  all + ring               0.103742  +0.0018 (+2%)    72.2s

  $ python extend_ring.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 1.0

  N = 2000 graphs   alpha = 1.0

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   48.1s
  ring only                0.755527 +0.6536 (+641%)   46.0s
  ring + edge              0.162202 +0.0602 (+59%)    40.2s
  all + ring               0.110654  +0.0087 (+9%)    35.6s

  $ python extend_ring.py --ckpt checkpoint_best_pcqm4mv2.pt --max-graphs 2000 --alpha 2.0

  N = 2000 graphs   alpha = 2.0

  configuration                 MAE   diff vs base     time
  ---------------------- ---------- --------------  -------
  baseline                 0.101972                   88.1s
  ring only                0.881572 +0.7796 (+765%)   76.0s
  ring + edge              0.247152 +0.1452 (+142%)   71.0s
  all + ring               0.159961 +0.0580 (+57%)    58.0s


============================================================================
6. Per-molecule study: handcraft.py
============================================================================
Runs the loaded model on five canonical aromatic molecules with documented
B3LYP HOMO-LUMO gaps from the literature. For each molecule, four forward
passes: baseline plus each extension applied IN ISOLATION (never two at
once). The "closest" column marks whichever configuration's prediction is
nearest the experimental gap; "a-priori" is the extension we predicted
ahead of time should help most given that molecule's graph structure.
Run uses sweep-optimal α for ring (0.05); other extensions at script
defaults (α-laplacian=1.0, α-rwpe=0.5).

  $ python handcraft.py --ckpt checkpoint_best_pcqm4mv2.pt --alpha-ring 0.05

  alphas: laplacian=1.0  rwpe=0.5  ring=0.05

  molecule          exp.   baseline   +laplac.      +rwpe      +ring     closest    a-priori
  ------------------------------------------------------------------------------------------
  naphthalene       4.40     4.8344     4.8303     4.8342     4.8351   laplacian        rwpe
  azulene           1.70     3.3331     3.3357     3.3076     3.3286        rwpe        rwpe
  anthracene        3.30     3.5862     3.5883     3.5793     3.5856        rwpe   laplacian
  phenanthrene      3.90     4.7476     4.7516     4.7495     4.7479    baseline   laplacian
  pyrene            3.30     3.8424     3.8431     3.8345     3.8408        rwpe        ring

  exp.       = literature B3LYP HOMO-LUMO gap (eV)
  baseline   = pretrained Graphormer, no extension
  +X         = same model + extension X applied (no other extensions active)
  closest    = which configuration's prediction is closest to exp.
  a-priori   = the extension we predicted should help most for this molecule
