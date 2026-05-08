## Plan: Graphormer Hyperparameter Study

Run a report-oriented hyperparameter study in Graphormer on WSL2/Linux GPU, starting from a stable baseline and then executing a controlled experiment matrix. This approach reduces setup risk, keeps experiments reproducible, and maps directly to a strong course report.

**Steps**
1. Phase 1: Environment Bring-up (blocking)
2. Set up WSL2 Ubuntu and install dependencies using [Original Codebase/Graphormer/install.sh](Original%20Codebase/Graphormer/install.sh).
3. Validate environment with quick CLI/import checks so training tools and graph libraries are confirmed working.
4. Phase 2: Baseline Reproduction (depends on Phase 1)
5. Choose a baseline script template:
   [Original Codebase/Graphormer/examples/property_prediction/zinc.sh](Original%20Codebase/Graphormer/examples/property_prediction/zinc.sh) for fastest iteration, or
   [Original Codebase/Graphormer/examples/property_prediction/pcqv2.sh](Original%20Codebase/Graphormer/examples/property_prediction/pcqv2.sh) for larger benchmark focus.
6. Run a short smoke training (reduced steps/epochs) to verify full pipeline: data load, model forward, checkpoint write, metric logging.
7. Run one full baseline and record final validation metric as the anchor for all comparisons.
8. Phase 3: Experiment Matrix Design (depends on baseline)
9. Define a compact, feasible grid (3-5 factors) with clear run budget.
10. Prioritize factors: learning rate, warmup steps, dropout/attention dropout, batch size/update frequency, and one architecture scale parameter.
11. Lock reproducibility controls: fixed seeds, consistent validation cadence, standardized run naming.
12. Phase 4: Controlled Runs (parallelizable)
13. Execute experiments in batches; parallelize across available GPU sessions when possible.
14. Track each run in one table: config, seed, runtime, memory usage, best metric, final metric.
15. Apply predefined fallback rules for failures (OOM, convergence issues, runtime faults) to keep the study on schedule.
16. Phase 5: Analysis and Reporting (depends on completed runs)
17. Build comparison tables/plots: baseline vs variants, sensitivity by factor, quality vs compute tradeoff.
18. Write findings and interpretation tied to Graphormer behavior.
19. Produce report-ready sections: setup, protocol, results, limitations, future work.
20. Phase 6: Optional Scale-up (after report baseline is secure)
21. Add a second dataset only after first pipeline is stable; reuse the same protocol for comparability.

**Relevant files**
- [Original Codebase/Graphormer/install.sh](Original%20Codebase/Graphormer/install.sh) - environment setup source of truth
- [Original Codebase/Graphormer/examples/property_prediction/zinc.sh](Original%20Codebase/Graphormer/examples/property_prediction/zinc.sh) - fastest baseline template
- [Original Codebase/Graphormer/examples/property_prediction/pcqv2.sh](Original%20Codebase/Graphormer/examples/property_prediction/pcqv2.sh) - larger benchmark template
- [Original Codebase/Graphormer/examples/property_prediction/hiv_pre.sh](Original%20Codebase/Graphormer/examples/property_prediction/hiv_pre.sh) - alternate benchmark template
- [Original Codebase/Graphormer/graphormer/tasks/graph_prediction.py](Original%20Codebase/Graphormer/graphormer/tasks/graph_prediction.py) - task wiring and argument flow
- [Original Codebase/Graphormer/graphormer/models/graphormer.py](Original%20Codebase/Graphormer/graphormer/models/graphormer.py) - architecture hyperparameter anchors
- [Original Codebase/Graphormer/graphormer/data/collator.py](Original%20Codebase/Graphormer/graphormer/data/collator.py) - batching behavior affecting memory/runtime
- [Original Codebase/Graphormer/README.md](Original%20Codebase/Graphormer/README.md) - benchmark context and expected targets

**Verification**
1. Environment installs cleanly in WSL2 and key imports pass.
2. Smoke run completes with checkpoint and validation output.
3. Baseline run is repeatable across at least 2 seeds with stable trend.
4. Every planned experiment has logged config and outcome (or documented failure reason).
5. Final report contains reproducibility details, comparison tables, and explicit limitations.

**Decisions**
- Included: hyperparameter-focused project path with report-first framing.
- Included: WSL2/Linux execution path due better compatibility.
- Deferred: first dataset choice (ZINC, PCQM4Mv2, or MolHIV).
- Excluded for now: Distributional Graphormer branch.

**Further Considerations**
1. Recommended dataset decision gate:
   Option A ZINC first for rapid iteration,
   Option B PCQM4Mv2 first for stronger benchmark relevance,
   Option C MolHIV for classification-focused narrative.
2. Recommended initial run budget: 8-15 runs before expanding.
3. Recommended report strategy: one deep, reproducible story over broad shallow coverage.
