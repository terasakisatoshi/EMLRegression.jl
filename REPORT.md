# Section 4.3 Faithful Rewrite Report

Date: 2026-04-17
Branch: `report-remaining-work`
Worktree: `.worktrees/report-remaining-work`

## Summary

Section 4.3 faithful rewrite remains the main implementation direction. The old choice-logit tree path has been replaced by a PyTorch-v16-style `EMLTree` core with paper-aligned targets, search/hardening training, paper-style snapping, and paper-style recovery metrics.

The latest blocker on `main` was a paper-budget training failure, not a modeling mismatch. Full `6000/2000` hardening could hit a Zygote shape error after the gates had already become highly saturated. That blocker is now fixed in this worktree.

Current work is in the measurement/reporting phase. The sweep and summary scripts are now wired to emit a Section 4.3 family table alongside the existing aggregate summary, but this worktree does not yet contain completed raw sweep artifacts to report final counts from.

```bash
~/.juliaup/bin/julia --project=. scripts/run_paper_headless_sweep.jl --jobs depth2,depth3,depth4,depth5_random,depth6_random,depth5_manual_noise12,depth6_manual_noise12
~/.juliaup/bin/julia --project=. scripts/summarize_results.jl
```

At the time of this report update, no local raw JSON sweep artifacts are present in `results/raw`, so completed family counts remain pending. The reporting path is ready to consume them once the sweep finishes.

## What Was Done

### 1. Model / Parameterization Rewrite

- Replaced the prior candidate-choice tree scaffold with a paper-faithful `EMLTree`.
- Standardized internal parameters to:
  - `leaf_logits[n_leaves, 3]`
  - `blend_logits[n_internal, 2]`
- Reworked forward evaluation to follow the PyTorch-v16 level-by-level tree semantics.
- Added manual initialization support from EML expressions.

Relevant files:

- [src/models/EMLLayer.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/src/models/EMLLayer.jl:1)
- [src/EMLRegression.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/src/EMLRegression.jl:1)
- [test/models/test_eml_layer.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/test/models/test_eml_layer.jl:1)

### 2. Paper-Aligned Targets / Training / Recovery

- Replaced target registry with `:eml_depth2` through `:eml_depth6`.
- Replaced the old training loop with paper-style `SEARCH -> HARDENING`.
- Replaced snapping / recovery with:
  - `analyze_snap`
  - `hard_project!`
  - `fit_success`
  - `symbol_success`
  - `stable_symbol_success`
- Updated experiment and summary scripts to the new schema.
- Added `scripts/run_paper_headless_sweep.jl` as the Julia analogue of the PyTorch headless sweep runner.
- Added a family-level `Section 4.3` summary output alongside `results/summaries/summary.csv`.

Relevant files:

- [src/targets/TargetRegistry.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/src/targets/TargetRegistry.jl:1)
- [src/training/TrainConfig.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/src/training/TrainConfig.jl:1)
- [src/training/TrainLoop.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/src/training/TrainLoop.jl:1)
- [src/snapping/Snap.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/src/snapping/Snap.jl:1)
- [src/eval/Metrics.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/src/eval/Metrics.jl:1)
- [src/eval/Recovery.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/src/eval/Recovery.jl:1)
- [scripts/run_experiment.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/scripts/run_experiment.jl:1)
- [scripts/run_suite.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/scripts/run_suite.jl:1)
- [scripts/summarize_results.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/scripts/summarize_results.jl:1)
- [scripts/run_paper_headless_sweep.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/scripts/run_paper_headless_sweep.jl:1)

## Gradient / Hardening Bug Investigation

### Symptom

The reduced paper-style sweep first exposed a structural failure in `depth4 random_hot`, and the later full paper-budget run exposed a second blocker in `depth2 biased`.

Reduced-budget reproducer:

```bash
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config /tmp/depth4_random_hot.toml --target eml_depth4 --seed 137
```

Observed reduced-budget error:

- Zygote `MethodError: no method matching -(::Nothing)`
- This occurred during backprop through the `EMLTree` forward path.

Paper-budget reproducer:

```bash
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config /tmp/pnas_d2_random-biased.toml --target eml_depth2 --seed 137
```

Observed paper-budget error:

- Zygote `DimensionMismatch`
- This occurred late in hardening, after the gate probabilities had already become highly saturated.

### Root Cause

The original faithful rewrite returned diagnostics from the forward pass in a way that caused Zygote to build tangents containing `Nothing` inside complex array gradients. After that path was stabilized, a second issue remained: the level-update implementation still used a shape-sensitive gradient path that broke at paper budget once hardening had saturated the gates.

The problematic paths were:

- multi-level forward collection logic
- complex broadcast differentiation through `eml`
- mixed tangent propagation where `Nothing` was not normalized away
- late-hardening level updates through per-column tuple assembly

### Fixes Applied

1. Reworked `EMLTree` forward evaluation to avoid the earlier recursive / mutation-heavy output collection path.
2. Split training-time evaluation into a dedicated `forward_with_regularizers` path that computes scalar regularizers directly instead of differentiating through aux collection.
3. Replaced the per-column level update path with matrix-wise blending and matrix-wise `eml` evaluation.
4. Rewrote `_complex_blend(::AbstractMatrix, ...)` as pure broadcast so Zygote no longer hits mutation-sensitive adjoints.
5. Flattened `eml_outputs` handling in the loss path so intermediate penalties operate on a simple vector.
6. Locked in regression tests for the formerly crashing training state and the paper-budget saturated-gate reproducer.

Regression tests:

- [test/training/test_train_loop.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/test/training/test_train_loop.jl:1)
- [test/models/test_eml_layer.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/report-remaining-work/test/models/test_eml_layer.jl:1)

Commit:

- `03d9b73 fix: unblock paper-budget hardening gradients`

## Current State

### Verified

Fresh verification command run successfully:

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'
```

What that means:

- the structural tuple-gradient crash is fixed
- the paper-budget hardening crash is fixed
- the training test suite is green

Additional regression coverage now exists for:

- nonfinite search steps incrementing restart counters instead of aborting immediately
- exact-one and saturated-gate blend behavior in `_complex_blend`
- paper-budget saturated gates remaining differentiable

### Partial Full-Sweep Results

The full sweep is still running. Confirmed partial results at this report snapshot:

- `pnas_d2_random-biased`: `8/8` fit, `8/8` symbolic, `4/8` stable symbolic, `nonfinite_steps=0`, `nan_restarts=0`
- `pnas_d2_random-uniform`: `8/8` fit, `8/8` symbolic, `2/8` stable symbolic, `nonfinite_steps=0`, `nan_restarts=0`
- `pnas_d2_random-xy_biased`: `8/8` fit, `8/8` symbolic, `2/8` stable symbolic, `nonfinite_steps=0`, `nan_restarts=0`
- `pnas_d2_random-random_hot`: `8/8` fit, `8/8` symbolic, `2/8` stable symbolic, `nonfinite_steps=0`, `nan_restarts=0`
- `pnas_d3_random-biased`: `0/16` fit, `0/16` symbolic, `0/16` stable symbolic, cumulative `nonfinite_steps=1546`, cumulative `nan_restarts=23`
- `pnas_d3_random-uniform`: `0/4` fit, `0/4` symbolic, `0/4` stable symbolic, cumulative `nonfinite_steps=347`, cumulative `nan_restarts=5`

This already shows one match and one mismatch versus the external PyTorch reference:

- match: `depth2` random discovery is effectively solved
- mismatch: `depth3` is currently underperforming the PyTorch README expectation of `17/64`

### Reduced Sweep Check

Reduced paper-style sweep run:

```bash
~/.juliaup/bin/julia --project=. scripts/run_paper_headless_sweep.jl --jobs depth2,depth3,depth4 --search-iters 5 --hardening-iters 2 --seeds 1
```

Observed raw diagnostics:

- 12 raw artifacts were written, one per depth/init-strategy combination for depths 2, 3, and 4.
- `depth4 random_hot` was the only reduced job that reported nonfinite activity: `nonfinite_steps=7`, `nan_restarts=0`, `failure_reason=no_failure`.
- Every other reduced job reported `nan_restarts=0` and `nonfinite_steps=0`.
- None of the reduced jobs reached `fit_success` or `stable_symbol_success` under the shortened iteration budget.

This confirmed the runner propagates the reduced-sweep overrides correctly, and justified moving on to the full paper-style sweep.

## Remaining Work

### Priority 1: Finish The Full Sweep

Let the current full paper-style sweep complete, then summarize it with:

```bash
~/.juliaup/bin/julia --project=. scripts/summarize_results.jl
```

The main deliverable is the measured tendency table for:

- `depth2`
- `depth3`
- `depth4`
- `depth5_random`
- `depth6_random`
- `depth5_manual_noise12`
- `depth6_manual_noise12`

### Priority 2: Numerical Stabilization Alignment

Match the PyTorch implementation more closely in the forward / optimization path:

- investigate why `depth3` is still showing large nonfinite/restart pressure under the paper budget
- compare hardening dynamics and temperature schedule against the PyTorch implementation
- inspect whether the current Julia objective is still too brittle once gates saturate
- re-evaluate clamp and restart thresholds if `depth4+` continue to underperform after the first full measurement

### Priority 3: Sweep Measurement

After the current sweep finishes:

1. rebuild `results/summaries/summary.csv`
2. rebuild `results/summaries/section_4_3_summary.csv`
3. compare each family against the PyTorch README / Supplementary tendencies
4. update this report with completed counts instead of partial counts
5. decide whether the next pass should target `depth3` stability specifically or move directly to `depth4+`

## Files Changed So Far

- `src/EMLRegression.jl`
- `src/models/EMLLayer.jl`
- `src/targets/TargetRegistry.jl`
- `src/training/TrainConfig.jl`
- `src/training/TrainLoop.jl`
- `src/snapping/Snap.jl`
- `src/eval/Metrics.jl`
- `src/eval/Recovery.jl`
- `scripts/run_experiment.jl`
- `scripts/run_suite.jl`
- `scripts/summarize_results.jl`
- `scripts/run_paper_headless_sweep.jl`
- `test/models/test_eml_layer.jl`
- `test/targets/test_targets.jl`
- `test/training/test_train_loop.jl`
- `test/snapping/test_snap.jl`
- `test/eval/test_recovery.jl`
- `test/integration/test_smoke_ln.jl`

## Bottom Line

The rewrite has crossed the structural milestone and the paper-budget hardening blocker is fixed. The current state is no longer "the sweep crashes immediately."

The next milestone is empirical: finish the full sweep, record the depth-wise recovery tendency, and then focus the next stabilization pass on the still-weak `depth3+` behavior.
