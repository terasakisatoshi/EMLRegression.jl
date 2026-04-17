# Section 4.3 Faithful Rewrite Report

Date: 2026-04-17
Branch: `section-4-3-faithful-rewrite`
Worktree: `.worktrees/section-4-3-faithful-rewrite`

## Summary

Section 4.3 faithful rewrite is in place as the main implementation direction. The old choice-logit tree path has been replaced by a PyTorch-v16-style `EMLTree` core with paper-aligned targets, search/hardening training, paper-style snapping, and paper-style recovery metrics.

The main blocking bug found during sweep work was a Zygote tuple-gradient failure in `depth4 random_hot`. That structural autodiff failure is now fixed. The remaining issue is numerical stability under wider paper-style domains and random-hot initialization.

Since the original report snapshot, two paper-style stabilization steps have been added:

- paper-style nonfinite restart accounting in the training loop
- paper-style `_BYPASS_THR` blending and gradient clipping

## What Was Done

### 1. Model / Parameterization Rewrite

- Replaced the prior candidate-choice tree scaffold with a paper-faithful `EMLTree`.
- Standardized internal parameters to:
  - `leaf_logits[n_leaves, 3]`
  - `blend_logits[n_internal, 2]`
- Reworked forward evaluation to follow the PyTorch-v16 level-by-level tree semantics.
- Added manual initialization support from EML expressions.

Relevant files:

- [src/models/EMLLayer.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/src/models/EMLLayer.jl:1)
- [src/EMLRegression.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/src/EMLRegression.jl:1)
- [test/models/test_eml_layer.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/test/models/test_eml_layer.jl:1)

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

Relevant files:

- [src/targets/TargetRegistry.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/src/targets/TargetRegistry.jl:1)
- [src/training/TrainConfig.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/src/training/TrainConfig.jl:1)
- [src/training/TrainLoop.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/src/training/TrainLoop.jl:1)
- [src/snapping/Snap.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/src/snapping/Snap.jl:1)
- [src/eval/Metrics.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/src/eval/Metrics.jl:1)
- [src/eval/Recovery.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/src/eval/Recovery.jl:1)
- [scripts/run_experiment.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/scripts/run_experiment.jl:1)
- [scripts/run_suite.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/scripts/run_suite.jl:1)
- [scripts/summarize_results.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/scripts/summarize_results.jl:1)
- [scripts/run_paper_headless_sweep.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/scripts/run_paper_headless_sweep.jl:1)

## Tuple-Gradient Bug Investigation

### Symptom

The reduced paper-style sweep exposed a reproducible failure in `depth4 random_hot` on the paper-style domain `1.0:0.1:3.0`.

Reproducer:

```bash
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config /tmp/depth4_random_hot.toml --target eml_depth4 --seed 137
```

Observed error:

- Zygote `MethodError: no method matching -(::Nothing)`
- This occurred during backprop through the `EMLTree` forward path.

### Root Cause

The original faithful rewrite returned diagnostics from the forward pass in a way that caused Zygote to build tangents containing `Nothing` inside complex array gradients. The failure was structural, not just a bad loss term.

The problematic path was:

- multi-level forward collection logic
- complex broadcast differentiation through `eml`
- mixed tangent propagation where `Nothing` was not normalized away

### Fixes Applied

1. Reworked `EMLTree` forward evaluation to avoid the earlier recursive / mutation-heavy output collection path.
2. Flattened `eml_outputs` handling in the loss path so intermediate penalties operate on a simple vector.
3. Added a custom reverse rule for `eml` in [src/EMLRegression.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/src/EMLRegression.jl:56) that coerces `nothing` tangents to zero.
4. Locked in a regression test that asserts the training call returns normally for the formerly crashing case.

Regression test:

- [test/training/test_train_loop.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/test/training/test_train_loop.jl:60)

## Current State

### Verified

Fresh verification commands run successfully:

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'
```

What that means:

- the structural tuple-gradient crash is fixed
- the rewrite test suite is green
- the paper headless sweep runner smoke test passes

Additional regression coverage now exists for:

- nonfinite search steps incrementing restart counters instead of aborting immediately
- exact-one gate bypass behavior in `_complex_blend`
- nested gradient clipping to max norm `1.0`

### Not Yet Solved

`depth4 random_hot` no longer crashes, but it can still terminate immediately with `failure_reason == nonfinite_detected`.

Current diagnosis:

- initial predictions remain finite
- intermediate outputs remain finite
- but the paper-style wide-domain random-hot initialization can produce very large finite outputs
- the resulting MSE overflows to `Inf`
- Julia currently treats this as immediate failure

This is now a numerical stability gap, not an autodiff bug.

After adding restart logic, `_BYPASS_THR`, and gradient clipping, the reduced Julia sweep on `depth2,depth3,depth4` with `--search-iters 50 --hardening-iters 20 --seeds 1` still shows:

- `depth4 random_hot`: `failure_reason=no_failure`, `nan_restarts=1`, `nonfinite_steps=70`
- all other reduced jobs: `nan_restarts=0`, `nonfinite_steps=0`
- all reduced jobs in this short sweep: `fit_success=false`, `stable_symbol_success=false`

So the current bottleneck is not the hard crash path anymore. It is overflow-prone optimization dynamics on the wide-domain `depth4 random_hot` case.

## Remaining Work

### Priority 1: Paper-Style Nonfinite Handling

Mirror the PyTorch-v16 behavior more closely in [src/training/TrainLoop.jl](/Users/terasaki/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-faithful-rewrite/src/training/TrainLoop.jl:1):

- implement `nan_streak`
- implement `nan_restart_patience`
- implement `max_nan_restarts`
- reload best soft state on repeated nonfinite steps
- continue training instead of aborting immediately on the first nonfinite loss

This is the largest remaining gap between current Julia behavior and the external reference.

### Priority 2: Numerical Stabilization Alignment

Match the PyTorch implementation more closely in the forward / optimization path:

- re-check the already added bypass threshold behavior equivalent to `_BYPASS_THR`
- re-check the already added gradient clipping equivalent to `clip_grad_norm_(..., 1.0)`
- re-check clamp behavior and loss overflow behavior under paper domains
- re-evaluate whether `eml_clamp=1e300` should remain the default for Julia runs without more aggressive restart logic
- inspect whether Julia `MSE` overflow is happening before restart logic can provide enough benefit

### Priority 3: Sweep Measurement

After the nonfinite/restart path is implemented:

1. rerun reduced `depth2,depth3,depth4` sweep
2. inspect raw JSON outputs for recovery tendency
3. expand to `depth5_random`, `depth6_random`, and manual-noise jobs
4. compare the observed recovery pattern against the PyTorch README and Supplementary expectations

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

The rewrite has crossed the main structural milestone: the new Section 4.3 core is in place, and the tuple-gradient failure that blocked paper-style sweep work is fixed.

The next milestone is numerical: make nonfinite handling and restart behavior paper-compatible enough that `depth 2-6` recovery tendencies can be measured rather than failing early on wide-domain random initializations.
