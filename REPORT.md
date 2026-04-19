# Section 4.3 Faithful Rewrite Report

Date: 2026-04-17
Branch: `section-4-3-remaining-work`
Worktree: `.worktrees/section-4-3-remaining-work`

## Summary

Section 4.3 faithful rewrite remains the main implementation direction. The old choice-logit tree path has been replaced by a PyTorch-v16-style `EMLTree` core with paper-aligned targets, search/hardening training, paper-style snapping, and paper-style recovery metrics.

The latest blocker on `main` was a paper-budget training failure, not a modeling mismatch. Full `6000/2000` hardening could hit a Zygote shape error after the gates had already become highly saturated. That blocker is now fixed in this worktree.

Current work is in the measurement/reporting phase. The sweep and summary scripts now emit a Section 4.3 family table alongside the aggregate summary, and this worktree contains a post-stabilization paper-budget sample rerun for `depth3` and `depth4` plus a `depth2` anchor run.

```bash
~/.juliaup/bin/julia --project=. scripts/run_paper_headless_sweep.jl --jobs depth2,depth3,depth4,depth5_random,depth6_random,depth5_manual_noise12,depth6_manual_noise12
~/.juliaup/bin/julia --project=. scripts/summarize_results.jl
```

The local raw artifacts are still far short of the full paper sweep, but they are enough to provide a tentative signal: the broad `depth3` forward/post-update and nonfinite-gradient loop is suppressed in this sample, and all sampled `depth3`/`depth4` random-init families now reach `fit_success=true`, `symbol_success=true`, and `stable_symbol_success=true`. The remaining gap has shifted away from one-seed recovery quality and toward wider seed confirmation plus the untouched `depth5+` families.

## What Was Done

### 1. Model / Parameterization Rewrite

- Replaced the prior candidate-choice tree scaffold with a paper-faithful `EMLTree`.
- Standardized internal parameters to:
  - `leaf_logits[n_leaves, 3]`
  - `blend_logits[n_internal, 2]`
- Reworked forward evaluation to follow the PyTorch-v16 level-by-level tree semantics.
- Added manual initialization support from EML expressions.

Relevant files:

- [src/models/EMLLayer.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/src/models/EMLLayer.jl:1)
- [src/EMLRegression.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/src/EMLRegression.jl:1)
- [test/models/test_eml_layer.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/test/models/test_eml_layer.jl:1)

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

- [src/targets/TargetRegistry.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/src/targets/TargetRegistry.jl:1)
- [src/training/TrainConfig.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/src/training/TrainConfig.jl:1)
- [src/training/TrainLoop.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/src/training/TrainLoop.jl:1)
- [src/snapping/Snap.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/src/snapping/Snap.jl:1)
- [src/eval/Metrics.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/src/eval/Metrics.jl:1)
- [src/eval/Recovery.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/src/eval/Recovery.jl:1)
- [scripts/run_experiment.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/scripts/run_experiment.jl:1)
- [scripts/run_suite.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/scripts/run_suite.jl:1)
- [scripts/summarize_results.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/scripts/summarize_results.jl:1)
- [scripts/run_paper_headless_sweep.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/scripts/run_paper_headless_sweep.jl:1)

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

- [test/training/test_train_loop.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/test/training/test_train_loop.jl:1)
- [test/models/test_eml_layer.jl](/Users/atelierarith/work/terasakisatoshi/EMLRegression.jl/.worktrees/section-4-3-remaining-work/test/models/test_eml_layer.jl:1)

Commit:

- `03d9b73 fix: unblock paper-budget hardening gradients`

## Current State

### Verified

Fresh verification commands run successfully:

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["models"])'
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["eval"])'
```

What that means:

- the structural tuple-gradient crash is fixed
- the paper-budget hardening crash is fixed
- the training test suite is green

Additional regression coverage now exists for:

- nonfinite search steps incrementing restart counters instead of aborting immediately
- exact-one and saturated-gate blend behavior in `_complex_blend`
- paper-budget saturated gates remaining differentiable

### Post-Stabilization Paper-Budget Sample Rerun

This worktree now has a small local paper-budget rerun:

- `pnas_d2_random-biased`: `1/1` fit, `1/1` symbolic, `0/1` stable symbolic, `nonfinite_steps=0`, `nonfinite_grad_steps=0`, `nan_restarts=0`
- `pnas_d3_random-biased`: `1/1` fit, `1/1` symbolic, `1/1` stable symbolic, `nonfinite_steps=0`, `nonfinite_grad_steps=0`, `nan_restarts=0`
- `pnas_d3_random-uniform`: `1/1` fit, `1/1` symbolic, `1/1` stable symbolic, `nonfinite_steps=0`, `nonfinite_grad_steps=0`, `nan_restarts=0`
- `pnas_d3_random-xy_biased`: `1/1` fit, `1/1` symbolic, `1/1` stable symbolic, `nonfinite_steps=0`, `nonfinite_grad_steps=0`, `nan_restarts=0`
- `pnas_d3_random-random_hot`: `1/1` fit, `1/1` symbolic, `1/1` stable symbolic, `nonfinite_steps=0`, `nonfinite_grad_steps=0`, `nan_restarts=0`
- `pnas_d4_random-biased`: `1/1` fit, `1/1` symbolic, `1/1` stable symbolic, `nonfinite_steps=0`, `nonfinite_grad_steps=0`, `nan_restarts=0`
- `pnas_d4_random-uniform`: `1/1` fit, `1/1` symbolic, `1/1` stable symbolic, `nonfinite_steps=0`, `nonfinite_grad_steps=0`, `nan_restarts=0`
- `pnas_d4_random-xy_biased`: `1/1` fit, `1/1` symbolic, `1/1` stable symbolic, `nonfinite_steps=0`, `nonfinite_grad_steps=0`, `nan_restarts=0`
- `pnas_d4_random-random_hot`: `1/1` fit, `1/1` symbolic, `1/1` stable symbolic, `nonfinite_steps=0`, `nonfinite_grad_steps=0`, `nan_restarts=0`, `failure_reason=no_failure`

This sample is too small to replace the paper-scale tendency table, but it is large enough to change the near-term debugging targets:

- `depth3` no longer shows recorded forward/post-update nonfinite, nonfinite-gradient, or restart pressure across the four random initialization families in this sample, and it is now `4/4` on `fit_success`, `symbol_success`, and `stable_symbol_success`.
- `depth4` is also `4/4` on `fit_success`, `symbol_success`, and `stable_symbol_success` in this sample with `nonfinite_steps=0`, `nonfinite_grad_steps=0`, and `nan_restarts=0`; `random_hot` is no longer a special overflow case.

Four training-path changes were active during this rerun:

- hardening now advances its temperature step even when an iteration fails; the regression suite checks that late hardening tau values continue to move rather than freezing
- the faithful loss path now defaults to linear uncertainty weighting (`uncertainty_power=1.0`) to match the current PyTorch reference more closely
- nonfinite gradient entries are scrubbed before clipping/update, and raw results now record those events separately as `nonfinite_grad_steps`
- gate probabilities now use a numerically stable sigmoid in both training and snapping paths, avoiding `exp` overflow in the hardening tail

One recovery-side change was active during the latest rerun:

- the snap beam now breaks near-equal candidates by lower ambiguity count, so fixed-seed `depth3` recovery no longer gets stuck on a higher-ambiguity exact projection when a lower-ambiguity exact projection is already in beam reach
- the snap beam now sizes its step budget from `analyze_snap` and allows larger ambiguity-cleanup passes on `depth4`-style fixtures, which is what moved the sampled `depth3` / `depth4` families from symbolic-only to stable symbolic

The summary pipeline also needed one robustness fix: failed runs can emit `hardening_iter = null`, so `scripts/summarize_results.jl` now loads that field as missing and emits `NaN` for group means when no hardening iteration exists instead of crashing during aggregation.

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

### Priority 1: Wider Confirmation Before The Expensive Full Sweep

The local sample rerun no longer points to `depth3` as a broad numerical-instability problem, a blind symbolic-recovery failure, or a one-seed stable-recovery failure. The next pass should confirm that the sampled `depth3` / `depth4` wins survive a wider seed slice, and only then rerun the expensive full paper-style sweep:

```bash
~/.juliaup/bin/julia --project=. scripts/summarize_results.jl
```

The eventual measurement deliverable is still the family tendency table for:

- `depth2`
- `depth3`
- `depth4`
- `depth5_random`
- `depth6_random`
- `depth5_manual_noise12`
- `depth6_manual_noise12`

### Priority 2: Numerical Stabilization Alignment

The current local evidence suggests three concrete follow-ups:

- confirm whether the new `depth3` / `depth4` one-seed stable-symbolic wins survive a wider seed sample, especially for `random_hot`
- move back out to the untouched `depth5_random`, `depth6_random`, `depth5_manual_noise12`, and `depth6_manual_noise12` families once the wider `depth3` / `depth4` slice is in place
- compare those wider-seed outcomes against the PyTorch implementation's published tendencies before paying for the full sweep

### Priority 3: Sweep Measurement

After the next stabilization pass:

1. rebuild `results/summaries/summary.csv`
2. rebuild `results/summaries/section_4_3_summary.csv`
3. compare each family against the PyTorch README / Supplementary tendencies
4. update this report with completed family counts instead of the current small-sample rerun
5. decide whether the next pass should stay on wider `depth3` / `depth4` confirmation or widen back out to the untouched `depth5+` jobs

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

The new local rerun changes the tentative prioritization: recorded forward/post-update and nonfinite-gradient `depth3` failures are suppressed in this sample, and stable symbolic recovery is now present across sampled `depth3` and `depth4` random-init families. The remaining gap is broader seed confirmation plus the untouched `depth5+` jobs. The full sweep still matters, but it should follow those confirmations rather than lead them.
