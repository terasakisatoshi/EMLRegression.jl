# Section 4.3 Reproduction Guide

## Environment Setup

```bash
~/.juliaup/bin/julia --version
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Test Command

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'
```

## Single Smoke Run

```bash
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target depth2_double_exp --seed 1
```

Expected output:

- one JSON artifact in `results/raw/`
- raw JSON includes `snap_status`, `structure_match`, `numerical_match`, `training_failure_reason`, `max_output_abs`, `max_node_abs`, `ambiguous_nodes`, and `validation_loss`
- rerunning the same `config-target-seed` combination overwrites that file

## Batch Run

```bash
~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/must_pass_depth2.toml
```

## Summary

```bash
~/.juliaup/bin/julia --project=. scripts/summarize_results.jl
```

Expected output:

- `results/summaries/summary.csv`
- summary columns include `success_rate`, `numerical_match_rate`, `snap_ok_rate`, `ambiguous_rate`, `structure_match_rate`, `training_failure_rate`, `mean_max_output_abs`, `mean_max_node_abs`, `mean_validation_loss`, `init_strategies`, and `target_noise_stds`
- summary includes every JSON currently present in `results/raw/`
- `challenge_depth4_sweep-longer_cool` is the current depth-4 tuned variant to inspect after `top-k` snapping search
- `challenge_depth4_init_sweep-zero_bias` is the current best initialization variant among the wider 8-seed depth-4 runs
- `challenge_depth4_basin_sweep-*` is the current basin-of-attraction style check starting from noisy target-tree initializations
- `challenge_depth5_basin_sweep-*` is the current depth-5 basin-of-attraction check
- `challenge_depth6_sweep-*` is the current negative control for blind recovery at depth 6

## Formula Export

```bash
~/.juliaup/bin/julia --project=. scripts/export_recovered_formulas.jl
```

## Output Locations

- raw experiment artifacts: `results/raw/`
- summary tables: `results/summaries/`
- generated figures: `results/figures/`

## Notes

- If you want a summary for only one suite run, archive or clean `results/raw/` before running `scripts/summarize_results.jl`.
- The current paper-aligned baseline recovers both `must_pass_depth2` targets with exact snapped structure.
- `must_pass_depth3_sweep-cooler_hardening` now reaches strict recovery, and depth-4 recovery should be checked through `challenge_depth4_sweep-longer_cool` rather than the untuned baseline.
- `challenge_depth4_init_sweep` widens the depth-4 evaluation to `8` seeds and multiple initialization strategies. In the current results, `zero_bias_to_inputs` reaches `6/8`, `small_gaussian` reaches `5/8`, and `margin_biased` reaches `3/8`.
- `challenge_depth4_basin_sweep` starts from the target tree plus Gaussian noise in logits. The current `depth4_nested` results are `8/8` for `σ=0.05`, `σ=0.10`, and `σ=0.25`.
- `challenge_depth5_sweep` currently remains a blind-recovery failure for both `small_gaussian` and `zero_bias_to_inputs`, with `0/8` strict recovery in each variant.
- `challenge_depth5_basin_sweep` starts from noisy target-tree logits for `depth5_affine_log`. The current `σ=0.05`, `σ=0.10`, and `σ=0.25` variants are all `8/8`.
- `challenge_depth6_sweep` is the current negative control. Both `small_gaussian` and `zero_bias_to_inputs` are `0/4`, which matches the paper's direction even though the sample size is much smaller.
- `run_experiment` now applies a `top-k` snapping search before final greedy refinement, so recovered formulas can be structurally correct even when plain argmax snapping would collapse to a shallow tree.
