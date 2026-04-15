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
- raw JSON includes `snap_status`, `structure_match`, `numerical_match`, `training_failure_reason`, `ambiguous_nodes`, and `validation_loss`
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
- summary columns include `success_rate`, `numerical_match_rate`, `snap_ok_rate`, `ambiguous_rate`, `structure_match_rate`, `training_failure_rate`, and `mean_validation_loss`
- summary includes every JSON currently present in `results/raw/`

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
- The current paper-aligned baseline recovers both `must_pass_depth2` targets with exact snapped structure, and `must_pass_depth3` now reaches numerical recovery without exact structure recovery, while `challenge_depth4` still fails. See `docs/first-baseline-report.md` for the current interpretation.
