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
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target ln --seed 1
```

Expected output:

- one JSON artifact in `results/raw/`

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

## Formula Export

```bash
~/.juliaup/bin/julia --project=. scripts/export_recovered_formulas.jl
```

## Output Locations

- raw experiment artifacts: `results/raw/`
- summary tables: `results/summaries/`
- generated figures: `results/figures/`
