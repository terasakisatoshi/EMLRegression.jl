# EMLRegression

Julia research repository for reproducing Section 4.3 of "All elementary functions from a single operator" with `Lux.jl`.

## Current Milestone

The current milestone is a CPU-first reproduction of trainable EML trees with complex-valued internal computation, staged optimization, snapping, and blind recovery evaluation across depths 2-4.

## Running Tests

```bash
~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'
```

## Smoke Experiment

```bash
~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target ln --seed 1
~/.juliaup/bin/julia --project=. scripts/summarize_results.jl
```

The supported path is CPU-first. GPU execution is not part of the current milestone.

## Results Layout

- `results/raw/`: per-run artifacts
- `results/summaries/`: aggregated metrics
- `results/figures/`: generated plots and figures
