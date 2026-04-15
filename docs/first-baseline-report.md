# First Baseline Report

## Scope

This baseline covers the current CPU-first reproduction scaffold for Section 4.3:

- `must_pass_depth2`
- `must_pass_depth3`
- `challenge_depth4`

## Commands Used

```bash
~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/must_pass_depth2.toml
~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/must_pass_depth3.toml
~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/challenge_depth4.toml
~/.juliaup/bin/julia --project=. scripts/summarize_results.jl
```

## Observations

- Raw JSON artifacts were produced successfully for all configured runs.
- Summary CSV was generated successfully at `results/summaries/summary.csv`.
- Blind recovery success is currently `0` for every target and depth in this baseline.

## Expected Reason For Failure

This is consistent with the current implementation stage:

- the `Lux` model is still a minimal structural placeholder,
- the training loop logs losses but does not yet perform meaningful gradient-based parameter updates,
- snapping always collapses to a trivial terminal choice,
- exported recovered formulas are therefore trivial and do not match the targets.

## Current Bottlenecks

1. The EML tree model does not yet represent the paper's full master-formula choice structure.
2. The training loop needs real optimization, not metric-only passes.
3. Snapping needs ambiguity handling and structured tree reconstruction.
4. Recovery should validate the snapped tree, not the pre-snap placeholder forward path only.

## Next Implementation Priorities

1. Replace the placeholder forward model with per-node branch-choice logits.
2. Add real optimizer updates and a distinct hardening objective.
3. Rebuild snapped trees into readable multi-node EML formulas.
4. Re-run the same suites and compare success-rate changes against this baseline.
