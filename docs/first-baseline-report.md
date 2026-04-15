# First Baseline Report

## Scope

This baseline covers the paper-aligned benchmark suites currently shipped in the CPU-first reproduction scaffold:

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

- Raw JSON artifacts were produced successfully for all 15 configured runs.
- Summary CSV was generated successfully at `results/summaries/summary.csv`.
- The new summary now reports numerical recovery, ambiguity, and structure-match rates per target.

## Current Baseline

| suite | target | runs | success_rate | ambiguous_rate | structure_match_rate | mean_validation_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `must_pass_depth2` | `depth2_double_exp` | 5 | 1.0 | 0.0 | 1.0 | 0.0 |
| `must_pass_depth2` | `depth2_exp` | 5 | 1.0 | 0.0 | 1.0 | 0.0 |
| `must_pass_depth3` | `depth3_log` | 3 | 1.0 | 0.6667 | 0.0 | 5.93e-32 |
| `challenge_depth4` | `depth4_nested` | 2 | 0.0 | 0.0 | 0.0 | 36.3271 |

## Expected Reason For Failure

This is still consistent with the current implementation stage:

- the tuned depth-2 suite now recovers both easiest targets with exact snapped structure,
- the tuned depth-3 suite now recovers a numerically exact equivalent formula for all configured seeds,
- depth-3 recovery is still structurally mismatched and often marked ambiguous under the current margin threshold,
- the depth-4 suite still falls outside the current optimization basin,
- exported formulas show the remaining gap is now concentrated in exact structure recovery and depth-4 search.

## Current Bottlenecks

1. The easiest paper-aligned settings are now reproducible through depth 3, but depth-3 success is only numerical, not structural.
2. The depth-4 target still needs stronger optimization or a different inductive bias.
3. The challenge suite is now decisive rather than ambiguous, which narrows the remaining problem to model/search quality.
4. The current benchmark family is paper-aligned, but the observed rates still diverge from the paper once exact structure recovery and depth 4 are considered.

## Next Implementation Priorities

1. Convert `must_pass_depth3` from numerical recovery to exact structure recovery.
2. Improve the depth-4 search so the challenge suite does better than a clean but wrong snap.
3. Decide whether ambiguity should be regularized away or treated as acceptable when the snapped formula is numerically exact.
4. Re-run the same suites and compare all four reported rates against this baseline.
