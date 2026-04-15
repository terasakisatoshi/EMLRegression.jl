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
| `must_pass_depth2` | `depth2_double_exp` | 5 | 1.0 | 0.0 | 0.0 | 0.0 |
| `must_pass_depth2` | `depth2_exp` | 5 | 0.0 | 0.0 | 0.0 | 1.0 |
| `must_pass_depth3` | `depth3_log` | 3 | 0.0 | 0.0 | 0.0 | 9.6460 |
| `challenge_depth4` | `depth4_nested` | 2 | 0.0 | 1.0 | 0.0 | 37.8314 |

## Expected Reason For Failure

This is still consistent with the current implementation stage:

- the model can now recover the easiest numerical target, but exact snapped structure recovery is still absent,
- depth 3 and depth 4 targets still fall outside the current optimization basin,
- ambiguity remains unresolved on the hardest suite,
- exported formulas show repeated near-miss trees rather than the target trees themselves.

## Current Bottlenecks

1. Numerical recovery can happen without structure recovery, so the snapping objective is still too weak.
2. The depth-3 and depth-4 targets need stronger optimization or better hardening schedules.
3. Ambiguous node selections remain common on the challenge suite.
4. The current benchmark family is paper-aligned, but the observed rates are still far from the paper's reported depth scaling.

## Next Implementation Priorities

1. Close the gap between numerical success and exact structure recovery on `must_pass_depth2`.
2. Improve hardening and snapping so `challenge_depth4` no longer stays fully ambiguous.
3. Raise `must_pass_depth3` above `0/3` before expanding the benchmark family further.
4. Re-run the same suites and compare all four reported rates against this baseline.
