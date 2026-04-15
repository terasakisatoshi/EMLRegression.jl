# Paper-Aligned Benchmark Redesign

**Date:** 2026-04-15

**Goal:** Replace the current named-function benchmark suites with paper-aligned EML-composed target suites so blind recovery rates can be compared meaningfully to Section 4.3.

## Scope

Included:

- replace named-function targets with EML-composed benchmark targets,
- define success by snapped-tree validation behavior,
- add benchmark metadata for structure match and ambiguity,
- rewrite suite configs and summaries around the paper-aligned targets,
- run depth-specific suites for paper-style recovery-rate measurement.

Not included:

- changing the core EML operator,
- adding non-EML baselines,
- broad symbolic simplification beyond target/export needs.

## Benchmark Definition

- `must_pass_depth2`, `must_pass_depth3`, and `challenge_depth4` are retained as suite names but their contents become paper-aligned EML-composed target sets.
- Each target is defined by a discrete EML tree, not by a named elementary function evaluator.
- The depth 2 suite contains targets intentionally chosen to be recoverable at shallow depth.
- The depth 3 suite contains targets first expressible or practically recoverable at depth 3.
- The depth 4 suite remains the hardest blind-recovery benchmark.

## Success Metric

- Primary metric: blind recovery success rate from random initialization.
- Success is determined from the snapped tree on independent validation samples.
- `snap_status` and `structure_match` are reported as secondary diagnostics.
- Low continuous-model loss alone is never sufficient.

## File Responsibilities

- `src/targets/TargetRegistry.jl`: paper benchmark target definitions backed by discrete EML trees.
- `src/trees/TreeTypes.jl`: shared target-tree and recovered-tree structures.
- `src/snapping/Snap.jl`: snapped-tree diagnostics including ambiguity and structure helpers.
- `src/eval/Metrics.jl`: richer experiment verdict fields.
- `src/eval/Recovery.jl`: snapped-tree-centered experiment evaluation.
- `scripts/run_suite.jl` and `scripts/summarize_results.jl`: benchmark execution and paper-style summary metrics.
- `experiments/configs/*.toml`: paper-aligned suite definitions.

## Success Criteria

This redesign is successful when:

- suite targets are discrete EML trees rather than named functions,
- experiment verdicts are based on snapped-tree validation,
- summary outputs report recovery rate, ambiguity rate, and structure-match rate,
- depth-specific benchmark runs support direct comparison with the paper’s qualitative trend.
