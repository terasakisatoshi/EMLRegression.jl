# Snap Diagnostics And Margin-Aware Search Design

**Date:** 2026-04-16

**Goal:** Make depth5 snapping failures observable at the active-node level and bias top-k recovery search toward the lowest-margin decisions before changing training again.

## Scope

This design covers two connected changes in the snapping pipeline:

- record per-active-node snapping diagnostics in raw experiment artifacts,
- make top-k snapping search expand the lowest-margin active node sides first.

Included:

- active-node `left_margin` and `right_margin` reporting,
- raw JSON persistence for diagnostics,
- neighbor ordering based on side-level margins,
- rerunning the `depth5_blind_margin` sweep and regenerating summaries.

Not included:

- new training losses or regularizers,
- summary-level aggregation of per-node diagnostics,
- broad refactors of the snapping module.

## Recommended Approach

Keep the existing beam objective and candidate evaluation intact, and only change the observability and expansion order around it.

Why this approach:

- it directly targets the current `snap = :ambiguous` failure mode,
- it lets us identify bottleneck nodes without changing recovery semantics,
- it isolates the effect of margin-aware search from later threshold sweeps.

## Architecture

### Active Snap Diagnostics

`src/snapping/Snap.jl` will expose a diagnostics helper that computes information only for nodes that are active under the snapped tree. Each diagnostic row will include:

- `node_id`,
- `left_choice`,
- `right_choice`,
- `left_margin`,
- `right_margin`,
- `min_margin`,
- `ambiguous`.

Inactive nodes stay excluded so diagnostics match the existing ambiguity semantics.

### Margin-Aware Neighbor Expansion

The current top-k search enumerates neighbors uniformly across active node sides. The updated search will first rank active `(node_id, side)` pairs by ascending margin, then enumerate alternative symbols from those sides in that order. Candidate scoring, beam pruning, and refinement remain unchanged so the only search-policy change is which branches are explored first.

### Raw Artifact Integration

`src/eval/Recovery.jl` will attach the diagnostics to the experiment outcome, and `scripts/run_experiment.jl` will write them into the raw JSON under `snap_diagnostics`. `scripts/summarize_results.jl` will remain unchanged for now.

## Data Flow

1. Train the continuous model as today.
2. Snap logits into a `RecoveredTree`.
3. Compute active-node diagnostics from the snapped tree and raw logits.
4. Use the same active-node margins to prioritize top-k neighbor generation.
5. Persist diagnostics in `results/raw/*.json`.
6. Rebuild summaries after rerunning the `depth5_blind_margin` sweep.

## File Responsibilities

- `src/snapping/Snap.jl`: diagnostics collection, side-level margin ranking, and neighbor ordering.
- `src/eval/Recovery.jl`: package diagnostics with the recovery outcome.
- `src/eval/Metrics.jl`: extend the verdict payload if needed to carry diagnostics.
- `scripts/run_experiment.jl`: serialize diagnostics into raw results.
- `test/snapping/test_snap.jl`: unit coverage for diagnostics and margin-aware ordering.
- `test/integration/test_smoke_ln.jl`: integration coverage for raw artifact shape.

## Testing Strategy

Add focused tests that prove:

- diagnostics report only active nodes and ignore inactive ambiguous nodes,
- per-side margins and snapped choices are serialized in raw results,
- neighbor ordering prefers the lowest-margin active side before higher-margin sides.

After implementation, run targeted tests for `snapping`, `eval`, and integration raw export, then rerun the `depth5_blind_margin` 8-seed sweep and regenerate summaries.

## Success Criteria

This design is successful when all of the following are true:

- raw experiment artifacts identify bottleneck active nodes by margin,
- depth5 snapping search preferentially explores low-margin branches,
- the new behavior is covered by tests,
- the updated `depth5_blind_margin` summary shows whether ambiguous failures decrease.
