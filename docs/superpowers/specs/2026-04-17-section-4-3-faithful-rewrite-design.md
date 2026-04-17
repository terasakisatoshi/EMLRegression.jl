# Section 4.3 Faithful Rewrite Design

**Date:** 2026-04-17

**Goal:** Replace the current Julia scaffold with a PyTorch v16 and Section 4.3 faithful EML tree implementation, including parameterization, training schedule, snapping, targets, and recovery metrics that track the paper's depth-2 to depth-6 behavior as closely as practical.

## Scope

This design replaces the current `choice-logit` tree scaffold with a new standard implementation derived from:

- Section 4.3 of the paper,
- `extern/SymbolicRegressionPackage/EML_toolkit/EmL_training/Log_fit.nb`,
- `extern/SymbolicRegressionPackage/EML_toolkit/EmL_training/PyTorch_v16_final/tree_prototype_torch_v16_final.py`,
- `extern/SymbolicRegressionPackage/EML_toolkit/EmL_training/PyTorch_v16_final/depth_2_to_6_headless.sh`,
- `extern/SymbolicRegressionPackage/EML_toolkit/EmL_training/PyTorch_v16_final/README.md`.

Included:

- full replacement of the internal tree parameterization,
- PyTorch-v16-style forward semantics,
- Section-4.3-style target functions for depths 2 through 6,
- two-stage training with search and hardening,
- paper-aligned snapping and uncertainty analysis,
- recovery metrics based on snapped numerical exactness and uncertainty counts,
- sweep support for random-start and manual-noise experiments.

Not included:

- backward compatibility with the current public API,
- preserving the current candidate-table tree abstraction,
- exhaustive symbolic verification or compiler reconstruction beyond what is needed for Section 4.3,
- wrapping the external Python implementation instead of reimplementing it in Julia.

## Recommended Approach

Replace the current implementation wholesale and make the PyTorch-v16-compatible model the new repository standard.

Why this approach:

- the current implementation differs from the paper at the most important level, the parameterization,
- attempting to preserve both designs would increase complexity and make results harder to interpret,
- the milestone explicitly prioritizes paper fidelity and recovery behavior over compatibility.

## Architecture

### Core Tree Representation

The new standard tree representation is a full binary tree determined only by `depth`.

The trainable parameters are:

- `leaf_logits::Matrix{Float64}` with shape `(2^depth, 3)`,
- `blend_logits::Matrix{Float64}` with shape `(2^depth - 1, 2)`.

The three leaf logits correspond to `{1, x, y}`.
The two blend logits for each internal node correspond to whether the left and right inputs use the constant `1` versus the corresponding child output.

This is not the current `TreeNodeSpec` candidate-source model.
It is a direct structural match to the PyTorch v16 trainer.

### Forward Semantics

Forward evaluation follows the PyTorch v16 implementation exactly in structure:

1. compute leaf probabilities with `softmax(leaf_logits / tau_leaf)`,
2. construct leaf mixtures from `{1, x, y}`,
3. evaluate the tree level by level,
4. compute gate probabilities with `sigmoid(blend_logits / tau_gate)`,
5. blend each internal-node input between `1` and the corresponding child,
6. apply `eml(left_input, right_input)` at every internal node,
7. clamp and sanitize intermediate complex values to avoid runaway overflow and NaN cascades.

The standard numeric mode is complex double precision and should be treated as the Julia analogue of `torch.complex128`.

### Targets

The new default target suite is the PyTorch v16 suite:

- `eml_depth2`,
- `eml_depth3`,
- `eml_depth4`,
- `eml_depth5`,
- `eml_depth6`.

These targets replace the current curated `RecoveredTree`-backed registry as the default Section 4.3 benchmark.

Training data uses the same policy as the PyTorch implementation:

- grid data on `[data_lo, data_hi]^2` for training,
- random points on `[gen_lo, gen_hi]^2` for generalization,
- retain only points whose complex evaluation lands on the real domain with finite real values and negligible imaginary part.

### Training

Training is a two-stage schedule:

1. `SEARCH`
   fixed high temperature,
   Adam,
   optimize the soft model,
   monitor plateaus and hard diagnostics.

2. `HARDENING`
   anneal temperature from `tau_search` to `tau_hard`,
   increase entropy and binarity penalties,
   reduce effective learning rate according to the PyTorch schedule,
   optionally finish with a short LBFGS polish.

The repository's new standard loss is the PyTorch v16 composite objective:

- data MSE,
- uncertainty-weighted leaf entropy penalty,
- uncertainty-weighted gate binarity penalty,
- intermediate-magnitude penalty.

The current Julia-specific complexity penalty, margin penalty, and custom snap search are removed from the default faithful pipeline.

### Snapping and Recovery

Snapping becomes a direct hard projection:

- leaves snap by argmax over the 3 leaf logits,
- gates snap by sign of the 2 blend logits.

Uncertainty is measured before projection using the same thresholds as the PyTorch implementation.

Primary recovery metrics become:

- `fit_success`,
- `symbol_success`,
- `stable_symbol_success`,
- `n_uncertain`.

The current `structure_match`-driven success definition is demoted from the primary benchmark path.

### Sweep Runner

The repository should gain a Julia-side sweep runner equivalent in intent to `depth_2_to_6_headless.sh`.

The first milestone sweep matrix is:

- depth 2 random-start,
- depth 3 random-start,
- depth 4 random-start,
- depth 5 random-start with `random_hot`,
- depth 6 random-start with `random_hot`,
- depth 5 random-start with `uniform`,
- depth 6 random-start with `uniform`,
- depth 5 manual exact tree plus noise `12`,
- depth 6 manual exact tree plus noise `12`.

The default iteration budgets should initially match the PyTorch script as closely as possible.

## Data Flow

1. select a paper-aligned target function and depth,
2. build training and generalization datasets by real-domain filtering,
3. initialize `leaf_logits` and `blend_logits` according to the chosen init strategy,
4. run search phase,
5. transition to hardening according to plateau and hard-diagnostic rules,
6. optionally run short LBFGS polish,
7. analyze snap uncertainty,
8. hard-project the model,
9. evaluate snapped-tree MSE and uncertainty-based success flags,
10. record per-seed summaries and aggregate recovery statistics.

## File Responsibilities

- `src/models/EMLLayer.jl` or a replacement file in `src/models/`: the new `EMLTree` core, parameter storage, forward semantics, hard projection, and snap analysis.
- `src/training/TrainConfig.jl`: PyTorch-v16-style configuration surface, including temperature schedule, penalties, restart controls, thresholds, and sweep controls.
- `src/training/TrainLoop.jl`: search phase, hardening phase, optional polish, diagnostics, and per-seed summaries.
- `src/targets/TargetRegistry.jl`: target function definitions, real-domain filtering, grid data, and generalization sampling.
- `src/eval/Recovery.jl`: `fit_success`, `symbol_success`, `stable_symbol_success`, uncertainty-aware summaries, and generalized evaluation helpers.
- `scripts/`: sweep entry points that mirror the `depth_2_to_6_headless.sh` experiment families.
- `docs/src/`: updated user-facing explanation that the repository now follows the PyTorch-v16-compatible Section 4.3 model rather than the older candidate-choice scaffold.

The existing tree-type files may be reduced or removed if they no longer serve the new model.

## Migration Decisions

The following current concepts are retired from the faithful path:

- candidate-table `MasterTree` as the primary trainable structure,
- node-wise source-choice beam search,
- greedy formula refinement after snapping,
- strict success defined primarily by recovered discrete structure match,
- current curated target registry as the benchmark standard for Section 4.3.

This is an intentional replacement, not an additive feature.

## Testing Strategy

Add tests that prove fidelity to the external reference design instead of only preserving the current scaffold.

Required unit-level coverage:

- leaf and gate tensor shapes for each depth,
- forward evaluation shape and determinism at fixed seeds,
- hard projection produces one-hot leaves and binary gates,
- snap uncertainty counting matches expected edge cases,
- real-domain filtering rejects invalid points and preserves valid ones,
- manual initialization from exact expressions populates logits correctly,
- search-to-hardening transition occurs under the configured conditions.

Required integration-level coverage:

- depth 2 random-start yields success under the paper-aligned budget,
- depth 3 and depth 4 recover at nonzero rates under the paper-aligned sweep,
- depth 5 random-start remains rare but nonzero under at least one supported init family,
- depth 6 random-start remains a negative control under the default budget,
- manual-noise initialization remains robust at depths 5 and 6 according to the chosen success metric.

## Success Criteria

This design is successful when the repository can truthfully claim all of the following:

- the default Section 4.3 model uses the PyTorch-v16-compatible `leaf_logits` and `blend_logits` representation,
- training follows the search-plus-hardening schedule rather than the previous Julia-specific scaffold,
- snapping is a direct hard projection with explicit uncertainty accounting,
- the default depth-2 to depth-6 targets and sweeps match the external reference setup,
- observed recovery behavior trends in the same direction as the external PyTorch materials,
- the older candidate-choice implementation is no longer the standard path for Section 4.3 experiments.

## Risks and Open Questions

- Julia complex autodiff may not match PyTorch exactly, especially around `log`, clamp behavior, and NaN handling.
- The precise balance of penalties and restart behavior may need tuning even if the structure matches.
- The existing repository docs and tests currently assume the older model and will need coordinated replacement.
- The current milestone measures directional agreement with the paper's depth-wise recovery behavior, not a bit-for-bit reimplementation guarantee.
