# Paper-Aligned EML Tree Training Design

**Date:** 2026-04-15

**Goal:** Replace the current scaffolded training pipeline with a paper-aligned EML tree model that supports recursive tree evaluation, real parameter updates, hardening, and node-wise snapping.

## Scope

This design updates the current Julia/Lux reproduction so that Section 4.3 is modeled as a trainable EML tree rather than a single soft terminal mixture.

Included:

- recursive evaluation of a full binary EML master tree,
- node-wise left/right source selection logits,
- real optimization updates during training,
- a hardening phase that sharpens continuous selections,
- node-wise snapping into a discrete recovered tree,
- downstream evaluation and formula export based on the recovered tree.

Not included:

- outer-loop operator discovery beyond Section 4.3,
- full symbolic proof or compiler reconstruction,
- unrelated repository refactors.

## Recommended Approach

Keep the repository surface mostly stable and replace the internals with a paper-aligned implementation.

Why this approach:

- it preserves the existing CLI and evaluation flow,
- it fixes the real blockers in order: model structure, optimization, then discretization,
- it allows staged verification without rewriting the whole repository.

## Architecture

### Tree Representation

`src/trees/` defines a real master tree, not only a placeholder search space. Each internal node has a stable identifier and candidate source tables for its left and right inputs. Candidate sources include `:const1`, input variables, and outputs of child subtrees where structurally valid.

### Lux Model

`src/models/EMLLayer.jl` evaluates the full tree recursively with complex arithmetic. Parameters are organized per node as `left_logits` and `right_logits`, aligned with the tree’s candidate tables. Forward evaluation uses soft selection during training and the standard `eml(a, b)` operator at every internal node.

### Training

`src/training/` runs two connected phases:

1. soft optimization with real parameter updates,
2. hardening that pushes node-wise selections toward discrete decisions.

`TrainConfig` expands to include optimizer and hardening controls such as learning rate, temperature, hardening start, hardening weight, and snap margin.

### Snapping and Recovery

`src/snapping/` snaps each node’s left and right selections independently. The output is a discrete `RecoveredTree`, not a flat terminal choice. Ambiguous selections are reported explicitly.

### Evaluation and Export

`src/eval/` judges success from the snapped tree’s numerical behavior on validation samples. `src/symbolics/` exports readable formulas by traversing the recovered tree structure.

## Data Flow

1. Build `MasterTree` for the requested depth and variables.
2. Initialize node-wise logits for every internal node.
3. Train with soft selection and record loss/health metrics.
4. Enter hardening to sharpen per-node distributions.
5. Snap node-wise choices into a discrete tree.
6. Evaluate the snapped tree against the target on independent samples.
7. Export recovered formulas and aggregate blind recovery statistics.

## File Responsibilities

- `src/trees/TreeTypes.jl`: structural types for master and recovered trees.
- `src/trees/MasterTree.jl`: tree construction and node/candidate indexing.
- `src/models/EMLLayer.jl`: recursive soft evaluation of the trainable tree.
- `src/training/TrainConfig.jl`: optimizer and hardening configuration.
- `src/training/TrainLoop.jl`: update loop, phase scheduling, and metrics.
- `src/snapping/Snap.jl`: node-wise discrete recovery and ambiguity checks.
- `src/symbolics/Export.jl`: recovered-tree to formula conversion.
- `src/eval/Recovery.jl`: validation-based blind recovery judgment.

## Testing Strategy

Add focused tests in the corresponding mirrored `test/` areas:

- recursive forward evaluation returns expected shapes and deterministic outputs,
- training updates logits from their initialized state,
- hardening increases node-wise selection sharpness,
- snapping reconstructs a valid discrete tree,
- recovered-tree evaluation and formula export match the snapped structure.

## Success Criteria

This design is successful when the repository can truthfully claim all of the following:

- the full tree, not a terminal mixture, is being trained,
- training performs real parameter updates,
- hardening changes model behavior rather than only recording metrics,
- snapping produces a discrete tree with per-node choices,
- evaluation is based on snapped-tree validation behavior.
