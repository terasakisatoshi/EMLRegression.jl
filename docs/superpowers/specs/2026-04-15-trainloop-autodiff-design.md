# TrainLoop Autodiff Design

**Date:** 2026-04-15

**Goal:** Replace the current finite-difference gradient path in `run_training` with reverse-mode autodiff so the Julia implementation is closer to the paper's Section 4.3 optimization setup.

## Context

The current training loop evaluates a real-valued loss, but computes parameter gradients by manually perturbing each logit entry. This differs materially from the paper's optimization setup, which uses Adam with complex-valued internal computation and autodiff-backed gradients. The existing implementation already has the right high-level structure:

- recursive EML tree evaluation
- logits per node and branch
- Adam updates
- optional hardening and complexity penalties

The main gap is gradient computation.

## Chosen Approach

Use `Zygote.gradient(loss_fn, ps)` to compute gradients for the existing `Lux.setup` parameter structure. Keep the current parameter layout, optimizer, hardening schedule, and recovery pipeline unchanged.

This is the smallest change that removes the finite-difference approximation while preserving the current API and experiments.

## Alternatives Considered

### 1. Direct `Zygote` integration

Pros:

- minimal surface-area change
- keeps `Lux` and `Optimisers` usage intact
- directly addresses the paper gap

Cons:

- adds an explicit dependency
- requires care that the loss remains real-valued

### 2. Broader `Lux` ADTypes refactor

Pros:

- cleaner long-term abstraction for switching AD backends

Cons:

- wider refactor than needed
- higher risk of unrelated breakage

### 3. `ForwardDiff` or `Enzyme`

Pros:

- could be useful later for backend comparisons

Cons:

- less direct fit for this migration
- more uncertainty with current complex and nested-parameter setup

## Design

### Gradient path

`run_training` will build the same scalar loss closure it already uses and call `Zygote.gradient` on the full parameter tree. The closure must return a real scalar composed from:

- model MSE
- optional hardening penalty
- optional complexity penalty

The finite-difference helpers will be removed after the new tests pass.

### Parameter structure

Keep the existing nested named-tuple structure returned by `Lux.setup`. The autodiff migration should not require converting parameters into another container type.

### Stability assumptions

This change does not yet integrate the existing stability helpers into the training loop. The scope is limited to replacing gradient computation. If autodiff exposes new instability in deeper experiments, stability handling can be a follow-up change.

### Verification strategy

Use TDD:

1. add a focused regression test that exercises a Zygote gradient over the training-loss shape
2. verify it fails before implementation
3. implement the minimal training-loop changes
4. run focused training tests
5. run the full test suite

## Risks

- Zygote may reject some mutating code patterns in the loss path.
- Complex-valued intermediate computation is acceptable, but the loss must stay real.
- Training behavior may shift numerically after replacing finite differences, so existing tests should assert behavior, not exact trajectories.

## Success Criteria

- `run_training` no longer uses finite-difference helpers for parameter gradients.
- Existing training and recovery tests still pass.
- A new regression test confirms autodiff can produce gradients for the model/loss structure used in training.
