# EML Lux Symbolic Regression Design

**Date:** 2026-04-15

**Goal:** Rebuild the paper's Section 4.3 pipeline in Julia as a serious reproduction repository, centered on trainable EML trees implemented with Lux.jl and evaluated by blind recovery success across depths 2-4.

## Scope

This design covers the first major milestone of the larger "full paper pipeline" effort:

- Reproduce the trainable EML-tree experiments from Section 4.3.
- Use complex-valued internal computation from the beginning.
- Target CPU-first execution.
- Organize the work as a research-grade repository, not a one-off script.

This design does not yet include:

- Full reconstruction of the paper's operator-discovery outer loop.
- Full symbolic verification of the entire paper.
- Full EML compiler reproduction.

Those can be added later without restructuring the repository.

## Success Criteria

The first milestone is considered successful when the repository can:

- Train EML trees of depths 2-4 against multiple target functions.
- Run multiple random seeds per target/depth pair.
- Perform a hardening phase and a final snapping step.
- Report blind recovery success rates, not only training loss.
- Save recovered discrete trees and evaluation summaries.
- Re-run experiments reproducibly from CLI entry points.

Success is judged in the paper's spirit:

- snapping must produce a discrete tree,
- the snapped tree must match the target numerically on independent validation points,
- success rates must be aggregated by depth and target.

Low MSE alone is not sufficient.

## Recommended Technical Stack

### Core modeling

- `Lux.jl`
- `Optimisers.jl`
- `Optimization.jl` or a thin custom training loop if Lux-native loops prove simpler

### Symbolic inspection and verification

- `Symbolics.jl`
- `SymbolicUtils.jl`

### Data handling and experiment management

- `ComponentArrays.jl` for parameter organization
- `JSON3.jl` and `CSV.jl` for result persistence
- `DataFrames.jl` for aggregation
- `StableRNGs.jl` for reproducibility

### Testing

- Julia standard `Test`

## Why Lux.jl Is the Primary Choice

Section 4.3 is fundamentally a continuous optimization problem over a structured expression tree. Lux is a better fit than SymbolicRegression.jl for this milestone because the paper's key move is not generic symbolic search over an operator set, but training a fixed EML-tree family with continuous parameters, then hardening and snapping those parameters into a discrete symbolic form.

SymbolicRegression.jl remains valuable later as a comparison baseline or for follow-up experiments, but it is not the best central engine for this reproduction target.

## Repository Layout

```text
.
├── Project.toml
├── Manifest.toml
├── README.md
├── src/
│   ├── EMLRegression.jl
│   ├── trees/
│   ├── models/
│   ├── training/
│   ├── snapping/
│   ├── targets/
│   ├── eval/
│   └── symbolics/
├── experiments/
│   ├── configs/
│   ├── suites/
│   └── notebooks/
├── scripts/
│   ├── run_experiment.jl
│   ├── run_suite.jl
│   ├── summarize_results.jl
│   └── export_recovered_formulas.jl
├── test/
├── results/
│   ├── raw/
│   ├── summaries/
│   └── figures/
└── docs/
    └── superpowers/
        ├── specs/
        └── plans/
```

## Architectural Decomposition

### 1. Tree representation

`src/trees/` owns the structural definition of the EML master tree.

Responsibilities:

- represent full binary trees of configurable depth,
- define the available terminal choices,
- define how each node combines its candidate inputs,
- provide deterministic conversion between parameter tensors and a tree instance.

This layer should not know about optimization or experiment bookkeeping.

### 2. Lux model layer

`src/models/` wraps the EML tree as a Lux-compatible trainable model.

Responsibilities:

- expose model parameters as logits,
- evaluate the tree with complex arithmetic,
- support batched forward passes,
- preserve a parameter layout that is easy to harden and inspect.

The model should make the paper's "master formula" concrete: each node chooses among `1`, input variables, and prior subtree outputs through trainable weights.

### 3. Training system

`src/training/` owns the optimization loop.

Responsibilities:

- ordinary Adam phase,
- hardening phase to push logits toward discrete selections,
- numeric stabilization for complex `exp` and `log`,
- seed control,
- checkpointing metrics and intermediate states.

This module must treat failure modes as first-class outputs. NaNs, branch-cut pathologies, exploding values, and non-snappable plateaus should be logged explicitly.

### 4. Snapping and discrete recovery

`src/snapping/` converts trained continuous parameters into discrete trees.

Responsibilities:

- map logits to hard choices,
- rebuild a discrete symbolic tree,
- report ambiguity when selections are not sufficiently separated,
- provide a serializable recovered formula object.

This is the key bridge between "neural training" and "symbolic recovery."

### 5. Target functions

`src/targets/` defines the target functions used in experiments.

Initial target set chosen for the project:

- `exp(x)`
- `ln(x)`
- `-x`
- `1/x`
- `x + y`
- `x * y`
- `x / y`
- `x^2`
- `sqrt(x)`
- `sin(x)`
- `cos(x)`
- `tan(x)`
- `log_x(y)`

For implementation sequencing, split these into:

- must-pass set: `exp`, `ln`, `-x`, `1/x`, `x+y`, `x*y`
- challenge set: `x/y`, `x^2`, `sqrt`, `sin`, `cos`, `tan`, `log_x(y)`

This keeps the experimental program ambitious without confusing "hard target" failures with infrastructure failures.

### 6. Evaluation

`src/eval/` owns the experiment-level verdicts.

Responsibilities:

- compute train/validation/extrapolation errors,
- check numerical agreement of snapped trees,
- aggregate blind recovery statistics by target and depth,
- emit human-readable and machine-readable summaries.

The evaluation contract should make it impossible to confuse "low loss" with "successful symbolic recovery."

### 7. Symbolic bridge

`src/symbolics/` connects recovered trees to symbolic tooling.

Responsibilities:

- convert discrete trees into symbolic forms,
- simplify or canonicalize where useful,
- support future verification work,
- support exporting readable recovered formulas.

This layer is intentionally downstream of the numerical pipeline. It should not block early experiments.

## Model Design

The core model is a full binary EML tree.

Each node computes:

```math
eml(a, b) = \exp(a) - \log(b)
```

The internal representation is complex from the start, using `ComplexF64`.

Each node receives trainable logits that determine which source feeds each branch. Candidate sources depend on tree level but conceptually include:

- the constant `1`,
- input variables such as `x` and `y`,
- outputs from available child subtrees.

Training uses soft selection. Recovery uses a staged pipeline:

1. optimize with Adam,
2. harden logits toward extreme selections,
3. snap to discrete choices,
4. validate the recovered tree numerically,
5. optionally inspect symbolically.

## Numerical Stability Strategy

Complex `exp` and `log` are the biggest technical risk.

The implementation should explicitly include:

- configurable sampling domains per target,
- guards for NaN and Inf propagation,
- magnitude monitoring for intermediate activations,
- optional clamping or stabilized transforms where they do not invalidate the experiment,
- structured logging of branch-cut-related failures.

The goal is not to pretend these issues do not exist. The goal is to make them observable and comparable across targets and depths.

## Experiment Protocol

### Main axes

- depth: 2, 3, 4
- target function
- random seed

### Per-run phases

1. initialize model
2. sample training/validation data
3. optimize with Adam
4. run hardening
5. snap
6. validate snapped tree
7. save run artifact

### Primary outputs

- final train loss
- final validation loss
- snap success or failure
- blind recovery success or failure
- recovered discrete tree
- readable formula export
- failure reason if unsuccessful

### Aggregate outputs

- success rate by target
- success rate by depth
- success rate by target and depth
- summary tables and plots

## CPU-First Execution Policy

The first implementation should prioritize deterministic CPU execution over early GPU support.

Reasons:

- debugging complex-number pathologies is easier on CPU,
- reproducibility is easier,
- depth 2-4 experiments are tractable enough to validate the pipeline before optimizing.

GPU support can be added later once the pipeline is numerically stable and evaluation semantics are settled.

## Testing Strategy

Tests should be organized by responsibility.

### Unit tests

- EML node evaluation
- tree forward pass
- parameter shape/layout invariants
- snapping behavior
- target function sampling

### Property-style tests

- identical seeds produce identical initializations
- discrete selections round-trip correctly
- snapped trees serialize and deserialize without changing meaning

### Experiment smoke tests

- a tiny CPU run for a simple target such as `ln(x)` completes end-to-end,
- result files are written correctly,
- aggregation scripts can summarize one or more runs.

## Key Risks

### 1. Numerical instability

Complex `log` and repeated `exp` may explode or become undefined on sampled domains.

Mitigation:

- start with tight domains,
- log intermediate ranges,
- add explicit run failure classification.

### 2. Ambiguous snapping

Training may converge to low loss without clearly discrete logits.

Mitigation:

- separate hardening as an explicit phase,
- record margin statistics at snap time,
- classify ambiguous runs distinctly from wrong recovered formulas.

### 3. Overly ambitious target set

Trigonometric and two-variable targets may fail before the infrastructure is mature.

Mitigation:

- maintain must-pass and challenge target tiers,
- treat the must-pass set as infrastructure validation.

## Expected Evolution Path

Once the Section 4.3 pipeline is stable, the same repository can grow into the broader "full paper pipeline" by adding:

- EML compiler machinery,
- symbolic verification modules,
- ablation studies,
- alternative operators such as EDL,
- outer-loop operator search.

The current architecture is chosen specifically to allow that growth without replacing the core layout.

## Approval State

This design is approved conversationally by the user as of 2026-04-15 and is ready to transition into implementation planning.
