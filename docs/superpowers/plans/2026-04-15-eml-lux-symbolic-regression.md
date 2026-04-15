# EML Lux Symbolic Regression Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Julia research repository that reproduces the paper's Section 4.3 trainable EML-tree experiments on CPU, using `Lux.jl` with complex-valued internal computation and reporting blind recovery rates for depths 2-4.

**Architecture:** The repository is split into a reusable library in `src/` and reproducible experiments in `experiments/`, `scripts/`, and `results/`. The core implementation is a `Lux.jl` trainable full-binary EML tree with a staged pipeline of optimization, hardening, snapping, and post-snap numerical validation.

**Tech Stack:** Julia, Lux.jl, Optimisers.jl, ComponentArrays.jl, Symbolics.jl, SymbolicUtils.jl, JSON3.jl, CSV.jl, DataFrames.jl, StableRNGs.jl, Test

---

## File Structure

### Core package files

- Create: `Project.toml`
- Create: `README.md`
- Create: `src/EMLRegression.jl`
- Create: `src/trees/TreeTypes.jl`
- Create: `src/trees/MasterTree.jl`
- Create: `src/models/EMLLayer.jl`
- Create: `src/training/TrainConfig.jl`
- Create: `src/training/TrainLoop.jl`
- Create: `src/training/Stability.jl`
- Create: `src/snapping/Snap.jl`
- Create: `src/targets/TargetRegistry.jl`
- Create: `src/eval/Metrics.jl`
- Create: `src/eval/Recovery.jl`
- Create: `src/symbolics/Export.jl`

### Experiment and CLI files

- Create: `experiments/configs/must_pass_depth2.toml`
- Create: `experiments/configs/must_pass_depth3.toml`
- Create: `experiments/configs/challenge_depth4.toml`
- Create: `scripts/run_experiment.jl`
- Create: `scripts/run_suite.jl`
- Create: `scripts/summarize_results.jl`
- Create: `scripts/export_recovered_formulas.jl`

### Tests

- Create: `test/runtests.jl`
- Create: `test/test_project_load.jl`
- Create: `test/trees/test_master_tree.jl`
- Create: `test/models/test_eml_layer.jl`
- Create: `test/training/test_stability.jl`
- Create: `test/training/test_train_loop.jl`
- Create: `test/snapping/test_snap.jl`
- Create: `test/targets/test_targets.jl`
- Create: `test/eval/test_recovery.jl`
- Create: `test/integration/test_smoke_ln.jl`

### Repository support files

- Create: `.gitignore`
- Create: `results/raw/.gitkeep`
- Create: `results/summaries/.gitkeep`
- Create: `results/figures/.gitkeep`

## Task 1: Bootstrap Julia Project Skeleton

**Files:**
- Create: `Project.toml`
- Create: `README.md`
- Create: `.gitignore`
- Create: `src/EMLRegression.jl`
- Create: `test/runtests.jl`
- Create: `test/test_project_load.jl`
- Create: `results/raw/.gitkeep`
- Create: `results/summaries/.gitkeep`
- Create: `results/figures/.gitkeep`

- [ ] **Step 1: Write the failing project-load test**

```julia
# test/test_project_load.jl
using Test

@testset "project loads" begin
    using EMLRegression
    @test isdefined(EMLRegression, :eml)
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL because `Project.toml` and package entrypoint do not exist yet.

- [ ] **Step 3: Create the minimal package skeleton**

```julia
# src/EMLRegression.jl
module EMLRegression

eml(x, y) = exp(x) - log(y)

end
```

Use `Project.toml` with the package name `EMLRegression` and add placeholder dependency entries only after the basic load path is working.

- [ ] **Step 4: Add repository hygiene**

Ignore:

```gitignore
/Manifest.toml
/results/raw/*
/results/summaries/*
/results/figures/*
!/results/raw/.gitkeep
!/results/summaries/.gitkeep
!/results/figures/.gitkeep
/tmp/
```

Document in `README.md`:

- project goal,
- current milestone,
- how to run tests,
- how results directories are used.

- [ ] **Step 5: Run tests again**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS for the package load test.

- [ ] **Step 6: Commit**

```bash
git add Project.toml README.md .gitignore src/EMLRegression.jl test/runtests.jl test/test_project_load.jl results/raw/.gitkeep results/summaries/.gitkeep results/figures/.gitkeep
git commit -m "chore: bootstrap EML regression project"
```

## Task 2: Implement Tree Types and Master Tree Structure

**Files:**
- Modify: `src/EMLRegression.jl`
- Create: `src/trees/TreeTypes.jl`
- Create: `src/trees/MasterTree.jl`
- Create: `test/trees/test_master_tree.jl`

- [ ] **Step 1: Write failing tests for tree structure**

```julia
# test/trees/test_master_tree.jl
using Test
using EMLRegression

@testset "master tree structure" begin
    tree = build_master_tree(depth=2, variables=(:x,))
    @test tree.depth == 2
    @test length(tree.nodes) == 3
    @test :const1 in tree.terminals
    @test :x in tree.terminals
end
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["trees"])'`
Expected: FAIL because `build_master_tree` and tree types do not exist.

- [ ] **Step 3: Implement tree types and constructors**

Define:

- `TerminalChoice`
- `NodeChoiceSpace`
- `MasterTree`
- `build_master_tree(; depth, variables)`

Minimal shape:

```julia
struct MasterTree
    depth::Int
    variables::Vector{Symbol}
    terminals::Vector{Symbol}
    nodes::Vector{Int}
end
```

Then refine it so each node has explicit left/right candidate spaces.

- [ ] **Step 4: Export the tree APIs from the package**

Re-export from `src/EMLRegression.jl`:

- `MasterTree`
- `build_master_tree`

- [ ] **Step 5: Run targeted tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["trees"])'`
Expected: PASS for structural tree tests.

- [ ] **Step 6: Commit**

```bash
git add src/EMLRegression.jl src/trees/TreeTypes.jl src/trees/MasterTree.jl test/trees/test_master_tree.jl
git commit -m "feat: add master tree structure"
```

## Task 3: Implement Complex EML Lux Layer

**Files:**
- Modify: `src/EMLRegression.jl`
- Create: `src/models/EMLLayer.jl`
- Create: `test/models/test_eml_layer.jl`

- [ ] **Step 1: Write failing tests for forward evaluation**

```julia
# test/models/test_eml_layer.jl
using Test
using Random
using Lux
using EMLRegression

@testset "EML layer forward pass" begin
    layer = EMLTreeLayer(build_master_tree(depth=2, variables=(:x,)))
    rng = Random.MersenneTwister(1)
    ps, st = Lux.setup(rng, layer)
    x = ComplexF64[1.0 + 0im, 2.0 + 0im]
    y, st2 = Lux.apply(layer, x, ps, st)
    @test length(y) == 2
    @test eltype(y) == ComplexF64
end
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["models"])'`
Expected: FAIL because `EMLTreeLayer` is not implemented.

- [ ] **Step 3: Implement the Lux layer**

Implement:

- `struct EMLTreeLayer <: Lux.AbstractExplicitLayer`
- `Lux.initialparameters`
- `Lux.initialstates`
- `Lux.apply`

Parameter design:

- logits per branch choice,
- grouped with `ComponentArray`,
- output type `ComplexF64`.

Minimal evaluation sketch:

```julia
weighted_choice(values, logits) = sum(softmax(logits) .* values)
eml(a, b) = exp(a) - log(b)
```

- [ ] **Step 4: Add shape and determinism coverage**

Extend tests to confirm:

- repeatable setup under fixed RNG,
- one-variable batches preserve batch length,
- forward pass does not allocate invalid real-only values.

- [ ] **Step 5: Run model tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["models"])'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/EMLRegression.jl src/models/EMLLayer.jl test/models/test_eml_layer.jl
git commit -m "feat: add Lux EML tree layer"
```

## Task 4: Add Numerical Stability Utilities

**Files:**
- Modify: `src/EMLRegression.jl`
- Create: `src/training/Stability.jl`
- Create: `test/training/test_stability.jl`

- [ ] **Step 1: Write failing tests for stability helpers**

```julia
# test/training/test_stability.jl
using Test
using EMLRegression

@testset "stability helpers" begin
    z = ComplexF64[1 + 0im, Inf + 0im]
    report = inspect_complex_values(z)
    @test report.has_inf
    @test !report.has_nan
end
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: FAIL because stability helpers do not exist.

- [ ] **Step 3: Implement stability helpers**

Implement:

- `inspect_complex_values`
- `finite_or_flag`
- `clamp_complex_magnitude`
- `FailureReason` enum-like constants

Return structured reports, not booleans only.

- [ ] **Step 4: Wire exports**

Expose stability helpers from the package root so later training code can consume them without reaching into internal files.

- [ ] **Step 5: Re-run training tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: PASS for stability-only tests.

- [ ] **Step 6: Commit**

```bash
git add src/EMLRegression.jl src/training/Stability.jl test/training/test_stability.jl
git commit -m "feat: add complex stability helpers"
```

## Task 5: Implement Target Registry and Sampling

**Files:**
- Modify: `src/EMLRegression.jl`
- Create: `src/targets/TargetRegistry.jl`
- Create: `test/targets/test_targets.jl`

- [ ] **Step 1: Write failing tests for target lookup and sampling**

```julia
# test/targets/test_targets.jl
using Test
using EMLRegression

@testset "target registry" begin
    target = get_target(:ln)
    xs = sample_domain(target, 16; rng_seed=1)
    ys = evaluate_target(target, xs)
    @test length(xs) == 16
    @test length(ys) == 16
    @test eltype(ys) == ComplexF64
end
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["targets"])'`
Expected: FAIL because target registry functions do not exist.

- [ ] **Step 3: Implement the target registry**

Define a `TargetSpec` with:

- symbolic name,
- arity,
- evaluator function,
- sampling domain,
- must-pass or challenge tier.

Implement initial targets:

- `exp`, `ln`, `neg`, `inv`, `add`, `mul`

Stub challenge targets with `@test_broken` coverage until they are implemented for real.

- [ ] **Step 4: Add domain-safe sampling**

Ensure domain sampling avoids trivial singularities for:

- `ln`
- `inv`
- `tan`
- `log_x(y)`

Use deterministic RNG via `StableRNGs.jl`.

- [ ] **Step 5: Re-run target tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["targets"])'`
Expected: PASS for must-pass targets.

- [ ] **Step 6: Commit**

```bash
git add src/EMLRegression.jl src/targets/TargetRegistry.jl test/targets/test_targets.jl
git commit -m "feat: add target registry and samplers"
```

## Task 6: Implement Training Loop and Hardening Phase

**Files:**
- Modify: `src/EMLRegression.jl`
- Create: `src/training/TrainConfig.jl`
- Create: `src/training/TrainLoop.jl`
- Create: `test/training/test_train_loop.jl`

- [ ] **Step 1: Write failing tests for one training step**

```julia
# test/training/test_train_loop.jl
using Test
using StableRNGs
using EMLRegression

@testset "training loop" begin
    cfg = TrainConfig(depth=2, target=:ln, batch_size=32, steps=2)
    result = run_training(cfg; rng=StableRNG(1))
    @test haskey(result.metrics, :train_loss)
    @test length(result.metrics[:train_loss]) == 2
end
```

- [ ] **Step 2: Run targeted tests to verify failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: FAIL because `TrainConfig` and `run_training` do not exist.

- [ ] **Step 3: Implement config and basic loop**

Define:

- `TrainConfig`
- `TrainingResult`
- `run_training`

Include:

- Adam phase,
- batched sampling,
- per-step loss logging,
- stability checks on activations and outputs.

- [ ] **Step 4: Add hardening**

Implement a second phase that increases discrete pressure on logits. Keep it explicit in config:

```julia
TrainConfig(; hardening_steps=100, hardening_weight=0.1)
```

The result object must separately log:

- ordinary phase metrics,
- hardening phase metrics,
- failure reason if aborted.

- [ ] **Step 5: Re-run training tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: PASS for stability and train-loop tests.

- [ ] **Step 6: Commit**

```bash
git add src/EMLRegression.jl src/training/TrainConfig.jl src/training/TrainLoop.jl test/training/test_train_loop.jl
git commit -m "feat: add EML training and hardening loops"
```

## Task 7: Implement Snapping and Recovered Tree Export

**Files:**
- Modify: `src/EMLRegression.jl`
- Create: `src/snapping/Snap.jl`
- Create: `src/symbolics/Export.jl`
- Create: `test/snapping/test_snap.jl`

- [ ] **Step 1: Write failing tests for snapping**

```julia
# test/snapping/test_snap.jl
using Test
using EMLRegression

@testset "snap trained logits" begin
    snapped = snap_logits([10.0, -10.0, -10.0])
    @test snapped == [1.0, 0.0, 0.0]
end
```

Add a second test ensuring ambiguous logits return an explicit non-success status rather than silently choosing.

- [ ] **Step 2: Run snapping tests to verify failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["snapping"])'`
Expected: FAIL because snapping code does not exist.

- [ ] **Step 3: Implement snapping logic**

Implement:

- `snap_logits`
- `snap_model`
- `RecoveredTree`
- ambiguity margins

`RecoveredTree` should be serializable to JSON.

- [ ] **Step 4: Add formula export**

Provide a first readable export, even if not fully simplified:

```julia
formula_string(recovered_tree) -> "eml(1, eml(eml(1, x), 1))"
```

- [ ] **Step 5: Re-run snapping tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["snapping"])'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/EMLRegression.jl src/snapping/Snap.jl src/symbolics/Export.jl test/snapping/test_snap.jl
git commit -m "feat: add snapping and recovered formula export"
```

## Task 8: Implement Recovery Metrics and End-to-End Smoke Test

**Files:**
- Modify: `src/EMLRegression.jl`
- Create: `src/eval/Metrics.jl`
- Create: `src/eval/Recovery.jl`
- Create: `test/eval/test_recovery.jl`
- Create: `test/integration/test_smoke_ln.jl`

- [ ] **Step 1: Write failing tests for recovery judgment**

```julia
# test/eval/test_recovery.jl
using Test
using EMLRegression

@testset "recovery judgment" begin
    verdict = recovery_verdict(
        train_loss=1e-12,
        validation_loss=1e-12,
        snap_status=:ok,
        numerical_match=true,
    )
    @test verdict.success
end
```

Also add a smoke test:

```julia
# test/integration/test_smoke_ln.jl
using Test
using StableRNGs
using EMLRegression

@testset "ln smoke run" begin
    cfg = TrainConfig(depth=2, target=:ln, steps=5, hardening_steps=2, batch_size=16)
    outcome = run_experiment(cfg; rng=StableRNG(1))
    @test haskey(outcome, :recovery)
end
```

- [ ] **Step 2: Run tests to verify failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL because recovery metrics and integration APIs are missing.

- [ ] **Step 3: Implement evaluation and experiment wrapper**

Implement:

- `recovery_verdict`
- `numerical_match`
- `run_experiment`

The recovery contract must distinguish:

- low loss but ambiguous snap,
- snapped but numerically wrong,
- snapped and numerically correct.

- [ ] **Step 4: Re-run full test suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS for all implemented tests, with challenge-target tests still isolated or marked broken as needed.

- [ ] **Step 5: Commit**

```bash
git add src/EMLRegression.jl src/eval/Metrics.jl src/eval/Recovery.jl test/eval/test_recovery.jl test/integration/test_smoke_ln.jl
git commit -m "feat: add recovery metrics and smoke experiment"
```

## Task 9: Add Experiment Configs and Batch CLI Scripts

**Files:**
- Create: `experiments/configs/must_pass_depth2.toml`
- Create: `experiments/configs/must_pass_depth3.toml`
- Create: `experiments/configs/challenge_depth4.toml`
- Create: `scripts/run_experiment.jl`
- Create: `scripts/run_suite.jl`
- Create: `scripts/summarize_results.jl`
- Create: `scripts/export_recovered_formulas.jl`

- [ ] **Step 1: Write failing smoke tests for CLI outputs**

Add to `test/integration/test_smoke_ln.jl` or split a new integration test that checks a run writes:

- one raw result JSON,
- one summary CSV row.

- [ ] **Step 2: Run the integration tests to verify failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'`
Expected: FAIL because scripts and persistence outputs do not exist.

- [ ] **Step 3: Implement experiment configs**

Use TOML configs with exact fields:

```toml
name = "must_pass_depth2"
depth = 2
targets = ["exp", "ln", "neg", "inv", "add", "mul"]
seeds = [1, 2, 3, 4, 5]
steps = 500
hardening_steps = 100
batch_size = 64
```

- [ ] **Step 4: Implement scripts**

Responsibilities:

- `scripts/run_experiment.jl`
  run one config-target-seed combination
- `scripts/run_suite.jl`
  run all combinations in a suite
- `scripts/summarize_results.jl`
  aggregate blind recovery rate by target and depth
- `scripts/export_recovered_formulas.jl`
  export snapped trees as readable formulas

- [ ] **Step 5: Re-run integration tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'`
Expected: PASS for raw output creation and summary generation.

- [ ] **Step 6: Commit**

```bash
git add experiments/configs/must_pass_depth2.toml experiments/configs/must_pass_depth3.toml experiments/configs/challenge_depth4.toml scripts/run_experiment.jl scripts/run_suite.jl scripts/summarize_results.jl scripts/export_recovered_formulas.jl test/integration/test_smoke_ln.jl
git commit -m "feat: add experiment configs and CLI runners"
```

## Task 10: Expand Challenge Targets and Symbolic Export Coverage

**Files:**
- Modify: `src/targets/TargetRegistry.jl`
- Modify: `src/symbolics/Export.jl`
- Modify: `test/targets/test_targets.jl`
- Modify: `test/eval/test_recovery.jl`

- [ ] **Step 1: Add failing tests for challenge targets**

Add explicit tests for:

- `x / y`
- `x^2`
- `sqrt(x)`
- `sin(x)`
- `cos(x)`
- `tan(x)`
- `log_x(y)`

For unstable targets, start with domain and shape assertions before expecting recovery success.

- [ ] **Step 2: Run target tests to verify failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["targets"])'`
Expected: FAIL because challenge targets are incomplete.

- [ ] **Step 3: Implement challenge target evaluators and safe domains**

Keep domains conservative and document each constraint in code comments and README notes.

- [ ] **Step 4: Extend symbolic export**

Add enough export support to print:

- unary target formulas,
- binary target formulas,
- constant `1`,
- variable terminals `x`, `y`.

- [ ] **Step 5: Re-run relevant tests**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["targets", "eval"])'`
Expected: PASS for evaluator correctness and export coverage.

- [ ] **Step 6: Commit**

```bash
git add src/targets/TargetRegistry.jl src/symbolics/Export.jl test/targets/test_targets.jl test/eval/test_recovery.jl
git commit -m "feat: add challenge targets and symbolic export coverage"
```

## Task 11: Document Reproduction Workflow

**Files:**
- Modify: `README.md`
- Create: `docs/reproduction-section-4-3.md`

- [ ] **Step 1: Write failing documentation checklist**

Create a checklist in `docs/reproduction-section-4-3.md` covering:

- environment setup,
- test command,
- one smoke experiment command,
- one suite command,
- one summary command,
- where outputs appear.

- [ ] **Step 2: Verify the commands are real**

Run, at minimum:

- `julia --project=. -e 'using Pkg; Pkg.test()'`
- `julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target ln --seed 1`
- `julia --project=. scripts/summarize_results.jl`

Expected: commands complete successfully or expose remaining implementation gaps that must be fixed before closing the task.

- [ ] **Step 3: Update README**

Include:

- repository purpose,
- current milestone status,
- exact commands for smoke and suite runs,
- note that CPU-first is the supported path.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/reproduction-section-4-3.md
git commit -m "docs: add section 4.3 reproduction guide"
```

## Task 12: Baseline Reproduction Run and Result Freeze

**Files:**
- Modify: `experiments/configs/must_pass_depth2.toml`
- Modify: `experiments/configs/must_pass_depth3.toml`
- Modify: `experiments/configs/challenge_depth4.toml`
- Write: `results/raw/*.json`
- Write: `results/summaries/*.csv`
- Write: `results/figures/*.png`
- Create: `docs/first-baseline-report.md`

- [ ] **Step 1: Run the must-pass suite**

Run: `julia --project=. scripts/run_suite.jl --config experiments/configs/must_pass_depth2.toml`
Expected: per-run JSON artifacts in `results/raw/`.

- [ ] **Step 2: Run the depth-3 suite**

Run: `julia --project=. scripts/run_suite.jl --config experiments/configs/must_pass_depth3.toml`
Expected: additional raw artifacts for depth 3.

- [ ] **Step 3: Run the challenge suite**

Run: `julia --project=. scripts/run_suite.jl --config experiments/configs/challenge_depth4.toml`
Expected: mixed success and failure artifacts, both preserved.

- [ ] **Step 4: Summarize results**

Run: `julia --project=. scripts/summarize_results.jl`
Expected: summary CSV/JSON grouped by depth and target.

- [ ] **Step 5: Write baseline report**

Capture:

- must-pass targets that recover,
- targets that fail,
- common failure reasons,
- next numerical bottlenecks.

- [ ] **Step 6: Commit**

```bash
git add experiments/configs/must_pass_depth2.toml experiments/configs/must_pass_depth3.toml experiments/configs/challenge_depth4.toml docs/first-baseline-report.md
git commit -m "results: record first section 4.3 baseline"
```

## Verification Checklist Before Claiming Completion

- [ ] `julia --project=. -e 'using Pkg; Pkg.test()'` passes
- [ ] `scripts/run_experiment.jl` can run one smoke case end-to-end
- [ ] `scripts/run_suite.jl` produces raw artifacts
- [ ] `scripts/summarize_results.jl` produces grouped summary outputs
- [ ] A snapped formula can be exported as a readable string
- [ ] Recovery reports distinguish low-loss failure from true recovery

## Notes for the Implementer

- Keep the first milestone CPU-only even if GPU support looks tempting.
- Do not broaden scope into operator discovery or full symbolic proof machinery during initial implementation.
- Treat numerical failures as data. Persist them with explicit reason codes.
- Keep challenge targets isolated from must-pass infrastructure validation.
- Prefer small commits exactly at task boundaries. Do not batch unrelated tasks together.
