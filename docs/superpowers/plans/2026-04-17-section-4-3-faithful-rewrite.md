# Section 4.3 Faithful Rewrite Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current Section 4.3 scaffold with a PyTorch-v16-compatible Julia implementation whose parameterization, training, snapping, targets, and recovery metrics are aligned with the paper and external reference materials.

**Architecture:** Remove the current candidate-choice tree as the default Section 4.3 model and replace it with a full-binary-tree `EMLTree` parameterized by `leaf_logits[n_leaves,3]` and `blend_logits[n_internal,2]`. Keep the repository's Julia CLI and results pipeline, but make forward semantics, loss, hardening, snapping, and sweep jobs match `extern/SymbolicRegressionPackage/EML_toolkit/EmL_training/PyTorch_v16_final` as closely as practical.

**Tech Stack:** Julia 1.12.6, Lux.jl, Zygote.jl, Optimisers.jl, StableRNGs.jl, JSON3.jl, CSV.jl, DataFrames.jl, Test, Documenter.jl

---

## File Map

- Modify: `src/EMLRegression.jl`
- Modify: `src/models/EMLLayer.jl`
- Modify: `src/training/TrainConfig.jl`
- Modify: `src/training/TrainLoop.jl`
- Modify: `src/targets/TargetRegistry.jl`
- Modify: `src/snapping/Snap.jl`
- Modify: `src/eval/Recovery.jl`
- Modify: `src/symbolics/Export.jl`
- Modify: `scripts/run_experiment.jl`
- Create: `scripts/run_paper_headless_sweep.jl`
- Modify: `README.md`
- Modify: `docs/src/paper.md`
- Modify: `docs/src/training.md`
- Modify: `docs/src/training-tutorial.md`
- Modify: `test/models/test_eml_layer.jl`
- Modify: `test/training/test_train_loop.jl`
- Modify: `test/targets/test_targets.jl`
- Modify: `test/snapping/test_snap.jl`
- Modify: `test/eval/test_recovery.jl`
- Modify: `test/integration/test_smoke_ln.jl`

## Reference Inputs

Use these as the implementation source of truth:

- `extern/SymbolicRegressionPackage/EML_toolkit/EmL_training/Log_fit.nb`
- `extern/SymbolicRegressionPackage/EML_toolkit/EmL_training/PyTorch_v16_final/tree_prototype_torch_v16_final.py`
- `extern/SymbolicRegressionPackage/EML_toolkit/EmL_training/PyTorch_v16_final/depth_2_to_6_headless.sh`
- `extern/SymbolicRegressionPackage/EML_toolkit/EmL_training/PyTorch_v16_final/README.md`
- `docs/superpowers/specs/2026-04-17-section-4-3-faithful-rewrite-design.md`

## Task 1: Replace The Model Core With PyTorch-v16-Compatible State

**Files:**
- Modify: `src/models/EMLLayer.jl`
- Modify: `src/EMLRegression.jl`
- Test: `test/models/test_eml_layer.jl`

- [ ] **Step 1: Write the failing model-layout tests**

```julia
@testset "paper-faithful tree parameter shapes" begin
    tree = EMLTree(depth=3)
    ps, st = Lux.setup(StableRNG(1), tree)
    @test size(ps.leaf_logits) == (8, 3)
    @test size(ps.blend_logits) == (7, 2)
end

@testset "paper-faithful forward returns root prediction and diagnostics" begin
    tree = EMLTree(depth=2)
    ps, st = Lux.setup(StableRNG(1), tree)
    x = ComplexF64[1.0 + 0im, 2.0 + 0im]
    y = ComplexF64[1.5 + 0im, 2.5 + 0im]
    pred, aux = Lux.apply(tree, (x, y), ps, st)
    @test length(pred) == 2
    @test haskey(aux, :leaf_probs)
    @test haskey(aux, :gate_probs)
    @test haskey(aux, :eml_outputs)
end
```

- [ ] **Step 2: Run the model tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["models"])'`
Expected: FAIL because the current model stores node-choice logits instead of `leaf_logits` and `blend_logits`, and it does not return PyTorch-style diagnostics.

- [ ] **Step 3: Rewrite the model state and initialization**

```julia
struct EMLTree <: Lux.AbstractLuxLayer
    depth::Int
    n_leaves::Int
    n_internal::Int
    eml_clamp::Float64
    init_strategy::Symbol
    init_scale::Float64
end

function Lux.initialparameters(rng::LehmerRNG, tree::EMLTree)
    leaf_init, gate_init = initialize_logits(rng, tree.depth; strategy=tree.init_strategy, scale=tree.init_scale)
    return (; leaf_logits=leaf_init, blend_logits=gate_init)
end
```

- [ ] **Step 4: Implement level-by-level forward evaluation**

```julia
function (tree::EMLTree)(xy, ps, st)
    x, y = to_complex_inputs(xy)
    leaf_probs = softmax_rows(ps.leaf_logits, st.tau_leaf)
    current_level = leaf_candidates(x, y) * leaf_probs'
    gate_probs_levels = Matrix{Float64}[]
    eml_outputs = Vector{Vector{ComplexF64}}()
    # loop level by level until one root remains
    return root, (; leaf_probs, gate_probs, eml_outputs, tau_leaf=st.tau_leaf, tau_gate=st.tau_gate)
end
```

- [ ] **Step 5: Match the PyTorch numerical behavior**

```julia
left_input = complex_blend(left_children, s_left)
right_input = complex_blend(right_children, s_right)
root = eml_exact(left_input, right_input)
root = sanitize_and_clamp(root, tree.eml_clamp)
```

- [ ] **Step 6: Add manual initialization helpers**

```julia
init_from_expr!(ps, expr; k=32.0)
init_from_blend_leaves!(ps, blend_bits, leaf_symbols; k=32.0)
add_init_noise!(rng, ps, noise_scale)
```

- [ ] **Step 7: Run the model tests to verify they pass**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["models"])'`
Expected: PASS for parameter shapes, forward diagnostics, and manual initialization shape checks.

- [ ] **Step 8: Commit**

```bash
git add src/models/EMLLayer.jl src/EMLRegression.jl test/models/test_eml_layer.jl
git commit -m "feat: replace section 4.3 model core with faithful eml tree"
```

## Task 2: Replace The Target Registry With Paper-Aligned Functions And Real-Domain Filtering

**Files:**
- Modify: `src/targets/TargetRegistry.jl`
- Test: `test/targets/test_targets.jl`

- [ ] **Step 1: Write the failing target tests**

```julia
@testset "paper-aligned targets are registered" begin
    for name in (:eml_depth2, :eml_depth3, :eml_depth4, :eml_depth5, :eml_depth6)
        target = get_target(name)
        @test target.name == name
    end
end

@testset "real-domain filtering rejects invalid complex-domain points" begin
    target = get_target(:eml_depth4)
    x = [0.1, 1.0, 2.0]
    y = [10.0, 1.5, 2.5]
    xf, yf, tf = EMLRegression.filter_real_domain(x, y, target.fn)
    @test length(xf) <= 3
    @test all(isfinite, real.(tf))
    @test all(abs.(imag.(tf)) .< 1e-12)
end
```

- [ ] **Step 2: Run the target tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["targets"])'`
Expected: FAIL because the current targets are recovered-tree fixtures, not the PyTorch-v16 function family with real-domain filtering.

- [ ] **Step 3: Replace the target definitions**

```julia
const TARGET_FUNCTIONS = Dict(
    :eml_depth2 => (fn_eml_depth2, "e - log(exp(y) - log(x))", 2),
    :eml_depth3 => (fn_eml_depth3, "e^e/(e^y - log(x))", 3),
    :eml_depth4 => (fn_eml_depth4, "log(e^x - log(y))", 4),
    :eml_depth5 => (fn_eml_depth5, "log(e-log(e^x - log(y)))", 5),
    :eml_depth6 => (fn_eml_depth6, "e - y*exp(e - exp(x))", 6),
)
```

- [ ] **Step 4: Implement data generation and filtering**

```julia
filter_real_domain(x, y, target_fn; imag_tol=1e-12)
make_grid_data(target_fn; lo=1.0, hi=3.0, step=0.1)
make_generalization_data(target_fn; lo=0.5, hi=5.0, n=4000, seed=12345)
```

- [ ] **Step 5: Run the target tests to verify they pass**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["targets"])'`
Expected: PASS for target lookup and real-domain filtering behavior.

- [ ] **Step 6: Commit**

```bash
git add src/targets/TargetRegistry.jl test/targets/test_targets.jl
git commit -m "feat: replace target registry with paper-aligned functions"
```

## Task 3: Replace The Training Loop With Search And Hardening

**Files:**
- Modify: `src/training/TrainConfig.jl`
- Modify: `src/training/TrainLoop.jl`
- Test: `test/training/test_train_loop.jl`

- [ ] **Step 1: Write the failing training-schedule tests**

```julia
@testset "search and hardening schedule transitions" begin
    cfg = TrainConfig(target=:eml_depth2, depth=2, search_iters=20, hardening_iters=10, eval_every=5)
    result = run_training(cfg; rng=StableRNG(1))
    @test haskey(result.metrics, :soft_rmse)
    @test haskey(result.metrics, :hard_rmse)
    @test haskey(result.metrics, :tau)
    @test result.summary[:hardening_iter] !== nothing
end
```

- [ ] **Step 2: Run the training tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: FAIL because the current loop has a Julia-specific scaffold and does not implement the PyTorch-v16 search/hardening schedule.

- [ ] **Step 3: Replace `TrainConfig` with faithful controls**

```julia
Base.@kwdef struct TrainConfig
    target::Symbol
    depth::Int
    init_strategy::Symbol = :all
    init_scale::Float64 = 1.0
    init_noise::Float64 = 0.0
    search_iters::Int = 6000
    hardening_iters::Int = 2000
    lr::Float64 = 1e-2
    tau_search::Float64 = 2.5
    tau_hard::Float64 = 0.01
    lam_ent_hard::Float64 = 2e-2
    lam_bin_hard::Float64 = 2e-2
    lam_inter::Float64 = 1e-4
    inter_threshold::Float64 = 50.0
    # plus patience and success-threshold fields
end
```

- [ ] **Step 4: Implement the composite loss**

```julia
total = data_loss + lam_ent * entropy + lam_bin * binarity + lam_inter * inter_penalty
```

- [ ] **Step 5: Implement the search phase**

```julia
phase = :search
tau = cfg.tau_search
optimizer = Optimisers.Adam(cfg.lr)
# monitor best soft loss, plateau counter, hard-trigger diagnostics
```

- [ ] **Step 6: Implement the hardening phase**

```julia
t = hard_step / max(cfg.hardening_iters, 1)
tau = cfg.tau_search * (cfg.tau_hard / cfg.tau_search) ^ (t ^ cfg.hardening_tau_power)
lam_ent = t * cfg.lam_ent_hard
lam_bin = t * cfg.lam_bin_hard
```

- [ ] **Step 7: Implement NaN restarts and optional LBFGS polish**

```julia
if !isfinite(total_loss)
    nan_restarts += 1
    restore_best_soft_state!()
end
```

- [ ] **Step 8: Run the training tests to verify they pass**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: PASS for search/hardening metrics, tau tracking, and restart-safe training.

- [ ] **Step 9: Commit**

```bash
git add src/training/TrainConfig.jl src/training/TrainLoop.jl test/training/test_train_loop.jl
git commit -m "feat: implement faithful search and hardening training loop"
```

## Task 4: Replace Snapping And Recovery Metrics

**Files:**
- Modify: `src/snapping/Snap.jl`
- Modify: `src/eval/Recovery.jl`
- Modify: `src/symbolics/Export.jl`
- Test: `test/snapping/test_snap.jl`
- Test: `test/eval/test_recovery.jl`

- [ ] **Step 1: Write the failing snap-and-recovery tests**

```julia
@testset "hard projection snaps leaves and gates" begin
    ps = (leaf_logits=[10.0 -10.0 -10.0; -10.0 10.0 -10.0], blend_logits=[10.0 -10.0])
    snapped = hard_project(ps; k=24.0)
    @test snapped.leaf_logits[1, 1] == 24.0
    @test snapped.blend_logits[1, 1] == 24.0
    @test snapped.blend_logits[1, 2] == -24.0
end

@testset "recovery summary exposes fit and symbol success" begin
    verdict = recovery_verdict(snap_mse=1e-24, n_uncertain=0, fit_success_thr=1e-6, success_thr=1e-20, max_uncertain_success=0)
    @test verdict.fit_success
    @test verdict.symbol_success
    @test verdict.stable_symbol_success
end
```

- [ ] **Step 2: Run the snapping and recovery tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["snapping", "eval"])'`
Expected: FAIL because the current code uses beam search and structure-match-based recovery rather than direct hard projection with uncertainty counts.

- [ ] **Step 3: Replace snap analysis and hard projection**

```julia
analyze_snap(ps; snap_threshold=0.01)
hard_project!(ps; k=24.0)
```

- [ ] **Step 4: Replace recovery verdict semantics**

```julia
fit_success = isfinite(snap_mse) && snap_mse < fit_success_thr
symbol_success = isfinite(snap_mse) && snap_mse < success_thr
stable_symbol_success = symbol_success && n_uncertain <= max_uncertain_success
```

- [ ] **Step 5: Update formula export to reflect snapped leaf and gate states**

```julia
export_mathematica(tree; discretize=true, snap_threshold=0.01)
```

- [ ] **Step 6: Run the snapping and recovery tests to verify they pass**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["snapping", "eval"])'`
Expected: PASS for hard projection, uncertainty counting, and three-level success metrics.

- [ ] **Step 7: Commit**

```bash
git add src/snapping/Snap.jl src/eval/Recovery.jl src/symbolics/Export.jl test/snapping/test_snap.jl test/eval/test_recovery.jl
git commit -m "feat: replace snapping and recovery with faithful metrics"
```

## Task 5: Replace The Experiment Entrypoints And Add The Paper Sweep

**Files:**
- Modify: `scripts/run_experiment.jl`
- Create: `scripts/run_paper_headless_sweep.jl`
- Test: `test/integration/test_smoke_ln.jl`

- [ ] **Step 1: Write the failing integration tests**

```julia
@testset "paper-faithful run_experiment returns summary fields" begin
    cfg = TrainConfig(target=:eml_depth2, depth=2, search_iters=20, hardening_iters=10)
    result = run_experiment(cfg; rng=StableRNG(1))
    @test haskey(result.recovery, :fit_success)
    @test haskey(result.recovery, :symbol_success)
    @test haskey(result.recovery, :stable_symbol_success)
end
```

- [ ] **Step 2: Run the integration tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'`
Expected: FAIL because `run_experiment` still assumes the old target registry, old snap path, and old recovery structure.

- [ ] **Step 3: Rewrite `run_experiment` around the faithful path**

```julia
training = run_training(cfg; rng=rng)
snapped = hard_project(copy(training.params); k=24.0)
snap_eval = evaluate(snapped, training.data.train...)
recovery = recovery_verdict(; snap_mse=snap_eval.mse, n_uncertain=training.snap_info.n_uncertain, ...)
```

- [ ] **Step 4: Add the Julia sweep runner mirroring the shell script**

```julia
const PAPER_JOBS = [
    (; target=:eml_depth2, depth=2, init_strategy=:all, seed0=137, seeds=8, search_iters=6000, hardening_iters=2000),
    (; target=:eml_depth3, depth=3, init_strategy=:all, seed0=137, seeds=16, search_iters=6000, hardening_iters=2000),
    # ...
]
```

- [ ] **Step 5: Run the integration tests to verify they pass**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'`
Expected: PASS for faithful `run_experiment` summaries.

- [ ] **Step 6: Smoke-run the new sweep entrypoint**

Run: `~/.juliaup/bin/julia --project=. scripts/run_paper_headless_sweep.jl --jobs depth2`
Expected: one or more `results/raw/` faithful summary artifacts with `fit_success`, `symbol_success`, `stable_symbol_success`, and `n_uncertain`.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_experiment.jl scripts/run_paper_headless_sweep.jl test/integration/test_smoke_ln.jl
git commit -m "feat: add faithful paper sweep runner"
```

## Task 6: Update Documentation And Verify The Depth Trends

**Files:**
- Modify: `README.md`
- Modify: `docs/src/paper.md`
- Modify: `docs/src/training.md`
- Modify: `docs/src/training-tutorial.md`

- [ ] **Step 1: Update docs to describe the new standard model**

```md
- Section 4.3 now uses `leaf_logits` and `blend_logits`
- training follows `SEARCH` then `HARDENING`
- snapping is direct hard projection, not beam search
```

- [ ] **Step 2: Run the focused depth-2 to depth-4 checks**

Run: `~/.juliaup/bin/julia --project=. scripts/run_paper_headless_sweep.jl --jobs depth2,depth3,depth4`
Expected: depth 2 succeeds consistently, depth 3 and depth 4 show nonzero recovery.

- [ ] **Step 3: Run the depth-5 and depth-6 random-start checks**

Run: `~/.juliaup/bin/julia --project=. scripts/run_paper_headless_sweep.jl --jobs depth5_random,depth6_random`
Expected: depth 5 is rare but nonzero under at least one family, depth 6 remains a negative control under the default budget.

- [ ] **Step 4: Run the manual-noise checks**

Run: `~/.juliaup/bin/julia --project=. scripts/run_paper_headless_sweep.jl --jobs depth5_manual_noise12,depth6_manual_noise12`
Expected: manual-noise initialization is robust, with depth 5 stable symbolic success and depth 6 exact fits that may still show uncertainty.

- [ ] **Step 5: Run the full test suite**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS across models, training, targets, snapping, eval, and integration tests.

- [ ] **Step 6: Build docs**

Run: `~/.juliaup/bin/julia --project=docs docs/make.jl`
Expected: PASS with updated Section 4.3 explanations.

- [ ] **Step 7: Commit**

```bash
git add README.md docs/src/paper.md docs/src/training.md docs/src/training-tutorial.md results/summaries
git commit -m "docs: describe faithful section 4.3 rewrite"
```

## Final Verification Checklist

- [ ] Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'`
- [ ] Run: `~/.juliaup/bin/julia --project=. scripts/run_paper_headless_sweep.jl --jobs depth2,depth3,depth4`
- [ ] Run: `~/.juliaup/bin/julia --project=. scripts/run_paper_headless_sweep.jl --jobs depth5_random,depth6_random`
- [ ] Run: `~/.juliaup/bin/julia --project=. scripts/run_paper_headless_sweep.jl --jobs depth5_manual_noise12,depth6_manual_noise12`
- [ ] Run: `~/.juliaup/bin/julia --project=docs docs/make.jl`

## Notes For Execution

- Do not preserve the old candidate-choice implementation unless a current step explicitly requires a short compatibility shim during migration.
- Keep the external PyTorch files open while implementing; this plan assumes direct line-by-line comparison against the reference behavior.
- Prefer deleting obsolete logic rather than layering the new faithful path on top of it.
- If Julia complex autodiff diverges from PyTorch behavior, fix numerical semantics first and tune penalties second.
