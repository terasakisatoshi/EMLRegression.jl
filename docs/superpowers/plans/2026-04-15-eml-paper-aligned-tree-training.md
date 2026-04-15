# Paper-Aligned EML Tree Training Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the current EMLRegression training pipeline so Section 4.3 is modeled as a recursively evaluated EML tree with real optimization, hardening, and node-wise snapping.

**Architecture:** Keep the public CLI flow intact while replacing the internal scaffold with a paper-aligned model. `src/trees/` defines the structural candidate graph, `src/models/` evaluates the full tree with node-wise logits, `src/training/` performs Adam plus hardening, and `src/snapping/` converts trained logits into a discrete `RecoveredTree` consumed by evaluation and formula export.

**Tech Stack:** Julia 1.12.6, Lux.jl, ComponentArrays.jl, StableRNGs.jl, Optimisers.jl, Test, JSON3.jl, CSV.jl, DataFrames.jl

---

## File Map

- Modify: `Project.toml`
- Modify: `src/EMLRegression.jl`
- Modify: `src/trees/TreeTypes.jl`
- Modify: `src/trees/MasterTree.jl`
- Modify: `src/models/EMLLayer.jl`
- Modify: `src/training/TrainConfig.jl`
- Modify: `src/training/TrainLoop.jl`
- Modify: `src/snapping/Snap.jl`
- Modify: `src/eval/Recovery.jl`
- Modify: `src/symbolics/Export.jl`
- Modify: `scripts/run_experiment.jl`
- Modify: `scripts/export_recovered_formulas.jl`
- Modify: `README.md`
- Modify: `test/models/test_eml_layer.jl`
- Modify: `test/training/test_train_loop.jl`
- Modify: `test/snapping/test_snap.jl`
- Modify: `test/eval/test_recovery.jl`
- Modify: `test/integration/test_smoke_ln.jl`
- Optional create if needed for focus: `test/trees/test_master_tree.jl`

### Task 1: Define A Real Master Tree And Discrete Recovery Types

**Files:**
- Modify: `src/trees/TreeTypes.jl`
- Modify: `src/trees/MasterTree.jl`
- Test: `test/trees/test_master_tree.jl`

- [ ] **Step 1: Write the failing tree-structure tests**

```julia
@testset "master tree candidate tables" begin
    tree = build_master_tree(depth=2, variables=(:x,))
    @test length(tree.nodes) == 3
    @test tree.root_id == 1
    @test :const1 in tree.nodes[2].left_candidates
    @test :x in tree.nodes[2].left_candidates
    @test :node_2 in tree.nodes[1].left_candidates
    @test :node_3 in tree.nodes[1].right_candidates
end

@testset "recovered tree stores node-wise choices" begin
    recovered = RecoveredTree(Dict(1 => (:node_2, :node_3), 2 => (:x, :const1), 3 => (:const1, :x)))
    @test recovered.choices[2] == (:x, :const1)
end
```

- [ ] **Step 2: Run the tree tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["trees"])'`
Expected: FAIL because `MasterTree` has no `root_id`, node candidate tables, or `RecoveredTree` choice map.

- [ ] **Step 3: Implement structural types and candidate indexing**

```julia
struct TreeNodeSpec
    id::Int
    left_candidates::Vector{Symbol}
    right_candidates::Vector{Symbol}
    left_child::Union{Int,Nothing}
    right_child::Union{Int,Nothing}
end

struct MasterTree
    depth::Int
    variables::Vector{Symbol}
    root_id::Int
    terminals::Vector{Symbol}
    nodes::Vector{TreeNodeSpec}
end

struct RecoveredTree
    choices::Dict{Int,Tuple{Symbol,Symbol}}
end
```

- [ ] **Step 4: Build deterministic candidate tables**

```julia
function build_master_tree(; depth::Integer, variables)
    terminals = Symbol[:const1, variables...]
    node_count = 2^depth - 1
    nodes = TreeNodeSpec[]
    for node_id in 1:node_count
        left_child = 2 * node_id <= node_count ? 2 * node_id : nothing
        right_child = 2 * node_id + 1 <= node_count ? 2 * node_id + 1 : nothing
        left_candidates = isnothing(left_child) ? copy(terminals) : vcat(copy(terminals), Symbol("node_$(left_child)"))
        right_candidates = isnothing(right_child) ? copy(terminals) : vcat(copy(terminals), Symbol("node_$(right_child)"))
        push!(nodes, TreeNodeSpec(node_id, left_candidates, right_candidates, left_child, right_child))
    end
    MasterTree(depth, Symbol[variables...], 1, terminals, nodes)
end
```

- [ ] **Step 5: Run the tree tests to verify they pass**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["trees"])'`
Expected: PASS for deterministic node indexing and recovered-tree shape.

- [ ] **Step 6: Commit**

```bash
git add src/trees/TreeTypes.jl src/trees/MasterTree.jl test/trees/test_master_tree.jl
git commit -m "feat: define paper-aligned master tree structure"
```

### Task 2: Replace Terminal Mixing With Recursive Node Evaluation

**Files:**
- Modify: `src/models/EMLLayer.jl`
- Modify: `src/EMLRegression.jl`
- Test: `test/models/test_eml_layer.jl`

- [ ] **Step 1: Write the failing model tests**

```julia
@testset "layer parameters are node-wise" begin
    layer = EMLTreeLayer(build_master_tree(depth=2, variables=(:x,)))
    ps, st = Lux.setup(StableRNG(1), layer)
    @test haskey(ps, :nodes)
    @test haskey(ps.nodes[1], :left_logits)
    @test haskey(ps.nodes[1], :right_logits)
end

@testset "layer evaluates recursively" begin
    tree = build_master_tree(depth=2, variables=(:x,))
    layer = EMLTreeLayer(tree)
    ps, st = Lux.setup(StableRNG(1), layer)
    y, _ = Lux.apply(layer, ComplexF64[0.5], ps, st)
    @test length(y) == 1
end
```

- [ ] **Step 2: Run the model tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["models"])'`
Expected: FAIL because parameters are still a single `logits` vector and the forward pass does not recurse through tree nodes.

- [ ] **Step 3: Implement node-wise parameters**

```julia
function Lux.initialparameters(rng::LehmerRNG, layer::EMLTreeLayer)
    node_params = map(layer.tree.nodes) do node
        (;
            left_logits=zeros(Float64, length(node.left_candidates)),
            right_logits=zeros(Float64, length(node.right_candidates)),
        )
    end
    return ComponentArray(nodes=node_params)
end
```

- [ ] **Step 4: Implement recursive forward evaluation**

```julia
function _evaluate_node(layer, node_id, x, ps)
    node = layer.tree.nodes[node_id]
    left_value = _soft_source_value(node.left_candidates, x, ps.nodes[node_id].left_logits)
    right_value = _soft_source_value(node.right_candidates, x, ps.nodes[node_id].right_logits)
    return eml(left_value, right_value)
end

function (layer::EMLTreeLayer)(x, ps, st)
    y = _evaluate_node(layer, layer.tree.root_id, _to_complex_input(x), ps)
    return y, st
end
```

- [ ] **Step 5: Extend `_soft_source_value` to resolve child-node references**

```julia
if startswith(String(symbol), "node_")
    child_id = parse(Int, split(String(symbol), "_")[2])
    value = _evaluate_node(layer, child_id, x, ps)
end
```

- [ ] **Step 6: Run the model tests to verify they pass**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["models"])'`
Expected: PASS for parameter layout and recursive forward shape checks.

- [ ] **Step 7: Commit**

```bash
git add src/models/EMLLayer.jl src/EMLRegression.jl test/models/test_eml_layer.jl
git commit -m "feat: evaluate eml trees recursively"
```

### Task 3: Add Real Optimization Updates To Training

**Files:**
- Modify: `Project.toml`
- Modify: `src/training/TrainConfig.jl`
- Modify: `src/training/TrainLoop.jl`
- Test: `test/training/test_train_loop.jl`

- [ ] **Step 1: Write the failing training-update tests**

```julia
@testset "training updates logits" begin
    cfg = TrainConfig(depth=2, target=:ln, batch_size=32, steps=3, learning_rate=1e-2)
    result = run_training(cfg; rng=StableRNG(1))
    @test any(!iszero, result.metrics[:train_loss])
    @test result.params.nodes[1].left_logits != zeros(length(result.params.nodes[1].left_logits))
end
```

- [ ] **Step 2: Run the training tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: FAIL because `TrainConfig` has no optimizer controls and `run_training` never updates `ps`.

- [ ] **Step 3: Add optimizer configuration**

```julia
Base.@kwdef struct TrainConfig
    depth::Int
    target::Symbol
    batch_size::Int
    steps::Int
    learning_rate::Float64 = 1e-2
    hardening_steps::Int = 0
    hardening_start::Int = steps + 1
    hardening_weight::Float64 = 0.1
    temperature::Float64 = 1.0
    margin_threshold::Float64 = 0.05
end
```

- [ ] **Step 4: Add Optimisers.jl and an update loop**

```julia
using Optimisers

opt = Optimisers.Adam(cfg.learning_rate)
opt_state = Optimisers.setup(opt, ps)

loss_fn(ps_current) = begin
    preds, _ = Lux.apply(layer, xs, ps_current, st)
    _mse(preds, ys)
end

grads = gradient(loss_fn, ps)[1]
opt_state, ps = Optimisers.update(opt_state, ps, grads)
```

- [ ] **Step 5: Track pre/post parameter snapshots in metrics**

```julia
metrics = Dict(
    :train_loss => train_loss,
    :hardening_loss => hardening_loss,
    :logit_margin => Float64[],
)
```

- [ ] **Step 6: Run the training tests to verify they pass**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: PASS with non-default node logits after training.

- [ ] **Step 7: Commit**

```bash
git add Project.toml src/training/TrainConfig.jl src/training/TrainLoop.jl test/training/test_train_loop.jl
git commit -m "feat: add optimizer-backed eml training loop"
```

### Task 4: Make Hardening Affect Training Dynamics

**Files:**
- Modify: `src/models/EMLLayer.jl`
- Modify: `src/training/TrainConfig.jl`
- Modify: `src/training/TrainLoop.jl`
- Test: `test/training/test_train_loop.jl`

- [ ] **Step 1: Write the failing hardening tests**

```julia
@testset "hardening sharpens selections" begin
    cfg = TrainConfig(depth=2, target=:ln, batch_size=32, steps=4, hardening_steps=3, hardening_start=3, hardening_weight=0.5)
    result = run_training(cfg; rng=StableRNG(1))
    @test !isempty(result.metrics[:hardening_loss])
    @test result.metrics[:logit_margin][end] > result.metrics[:logit_margin][1]
end
```

- [ ] **Step 2: Run the training tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: FAIL because hardening currently only appends a constant metric and does not change the model.

- [ ] **Step 3: Implement a hardening penalty**

```julia
function _hardening_penalty(ps)
    penalties = Float64[]
    for node in ps.nodes
        push!(penalties, 1.0 - maximum(softmax(node.left_logits)))
        push!(penalties, 1.0 - maximum(softmax(node.right_logits)))
    end
    return sum(penalties)
end
```

- [ ] **Step 4: Schedule hardening after `hardening_start`**

```julia
hardening_active = step >= cfg.hardening_start
total_loss = mse_loss + (hardening_active ? cfg.hardening_weight * _hardening_penalty(ps_current) : 0.0)
push!(hardening_loss, hardening_active ? total_loss - mse_loss : 0.0)
push!(logit_margin, _mean_logit_margin(ps))
```

- [ ] **Step 5: Optionally anneal temperature during hardening**

```julia
temperature = hardening_active ? max(cfg.temperature * 0.5, 0.1) : cfg.temperature
```

- [ ] **Step 6: Run the training tests to verify they pass**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
Expected: PASS with increasing margin and non-empty hardening metrics.

- [ ] **Step 7: Commit**

```bash
git add src/models/EMLLayer.jl src/training/TrainConfig.jl src/training/TrainLoop.jl test/training/test_train_loop.jl
git commit -m "feat: harden node selections during training"
```

### Task 5: Snap Per-Node Choices Into A Discrete Recovered Tree

**Files:**
- Modify: `src/snapping/Snap.jl`
- Modify: `src/symbolics/Export.jl`
- Modify: `src/eval/Recovery.jl`
- Test: `test/snapping/test_snap.jl`
- Test: `test/eval/test_recovery.jl`

- [ ] **Step 1: Write the failing snapping and export tests**

```julia
@testset "snap_model returns node-wise choices" begin
    layer = EMLTreeLayer(build_master_tree(depth=2, variables=(:x,)))
    ps = (nodes=[(left_logits=[0.0, 3.0], right_logits=[2.0, 0.0])],)
    recovered = snap_model(layer, ps; margin_threshold=0.1)
    @test recovered.choices[1] == (:x, :const1)
end

@testset "formula export traverses the recovered tree" begin
    recovered = RecoveredTree(Dict(1 => (:node_2, :const1), 2 => (:x, :const1)))
    @test occursin("eml", formula_string(recovered))
    @test occursin("x", formula_string(recovered))
end
```

- [ ] **Step 2: Run snapping and recovery tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["snapping", "eval"])'`
Expected: FAIL because snapping only chooses one terminal globally and export/eval do not traverse a tree.

- [ ] **Step 3: Implement node-wise snapping**

```julia
function snap_model(layer::EMLTreeLayer, ps; margin_threshold=0.0)
    choices = Dict{Int,Tuple{Symbol,Symbol}}()
    for node in layer.tree.nodes
        left = _snap_choice(node.left_candidates, ps.nodes[node.id].left_logits, margin_threshold)
        right = _snap_choice(node.right_candidates, ps.nodes[node.id].right_logits, margin_threshold)
        choices[node.id] = (left, right)
    end
    return RecoveredTree(choices)
end
```

- [ ] **Step 4: Implement discrete recovered-tree evaluation and export**

```julia
function evaluate_recovered(tree::RecoveredTree, master::MasterTree, x)
    _eval_recovered_node(tree, master, master.root_id, x)
end

function formula_string(tree::RecoveredTree, master::MasterTree)
    _formula_for_node(tree, master, master.root_id)
end
```

- [ ] **Step 5: Report ambiguity explicitly**

```julia
struct SnapChoice
    symbol::Symbol
    margin::Float64
    ambiguous::Bool
end
```

- [ ] **Step 6: Run snapping and recovery tests to verify they pass**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["snapping", "eval"])'`
Expected: PASS for per-node snapping, formula export, and recovered-tree evaluation.

- [ ] **Step 7: Commit**

```bash
git add src/snapping/Snap.jl src/symbolics/Export.jl src/eval/Recovery.jl test/snapping/test_snap.jl test/eval/test_recovery.jl
git commit -m "feat: snap eml trees into discrete node choices"
```

### Task 6: Reconnect The CLI, Integration Path, And Docs

**Files:**
- Modify: `scripts/run_experiment.jl`
- Modify: `scripts/export_recovered_formulas.jl`
- Modify: `test/integration/test_smoke_ln.jl`
- Modify: `README.md`

- [ ] **Step 1: Write the failing integration test updates**

```julia
@testset "ln smoke run exports node-wise recovered tree" begin
    outcome = run_experiment(TrainConfig(depth=2, target=:ln, batch_size=16, steps=2); rng=StableRNG(1))
    @test haskey(outcome.recovered_tree.choices, 1)
    @test outcome.recovery.snap_status in (:ok, :ambiguous)
end
```

- [ ] **Step 2: Run integration tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'`
Expected: FAIL because integration still assumes flat recovered formulas.

- [ ] **Step 3: Update experiment and export scripts**

```julia
recovered = snap_model(layer, training.params; margin_threshold=cfg.margin_threshold)
formula = formula_string(recovered, layer.tree)
preds = evaluate_recovered(recovered, layer.tree, xs)
```

- [ ] **Step 4: Update README examples and caveats**

```md
- hardening now changes node-wise logits during late training,
- snapped formulas are reconstructed trees rather than single terminals,
- `results/raw/*.json` still overwrite by `config-target-seed`.
```

- [ ] **Step 5: Run focused integration, then full suite**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'`
Expected: PASS for smoke integration.

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS for the full repository test suite.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_experiment.jl scripts/export_recovered_formulas.jl test/integration/test_smoke_ln.jl README.md
git commit -m "docs: reconnect paper-aligned tree pipeline"
```

## Final Verification Checklist

- [ ] `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.instantiate()'`
- [ ] `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["trees"])'`
- [ ] `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["models"])'`
- [ ] `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["training"])'`
- [ ] `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["snapping", "eval"])'`
- [ ] `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'`
- [ ] `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test()'`
- [ ] `~/.juliaup/bin/julia --project=. scripts/run_experiment.jl --config experiments/configs/must_pass_depth2.toml --target ln --seed 1`
- [ ] `~/.juliaup/bin/julia --project=. scripts/export_recovered_formulas.jl`

## Notes

- Use `@test-driven-development` for every task. Do not write production code before the targeted failing test exists and is observed failing.
- Keep each commit scoped to the task boundary above; do not bundle tree/model/training/snapping changes in a single commit.
- If Lux gradient plumbing becomes awkward, prefer the smallest additional dependency that keeps the training loop explicit. Do not switch frameworks.
- Preserve the current `results/raw/` overwrite semantics unless the user asks to change experiment artifact layout.
