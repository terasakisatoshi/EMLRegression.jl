# Snap Diagnostics And Margin-Aware Search Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add active-node snap diagnostics to raw results and make top-k snapping search prioritize low-margin branches so depth5 ambiguous failures become observable and easier to reduce.

**Architecture:** Keep the recovery pipeline intact while extending `src/snapping/Snap.jl` with one diagnostics path and one search-ordering path. Diagnostics flow from snapping into `run_experiment` raw JSON output, while search still uses the same beam scoring and refinement logic after reordering side expansions by ascending margin.

**Tech Stack:** Julia 1.12.6, Lux.jl, JSON3.jl, Test, CSV.jl, DataFrames.jl

---

## File Map

- Modify: `src/snapping/Snap.jl`
- Modify: `src/eval/Recovery.jl`
- Modify: `src/eval/Metrics.jl`
- Modify: `scripts/run_experiment.jl`
- Modify: `test/snapping/test_snap.jl`
- Modify: `test/integration/test_smoke_ln.jl`
- Re-run: `experiments/configs/challenge_depth5_sweep.toml`
- Re-run: `scripts/summarize_results.jl`

### Task 1: Add Failing Snap Diagnostics Tests

**Files:**
- Modify: `test/snapping/test_snap.jl`
- Test: `test/snapping/test_snap.jl`

- [ ] **Step 1: Write the failing diagnostics test**

```julia
@testset "snap diagnostics report active bottleneck margins" begin
    tree = build_master_tree(depth=2, variables=(:x,))
    layer = EMLTreeLayer(tree)
    ps = (
        nodes=(
            (left_logits=[0.0, 0.0, 0.2], right_logits=[0.3, 0.2, 0.0]),
            (left_logits=[0.9, 0.8], right_logits=[0.4, 0.1]),
            (left_logits=[0.0, 0.0], right_logits=[0.0, 0.0]),
        ),
    )

    diagnostics = EMLRegression.snap_diagnostics(layer, ps; margin_threshold=0.15)

    @test length(diagnostics) == 2
    @test diagnostics[1].node_id == 1
    @test diagnostics[1].left_choice == :node_2
    @test diagnostics[1].left_margin ≈ 0.2
    @test diagnostics[1].right_margin ≈ 0.1
    @test diagnostics[1].ambiguous
end
```

- [ ] **Step 2: Run the snapping tests to verify they fail**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["snapping"])'`
Expected: FAIL because `snap_diagnostics` does not exist yet.

- [ ] **Step 3: Add a failing margin-aware ordering test**

```julia
@testset "margin-aware neighbors expand the lowest-margin side first" begin
    tree = build_master_tree(depth=2, variables=(:x,))
    layer = EMLTreeLayer(tree)
    ps = (
        nodes=(
            (left_logits=[0.0, 0.0, 0.2], right_logits=[1.0, 0.0, 0.0]),
            (left_logits=[1.0, 0.9], right_logits=[0.8, 0.1]),
            (left_logits=[0.0, 0.0], right_logits=[0.0, 0.0]),
        ),
    )
    tree0 = snap_model(layer, ps)

    neighbors = EMLRegression._top_k_neighbors(tree0, layer, ps; top_k=2)

    @test neighbors[1].choices[1][1] != tree0.choices[1][1]
end
```

- [ ] **Step 4: Re-run snapping tests to verify the new failure**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["snapping"])'`
Expected: FAIL because neighbor ordering is still uniform.

### Task 2: Implement Diagnostics And Margin-Aware Search

**Files:**
- Modify: `src/snapping/Snap.jl`
- Modify: `src/eval/Recovery.jl`
- Modify: `src/eval/Metrics.jl`

- [ ] **Step 1: Add active-node diagnostics helpers**

```julia
function snap_diagnostics(layer::EMLTreeLayer, ps; margin_threshold=0.0)
    tree = snap_model(layer, ps; margin_threshold=margin_threshold)
    diagnostics = NamedTuple[]
    for node_id in sort!(collect(_active_node_ids(tree)))
        node_ps = ps.nodes[node_id]
        left_choice, right_choice = tree.choices[node_id]
        left_margin = _choice_margin(node_ps.left_logits)
        right_margin = _choice_margin(node_ps.right_logits)
        push!(diagnostics, (
            node_id=node_id,
            left_choice=left_choice,
            right_choice=right_choice,
            left_margin=left_margin,
            right_margin=right_margin,
            min_margin=min(left_margin, right_margin),
            ambiguous=min(left_margin, right_margin) < margin_threshold || left_margin < margin_threshold || right_margin < margin_threshold,
        ))
    end
    diagnostics
end
```

- [ ] **Step 2: Make neighbor expansion margin-aware**

```julia
function _ranked_active_sides(tree, layer, ps)
    sides = NamedTuple[]
    for node_id in sort!(collect(_active_node_ids(tree)))
        node_ps = ps.nodes[node_id]
        push!(sides, (node_id=node_id, side=:left, margin=_choice_margin(node_ps.left_logits)))
        push!(sides, (node_id=node_id, side=:right, margin=_choice_margin(node_ps.right_logits)))
    end
    sort!(sides; by=entry -> (entry.margin, entry.node_id, entry.side === :left ? 0 : 1))
end
```

- [ ] **Step 3: Reuse diagnostics in recovery output**

```julia
diagnostics = snap_diagnostics(layer, training.params; margin_threshold=cfg.margin_threshold)
```

- [ ] **Step 4: Extend the recovery payload to carry diagnostics**

```julia
struct RecoveryVerdict
    ...
    snap_diagnostics
end
```

- [ ] **Step 5: Run snapping tests to verify they pass**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["snapping"])'`
Expected: PASS for diagnostics and neighbor ordering.

### Task 3: Add Failing Raw Export Coverage

**Files:**
- Modify: `test/integration/test_smoke_ln.jl`
- Modify: `scripts/run_experiment.jl`

- [ ] **Step 1: Write the failing raw export test**

```julia
@test "snap_diagnostics" in keys(run1_json)
@test run1_json["snap_diagnostics"][1]["node_id"] == 1
@test "left_margin" in keys(run1_json["snap_diagnostics"][1])
```

- [ ] **Step 2: Run the integration test to verify it fails**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'`
Expected: FAIL because raw JSON does not include `snap_diagnostics`.

- [ ] **Step 3: Serialize diagnostics into raw results**

```julia
"snap_diagnostics" => [
    Dict(
        "node_id" => diag.node_id,
        "left_choice" => String(diag.left_choice),
        "right_choice" => String(diag.right_choice),
        "left_margin" => diag.left_margin,
        "right_margin" => diag.right_margin,
        "min_margin" => diag.min_margin,
        "ambiguous" => diag.ambiguous,
    )
    for diag in outcome.recovery.snap_diagnostics
],
```

- [ ] **Step 4: Re-run the integration test to verify it passes**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'`
Expected: PASS with the new raw artifact field.

### Task 4: Verify And Re-Run Depth5 Sweep

**Files:**
- Re-run: `experiments/configs/challenge_depth5_sweep.toml`
- Re-run: `scripts/summarize_results.jl`

- [ ] **Step 1: Run targeted verification**

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["snapping"])'`
Expected: PASS

Run: `~/.juliaup/bin/julia --project=. -e 'using Pkg; Pkg.test(test_args=["integration"])'`
Expected: PASS

- [ ] **Step 2: Run the `depth5_blind_margin` 8-seed sweep**

Run: `~/.juliaup/bin/julia --project=. scripts/run_suite.jl --config experiments/configs/challenge_depth5_sweep.toml --variant depth5_blind_margin`
Expected: eight raw result files refreshed under `results/raw/`.

- [ ] **Step 3: Rebuild summary output**

Run: `~/.juliaup/bin/julia --project=. scripts/summarize_results.jl`
Expected: `results/summaries/summary.csv` updated with refreshed `ambiguous_rate`.

- [ ] **Step 4: Inspect the refreshed depth5 summary rows**

Run: `rg -n "depth5_blind_margin|depth5_affine_log" results/summaries/summary.csv`
Expected: visible updated `ambiguous_rate`, `snap_ok_rate`, and recovery rates for the rerun.
