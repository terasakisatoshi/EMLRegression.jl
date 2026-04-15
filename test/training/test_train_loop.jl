using Test
using StableRNGs
using Lux
using EMLRegression

@testset "training loop" begin
    cfg = TrainConfig(depth=2, target=:ln, batch_size=32, steps=3, learning_rate=1e-2)
    result = run_training(cfg; rng=StableRNG(1))
    @test haskey(result.metrics, :train_loss)
    @test haskey(result.metrics, :logit_margin)
    @test haskey(result.metrics, :max_node_abs)
    @test length(result.metrics[:train_loss]) == 3
    @test any(!iszero, result.metrics[:train_loss])
    @test result.params.nodes[1].left_logits != zeros(length(result.params.nodes[1].left_logits))
end

@testset "hardening sharpens selections" begin
    cfg = TrainConfig(
        depth=2,
        target=:ln,
        batch_size=32,
        steps=4,
        hardening_steps=3,
        hardening_start=3,
        hardening_weight=0.5,
        learning_rate=1e-2,
    )
    result = run_training(cfg; rng=StableRNG(1))
    @test !isempty(result.metrics[:hardening_loss])
    @test length(result.metrics[:logit_margin]) == cfg.steps + cfg.hardening_steps
    @test result.metrics[:hardening_loss] != fill(cfg.hardening_weight, cfg.hardening_steps)
    @test result.metrics[:logit_margin][end] > result.metrics[:logit_margin][1]
end

@testset "hardening defaults to the final steps" begin
    cfg = TrainConfig(
        depth=2,
        target=:ln,
        batch_size=32,
        steps=4,
        hardening_steps=3,
        learning_rate=1e-2,
    )
    result = run_training(cfg; rng=StableRNG(1))
    @test length(result.metrics[:hardening_loss]) == cfg.hardening_steps
end

@testset "complexity penalty prefers simpler selections" begin
    layer = EMLTreeLayer(build_master_tree(depth=2, variables=(:x,)))
    simple = (
        nodes=(
            (left_logits=[8.0, -8.0, -8.0], right_logits=[8.0, -8.0, -8.0]),
            (left_logits=[8.0, -8.0], right_logits=[8.0, -8.0]),
            (left_logits=[8.0, -8.0], right_logits=[8.0, -8.0]),
        ),
    )
    complex = (
        nodes=(
            (left_logits=[-8.0, 8.0, -8.0], right_logits=[-8.0, -8.0, 8.0]),
            (left_logits=[-8.0, 8.0], right_logits=[-8.0, 8.0]),
            (left_logits=[-8.0, 8.0], right_logits=[-8.0, 8.0]),
        ),
    )
    @test EMLRegression._complexity_penalty(layer, simple) < EMLRegression._complexity_penalty(layer, complex)
end

@testset "autodiff gradient matches parameter tree" begin
    rng = StableRNG(1)
    layer = EMLTreeLayer(build_master_tree(depth=2, variables=(:x,)))
    ps, st = Lux.setup(rng, layer)
    xs = ComplexF64.([0.3, 0.7, 1.1])
    ys = eml(xs, fill(1.0 + 0.0im, length(xs)))

    grads = EMLRegression._autodiff_gradient(ps) do ps_current
        preds, _ = Lux.apply(layer, xs, ps_current, st)
        EMLRegression._mse(preds, ys)
    end

    @test length(grads.nodes) == length(ps.nodes)
    @test length(grads.nodes[1].left_logits) == length(ps.nodes[1].left_logits)
    @test length(grads.nodes[1].right_logits) == length(ps.nodes[1].right_logits)
    @test any(!iszero, grads.nodes[1].left_logits)
    @test any(!iszero, grads.nodes[1].right_logits)
end

@testset "training loop clamps model outputs before recording metrics" begin
    cfg = TrainConfig(
        depth=2,
        target=:depth2_exp,
        batch_size=32,
        steps=3,
        learning_rate=1e-2,
        stability_limit=0.25,
    )
    result = run_training(cfg; rng=StableRNG(1))
    @test haskey(result.metrics, :max_output_abs)
    @test all(<=((0.25 + 1e-9)), result.metrics[:max_output_abs])
end

@testset "training loop keeps exp-heavy batches finite after exp clamping" begin
    overflow_target = TargetSpec(
        :overflow_probe,
        1,
        :must_pass,
        2,
        RecoveredTree(Dict(1 => (:const1, :const1), 2 => (:const1, :const1), 3 => (:const1, :const1))),
        (rng, n) -> ComplexF64.(fill(1000.0, n)),
    )
    EMLRegression.TARGETS[:overflow_probe] = overflow_target
    try
        cfg = TrainConfig(
            depth=2,
            target=:overflow_probe,
            batch_size=8,
            steps=2,
            learning_rate=1e-2,
        )
        result = run_training(cfg; rng=StableRNG(1))
        @test result.failure_reason == EMLRegression.no_failure
        @test all(isfinite, result.metrics[:max_output_abs])
    finally
        delete!(EMLRegression.TARGETS, :overflow_probe)
    end
end

@testset "initialization strategies bias root selections differently" begin
    tree = build_master_tree(depth=3, variables=(:x,))

    gaussian, _ = Lux.setup(StableRNG(1), EMLTreeLayer(tree; init_strategy=:small_gaussian))
    terminal, _ = Lux.setup(StableRNG(1), EMLTreeLayer(tree; init_strategy=:zero_bias_to_inputs))
    subtree, _ = Lux.setup(StableRNG(1), EMLTreeLayer(tree; init_strategy=:subtree_favoring))
    margin, _ = Lux.setup(StableRNG(1), EMLTreeLayer(tree; init_strategy=:margin_biased))

    @test length(gaussian.nodes) == length(tree.nodes)
    @test terminal.nodes[1].left_logits[1] > terminal.nodes[1].left_logits[end]
    @test subtree.nodes[1].left_logits[end] > subtree.nodes[1].left_logits[1]
    @test maximum(margin.nodes[1].left_logits) - minimum(margin.nodes[1].left_logits) >
          maximum(gaussian.nodes[1].left_logits) - minimum(gaussian.nodes[1].left_logits)
end
