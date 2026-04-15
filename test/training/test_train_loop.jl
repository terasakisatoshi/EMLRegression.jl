using Test
using StableRNGs
using EMLRegression

@testset "training loop" begin
    cfg = TrainConfig(depth=2, target=:ln, batch_size=32, steps=3, learning_rate=1e-2)
    result = run_training(cfg; rng=StableRNG(1))
    @test haskey(result.metrics, :train_loss)
    @test haskey(result.metrics, :logit_margin)
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
