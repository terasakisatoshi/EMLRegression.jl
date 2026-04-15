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
