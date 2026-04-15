using Test
using StableRNGs
using EMLRegression

@testset "ln smoke run" begin
    cfg = TrainConfig(depth=2, target=:ln, steps=5, hardening_steps=2, batch_size=16)
    outcome = run_experiment(cfg; rng=StableRNG(1))
    @test haskey(outcome, :recovery)
end
