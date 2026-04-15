using Test
using StableRNGs
using EMLRegression

@testset "ln smoke run" begin
    cfg = TrainConfig(depth=2, target=:ln, steps=5, hardening_steps=2, batch_size=16)
    outcome = run_experiment(cfg; rng=StableRNG(1))
    @test haskey(outcome, :recovery)
    @test haskey(outcome.recovered_tree.choices, 1)
    @test outcome.recovery.snap_status in (:ok, :ambiguous)
end
