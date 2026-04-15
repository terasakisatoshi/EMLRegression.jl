using Test
using StableRNGs
using EMLRegression

const EXPECTED_SUITE_TARGETS = Dict(
    "must_pass_depth2" => [:depth2_exp, :depth2_double_exp],
    "must_pass_depth3" => [:depth3_log],
    "challenge_depth4" => [:depth4_nested],
)

@testset "paper benchmark suite configs" begin
    for (suite, targets) in EXPECTED_SUITE_TARGETS
        cfg_text = read(joinpath(@__DIR__, "..", "..", "experiments", "configs", "$(suite).toml"), String)
        for target in targets
            @test occursin("\"$(target)\"", cfg_text)
        end
    end
end

@testset "paper target smoke run" begin
    cfg = TrainConfig(depth=2, target=:depth2_exp, steps=5, hardening_steps=2, batch_size=16)
    outcome = run_experiment(cfg; rng=StableRNG(1))
    @test haskey(outcome, :recovery)
    @test haskey(outcome.recovered_tree.choices, 1)
    @test outcome.recovery.snap_status in (:ok, :ambiguous)
end
