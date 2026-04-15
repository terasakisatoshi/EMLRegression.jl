using Test
using StableRNGs
using CSV
using DataFrames
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

@testset "summary script reports paper recovery rates" begin
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    summary_script = joinpath(repo_root, "scripts", "summarize_results.jl")
    mktempdir() do dir
        raw_dir = joinpath(dir, "results", "raw")
        mkpath(raw_dir)

        write(joinpath(raw_dir, "run1.json"), """
        {"config_name":"must_pass_depth2","depth":2,"target":"depth2_exp","tier":"must_pass","seed":1,"success":true,"reason":"recovered","snap_status":"ok","structure_match":true,"ambiguous_nodes":0,"validation_loss":0.0,"formula":"eml(x, 1)","train_loss":0.0,"hardening_loss":0.0}
        """)
        write(joinpath(raw_dir, "run2.json"), """
        {"config_name":"must_pass_depth2","depth":2,"target":"depth2_exp","tier":"must_pass","seed":2,"success":false,"reason":"ambiguous","snap_status":"ambiguous","structure_match":false,"ambiguous_nodes":1,"validation_loss":0.5,"formula":"1","train_loss":0.1,"hardening_loss":0.2}
        """)

        run(Cmd(`$(Base.julia_cmd()) --project=$(repo_root) $(summary_script)`; dir=dir))

        summary = CSV.read(joinpath(dir, "results", "summaries", "summary.csv"), DataFrame)
        @test "success_rate" in names(summary)
        @test "ambiguous_rate" in names(summary)
        @test "structure_match_rate" in names(summary)
        @test "mean_validation_loss" in names(summary)
        @test summary[1, :success_rate] ≈ 0.5
        @test summary[1, :ambiguous_rate] ≈ 0.5
        @test summary[1, :structure_match_rate] ≈ 0.5
        @test summary[1, :mean_validation_loss] ≈ 0.25
    end
end
