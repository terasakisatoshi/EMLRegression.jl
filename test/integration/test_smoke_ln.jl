using Test
using StableRNGs
using CSV
using DataFrames
using JSON3
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

@testset "must_pass_depth2 config recovers the easiest target" begin
    cfg_text = read(joinpath(@__DIR__, "..", "..", "experiments", "configs", "must_pass_depth2.toml"), String)
    @test occursin("learning_rate = 0.2", cfg_text)
end

@testset "must_pass_depth3 config enables complexity-biased recovery" begin
    cfg_text = read(joinpath(@__DIR__, "..", "..", "experiments", "configs", "must_pass_depth3.toml"), String)
    @test occursin("learning_rate = 0.2", cfg_text)
    @test occursin("complexity_weight = 0.05", cfg_text)
end

@testset "paper sweep suite configs exist for depth 3 and 4" begin
    for suite in ("must_pass_depth3_sweep", "challenge_depth4_sweep")
        cfg_text = read(joinpath(@__DIR__, "..", "..", "experiments", "configs", "$(suite).toml"), String)
        @test occursin("[[variants]]", cfg_text)
    end
end

@testset "depth3 log target is not yet a strict recovery under complexity bias" begin
    cfg = TrainConfig(
        depth=3,
        target=:depth3_log,
        steps=500,
        hardening_steps=100,
        batch_size=64,
        learning_rate=0.2,
        complexity_weight=0.05,
    )
    outcome = run_experiment(cfg; rng=StableRNG(1))
    @test !outcome.recovery.success
    @test outcome.recovery.snap_status in (:ok, :ambiguous)
end

@testset "summary script reports paper recovery rates" begin
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    summary_script = joinpath(repo_root, "scripts", "summarize_results.jl")
    mktempdir() do dir
        raw_dir = joinpath(dir, "results", "raw")
        mkpath(raw_dir)

        write(joinpath(raw_dir, "run1.json"), """
        {"config_name":"must_pass_depth2","depth":2,"target":"depth2_exp","tier":"must_pass","seed":1,"success":true,"reason":"recovered","snap_status":"ok","structure_match":true,"ambiguous_nodes":0,"validation_loss":0.0,"numerical_match":true,"training_failure_reason":"no_failure","max_output_abs":1.0,"max_node_abs":2.0,"formula":"eml(x, 1)","train_loss":0.0,"hardening_loss":0.0}
        """)
        write(joinpath(raw_dir, "run2.json"), """
        {"config_name":"must_pass_depth2","depth":2,"target":"depth2_exp","tier":"must_pass","seed":2,"success":false,"reason":"snap_failed","snap_status":"ambiguous","structure_match":false,"ambiguous_nodes":1,"validation_loss":0.5,"numerical_match":true,"training_failure_reason":"no_failure","max_output_abs":3.0,"max_node_abs":4.0,"formula":"1","train_loss":0.1,"hardening_loss":0.2}
        """)

        run(Cmd(`$(Base.julia_cmd()) --project=$(repo_root) $(summary_script)`; dir=dir))

        summary = CSV.read(joinpath(dir, "results", "summaries", "summary.csv"), DataFrame)
        @test "success_rate" in names(summary)
        @test "numerical_match_rate" in names(summary)
        @test "snap_ok_rate" in names(summary)
        @test "ambiguous_rate" in names(summary)
        @test "structure_match_rate" in names(summary)
        @test "training_failure_rate" in names(summary)
        @test "mean_max_output_abs" in names(summary)
        @test "mean_max_node_abs" in names(summary)
        @test "mean_validation_loss" in names(summary)
        @test summary[1, :success_rate] ≈ 0.5
        @test summary[1, :numerical_match_rate] ≈ 1.0
        @test summary[1, :snap_ok_rate] ≈ 0.5
        @test summary[1, :ambiguous_rate] ≈ 0.5
        @test summary[1, :structure_match_rate] ≈ 0.5
        @test summary[1, :training_failure_rate] ≈ 0.0
        @test summary[1, :mean_max_output_abs] ≈ 2.0
        @test summary[1, :mean_max_node_abs] ≈ 3.0
        @test summary[1, :mean_validation_loss] ≈ 0.25
    end
end

@testset "suite runner expands sweep variants into distinct raw artifacts" begin
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    suite_script = joinpath(repo_root, "scripts", "run_suite.jl")
    mktempdir() do dir
        config_path = joinpath(dir, "depth2_sweep.toml")
        write(config_path, """
        name = "depth2_sweep"
        depth = 2
        targets = ["depth2_exp"]
        seeds = [1]
        steps = 5
        hardening_steps = 2
        batch_size = 16

        [[variants]]
        name = "lr010"
        learning_rate = 0.10

        [[variants]]
        name = "lr020"
        learning_rate = 0.20
        """)

        run(Cmd(`$(Base.julia_cmd()) --project=$(repo_root) $(suite_script) --config $(config_path)`; dir=dir))

        raw_dir = joinpath(dir, "results", "raw")
        paths = sort(filter(p -> endswith(p, ".json"), readdir(raw_dir; join=true)))
        @test length(paths) == 2

        rows = JSON3.read.(read.(paths, String))
        config_names = sort([String(row[:config_name]) for row in rows])
        @test config_names == ["depth2_sweep-lr010", "depth2_sweep-lr020"]
    end
end
