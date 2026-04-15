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
    "challenge_depth5" => [:depth5_affine_log],
    "challenge_depth6" => [:depth6_inverse_logy],
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

@testset "paper sweep suite configs exist for depth 3 through 6" begin
    for suite in (
        "must_pass_depth3_sweep",
        "challenge_depth4_sweep",
        "challenge_depth4_init_sweep",
        "challenge_depth5_sweep",
        "challenge_depth5_basin_sweep",
        "challenge_depth6_sweep",
    )
        cfg_text = read(joinpath(@__DIR__, "..", "..", "experiments", "configs", "$(suite).toml"), String)
        @test occursin("[[variants]]", cfg_text)
    end
end

@testset "challenge depth4 sweep includes a longer tuned variant" begin
    cfg_text = read(joinpath(@__DIR__, "..", "..", "experiments", "configs", "challenge_depth4_sweep.toml"), String)
    @test occursin("name = \"longer_cool\"", cfg_text)
    @test occursin("steps = 600", cfg_text)
    @test occursin("learning_rate = 0.05", cfg_text)
end

@testset "challenge depth4 init sweep covers multiple initialization strategies" begin
    cfg_text = read(joinpath(@__DIR__, "..", "..", "experiments", "configs", "challenge_depth4_init_sweep.toml"), String)
    @test occursin("init_strategy = \"zero_bias_to_inputs\"", cfg_text)
    @test occursin("init_strategy = \"margin_biased\"", cfg_text)
    @test occursin("seeds = [1, 2, 3, 4, 5, 6, 7, 8]", cfg_text)
end

@testset "challenge depth4 basin sweep covers noisy target initialization" begin
    cfg_text = read(joinpath(@__DIR__, "..", "..", "experiments", "configs", "challenge_depth4_basin_sweep.toml"), String)
    @test occursin("init_strategy = \"target_tree_noise\"", cfg_text)
    @test occursin("target_noise_std = 0.05", cfg_text)
    @test occursin("target_noise_std = 0.25", cfg_text)
end

@testset "challenge depth5 basin sweep covers noisy target initialization" begin
    cfg_text = read(joinpath(@__DIR__, "..", "..", "experiments", "configs", "challenge_depth5_basin_sweep.toml"), String)
    @test occursin("init_strategy = \"target_tree_noise\"", cfg_text)
    @test occursin("target_noise_std = 0.05", cfg_text)
    @test occursin("target_noise_std = 0.25", cfg_text)
end

@testset "challenge depth5 sweep includes a depth-aware blind initialization" begin
    cfg_text = read(joinpath(@__DIR__, "..", "..", "experiments", "configs", "challenge_depth5_sweep.toml"), String)
    @test occursin("name = \"depth5_blind_bias\"", cfg_text)
    @test occursin("init_strategy = \"depth5_blind_bias\"", cfg_text)
end

@testset "challenge depth6 sweep is configured as a negative control" begin
    cfg_text = read(joinpath(@__DIR__, "..", "..", "experiments", "configs", "challenge_depth6_sweep.toml"), String)
    @test occursin("targets = [\"depth6_inverse_logy\"]", cfg_text)
    @test occursin("name = \"zero_bias\"", cfg_text)
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

@testset "depth4 nested target can recover under the longer cool schedule" begin
    cfg = TrainConfig(
        depth=4,
        target=:depth4_nested,
        steps=600,
        hardening_steps=150,
        batch_size=64,
        learning_rate=0.05,
        hardening_weight=0.2,
        temperature=1.0,
    )
    outcome = run_experiment(cfg; rng=StableRNG(2))
    @test outcome.recovery.success
    @test outcome.recovery.structure_match
end

@testset "depth4 top-k snapping search can recover another longer cool seed" begin
    cfg = TrainConfig(
        depth=4,
        target=:depth4_nested,
        steps=600,
        hardening_steps=150,
        batch_size=64,
        learning_rate=0.05,
        hardening_weight=0.2,
        temperature=1.0,
    )
    outcome = run_experiment(cfg; rng=StableRNG(1))
    @test outcome.recovery.success
    @test outcome.recovery.structure_match
end

@testset "depth4 target-tree basin initialization recovers under moderate noise" begin
    cfg = TrainConfig(
        depth=4,
        target=:depth4_nested,
        steps=300,
        hardening_steps=75,
        batch_size=64,
        learning_rate=0.05,
        hardening_weight=0.2,
        temperature=1.0,
        init_strategy=:target_tree_noise,
        target_noise_std=0.05,
    )
    outcome = run_experiment(cfg; rng=StableRNG(1))
    @test outcome.recovery.success
    @test outcome.recovery.structure_match
end

@testset "depth5 affine-log target recovers from target-tree basin initialization" begin
    cfg = TrainConfig(
        depth=5,
        target=:depth5_affine_log,
        steps=400,
        hardening_steps=100,
        batch_size=64,
        learning_rate=0.05,
        hardening_weight=0.2,
        temperature=1.0,
        init_strategy=:target_tree_noise,
        target_noise_std=0.05,
    )
    outcome = run_experiment(cfg; rng=StableRNG(1))
    @test outcome.recovery.success
    @test outcome.recovery.structure_match
end

@testset "depth6 inverse-logy target remains a blind-recovery negative control" begin
    cfg = TrainConfig(
        depth=6,
        target=:depth6_inverse_logy,
        steps=1000,
        hardening_steps=250,
        batch_size=64,
        learning_rate=0.03,
        hardening_weight=0.2,
        temperature=1.0,
        init_strategy=:small_gaussian,
    )
    outcome = run_experiment(cfg; rng=StableRNG(1))
    @test !outcome.recovery.success
end

@testset "summary script reports paper recovery rates" begin
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    summary_script = joinpath(repo_root, "scripts", "summarize_results.jl")
    mktempdir() do dir
        raw_dir = joinpath(dir, "results", "raw")
        mkpath(raw_dir)

        write(joinpath(raw_dir, "run1.json"), """
        {"config_name":"must_pass_depth2","depth":2,"target":"depth2_exp","tier":"must_pass","seed":1,"init_strategy":"small_gaussian","success":true,"reason":"recovered","snap_status":"ok","structure_match":true,"ambiguous_nodes":0,"validation_loss":0.0,"numerical_match":true,"training_failure_reason":"no_failure","max_output_abs":1.0,"max_node_abs":2.0,"formula":"eml(x, 1)","train_loss":0.0,"hardening_loss":0.0}
        """)
        write(joinpath(raw_dir, "run2.json"), """
        {"config_name":"must_pass_depth2","depth":2,"target":"depth2_exp","tier":"must_pass","seed":2,"init_strategy":"subtree_favoring","success":false,"reason":"snap_failed","snap_status":"ambiguous","structure_match":false,"ambiguous_nodes":1,"validation_loss":0.5,"numerical_match":true,"training_failure_reason":"no_failure","max_output_abs":3.0,"max_node_abs":4.0,"formula":"1","train_loss":0.1,"hardening_loss":0.2}
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
        @test "init_strategies" in names(summary)
        @test summary[1, :success_rate] ≈ 0.5
        @test summary[1, :init_strategies] == "small_gaussian,subtree_favoring"
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
        init_strategy = "subtree_favoring"
        """)

        run(Cmd(`$(Base.julia_cmd()) --project=$(repo_root) $(suite_script) --config $(config_path)`; dir=dir))

        raw_dir = joinpath(dir, "results", "raw")
        paths = sort(filter(p -> endswith(p, ".json"), readdir(raw_dir; join=true)))
        @test length(paths) == 2

        rows = JSON3.read.(read.(paths, String))
        config_names = sort([String(row[:config_name]) for row in rows])
        @test config_names == ["depth2_sweep-lr010", "depth2_sweep-lr020"]
        strategies = sort([String(row[:init_strategy]) for row in rows])
        @test strategies == ["small_gaussian", "subtree_favoring"]
    end
end
