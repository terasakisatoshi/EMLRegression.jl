using Test
using StableRNGs
using CSV
using DataFrames
using JSON3
using EMLRegression

@testset "paper target smoke run" begin
    cfg = TrainConfig(
        depth=2,
        target=:eml_depth2,
        search_iters=10,
        hardening_iters=4,
        eval_every=2,
        tail_eval_every=1,
        data_lo=1.0,
        data_hi=1.2,
        data_step=0.1,
    )
    outcome = run_experiment(cfg; rng=StableRNG(1))

    @test haskey(outcome, :training)
    @test haskey(outcome, :recovery)
    @test haskey(outcome, :snap)
    @test outcome.recovery.n_uncertain == outcome.snap.n_uncertain
    @test isfinite(outcome.recovery.snap_mse) || isnan(outcome.recovery.snap_mse)
end

@testset "summary script aggregates paper-style metrics" begin
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    summary_script = joinpath(repo_root, "scripts", "summarize_results.jl")
    mktempdir() do dir
        raw_dir = joinpath(dir, "results", "raw")
        mkpath(raw_dir)

        write(joinpath(raw_dir, "run1.json"), """
        {"config_name":"pnas_d2_random-biased","family":"depth2","depth":2,"target":"eml_depth2","tier":"must_pass","seed":1,"init_strategy":"biased","fit_success":true,"symbol_success":false,"stable_symbol_success":false,"success":true,"n_uncertain":1,"snap_mse":1e-8,"snap_rmse":1e-4,"snap_max_real":1e-3,"snap_max_imag":0.0,"hardening_iter":10}
        """)
        write(joinpath(raw_dir, "run2.json"), """
        {"config_name":"pnas_d2_random-random_hot","family":"depth2","depth":2,"target":"eml_depth2","tier":"must_pass","seed":2,"init_strategy":"random_hot","fit_success":false,"symbol_success":false,"stable_symbol_success":false,"success":false,"n_uncertain":3,"snap_mse":1e-2,"snap_rmse":1e-1,"snap_max_real":1e-1,"snap_max_imag":0.0,"hardening_iter":12}
        """)

        run(Cmd(`$(Base.julia_cmd()) --project=$(repo_root) $(summary_script)`; dir=dir))

        summary = CSV.read(joinpath(dir, "results", "summaries", "summary.csv"), DataFrame)
        section = CSV.read(joinpath(dir, "results", "summaries", "section_4_3_summary.csv"), DataFrame)
        @test "fit_success_rate" in names(summary)
        @test "symbol_success_rate" in names(summary)
        @test "stable_symbol_success_rate" in names(summary)
        @test "mean_n_uncertain" in names(summary)
        @test nrow(summary) == 2
        @test Set(summary.config_name) == Set(["pnas_d2_random-biased", "pnas_d2_random-random_hot"])
        @test Set(summary.init_strategies) == Set(["biased", "random_hot"])
        @test Set(section.job_name) == Set(["depth2"])
        @test Set(section.init_strategy) == Set(["biased", "random_hot"])
        @test sum(section.fit_count) == 1
        @test sum(section.runs) == 2
    end
end

@testset "suite runner expands variants into distinct raw artifacts" begin
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    suite_script = joinpath(repo_root, "scripts", "run_suite.jl")
    mktempdir() do dir
        config_path = joinpath(dir, "depth2_sweep.toml")
        write(config_path, """
        name = "depth2_sweep"
        depth = 2
        targets = ["eml_depth2"]
        seeds = [1]
        search_iters = 5
        hardening_iters = 2
        data_lo = 1.0
        data_hi = 1.2
        data_step = 0.1

        [[variants]]
        name = "lr010"
        lr = 0.01

        [[variants]]
        name = "hot"
        init_strategy = "random_hot"
        """)

        run(Cmd(`$(Base.julia_cmd()) --project=$(repo_root) $(suite_script) --config $(config_path)`; dir=dir))

        raw_dir = joinpath(dir, "results", "raw")
        paths = sort(filter(p -> endswith(p, ".json"), readdir(raw_dir; join=true)))
        @test length(paths) == 2

        rows = JSON3.read.(read.(paths, String))
        config_names = sort([String(row[:config_name]) for row in rows])
        @test config_names == ["depth2_sweep-hot", "depth2_sweep-lr010"]
    end
end

@testset "raw experiment results include snap info" begin
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    experiment_script = joinpath(repo_root, "scripts", "run_experiment.jl")
    mktempdir() do dir
        config_path = joinpath(dir, "depth2_single.toml")
        write(config_path, """
        name = "depth2_single"
        family = "depth2"
        depth = 2
        search_iters = 5
        hardening_iters = 2
        data_lo = 1.0
        data_hi = 1.2
        data_step = 0.1
        """)

        run(Cmd(`$(Base.julia_cmd()) --project=$(repo_root) $(experiment_script) --config $(config_path) --target eml_depth2 --seed 1`; dir=dir))

        paths = filter(p -> endswith(p, ".json"), readdir(joinpath(dir, "results", "raw"); join=true))
        @test length(paths) == 1

        row = JSON3.read(read(only(paths), String))
        @test haskey(row, :fit_success)
        @test haskey(row, :symbol_success)
        @test haskey(row, :stable_symbol_success)
        @test haskey(row, :family)
        @test String(row[:family]) == "depth2"
        @test haskey(row, :failure_reason)
        @test haskey(row, :nan_restarts)
        @test haskey(row, :nonfinite_steps)
        @test haskey(row, :snap_info)
        @test haskey(row[:snap_info], :n_uncertain)
    end
end

@testset "paper headless sweep runner writes depth2 artifacts" begin
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    sweep_script = joinpath(repo_root, "scripts", "run_paper_headless_sweep.jl")
    mktempdir() do dir
        run(Cmd(`$(Base.julia_cmd()) --project=$(repo_root) $(sweep_script) --jobs depth2 --search-iters 5 --hardening-iters 2 --seeds 1`; dir=dir))

        raw_dir = joinpath(dir, "results", "raw")
        paths = filter(p -> endswith(p, ".json"), readdir(raw_dir; join=true))
        @test length(paths) == 4

        row = JSON3.read(read(first(paths), String))
        @test String(row[:target]) == "eml_depth2"
        @test haskey(row, :fit_success)
        @test haskey(row, :stable_symbol_success)
    end
end
