using Test
using EMLRegression
using JSON3

include(joinpath(@__DIR__, "..", "..", "scripts", "summarize_results.jl"))

@testset "paper-style success thresholds" begin
    verdict = recovery_verdict(
        snap_mse=1.0e-8,
        snap_max_real=1.0e-5,
        snap_max_imag=0.0,
        n_uncertain=0,
        fit_success_thr=1.0e-6,
        success_thr=1.0e-20,
        max_uncertain_success=0,
        hardening_iter=21,
    )

    @test verdict.fit_success
    @test !verdict.symbol_success
    @test !verdict.stable_symbol_success
    @test verdict.success
    @test verdict.hardening_iter == 21
end

@testset "paper sweep summary includes all section 4.3 families" begin
    mktempdir() do dir
        raw_dir = joinpath(dir, "results", "raw")
        mkpath(raw_dir)

        fixtures = [
            ("pnas_d2_random-biased", "biased", 2, "eml_depth2"),
            ("pnas_d3_random-random_hot", "random_hot", 3, "eml_depth3"),
            ("pnas_d4_random-uniform", "uniform", 4, "eml_depth4"),
            ("pnas_d5_random64_random_hot", "random_hot", 5, "eml_depth5"),
            ("pnas_d6_random64_uniform", "uniform", 6, "eml_depth6"),
            ("pnas_d5_manual_noise12", "manual", 5, "eml_depth5"),
            ("pnas_d6_manual_noise12", "manual", 6, "eml_depth6"),
        ]

        for (i, (config_name, init_strategy, depth, target)) in enumerate(fixtures)
            row = Dict(
                :config_name => config_name,
                :depth => depth,
                :target => target,
                :tier => "must_pass",
                :seed => i,
                :init_strategy => init_strategy,
                :fit_success => true,
                :symbol_success => true,
                :stable_symbol_success => i % 2 == 0,
                :success => true,
                :n_uncertain => 0,
                :nonfinite_steps => 0,
                :nan_restarts => 0,
                :snap_mse => 1.0e-8,
                :snap_rmse => 1.0e-4,
                :snap_max_real => 1.0e-5,
                :snap_max_imag => 0.0,
                :hardening_iter => 10,
            )
            open(joinpath(raw_dir, "run$(i).json"), "w") do io
                JSON3.write(io, row)
            end
        end

        rows = summarize_section_4_3(raw_dir)
        families = Set(String.(rows.job_name))
        @test "depth2" in families
        @test "depth3" in families
        @test "depth4" in families
        @test "depth5_random" in families
        @test "depth6_random" in families
        @test "depth5_manual_noise12" in families
        @test "depth6_manual_noise12" in families

        grouped = Dict((String(row.family), String(row.init_strategy)) => row for row in eachrow(rows))
        @test grouped[("depth2", "biased")].fit_count == 1
        @test grouped[("depth3", "random_hot")].symbol_count == 1
        @test grouped[("depth5_manual_noise12", "manual")].runs == 1
        @test grouped[("depth6_manual_noise12", "manual")].stable_symbol_count == 0
    end
end

@testset "stable_symbol_success requires zero uncertainty" begin
    verdict = recovery_verdict(
        snap_mse=1.0e-24,
        snap_max_real=1.0e-12,
        snap_max_imag=0.0,
        n_uncertain=2,
        fit_success_thr=1.0e-6,
        success_thr=1.0e-20,
        max_uncertain_success=0,
    )

    @test verdict.fit_success
    @test verdict.symbol_success
    @test !verdict.stable_symbol_success
    @test verdict.n_uncertain == 2
end
