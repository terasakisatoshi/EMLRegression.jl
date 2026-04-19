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

@testset "paper sweep summary tolerates missing hardening iteration" begin
    mktempdir() do dir
        raw_dir = joinpath(dir, "results", "raw")
        mkpath(raw_dir)

        row = Dict(
            :config_name => "pnas_d4_random-random_hot",
            :family => "depth4",
            :depth => 4,
            :target => "eml_depth4",
            :tier => "challenge",
            :seed => 137,
            :init_strategy => "random_hot",
            :fit_success => false,
            :symbol_success => false,
            :stable_symbol_success => false,
            :success => false,
            :n_uncertain => 46,
            :nonfinite_steps => 5000,
            :nonfinite_grad_steps => 17,
            :nan_restarts => 100,
            :snap_mse => 176.5,
            :snap_rmse => 13.28,
            :snap_max_real => 14.67,
            :snap_max_imag => 0.0,
            :hardening_iter => nothing,
        )
        open(joinpath(raw_dir, "run.json"), "w") do io
            JSON3.write(io, row)
        end

        loaded = load_raw_results(raw_dir)
        @test ismissing(loaded.hardening_iter[1])

        summary, section, raw_count = summarize_all_results(raw_dir)
        @test raw_count == 1
        @test nrow(summary) == 1
        @test isnan(summary.mean_hardening_iter[1])
        @test summary.mean_nonfinite_grad_steps[1] == 17.0
        @test nrow(section) == 1
        @test section.nonfinite_steps[1] == 5000
        @test section.nonfinite_grad_steps[1] == 17
        @test section.nan_restarts[1] == 100
    end
end

@testset "section 4.3 summary classifies by config name, not stored family metadata" begin
    mktempdir() do dir
        raw_dir = joinpath(dir, "results", "raw")
        mkpath(raw_dir)

        rows = [
            Dict(
                :config_name => "pnas_d2_random-biased",
                :family => "stale_depth2_label",
                :depth => 2,
                :target => "eml_depth2",
                :tier => "must_pass",
                :seed => 137,
                :init_strategy => "biased",
                :fit_success => true,
                :symbol_success => true,
                :stable_symbol_success => false,
                :success => true,
                :n_uncertain => 0,
                :nonfinite_steps => 0,
                :nonfinite_grad_steps => 0,
                :nan_restarts => 0,
                :snap_mse => 1.0e-8,
                :snap_rmse => 1.0e-4,
                :snap_max_real => 1.0e-5,
                :snap_max_imag => 0.0,
                :hardening_iter => 6001,
            ),
            Dict(
                :config_name => "depth2_single",
                :family => "depth2",
                :depth => 2,
                :target => "eml_depth2",
                :tier => "must_pass",
                :seed => 1,
                :init_strategy => "biased",
                :fit_success => false,
                :symbol_success => false,
                :stable_symbol_success => false,
                :success => false,
                :n_uncertain => 9,
                :nonfinite_steps => 0,
                :nonfinite_grad_steps => 0,
                :nan_restarts => 0,
                :snap_mse => 1.14,
                :snap_rmse => 1.07,
                :snap_max_real => 1.2,
                :snap_max_imag => 0.0,
                :hardening_iter => 6,
            ),
        ]

        for (idx, row) in enumerate(rows)
            open(joinpath(raw_dir, "run$(idx).json"), "w") do io
                JSON3.write(io, row)
            end
        end

        section = summarize_section_4_3(raw_dir)
        @test nrow(section) == 1
        @test section.family[1] == "depth2"
        @test section.init_strategy[1] == "biased"
        @test section.fit_count[1] == 1
        @test section.runs[1] == 1
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

@testset "snap refinement can flip an ambiguous gate to recover the right tree" begin
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        snap_threshold=0.01,
        data_lo=1.0,
        data_hi=1.6,
        data_step=0.1,
    )
    target = get_target(cfg.target)
    x_train, y_train, t_train = make_grid_data(target.fn; lo=cfg.data_lo, hi=cfg.data_hi, step=cfg.data_step)
    tree = EMLTree(depth=cfg.depth, eml_clamp=cfg.eml_clamp)
    ps = (
        leaf_logits=[
            24.0 -24.0 -24.0;
            24.0 -24.0 -24.0;
            -24.0 -24.0 24.0;
            -24.0 24.0 -24.0;
        ],
        blend_logits=[
            24.0 24.0;
            -24.0 -24.0;
            24.0 0.1;
        ],
    )

    snap_before = analyze_snap(ps; snap_threshold=cfg.snap_threshold)
    snapped_before = hard_project(ps)
    mse_before, _, _ = EMLRegression.evaluate(tree, snapped_before, NamedTuple(), x_train, y_train, t_train; tau=cfg.tau_hard)

    refined = EMLRegression._refine_snap_projection(
        tree,
        ps,
        NamedTuple(),
        x_train,
        y_train,
        t_train;
        tau=cfg.tau_hard,
        snap_threshold=cfg.snap_threshold,
    )
    mse_after, _, _ = EMLRegression.evaluate(tree, refined.snapped_params, NamedTuple(), x_train, y_train, t_train; tau=cfg.tau_hard)

    @test snap_before.n_uncertain == 1
    @test refined.snap_info.n_uncertain == 0
    @test mse_after < mse_before
    @test mse_after < 1.0e-20
end

@testset "snap refinement leaves already certain projections unchanged" begin
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        snap_threshold=0.01,
        data_lo=1.0,
        data_hi=1.6,
        data_step=0.1,
    )
    target = get_target(cfg.target)
    x_train, y_train, t_train = make_grid_data(target.fn; lo=cfg.data_lo, hi=cfg.data_hi, step=cfg.data_step)
    tree = EMLTree(depth=cfg.depth, eml_clamp=cfg.eml_clamp)
    ps = (
        leaf_logits=[
            24.0 -24.0 -24.0;
            24.0 -24.0 -24.0;
            -24.0 -24.0 24.0;
            -24.0 24.0 -24.0;
        ],
        blend_logits=[
            24.0 24.0;
            -24.0 -24.0;
            24.0 -24.0;
        ],
    )

    mse_before, _, _ = EMLRegression.evaluate(tree, hard_project(ps), NamedTuple(), x_train, y_train, t_train; tau=cfg.tau_hard)
    refined = EMLRegression._refine_snap_projection(
        tree,
        ps,
        NamedTuple(),
        x_train,
        y_train,
        t_train;
        tau=cfg.tau_hard,
        snap_threshold=cfg.snap_threshold,
    )
    mse_after, _, _ = EMLRegression.evaluate(tree, refined.snapped_params, NamedTuple(), x_train, y_train, t_train; tau=cfg.tau_hard)

    @test refined.snap_info.n_uncertain == 0
    @test !refined.improved
    @test mse_after == mse_before
end

@testset "snap refinement can escape greedy traps with coupled discrete moves" begin
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        snap_threshold=0.01,
        data_lo=1.0,
        data_hi=1.6,
        data_step=0.1,
    )
    target = get_target(cfg.target)
    x_train, y_train, t_train = make_grid_data(target.fn; lo=cfg.data_lo, hi=cfg.data_hi, step=cfg.data_step)
    tree = EMLTree(depth=cfg.depth, eml_clamp=cfg.eml_clamp)
    ps = (
        leaf_logits=[
            24.0 -24.0 -24.0;
            24.0 -24.0 -24.0;
            1.0 -2.0 -0.5;
            -2.0 -1.0 2.0;
        ],
        blend_logits=[
            24.0 24.0;
            1.0 -24.0;
            24.0 -24.0;
        ],
    )

    snapped_before = hard_project(ps)
    mse_before, _, _ = EMLRegression.evaluate(tree, snapped_before, NamedTuple(), x_train, y_train, t_train; tau=cfg.tau_hard)

    refined = EMLRegression._refine_snap_projection(
        tree,
        ps,
        NamedTuple(),
        x_train,
        y_train,
        t_train;
        tau=cfg.tau_hard,
        snap_threshold=cfg.snap_threshold,
    )
    mse_after, _, _ = EMLRegression.evaluate(tree, refined.snapped_params, NamedTuple(), x_train, y_train, t_train; tau=cfg.tau_hard)

    @test mse_after < mse_before
    @test mse_after < 1.0e-20
    @test refined.improved
    @test refined.snap_info.n_uncertain == 0
end

@testset "snap refinement can revisit low-margin stable gates" begin
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        snap_threshold=0.01,
        data_lo=1.0,
        data_hi=1.6,
        data_step=0.1,
    )
    target = get_target(cfg.target)
    x_train, y_train, t_train = make_grid_data(target.fn; lo=cfg.data_lo, hi=cfg.data_hi, step=cfg.data_step)
    tree = EMLTree(depth=cfg.depth, eml_clamp=cfg.eml_clamp)
    ps = (
        leaf_logits=[
            -0.2 -5.5 -6.0;
            24.0 -24.0 -24.0;
            -6.0 -0.1 5.0;
            -5.0 6.0 -0.1;
        ],
        blend_logits=[
            24.0 24.0;
            5.0 -24.0;
            4.8 -24.0;
        ],
    )

    @test analyze_snap(ps; snap_threshold=cfg.snap_threshold).n_uncertain == 0

    snapped_before = hard_project(ps)
    mse_before, _, _ = EMLRegression.evaluate(tree, snapped_before, NamedTuple(), x_train, y_train, t_train; tau=cfg.tau_hard)

    refined = EMLRegression._refine_snap_projection(
        tree,
        ps,
        NamedTuple(),
        x_train,
        y_train,
        t_train;
        tau=cfg.tau_hard,
        snap_threshold=cfg.snap_threshold,
    )
    mse_after, _, _ = EMLRegression.evaluate(tree, refined.snapped_params, NamedTuple(), x_train, y_train, t_train; tau=cfg.tau_hard)

    @test mse_after < mse_before
    @test mse_after < 1.0e-20
    @test refined.improved
    @test refined.snap_info.n_uncertain == 0
end

@testset "snap refinement can flip confident gates on depth3-style fixtures" begin
    cfg = TrainConfig(
        target=:eml_depth3,
        depth=3,
        snap_threshold=0.01,
        data_lo=1.0,
        data_hi=3.0,
        data_step=0.1,
    )
    target = get_target(cfg.target)
    x_train, y_train, t_train = make_grid_data(target.fn; lo=cfg.data_lo, hi=cfg.data_hi, step=cfg.data_step)
    tree = EMLTree(depth=cfg.depth, eml_clamp=cfg.eml_clamp)
    ps = (
        leaf_logits=[
            1.3574 5.9571 -7.2174;
            1.0212 -12.2938 1.0316;
            -6.1963 -6.8260 9.1048;
            -2.8762 15.5771 -5.3279;
            1.2469 6.8389 -3.6049;
            0.0966 4.7683 2.0959;
            0.0427 8.2359 -5.4291;
            -5.5683 -9.8341 6.1036;
        ],
        blend_logits=[
            0.0399 -5.3467;
            -8.0269 -11.8398;
            0.0889 0.0178;
            4.6150 -7.8325;
            -2.3970 -6.0851;
            -4.3508 4.9737;
            0.0227 -2.7662;
        ],
    )

    mse_before, _, _ = EMLRegression.evaluate(tree, hard_project(ps), NamedTuple(), x_train, y_train, t_train; tau=cfg.tau_hard)
    refined = EMLRegression._refine_snap_projection(
        tree,
        ps,
        NamedTuple(),
        x_train,
        y_train,
        t_train;
        tau=cfg.tau_hard,
        snap_threshold=cfg.snap_threshold,
    )
    mse_after, _, _ = EMLRegression.evaluate(tree, refined.snapped_params, NamedTuple(), x_train, y_train, t_train; tau=cfg.tau_hard)

    @test mse_after < mse_before
    @test mse_after < 1.0e-20
    @test refined.improved
end
