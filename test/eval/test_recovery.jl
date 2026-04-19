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
    @test refined.snap_info.n_uncertain <= 5
end

@testset "snap refinement uses enough steps for depth4-style ambiguity tails" begin
    cfg = TrainConfig(
        target=:eml_depth4,
        depth=4,
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
            -0.512485 6.64292 -3.41389;
            4.49966 -4.77304 -2.11683;
            5.2688 -5.31997 -2.9113;
            -0.432724 8.31214 -3.15435;
            5.28644 -5.56305 -0.42882;
            -0.503572 8.35143 -3.40652;
            -0.754014 9.27262 -2.45755;
            -1.95989 -8.14263 5.91611;
            4.17932 -5.69207 2.90461;
            -0.768972 5.7368 -8.10949;
            -3.82957 9.41661 -7.31033;
            -1.62481 -4.46154 9.74047;
            -0.325801 -2.2604 8.14683;
            5.79554 -7.27372 -2.37155;
            -2.95213 -6.47571 5.39465;
            -3.66313 8.39775 -3.82926;
        ],
        blend_logits=[
            -1.31074 4.22002;
            5.50001 4.52605;
            4.98679 2.45627;
            -4.23223 5.44572;
            1.69584 0.922219;
            -6.99418 -8.09909;
            -1.66309 6.74744;
            6.48962 5.99381;
            0.729457 5.10046;
            2.1059 2.12909;
            0.0188237 -0.983565;
            2.52353 6.02657;
            3.67198 3.73838;
            -1.38522 -1.25672;
            0.978311 -1.62561e-5;
        ],
    )

    @test analyze_snap(ps; snap_threshold=cfg.snap_threshold).n_uncertain >= 14

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

    @test refined.snap_mse < 1.0e-20
    @test refined.snap_info.n_uncertain <= 2
end

@testset "snap refinement widens step budget for high-ambiguity depth4 fixtures" begin
    ps = (
        leaf_logits=[
            -0.232933 1.41924 0.709155;
            2.78798 -0.137289 -0.717885;
            -0.955346 -2.39698 4.11686;
            5.65093 -1.32113 0.669707;
            1.35761 1.55326 4.46477;
            -0.612947 5.45111 -3.54724;
            6.429 -2.43043 -3.23842;
            -5.59415 9.01897 -0.935535;
            4.05003 0.237906 -1.1404;
            -0.671642 6.10979 -6.3139;
            -3.20158 9.13125 1.43073;
            -0.261542 -3.45158 9.72861;
            -3.16728 0.67682 10.4655;
            5.80527 -3.94578 -2.63892;
            5.15895 -5.9142 -3.30861;
            -2.84948 -1.3956 11.9517;
        ],
        blend_logits=[
            1.7947 2.50644;
            3.32213 -3.47787;
            2.991 0.325559;
            7.31052 -8.47645;
            4.04208 -1.94489;
            -3.86552 -3.56516;
            5.28534 6.201;
            5.86424 -6.69248;
            0.445503 -6.28349;
            0.452591 9.0285;
            2.08225 -2.50228;
            7.17528 -5.1814;
            4.95944 -2.75242;
            -1.03382 7.51979;
            1.25486 -4.09018;
        ],
    )

    @test analyze_snap(ps; snap_threshold=0.01).n_uncertain >= 17
    @test EMLRegression._default_snap_max_steps(ps; snap_threshold=0.01) == 30
end
