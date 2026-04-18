using Test
using Lux
using StableRNGs
using Zygote
using EMLRegression

@testset "search and hardening schedule transitions" begin
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        search_iters=20,
        hardening_iters=10,
        eval_every=5,
        data_lo=1.0,
        data_hi=1.2,
        data_step=0.1,
    )
    result = run_training(cfg; rng=StableRNG(1))
    @test haskey(result.metrics, :soft_rmse)
    @test haskey(result.metrics, :hard_rmse)
    @test haskey(result.metrics, :tau)
    @test result.summary[:hardening_iter] !== nothing
    @test length(result.metrics[:soft_rmse]) > 0
end

@testset "training loop records finite losses on paper-aligned target" begin
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        search_iters=10,
        hardening_iters=5,
        eval_every=5,
        data_lo=1.0,
        data_hi=1.2,
        data_step=0.1,
    )
    result = run_training(cfg; rng=StableRNG(1))
    @test all(isfinite, result.metrics[:soft_rmse])
    @test all(isfinite, result.metrics[:tau])
end

@testset "training loop uses the PyTorch uncertainty exponent" begin
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        search_iters=1,
        hardening_iters=0,
        lr=0.0,
        eval_every=1,
        data_lo=1.0,
        data_hi=1.2,
        data_step=0.1,
    )

    result = run_training(cfg; rng=StableRNG(1))

    target = get_target(cfg.target)
    x_train, y_train, t_train = make_grid_data(target.fn; lo=cfg.data_lo, hi=cfg.data_hi, step=cfg.data_step)
    tree = EMLTree(depth=cfg.depth, init_strategy=cfg.init_strategy, init_scale=cfg.init_scale, eml_clamp=cfg.eml_clamp)
    rng = StableRNG(1)
    ps, _ = Lux.setup(rng, tree)
    EMLRegression._apply_initialization!(rng, ps, cfg)

    _, regs_power_1 = EMLRegression.forward_with_regularizers(
        tree,
        (x_train, y_train),
        ps;
        tau_leaf=cfg.tau_search,
        tau_gate=cfg.tau_search,
        inter_threshold=cfg.inter_threshold,
        uncertainty_power=1.0,
    )
    _, regs_power_2 = EMLRegression.forward_with_regularizers(
        tree,
        (x_train, y_train),
        ps;
        tau_leaf=cfg.tau_search,
        tau_gate=cfg.tau_search,
        inter_threshold=cfg.inter_threshold,
        uncertainty_power=2.0,
    )

    @test result.summary[:uncertainty_power] == 1.0
    @test isapprox(result.metrics[:entropy][1], regs_power_1.entropy; rtol=1e-12, atol=1e-12)
    @test isapprox(result.metrics[:binarity][1], regs_power_1.binarity; rtol=1e-12, atol=1e-12)
    @test !isapprox(result.metrics[:entropy][1], regs_power_2.entropy; rtol=1e-12, atol=1e-12) ||
          !isapprox(result.metrics[:binarity][1], regs_power_2.binarity; rtol=1e-12, atol=1e-12)
end

@testset "training accepts manual expression initialization" begin
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        init_strategy=:manual,
        init_expr="EML[1, EML[x, y]]",
        init_noise=0.0,
        search_iters=4,
        hardening_iters=2,
        eval_every=2,
        tail_eval_every=1,
        data_lo=1.0,
        data_hi=1.2,
        data_step=0.1,
    )
    result = run_training(cfg; rng=StableRNG(1))
    @test !isempty(result.metrics[:soft_rmse])
    @test result.failure_reason == EMLRegression.no_failure
end

@testset "run_training restores best hard checkpoint instead of last params" begin
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        search_iters=4,
        hardening_iters=4,
        eval_every=1,
        tail_eval_every=1,
        data_lo=1.0,
        data_hi=1.2,
        data_step=0.1,
    )

    target = get_target(cfg.target)
    x_train, y_train, t_train = make_grid_data(target.fn; lo=cfg.data_lo, hi=cfg.data_hi, step=cfg.data_step)
    tree = EMLTree(depth=cfg.depth, init_strategy=cfg.init_strategy, init_scale=cfg.init_scale, eml_clamp=cfg.eml_clamp)
    _, st = Lux.setup(StableRNG(1), tree)

    result = run_training(cfg; rng=StableRNG(1))
    hard_mse, _, _ = EMLRegression.evaluate(tree, result.params, st, x_train, y_train, t_train; tau=cfg.tau_hard)

    @test haskey(result.summary, :best_hard_mse)
    @test haskey(result.summary, :final_hard_mse)
    @test hard_mse == get(result.summary, :final_hard_mse, Inf)
    @test get(result.summary, :final_hard_mse, Inf) == get(result.summary, :best_hard_mse, -Inf)
end

@testset "run_training can enter hardening before search budget on plateau" begin
    # These settings keep the search loss nearly flat so the plateau detector should trip early.
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        search_iters=40,
        hardening_iters=4,
        eval_every=1,
        patience=2,
        patience_threshold=Inf,
        plateau_rtol=1.0,
        data_lo=1.0,
        data_hi=1.1,
        data_step=0.1,
    )

    result = run_training(cfg; rng=StableRNG(1))

    @test result.summary[:hardening_iter] !== nothing
    @test result.summary[:hardening_iter] < cfg.search_iters
    @test get(result.summary, :hardening_reason, nothing) == :plateau
end

@testset "run_training can enter hardening from repeated exact hard evals" begin
    # This manual init starts at an exact hard solution so the repeated hard-eval trigger should fire quickly.
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        init_strategy=:manual,
        init_expr="EML[1, EML[x, y]]",
        init_noise=0.0,
        search_iters=20,
        hardening_iters=4,
        eval_every=1,
        hard_trigger_mse=0.03,
        hard_trigger_count=2,
        data_lo=1.0,
        data_hi=1.2,
        data_step=0.1,
    )

    result = run_training(cfg; rng=StableRNG(1))

    @test result.summary[:hardening_iter] !== nothing
    @test result.summary[:hardening_iter] <= 3
    @test get(result.summary, :hardening_reason, nothing) == :hard_trigger
end

@testset "run_training returns hardened params even when no hard eval runs" begin
    cfg_with_hardening = TrainConfig(
        target=:eml_depth2,
        depth=2,
        search_iters=1,
        hardening_iters=1,
        eval_every=100,
        tail_eval_every=100,
        data_lo=1.0,
        data_hi=1.2,
        data_step=0.1,
    )
    cfg_search_only = TrainConfig(
        target=:eml_depth2,
        depth=2,
        search_iters=1,
        hardening_iters=0,
        eval_every=100,
        tail_eval_every=100,
        data_lo=1.0,
        data_hi=1.2,
        data_step=0.1,
    )

    hardening_result = run_training(cfg_with_hardening; rng=StableRNG(1))
    search_only_result = run_training(cfg_search_only; rng=StableRNG(1))

    @test hardening_result.summary[:hardening_iter] == 2
    @test hardening_result.summary[:best_hard_mse] == Inf
    @test isempty(hardening_result.metrics[:hard_rmse])
    @test hardening_result.params != search_only_result.params
end

@testset "first finite best resets plateau counter before hardening" begin
    best_soft_loss, plateau_counter, is_new_best = EMLRegression._update_best_soft_loss(Inf, 0.25, 7, 1.0e-3)

    @test best_soft_loss == 0.25
    @test plateau_counter == 0
    @test is_new_best
end

@testset "depth4 random_hot training returns without tuple-gradient exception" begin
    cfg = TrainConfig(
        target=:eml_depth4,
        depth=4,
        init_strategy=:random_hot,
        search_iters=50,
        hardening_iters=20,
        eval_every=2,
        data_lo=1.0,
        data_hi=3.0,
        data_step=0.1,
    )
    result = run_training(cfg; rng=StableRNG(137))
    @test result isa TrainingResult
    @test result.failure_reason in (EMLRegression.no_failure, EMLRegression.nonfinite_detected)
end

@testset "nonfinite search steps trigger restart accounting instead of immediate abort" begin
    cfg = TrainConfig(
        target=:eml_depth4,
        depth=4,
        init_strategy=:random_hot,
        search_iters=6,
        hardening_iters=0,
        eval_every=2,
        data_lo=1.0,
        data_hi=3.0,
        data_step=0.1,
        nan_restart_patience=1,
        max_nan_restarts=2,
    )
    result = run_training(cfg; rng=StableRNG(137))
    @test haskey(result.summary, :nan_restarts)
    @test haskey(result.summary, :nonfinite_steps)
    @test result.summary[:nan_restarts] > 0
    @test result.summary[:nonfinite_steps] > 0
end

@testset "gradient clipping rescales oversized nested gradients" begin
    grads = (
        leaf_logits=fill(3.0, 2, 3),
        blend_logits=fill(4.0, 1, 2),
    )
    clipped = EMLRegression._clip_gradients(grads, 1.0)
    sq_norm = sum(abs2, clipped.leaf_logits) + sum(abs2, clipped.blend_logits)
    @test sqrt(sq_norm) <= 1.0 + 1.0e-12
    @test clipped.leaf_logits[1, 1] < grads.leaf_logits[1, 1]
    @test clipped.blend_logits[1, 1] < grads.blend_logits[1, 1]
end

@testset "paper-budget saturated gates remain differentiable" begin
    cfg = TrainConfig(
        target=:eml_depth2,
        depth=2,
        data_lo=1.0,
        data_hi=3.0,
        data_step=0.1,
        lam_inter=1.0e-4,
        inter_threshold=50.0,
    )
    target = get_target(cfg.target)
    x_train, y_train, t_train = make_grid_data(target.fn; lo=cfg.data_lo, hi=cfg.data_hi, step=cfg.data_step)
    tree = EMLTree(depth=2, init_strategy=:biased)
    ps = (
        leaf_logits=[
            5.21454 0.289582 -6.85149;
            -4.70773 -4.0905 9.98371;
            -8.28137 -8.53061 13.5623;
            0.183201 5.60814 -7.89822
        ],
        blend_logits=[
            3.57914 -8.22044;
            -12.9837 -3.10144;
            6.15721 -10.6534
        ],
    )
    tau = 0.08250917849561924
    loss(ps_current) = begin
        pred, regs = EMLRegression.forward_with_regularizers(
            tree,
            (x_train, y_train),
            ps_current;
            tau_leaf=tau,
            tau_gate=tau,
            inter_threshold=cfg.inter_threshold,
        )
        data_loss = EMLRegression._safe_mean_abs2(pred .- t_train)
        data_loss + 0.014 * regs.entropy + 0.014 * regs.binarity + cfg.lam_inter * regs.inter_penalty
    end

    @test isfinite(loss(ps))
    grads = only(Zygote.gradient(loss, ps))
    @test size(grads.leaf_logits) == size(ps.leaf_logits)
    @test size(grads.blend_logits) == size(ps.blend_logits)
end
