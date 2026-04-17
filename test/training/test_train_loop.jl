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
