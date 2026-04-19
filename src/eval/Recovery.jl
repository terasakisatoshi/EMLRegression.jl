using StableRNGs: StableRNG

"""
    run_experiment(cfg; rng=StableRNG(1))

学習後に snap 解析と hard projection を行い、paper-style の success 指標を返します。
"""
function run_experiment(cfg::TrainConfig; rng=StableRNG(1))
    training = run_training(cfg; rng=rng)
    target = get_target(cfg.target)
    x_train, y_train, t_train = make_grid_data(target.fn; lo=cfg.data_lo, hi=cfg.data_hi, step=cfg.data_step)

    tree = EMLTree(depth=cfg.depth, eml_clamp=cfg.eml_clamp, init_strategy=cfg.init_strategy, init_scale=cfg.init_scale)
    pre_projected = _projected_snap_metrics(
        tree,
        training.params,
        training.state,
        x_train,
        y_train,
        t_train;
        tau=cfg.tau_hard,
        snap_threshold=cfg.snap_threshold,
    )
    pre_verdict = recovery_verdict(
        snap_mse=pre_projected.snap_mse,
        snap_max_real=pre_projected.snap_max_real,
        snap_max_imag=pre_projected.snap_max_imag,
        n_uncertain=pre_projected.snap_info.n_uncertain,
        fit_success_thr=cfg.fit_success_thr,
        success_thr=cfg.success_thr,
        max_uncertain_success=cfg.max_uncertain_success,
        hardening_iter=training.summary[:hardening_iter],
    )

    refined = _refine_snap_projection(
        tree,
        training.params,
        training.state,
        x_train,
        y_train,
        t_train;
        tau=cfg.tau_hard,
        snap_threshold=cfg.snap_threshold,
    )

    verdict = recovery_verdict(
        snap_mse=refined.snap_mse,
        snap_max_real=refined.snap_max_real,
        snap_max_imag=refined.snap_max_imag,
        n_uncertain=refined.snap_info.n_uncertain,
        fit_success_thr=cfg.fit_success_thr,
        success_thr=cfg.success_thr,
        max_uncertain_success=cfg.max_uncertain_success,
        hardening_iter=training.summary[:hardening_iter],
    )

    return (
        training=training,
        pre_recovery=pre_verdict,
        pre_snap=pre_projected.snap_info,
        pre_snapped_params=pre_projected.snapped_params,
        recovery=verdict,
        snap=refined.snap_info,
        snapped_params=refined.snapped_params,
    )
end
