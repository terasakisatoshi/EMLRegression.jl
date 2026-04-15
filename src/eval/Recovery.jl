using StableRNGs: StableRNG

"""
    run_experiment(cfg; rng=StableRNG(1))

学習、snapping、blind recovery 判定までをまとめて実行します。
"""
function run_experiment(cfg::TrainConfig; rng=StableRNG(1))
    training = run_training(cfg; rng=rng)
    target = get_target(cfg.target)
    variables = target.arity == 1 ? (:x,) : (:x, :y)
    master_tree = build_master_tree(depth=cfg.depth, variables=variables)
    layer = EMLTreeLayer(master_tree)
    recovered = snap_model(layer, training.params)
    recovered_snap_status = snap_status(layer, training.params; margin_threshold=cfg.margin_threshold)

    xs = sample_domain(target, cfg.batch_size; rng_seed=cfg.steps + 1)
    ys = evaluate_target(target, xs)
    preds = evaluate_recovered(recovered, master_tree, xs)

    verdict = recovery_verdict(
        train_loss=isempty(training.metrics[:train_loss]) ? Inf : training.metrics[:train_loss][end],
        validation_loss=_mse(preds, ys),
        snap_status=recovered_snap_status,
        numerical_match=numerical_match(preds, ys),
    )

    return (
        training=training,
        recovery=verdict,
        recovered_tree=recovered,
        master_tree=master_tree,
    )
end
