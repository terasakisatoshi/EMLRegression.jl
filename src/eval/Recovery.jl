using StableRNGs: StableRNG

"""
    run_experiment(cfg; rng=StableRNG(1))

学習、snapping、blind recovery 判定までをまとめて実行します。
"""
function run_experiment(cfg::TrainConfig; rng=StableRNG(1))
    training = run_training(cfg; rng=rng)
    target = get_target(cfg.target)
    variables = target.arity == 1 ? (:x,) : (:x, :y)
    master_tree = build_master_tree(depth=target.depth, variables=variables)
    layer = EMLTreeLayer(master_tree; init_strategy=cfg.init_strategy)
    xs = sample_domain(target, cfg.batch_size; rng_seed=cfg.steps + 1)
    ys = evaluate_target(target, xs)
    recovered = search_recovered_tree(layer, training.params, master_tree, xs, ys)
    recovered_snap_status = snap_status(layer, training.params; margin_threshold=cfg.margin_threshold)
    ambiguous_nodes = ambiguous_node_count(layer, training.params; margin_threshold=cfg.margin_threshold)
    diagnostics = snap_diagnostics(layer, training.params; margin_threshold=cfg.margin_threshold)
    preds = evaluate_recovered(recovered, master_tree, xs)
    validation_loss = _mse(preds, ys)

    verdict = recovery_verdict(
        train_loss=isempty(training.metrics[:train_loss]) ? Inf : training.metrics[:train_loss][end],
        validation_loss=validation_loss,
        snap_status=recovered_snap_status,
        structure_match=structure_match(target.tree, recovered),
        ambiguous_nodes=ambiguous_nodes,
        numerical_match=numerical_match(preds, ys),
        training_failure_reason=training.failure_reason,
        snap_diagnostics=diagnostics,
    )

    return (
        training=training,
        recovery=verdict,
        recovered_tree=recovered,
        master_tree=master_tree,
        target_tree=target.tree,
    )
end
