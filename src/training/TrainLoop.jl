using StableRNGs: StableRNG
using Lux

function run_training(cfg::TrainConfig; rng=StableRNG(1))
    target = get_target(cfg.target)
    layer = EMLTreeLayer(build_master_tree(depth=cfg.depth, variables=(:x,)))
    ps, st = Lux.setup(rng, layer)

    train_loss = Float64[]
    hardening_loss = Float64[]

    for step in 1:cfg.steps
        xs = sample_domain(target, cfg.batch_size; rng_seed=step)
        ys = evaluate_target(target, xs)
        preds, st = Lux.apply(layer, xs, ps, st)
        loss = _mse(preds, ys)
        push!(train_loss, loss)
    end

    for step in 1:cfg.hardening_steps
        push!(hardening_loss, cfg.hardening_weight)
    end

    metrics = Dict(:train_loss => train_loss, :hardening_loss => hardening_loss)
    return TrainingResult(cfg, metrics, no_failure, ps, st)
end

function _mse(preds, ys)
    diffs = preds .- ys
    return sum(abs2, diffs) / max(length(diffs), 1)
end
