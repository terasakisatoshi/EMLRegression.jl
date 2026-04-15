using StableRNGs: StableRNG
using Lux
using Optimisers

"""
    run_training(cfg; rng=StableRNG(1))

現在の最小 EML モデルで学習ループを実行し、`TrainingResult` を返します。
"""
function run_training(cfg::TrainConfig; rng=StableRNG(1))
    target = get_target(cfg.target)
    variables = target.arity == 1 ? (:x,) : (:x, :y)
    layer = EMLTreeLayer(build_master_tree(depth=cfg.depth, variables=variables))
    ps, st = Lux.setup(rng, layer)
    opt = Optimisers.Adam(cfg.learning_rate)
    opt_state = Optimisers.setup(opt, ps)

    train_loss = Float64[]
    hardening_loss = Float64[]
    logit_margin = Float64[]

    for step in 1:cfg.steps
        xs = sample_domain(target, cfg.batch_size; rng_seed=step)
        ys = evaluate_target(target, xs)
        grads = _finite_difference_gradient(ps) do ps_current
            preds, _ = Lux.apply(layer, xs, ps_current, st)
            _mse(preds, ys)
        end
        opt_state, ps = Optimisers.update(opt_state, ps, grads)
        preds, st = Lux.apply(layer, xs, ps, st)
        loss = _mse(preds, ys)
        push!(train_loss, loss)
        push!(logit_margin, _mean_logit_margin(ps))
    end

    for step in 1:cfg.hardening_steps
        push!(hardening_loss, cfg.hardening_weight)
    end

    metrics = Dict(
        :train_loss => train_loss,
        :hardening_loss => hardening_loss,
        :logit_margin => logit_margin,
    )
    return TrainingResult(cfg, metrics, no_failure, ps, st)
end

"""
    _mse(preds, ys)

複素値対応の平均二乗誤差です。
"""
function _mse(preds, ys)
    diffs = preds .- ys
    return sum(abs2, diffs) / max(length(diffs), 1)
end

function _finite_difference_gradient(loss_fn, ps; eps=1e-4)
    node_grads = Tuple(
        (
            left_logits=_finite_difference_array(loss_fn, ps, node_id, :left_logits; eps=eps),
            right_logits=_finite_difference_array(loss_fn, ps, node_id, :right_logits; eps=eps),
        ) for node_id in eachindex(ps.nodes)
    )
    return (; nodes=node_grads)
end

function _finite_difference_array(loss_fn, ps, node_id::Int, field::Symbol; eps=1e-4)
    values = getproperty(ps.nodes[node_id], field)
    grads = similar(values)
    for idx in eachindex(values)
        ps_plus = _perturb_params(ps, node_id, field, idx, eps)
        ps_minus = _perturb_params(ps, node_id, field, idx, -eps)
        grads[idx] = (loss_fn(ps_plus) - loss_fn(ps_minus)) / (2eps)
    end
    return grads
end

function _perturb_params(ps, node_id::Int, field::Symbol, idx::Int, delta::Float64)
    ps_copy = deepcopy(ps)
    getproperty(ps_copy.nodes[node_id], field)[idx] += delta
    return ps_copy
end

function _mean_logit_margin(ps)
    margins = Float64[]
    for node in ps.nodes
        push!(margins, _logit_margin(node.left_logits))
        push!(margins, _logit_margin(node.right_logits))
    end
    return isempty(margins) ? 0.0 : sum(margins) / length(margins)
end

function _logit_margin(logits)
    sorted = sort(collect(logits); rev=true)
    return length(sorted) < 2 ? sorted[1] : sorted[1] - sorted[2]
end
