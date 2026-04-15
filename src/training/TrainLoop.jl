using ChainRulesCore
using StableRNGs: StableRNG
using Lux
using Optimisers
using Zygote

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

    total_steps = cfg.steps + cfg.hardening_steps
    hardening_start = _effective_hardening_start(cfg)

    for step in 1:total_steps
        xs = sample_domain(target, cfg.batch_size; rng=rng)
        ys = evaluate_target(target, xs)
        hardening_active = step >= hardening_start
        temperature = _hardening_temperature(cfg, step, hardening_start, hardening_active)
        grads = _autodiff_gradient(ps) do ps_current
            preds, _ = Lux.apply(layer, xs, ps_current, st)
            mse_loss = _mse(preds, ys)
            hardening_term = hardening_active ? cfg.hardening_weight * _hardening_penalty(ps_current; temperature=temperature) : 0.0
            complexity_term = cfg.complexity_weight * _complexity_penalty(layer, ps_current; temperature=temperature)
            mse_loss + hardening_term + complexity_term
        end
        opt_state, ps = Optimisers.update(opt_state, ps, grads)
        preds, st = Lux.apply(layer, xs, ps, st)
        loss = _mse(preds, ys)
        push!(train_loss, loss)
        push!(logit_margin, _mean_logit_margin(ps))
        if hardening_active
            push!(hardening_loss, cfg.hardening_weight * _hardening_penalty(ps; temperature=temperature))
        end
    end

    metrics = Dict(
        :train_loss => train_loss,
        :hardening_loss => hardening_loss,
        :logit_margin => logit_margin,
    )
    return TrainingResult(cfg, metrics, no_failure, ps, st)
end

function _effective_hardening_start(cfg::TrainConfig)
    if cfg.hardening_steps <= 0
        return typemax(Int)
    elseif cfg.hardening_start == typemax(Int)
        return cfg.steps + 1
    else
        return cfg.hardening_start
    end
end

"""
    _mse(preds, ys)

複素値対応の平均二乗誤差です。
"""
function _mse(preds, ys)
    diffs = preds .- ys
    return sum(abs2, diffs) / max(length(diffs), 1)
end

function _autodiff_gradient(loss_fn, ps)
    grads = only(Zygote.gradient(loss_fn, ps))
    isnothing(grads) && error("autodiff returned no gradients for training parameters")
    return grads
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

function _hardening_penalty(ps; temperature::Float64=1.0)
    penalties = (
        1.0 - maximum(_softmax_probabilities(logits; temperature=temperature))
        for node in ps.nodes
        for logits in (node.left_logits, node.right_logits)
    )
    return sum(penalties)
end

function _complexity_penalty(layer::EMLTreeLayer, ps; temperature::Float64=1.0)
    penalties = (
        begin
            probabilities = _softmax_probabilities(logits; temperature=temperature)
            sum(probability * _candidate_cost(symbol) for (probability, symbol) in zip(probabilities, candidates))
        end
        for node in layer.tree.nodes
        for (logits, candidates) in (
            (ps.nodes[node.id].left_logits, node.left_candidates),
            (ps.nodes[node.id].right_logits, node.right_candidates),
        )
    )
    return sum(penalties)
end

function _candidate_cost(symbol::Symbol)
    if symbol === :const1
        return 0.0
    elseif symbol === :x || symbol === :y
        return 1.0
    elseif startswith(String(symbol), "node_")
        return 2.0
    else
        return 1.0
    end
end

ChainRulesCore.@non_differentiable _candidate_cost(::Any...)

function _softmax_probabilities(logits; temperature::Float64=1.0)
    scaled = logits ./ max(temperature, 1.0e-6)
    shifted = scaled .- maximum(scaled)
    weights = exp.(shifted)
    return weights ./ sum(weights)
end

function _hardening_temperature(cfg::TrainConfig, step::Int, hardening_start::Int, hardening_active::Bool)
    if !hardening_active
        return cfg.temperature
    end
    hardening_step = max(step - hardening_start, 0)
    return max(cfg.temperature * (0.5 ^ hardening_step), 0.1)
end
