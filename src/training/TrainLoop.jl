using StableRNGs: StableRNG
using Lux
using Optimisers
using Zygote

function run_training(cfg::TrainConfig; rng=StableRNG(1))
    target = get_target(cfg.target)
    x_train, y_train, t_train = make_grid_data(target.fn; lo=cfg.data_lo, hi=cfg.data_hi, step=cfg.data_step)
    tree = EMLTree(depth=cfg.depth, init_strategy=cfg.init_strategy, init_scale=cfg.init_scale, eml_clamp=cfg.eml_clamp)
    ps, st = Lux.setup(rng, tree)
    _apply_initialization!(rng, ps, cfg)

    opt = Optimisers.Adam(cfg.lr)
    opt_state = Optimisers.setup(opt, ps)

    metrics = Dict(
        :soft_rmse => Float64[],
        :hard_rmse => Float64[],
        :tau => Float64[],
        :entropy => Float64[],
        :binarity => Float64[],
    )
    summary = Dict{Symbol,Any}(
        :hardening_iter => nothing,
        :nan_restarts => 0,
        :nonfinite_steps => 0,
    )

    best_soft_loss = Inf
    best_soft_state = deepcopy(ps)
    phase = :search
    hardening_iter = nothing
    hard_step = 0
    hard_success_streak = 0
    nan_streak = 0
    nan_restarts = 0
    failure_reason = no_failure

    total_iters = cfg.search_iters + cfg.hardening_iters
    for it in 1:total_iters
        if cfg.max_nan_restarts > 0 && nan_restarts >= cfg.max_nan_restarts
            failure_reason = nonfinite_detected
            break
        end

        if phase === :search && it > cfg.search_iters
            phase = :hardening
            hardening_iter = it
            summary[:hardening_iter] = it
            hard_step = 0
            ps = deepcopy(best_soft_state)
            opt_state = Optimisers.setup(opt, ps)
        end

        tau, lam_ent, lam_bin, lr_mult = _schedule(cfg, phase, hard_step)
        opt = Optimisers.Adam(cfg.lr * lr_mult)

        grads = _autodiff_gradient(ps) do ps_current
            pred, aux = Lux.apply(tree, (x_train, y_train), ps_current, _state_with_tau(st, tau))
            total, _, _, _, _, _, _ = compute_losses(
                pred,
                t_train,
                aux.leaf_probs,
                aux.gate_probs,
                aux.eml_outputs,
                lam_ent,
                lam_bin,
                cfg.lam_inter,
                cfg.inter_threshold,
            )
            total
        end
        grads = _clip_gradients(grads, cfg.grad_clip_norm)
        opt_state, ps = Optimisers.update(opt_state, ps, grads)

        pred, aux = Lux.apply(tree, (x_train, y_train), ps, _state_with_tau(st, tau))
        total, data_loss, entropy, binarity, _, _, _ = compute_losses(
            pred,
            t_train,
            aux.leaf_probs,
            aux.gate_probs,
            aux.eml_outputs,
            lam_ent,
            lam_bin,
            cfg.lam_inter,
            cfg.inter_threshold,
        )

        if !isfinite(total)
            summary[:nonfinite_steps] += 1
            nan_streak += 1
            if nan_streak >= cfg.nan_restart_patience
                ps = deepcopy(best_soft_state)
                opt_state = Optimisers.setup(opt, ps)
                nan_streak = 0
                nan_restarts += 1
                summary[:nan_restarts] = nan_restarts
            end
            continue
        end

        nan_streak = 0
        soft_loss = float(real(data_loss))
        push!(metrics[:soft_rmse], sqrt(max(soft_loss, 0.0)))
        push!(metrics[:tau], tau)
        push!(metrics[:entropy], float(real(entropy)))
        push!(metrics[:binarity], float(real(binarity)))

        if soft_loss < best_soft_loss
            best_soft_loss = soft_loss
            best_soft_state = deepcopy(ps)
        end

        if it % max(cfg.eval_every, 1) == 0 || (phase === :hardening && tau <= cfg.tail_eval_tau && it % max(cfg.tail_eval_every, 1) == 0)
            hard_mse, _, _ = evaluate(tree, ps, st, x_train, y_train, t_train; tau=cfg.tau_hard)
            push!(metrics[:hard_rmse], sqrt(max(hard_mse, 0.0)))
            if phase === :hardening && isfinite(hard_mse) && hard_mse < cfg.success_thr
                hard_success_streak += 1
                hard_success_streak >= cfg.early_stop_count && break
            else
                hard_success_streak = 0
            end
        end

        phase === :hardening && (hard_step += 1)
    end

    summary[:hardening_iter] = hardening_iter
    return TrainingResult(cfg, metrics, failure_reason, ps, st, summary)
end

function _apply_initialization!(rng, ps, cfg::TrainConfig)
    if cfg.init_strategy === :manual
        isnothing(cfg.init_expr) && throw(ArgumentError("init_expr is required when init_strategy=:manual"))
        init_from_expr!(ps, cfg.init_expr)
    end
    if cfg.init_noise > 0
        ps.leaf_logits .+= randn(rng, size(ps.leaf_logits)) .* cfg.init_noise
        ps.blend_logits .+= randn(rng, size(ps.blend_logits)) .* cfg.init_noise
    end
    return ps
end

function compute_losses(
    pred,
    target,
    leaf_probs,
    gate_probs,
    eml_outputs,
    lam_ent,
    lam_bin,
    lam_inter,
    inter_threshold;
    lam_sparse::Float64=0.0,
    uncertainty_power::Float64=2.0,
)
    data_loss = _mse(pred, target)
    eps = 1.0e-12

    leaf_max = vec(maximum(leaf_probs; dims=2))
    leaf_unc = clamp.((1.0 .- leaf_max) ./ (2.0 / 3.0), 0.0, 1.0) .^ uncertainty_power
    leaf_ent = vec(-sum(leaf_probs .* log.(leaf_probs .+ eps); dims=2))
    entropy = isempty(leaf_ent) ? 0.0 : sum(leaf_ent .* leaf_unc) / length(leaf_ent)

    gate_unc = clamp.(1.0 .- abs.(2.0 .* gate_probs .- 1.0), 0.0, 1.0) .^ uncertainty_power
    gate_bin = gate_probs .* (1.0 .- gate_probs)
    binarity = isempty(gate_bin) ? 0.0 : sum(gate_bin .* gate_unc) / length(gate_bin)

    inter_penalty = 0.0
    if lam_inter > 0 && !isempty(eml_outputs)
        excess = max.(abs.(eml_outputs) .- inter_threshold, 0.0)
        inter_penalty = sum(abs2, excess) / length(excess)
    end

    sparse = isempty(gate_probs) ? 0.0 : sum(1.0 .- gate_probs) / length(gate_probs)
    total = data_loss + lam_ent * entropy + lam_bin * binarity + lam_inter * inter_penalty + lam_sparse * sparse
    ambiguity = isempty(gate_unc) ? mean(leaf_unc) : (sum(leaf_unc) + sum(gate_unc)) / (length(leaf_unc) + length(gate_unc))
    return total, data_loss, entropy, binarity, inter_penalty, sparse, ambiguity
end

function evaluate(tree::EMLTree, ps, st, x_data, y_data, targets; tau::Float64=0.01)
    pred, _ = Lux.apply(tree, (x_data, y_data), ps, _state_with_tau(st, tau))
    mse = _mse(pred, targets)
    max_real = maximum(abs.(real.(pred) .- real.(targets)))
    max_imag = maximum(abs.(imag.(pred)))
    return mse, max_real, max_imag
end

function _state_with_tau(st, tau::Float64)
    return (; tau_leaf=tau, tau_gate=tau, leaf_probs=get(st, :leaf_probs, nothing), gate_probs=get(st, :gate_probs, nothing), eml_outputs=get(st, :eml_outputs, nothing))
end

function _schedule(cfg::TrainConfig, phase::Symbol, hard_step::Int)
    if phase === :search
        return cfg.tau_search, 0.0, 0.0, 1.0
    end
    t = hard_step / max(cfg.hardening_iters, 1)
    t_tau = t ^ cfg.hardening_tau_power
    tau = cfg.tau_search * (cfg.tau_hard / cfg.tau_search) ^ t_tau
    lam_ent = t * cfg.lam_ent_hard
    lam_bin = t * cfg.lam_bin_hard
    lr_mult = max(cfg.hardening_lr_floor, (1.0 - t)^2)
    return tau, lam_ent, lam_bin, lr_mult
end

function _mse(preds, ys)
    diffs = preds .- ys
    return sum(abs2, diffs) / max(length(diffs), 1)
end

function _autodiff_gradient(loss_fn, ps)
    grads = only(Zygote.gradient(loss_fn, ps))
    isnothing(grads) && error("autodiff returned no gradients for training parameters")
    return grads
end

function _clip_gradients(grads, max_norm::Real)
    max_norm <= 0 && return grads
    total_norm = sqrt(_grad_sqnorm(grads))
    (!isfinite(total_norm) || total_norm <= max_norm) && return grads
    scale = max_norm / (total_norm + eps(Float64))
    return _scale_grads(grads, scale)
end

_grad_sqnorm(::Nothing) = 0.0
_grad_sqnorm(x::Number) = abs2(x)
_grad_sqnorm(x::AbstractArray) = sum(abs2, x)
_grad_sqnorm(xs::Tuple) = sum(_grad_sqnorm, xs)
_grad_sqnorm(xs::NamedTuple) = sum(_grad_sqnorm, values(xs))

_scale_grads(::Nothing, _) = nothing
_scale_grads(x::Number, scale::Real) = x * scale
_scale_grads(x::AbstractArray, scale::Real) = x .* scale
_scale_grads(xs::Tuple, scale::Real) = map(x -> _scale_grads(x, scale), xs)
function _scale_grads(xs::NamedTuple, scale::Real)
    return NamedTuple{keys(xs)}(map(x -> _scale_grads(x, scale), values(xs)))
end
