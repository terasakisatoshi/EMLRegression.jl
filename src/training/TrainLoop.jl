using StableRNGs: StableRNG
using Lux
using Optimisers
using Zygote

function run_training(cfg::TrainConfig; rng=nothing, capture_diagnostics::Bool=false)
    rng = isnothing(rng) ? StableRNG(cfg.seed) : rng
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
        :nonfinite_grad_steps => 0,
    )
    diagnostics = Dict{Symbol,Vector{Any}}()

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
        phase === :hardening && (hard_step += 1)
        _adjust_optimizer_lr!(opt_state, cfg.lr * lr_mult)
        tau_state = _state_with_tau(st, tau)

        # Run the faithful composite loss once before autodiff so nonfinite states
        # are rejected by the restart logic instead of reaching Zygote.
        pred, regs = forward_with_regularizers(
            tree,
            (x_train, y_train),
            ps;
            tau_leaf=tau,
            tau_gate=tau,
            inter_threshold=cfg.inter_threshold,
        )
        data_loss = _data_loss(pred, t_train)
        total = _combine_losses(data_loss, regs.entropy, regs.binarity, regs.inter_penalty, lam_ent, lam_bin, cfg.lam_inter)
        entropy = regs.entropy
        binarity = regs.binarity
        if capture_diagnostics
            _push_diagnostic!(
                diagnostics,
                _training_diagnostics_snapshot(phase, it, tau_state, ps, total),
                cfg.diagnostics_limit,
            )
        end

        if !(isfinite(total) && isfinite(data_loss))
            summary[:nonfinite_steps] += 1
            nan_streak += 1
            if should_restart(nan_streak, nan_restarts, cfg)
                ps = deepcopy(best_soft_state)
                opt_state = Optimisers.setup(opt, ps)
                nan_streak = 0
                nan_restarts += 1
                hard_success_streak = 0
                summary[:nan_restarts] = nan_restarts
            end
            continue
        end

        grads = _autodiff_gradient(ps) do ps_current
            pred, regs = forward_with_regularizers(
                tree,
                (x_train, y_train),
                ps_current;
                tau_leaf=tau,
                tau_gate=tau,
                inter_threshold=cfg.inter_threshold,
            )
            data_loss = _data_loss(pred, t_train)
            _combine_losses(data_loss, regs.entropy, regs.binarity, regs.inter_penalty, lam_ent, lam_bin, cfg.lam_inter)
        end
        if any_bad_grad(grads)
            summary[:nonfinite_grad_steps] += 1
        end
        grads = _sanitize_grads(grads)
        grads = _clip_gradients(grads, cfg.grad_clip_norm)
        opt_state, ps = Optimisers.update(opt_state, ps, grads)

        pred, regs = forward_with_regularizers(
            tree,
            (x_train, y_train),
            ps;
            tau_leaf=tau,
            tau_gate=tau,
            inter_threshold=cfg.inter_threshold,
        )
        data_loss = _data_loss(pred, t_train)
        total = _combine_losses(data_loss, regs.entropy, regs.binarity, regs.inter_penalty, lam_ent, lam_bin, cfg.lam_inter)
        entropy = regs.entropy
        binarity = regs.binarity
        if capture_diagnostics
            _push_diagnostic!(
                diagnostics,
                _training_diagnostics_snapshot(phase, it, tau_state, ps, total),
                cfg.diagnostics_limit,
            )
        end

        if !(isfinite(total) && isfinite(data_loss))
            summary[:nonfinite_steps] += 1
            nan_streak += 1
            if should_restart(nan_streak, nan_restarts, cfg)
                ps = deepcopy(best_soft_state)
                opt_state = Optimisers.setup(opt, ps)
                nan_streak = 0
                nan_restarts += 1
                hard_success_streak = 0
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

    end

    summary[:hardening_iter] = hardening_iter
    return TrainingResult(cfg, metrics, failure_reason, ps, st, summary, summary[:nonfinite_steps], nan_restarts, diagnostics)
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
    uncertainty_power::Float64=1.0,
)
    data_loss = _data_loss(pred, target)
    entropy, binarity, inter_penalty, sparse, ambiguity = _regularization_stats(
        leaf_probs,
        gate_probs,
        eml_outputs,
        inter_threshold;
        lam_inter=lam_inter,
        uncertainty_power=uncertainty_power,
    )
    total = _combine_losses(data_loss, entropy, binarity, inter_penalty, lam_ent, lam_bin, lam_inter) + lam_sparse * sparse
    return total, data_loss, entropy, binarity, inter_penalty, sparse, ambiguity
end

function evaluate(tree::EMLTree, ps, st, x_data, y_data, targets; tau::Float64=0.01)
    pred, _ = forward_with_aux(tree, (x_data, y_data), ps; tau_leaf=tau, tau_gate=tau)
    mse = _safe_mean_abs2(pred .- targets)
    max_real = maximum(abs.(real.(pred) .- real.(targets)))
    max_imag = maximum(abs.(imag.(pred)))
    return mse, max_real, max_imag
end

function _state_with_tau(st, tau::Float64)
    return (; tau_leaf=tau, tau_gate=tau, leaf_probs=get(st, :leaf_probs, nothing), gate_probs=get(st, :gate_probs, nothing), eml_outputs=get(st, :eml_outputs, nothing))
end

function _training_diagnostics_snapshot(phase_name, iter, tau_state, ps, loss)
    max_gate_logit = isempty(ps.blend_logits) ? 0.0 : maximum(abs, ps.blend_logits)
    max_leaf_logit = isempty(ps.leaf_logits) ? 0.0 : maximum(abs, ps.leaf_logits)
    return (
        phase=phase_name,
        iter=iter,
        tau_leaf=tau_state.tau_leaf,
        tau_gate=tau_state.tau_gate,
        max_gate_logit=max_gate_logit,
        max_leaf_logit=max_leaf_logit,
        loss=loss,
    )
end

function _push_diagnostic!(trace::Dict{Symbol,Vector{Any}}, snapshot, limit::Int)
    limit <= 0 && return trace
    for (key, value) in pairs(snapshot)
        values = get!(trace, key) do
            Any[]
        end
        if length(values) >= limit
            popfirst!(values)
        end
        push!(values, value)
    end
    return trace
end

function _adjust_optimizer_lr!(opt_state, eta::Real)
    Optimisers.adjust!(opt_state, eta)
    return opt_state
end

function _optimizer_learning_rates(opt_state)
    rates = Float64[]
    _collect_optimizer_learning_rates!(rates, opt_state)
    return rates
end

function _collect_optimizer_learning_rates!(rates::Vector{Float64}, state::Optimisers.Leaf)
    if hasproperty(state.rule, :eta)
        push!(rates, Float64(getproperty(state.rule, :eta)))
    end
    return rates
end

_collect_optimizer_learning_rates!(rates::Vector{Float64}, state::NamedTuple) = (foreach(v -> _collect_optimizer_learning_rates!(rates, v), values(state)); rates)
_collect_optimizer_learning_rates!(rates::Vector{Float64}, state::Tuple) = (foreach(v -> _collect_optimizer_learning_rates!(rates, v), state); rates)
_collect_optimizer_learning_rates!(rates::Vector{Float64}, _) = rates

function _schedule(cfg::TrainConfig, phase::Symbol, hard_step::Int)
    if phase === :search
        return cfg.tau_search, 0.0, 0.0, 1.0
    end
    hardening_taus = _hardening_tau_schedule(cfg)
    idx = clamp(hard_step + 1, 1, max(length(hardening_taus), 1))
    tau = isempty(hardening_taus) ? cfg.tau_hard : hardening_taus[idx]
    t = cfg.hardening_iters <= 1 ? 1.0 : hard_step / (cfg.hardening_iters - 1)
    lam_ent = t * cfg.lam_ent_hard
    lam_bin = t * cfg.lam_bin_hard
    lr_mult = max(cfg.hardening_lr_floor, (1.0 - t)^2)
    return tau, lam_ent, lam_bin, lr_mult
end

function _hardening_tau_schedule(cfg::TrainConfig)
    if cfg.hardening_iters <= 0
        return Float64[]
    elseif cfg.hardening_iters == 1
        return [cfg.tau_hard]
    end
    t = range(0.0, 1.0, length=cfg.hardening_iters) .^ cfg.hardening_tau_power
    return exp10.(log10(cfg.tau_search) .+ t .* (log10(cfg.tau_hard) - log10(cfg.tau_search)))
end

function collect_tau_schedule(cfg::TrainConfig)
    return vcat(fill(cfg.tau_search, cfg.search_iters), _hardening_tau_schedule(cfg))
end

function _finite_residuals(preds, ys)
    diffs = preds .- ys
    residuals_finite = all(isfinite, real.(diffs)) && all(isfinite, imag.(diffs))
    return diffs, residuals_finite
end

function _data_loss(pred, target)
    diffs, residuals_finite = _finite_residuals(pred, target)
    return residuals_finite ? _safe_mean_abs2(diffs) : Inf
end

function _combine_losses(data_loss, entropy, binarity, inter_penalty, lam_ent, lam_bin, lam_inter)
    return data_loss + lam_ent * entropy + lam_bin * binarity + lam_inter * inter_penalty
end

function _regularization_stats(
    leaf_probs,
    gate_probs,
    eml_outputs,
    inter_threshold;
    lam_inter::Float64=0.0,
    uncertainty_power::Float64=1.0,
)
    eps = 1.0e-12

    if isempty(leaf_probs)
        entropy = 0.0
        leaf_unc = Float64[]
    else
        leaf_max = vec(maximum(leaf_probs; dims=2))
        leaf_unc = clamp.((1.0 .- leaf_max) ./ (2.0 / 3.0), 0.0, 1.0) .^ uncertainty_power
        leaf_ent = vec(-sum(leaf_probs .* log.(leaf_probs .+ eps); dims=2))
        entropy = isempty(leaf_ent) ? 0.0 : sum(leaf_ent .* leaf_unc) / length(leaf_ent)
    end

    if isempty(gate_probs)
        gate_unc = Float64[]
        gate_bin = Float64[]
        binarity = 0.0
    else
        gate_unc = clamp.(1.0 .- abs.(2.0 .* gate_probs .- 1.0), 0.0, 1.0) .^ uncertainty_power
        gate_bin = gate_probs .* (1.0 .- gate_probs)
        binarity = isempty(gate_bin) ? 0.0 : sum(gate_bin .* gate_unc) / length(gate_bin)
    end

    inter_penalty = 0.0
    if lam_inter > 0 && !isempty(eml_outputs)
        excess = max.(abs.(eml_outputs) .- inter_threshold, 0.0)
        inter_penalty = sum(abs2, excess) / length(excess)
    end

    sparse = isempty(gate_probs) ? 0.0 : sum(1.0 .- gate_probs) / length(gate_probs)
    ambiguity = isempty(gate_unc) ? (isempty(leaf_unc) ? 0.0 : sum(leaf_unc) / length(leaf_unc)) : (sum(leaf_unc) + sum(gate_unc)) / (length(leaf_unc) + length(gate_unc))
    return entropy, binarity, inter_penalty, sparse, ambiguity
end

function forward_with_regularizers(
    tree::EMLTree,
    xy,
    ps;
    tau_leaf::Float64=1.0,
    tau_gate::Float64=1.0,
    inter_threshold::Float64=50.0,
    uncertainty_power::Float64=1.0,
)
    x, y = _to_complex_pair(xy)
    leaf_probs = _softmax_rows(ps.leaf_logits, tau_leaf)
    candidates = hcat(
        fill(1.0 + 0.0im, length(x)),
        x,
        y,
    )
    current_level = Matrix{ComplexF64}(candidates * leaf_probs')
    node_idx = 1
    eps = 1.0e-12

    if isempty(leaf_probs)
        entropy = 0.0
        leaf_unc_sum = 0.0
        leaf_unc_count = 0
    else
        leaf_max = vec(maximum(leaf_probs; dims=2))
        leaf_unc = clamp.((1.0 .- leaf_max) ./ (2.0 / 3.0), 0.0, 1.0) .^ uncertainty_power
        leaf_ent = vec(-sum(leaf_probs .* log.(leaf_probs .+ eps); dims=2))
        entropy = isempty(leaf_ent) ? 0.0 : sum(leaf_ent .* leaf_unc) / length(leaf_ent)
        leaf_unc_sum = sum(leaf_unc)
        leaf_unc_count = length(leaf_unc)
    end

    gate_binarity_sum = 0.0
    gate_unc_sum = 0.0
    gate_count = 0
    sparse_sum = 0.0
    inter_penalty_sum = 0.0
    inter_count = 0

    while size(current_level, 2) > 1
        n_pairs = size(current_level, 2) ÷ 2
        raw = @view ps.blend_logits[node_idx:(node_idx + n_pairs - 1), :]
        gates = clamp_probs.(1.0 ./ (1.0 .+ exp.(-(raw ./ max(tau_gate, 1.0e-6)))))
        left_children = @view current_level[:, 1:2:size(current_level, 2)]
        right_children = @view current_level[:, 2:2:size(current_level, 2)]
        left_input = _complex_blend(left_children, vec(gates[:, 1]), tree.eml_clamp)
        right_input = _complex_blend(right_children, vec(gates[:, 2]), tree.eml_clamp)
        next_level = _sanitize_and_clamp_matrix(eml(left_input, right_input), tree.eml_clamp)
        current_level = next_level
        gate_unc = clamp.(1.0 .- abs.(2.0 .* gates .- 1.0), 0.0, 1.0) .^ uncertainty_power
        gate_bin = gates .* (1.0 .- gates)
        gate_binarity_sum += sum(gate_bin .* gate_unc)
        gate_unc_sum += sum(gate_unc)
        gate_count += length(gate_bin)
        sparse_sum += sum(1.0 .- gates)

        if !isempty(next_level)
            eml_outputs = vec(next_level)
            excess = max.(abs.(eml_outputs) .- inter_threshold, 0.0)
            inter_penalty_sum += sum(abs2, excess)
            inter_count += length(excess)
        end
        node_idx += n_pairs
    end

    pred = vec(current_level[:, 1])
    binarity = gate_count == 0 ? 0.0 : gate_binarity_sum / gate_count
    inter_penalty = inter_count == 0 ? 0.0 : inter_penalty_sum / inter_count
    sparse = gate_count == 0 ? 0.0 : sparse_sum / gate_count
    ambiguity_count = leaf_unc_count + gate_count
    ambiguity = ambiguity_count == 0 ? 0.0 : (leaf_unc_sum + gate_unc_sum) / ambiguity_count
    return pred, (; entropy, binarity, inter_penalty, sparse, ambiguity)
end

function _safe_mean_abs2(diffs)
    scale = 0.0
    sumsq = 1.0
    count = 0
    for z in diffs
        mag = abs(z)
        isfinite(mag) || return Inf
        if mag > 0.0
            if scale < mag
                ratio = scale / mag
                sumsq = 1.0 + sumsq * ratio * ratio
                scale = mag
            else
                ratio = mag / scale
                sumsq += ratio * ratio
            end
        end
        count += 1
    end
    count == 0 && return 0.0
    total = scale * (scale * (sumsq / count))
    isfinite(total) || return Inf
    return total
end

function _autodiff_gradient(loss_fn, ps)
    grads = only(Zygote.gradient(loss_fn, ps))
    isnothing(grads) && error("autodiff returned no gradients for training parameters")
    return grads
end

function _clip_gradients(grads, max_norm::Real)
    grads = _sanitize_grads(grads)
    max_norm <= 0 && return grads
    total_norm = sqrt(_grad_sqnorm(grads))
    (!isfinite(total_norm) || total_norm <= max_norm) && return grads
    scale = max_norm / (total_norm + eps(Float64))
    return _scale_grads(grads, scale)
end

_sanitize_grads(::Nothing) = nothing
_sanitize_grads(x::Number) = isfinite(x) ? x : zero(x)
_sanitize_grads(x::AbstractArray) = map(v -> isfinite(v) ? v : zero(v), x)
_sanitize_grads(xs::Tuple) = map(_sanitize_grads, xs)
_sanitize_grads(xs::NamedTuple) = map(_sanitize_grads, xs)
_sanitize_grads(x) = x

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
