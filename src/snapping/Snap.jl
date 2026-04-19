"""
    snap_logits(logits; k=24.0)

leaf logits を argmax で hard choice に射影します。
"""
function snap_logits(logits; k::Float64=24.0)
    snapped = fill(-k, length(logits))
    snapped[argmax(logits)] = k
    return snapped
end

"""
    analyze_snap(ps; snap_threshold=0.01)

softmax/sigmoid 確率を見て、閾値以内に hard snap できない leaf/gate を数えます。
"""
function analyze_snap(ps; snap_threshold::Float64=0.01)
    leaf_probs = _softmax_rows(ps.leaf_logits, 1.0)
    gate_probs = _gate_probs(ps.blend_logits, 1.0)

    names = ("1", "x", "y")
    uncertain_leaves = String[]
    for i in axes(leaf_probs, 1)
        probs = vec(leaf_probs[i, :])
        max_p, max_idx = findmax(probs)
        if max_p < 1.0 - snap_threshold
            push!(
                uncertain_leaves,
                "leaf[$(i - 1)]: best=$(names[max_idx])($(round(max_p; digits=4))) probs=$(round.(probs; digits=4))",
            )
        end
    end

    uncertain_gates = String[]
    for i in axes(gate_probs, 1)
        for j in axes(gate_probs, 2)
            p = gate_probs[i, j]
            if snap_threshold < p < 1.0 - snap_threshold
                side = j == 1 ? "left" : "right"
                push!(uncertain_gates, "gate[$(i - 1)].$(side): prob=$(round(p; digits=4))")
            end
        end
    end

    return (
        uncertain_leaves=uncertain_leaves,
        uncertain_gates=uncertain_gates,
        n_uncertain=length(uncertain_leaves) + length(uncertain_gates),
    )
end

"""
    hard_project(ps; k=24.0)

学習済みロジットを PyTorch v16 と同じ規則で hard projection した新しい parameter set を返します。
"""
function hard_project(ps; k::Float64=24.0)
    projected = (
        leaf_logits=copy(ps.leaf_logits),
        blend_logits=copy(ps.blend_logits),
    )
    return hard_project!(projected; k=k)
end

"""
    hard_project!(ps; k=24.0)

leaf は argmax、gate は sign で `±k` に射影します。
"""
function hard_project!(ps; k::Float64=24.0)
    for i in axes(ps.leaf_logits, 1)
        ps.leaf_logits[i, :] .= snap_logits(vec(ps.leaf_logits[i, :]); k=k)
    end
    ps.blend_logits .= ifelse.(ps.blend_logits .>= 0.0, k, -k)
    return ps
end

function _copy_params(ps)
    return (
        leaf_logits=copy(ps.leaf_logits),
        blend_logits=copy(ps.blend_logits),
    )
end

function _uncertain_snap_moves(ps; snap_threshold::Float64=0.01, k::Float64=24.0)
    leaf_probs = _softmax_rows(ps.leaf_logits, 1.0)
    gate_probs = _gate_probs(ps.blend_logits, 1.0)
    moves = Tuple[]

    for i in axes(leaf_probs, 1)
        probs = vec(leaf_probs[i, :])
        maximum(probs) >= 1.0 - snap_threshold && continue
        for choice in axes(ps.leaf_logits, 2)
            push!(moves, (:leaf, i, choice, k))
        end
    end

    for i in axes(gate_probs, 1)
        for j in axes(gate_probs, 2)
            p = gate_probs[i, j]
            if snap_threshold < p < 1.0 - snap_threshold
                push!(moves, (:gate, i, j, -k))
                push!(moves, (:gate, i, j, k))
            end
        end
    end

    return moves
end

function _apply_snap_move(ps, move)
    kind, i, j, value = move
    candidate = _copy_params(ps)
    if kind === :leaf
        candidate.leaf_logits[i, :] .= -value
        candidate.leaf_logits[i, j] = value
    else
        candidate.blend_logits[i, j] = value
    end
    return candidate
end

function _projected_snap_metrics(tree::EMLTree, ps, st, x_train, y_train, t_train; tau::Float64=0.01, k::Float64=24.0)
    snapped = hard_project(ps; k=k)
    snap_mse, snap_max_real, snap_max_imag = evaluate(tree, snapped, st, x_train, y_train, t_train; tau=tau)
    return (
        snapped_params=snapped,
        snap_mse=snap_mse,
        snap_max_real=snap_max_real,
        snap_max_imag=snap_max_imag,
    )
end

function _refine_snap_projection(
    tree::EMLTree,
    ps,
    st,
    x_train,
    y_train,
    t_train;
    tau::Float64=0.01,
    snap_threshold::Float64=0.01,
    k::Float64=24.0,
    max_steps::Int=8,
)
    best_params = _copy_params(ps)
    best_metrics = _projected_snap_metrics(tree, best_params, st, x_train, y_train, t_train; tau=tau, k=k)
    improved = false

    for _ in 1:max_steps
        step_best_params = best_params
        step_best_metrics = best_metrics

        for move in _uncertain_snap_moves(best_params; snap_threshold=snap_threshold, k=k)
            candidate = _apply_snap_move(best_params, move)
            candidate_metrics = _projected_snap_metrics(tree, candidate, st, x_train, y_train, t_train; tau=tau, k=k)
            if candidate_metrics.snap_mse + 1.0e-18 < step_best_metrics.snap_mse
                step_best_params = candidate
                step_best_metrics = candidate_metrics
            end
        end

        step_best_metrics.snap_mse + 1.0e-18 < best_metrics.snap_mse || break
        best_params = step_best_params
        best_metrics = step_best_metrics
        improved = true
    end

    return (
        params=best_params,
        snapped_params=best_metrics.snapped_params,
        snap_info=analyze_snap(best_params; snap_threshold=snap_threshold),
        snap_mse=best_metrics.snap_mse,
        snap_max_real=best_metrics.snap_max_real,
        snap_max_imag=best_metrics.snap_max_imag,
        improved=improved,
    )
end
