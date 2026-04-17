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
