struct RecoveredTree
    terminals::Vector{Symbol}
    weights::Vector{Float64}
end

function snap_logits(logits; margin_threshold=0.0)
    max_index = argmax(logits)
    snapped = zeros(Float64, length(logits))
    snapped[max_index] = 1.0
    return snapped
end

function snap_model(layer::EMLTreeLayer, ps; margin_threshold=0.0)
    weights = snap_logits(collect(ps.logits); margin_threshold=margin_threshold)
    return RecoveredTree(copy(layer.tree.terminals), weights)
end
