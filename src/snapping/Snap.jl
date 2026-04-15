"""
    snap_logits(logits; margin_threshold=0.0)

最大ロジットを 1、他を 0 にする単純な snapping です。
"""
function snap_logits(logits; margin_threshold=0.0)
    max_index = argmax(logits)
    snapped = zeros(Float64, length(logits))
    snapped[max_index] = 1.0
    return snapped
end

"""
    snap_model(layer, ps; margin_threshold=0.0)

モデルのロジットから `RecoveredTree` を作ります。
"""
function snap_model(layer::EMLTreeLayer, ps; margin_threshold=0.0)
    weights = snap_logits(collect(ps.logits); margin_threshold=margin_threshold)
    return RecoveredTree(copy(layer.tree.terminals), weights)
end

"""
    evaluate_recovered(tree, x)

現在の最小 EML layer に対応する `RecoveredTree` を評価します。
"""
function evaluate_recovered(tree::RecoveredTree, x)
    terminal = _selected_terminal(tree)
    values = _recovered_terminal_values(terminal, x)
    ones_input = fill(1.0 + 0.0im, length(_recovered_sample_vector(x)))
    return eml(values, ones_input)
end

function _selected_terminal(tree::RecoveredTree)
    selected = findfirst(==(1.0), tree.weights)
    return isnothing(selected) ? :const1 : tree.terminals[selected]
end

function _recovered_terminal_values(symbol::Symbol, x)
    if symbol === :const1
        return fill(1.0 + 0.0im, length(_recovered_sample_vector(x)))
    elseif symbol === :x
        return x isa Tuple ? x[1] : x
    elseif symbol === :y
        return x isa Tuple ? x[2] : throw(ArgumentError("terminal y requires binary input"))
    else
        throw(ArgumentError("unsupported terminal $(symbol)"))
    end
end

_recovered_sample_vector(x::Tuple) = x[1]
_recovered_sample_vector(x) = x
