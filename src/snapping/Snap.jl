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

モデルのロジットから node-wise な `RecoveredTree` を作ります。
"""
function snap_model(layer::EMLTreeLayer, ps; margin_threshold=0.0)
    choices = Dict{Int,Tuple{Symbol,Symbol}}()
    for node in layer.tree.nodes
        left = _snap_choice(node.left_candidates, ps.nodes[node.id].left_logits; margin_threshold=margin_threshold)
        right = _snap_choice(node.right_candidates, ps.nodes[node.id].right_logits; margin_threshold=margin_threshold)
        choices[node.id] = (left, right)
    end
    return RecoveredTree(choices)
end

function snap_status(layer::EMLTreeLayer, ps; margin_threshold=0.0)
    for node in layer.tree.nodes
        if _choice_margin(ps.nodes[node.id].left_logits) < margin_threshold
            return :ambiguous
        end
        if _choice_margin(ps.nodes[node.id].right_logits) < margin_threshold
            return :ambiguous
        end
    end
    return :ok
end

function ambiguous_node_count(layer::EMLTreeLayer, ps; margin_threshold=0.0)
    ambiguous = 0
    for node in layer.tree.nodes
        if _choice_margin(ps.nodes[node.id].left_logits) < margin_threshold || _choice_margin(ps.nodes[node.id].right_logits) < margin_threshold
            ambiguous += 1
        end
    end
    return ambiguous
end

function structure_match(expected::RecoveredTree, actual::RecoveredTree)
    return expected.choices == actual.choices
end

function _snap_choice(candidates, logits; margin_threshold=0.0)
    max_index = argmax(logits)
    return candidates[max_index]
end

function _choice_margin(logits)
    sorted = sort(collect(logits); rev=true)
    return length(sorted) < 2 ? Inf : sorted[1] - sorted[2]
end

"""
    evaluate_recovered(tree, master, x)

離散 recovered tree を master tree 上で評価します。
"""
function evaluate_recovered(tree::RecoveredTree, master::MasterTree, x)
    x_complex = _to_recovered_input(x)
    return _evaluate_recovered_node(tree, master, master.root_id, x_complex)
end

function evaluate_recovered(tree::RecoveredTree, x)
    terminal = _selected_terminal(tree)
    values = _recovered_terminal_values(terminal, x)
    ones_input = fill(1.0 + 0.0im, length(_recovered_sample_vector(x)))
    return eml(values, ones_input)
end

function _evaluate_recovered_node(tree::RecoveredTree, master::MasterTree, node_id::Int, x)
    left_symbol, right_symbol = tree.choices[node_id]
    left_value = _resolved_recovered_symbol(tree, master, left_symbol, x)
    right_value = _resolved_recovered_symbol(tree, master, right_symbol, x)
    return eml(left_value, right_value)
end

function _resolved_recovered_symbol(tree::RecoveredTree, master::MasterTree, symbol::Symbol, x)
    if symbol === :const1 || symbol === :x || symbol === :y
        return _recovered_terminal_values(symbol, x)
    elseif startswith(String(symbol), "node_")
        child_id = parse(Int, split(String(symbol), "_")[2])
        return _evaluate_recovered_node(tree, master, child_id, x)
    else
        throw(ArgumentError("unsupported recovered symbol $(symbol)"))
    end
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

_to_recovered_input(x::Tuple) = map(v -> ComplexF64.(v), x)
_to_recovered_input(x) = ComplexF64.(x)

_recovered_sample_vector(x::Tuple) = x[1]
_recovered_sample_vector(x) = x
