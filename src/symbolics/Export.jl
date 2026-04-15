"""
    formula_string(tree)

`RecoveredTree` の選択結果を簡易な文字列表現へ変換します。
"""
function formula_string(tree::RecoveredTree)
    terminal = _selected_terminal(tree)
    terminal_str = terminal === :const1 ? "1" : String(terminal)
    return "eml($(terminal_str), 1)"
end

function formula_string(tree::RecoveredTree, master::MasterTree)
    return _formula_for_node(tree, master.root_id)
end

function _formula_for_node(tree::RecoveredTree, node_id::Int)
    left_symbol, right_symbol = tree.choices[node_id]
    left = _formula_for_symbol(tree, left_symbol)
    right = _formula_for_symbol(tree, right_symbol)
    return "eml($(left), $(right))"
end

function _formula_for_symbol(tree::RecoveredTree, symbol::Symbol)
    if symbol === :const1
        return "1"
    elseif symbol === :x || symbol === :y
        return String(symbol)
    elseif startswith(String(symbol), "node_")
        child_id = parse(Int, split(String(symbol), "_")[2])
        return _formula_for_node(tree, child_id)
    else
        throw(ArgumentError("unsupported recovered symbol $(symbol)"))
    end
end
