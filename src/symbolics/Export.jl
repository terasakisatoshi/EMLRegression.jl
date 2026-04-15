function formula_string(tree::RecoveredTree)
    selected = findfirst(==(1.0), tree.weights)
    terminal = isnothing(selected) ? :const1 : tree.terminals[selected]
    return terminal === :const1 ? "1" : String(terminal)
end
