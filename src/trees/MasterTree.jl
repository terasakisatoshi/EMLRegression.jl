function build_master_tree(; depth::Integer, variables)
    depth >= 1 || throw(ArgumentError("depth must be at least 1"))

    vars = Symbol[variables...]
    terminals = vcat(Symbol[:const1], vars)
    candidate_symbols = copy(terminals)
    node_count = 2^depth - 1
    nodes = [NodeChoiceSpace(copy(candidate_symbols), copy(candidate_symbols)) for _ in 1:node_count]

    return MasterTree(depth, vars, terminals, nodes)
end
