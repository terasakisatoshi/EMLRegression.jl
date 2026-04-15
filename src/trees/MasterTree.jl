"""
    build_master_tree(; depth, variables)

指定した深さと変数集合から、完全二分木ベースの master tree を構築します。
"""
function build_master_tree(; depth::Integer, variables)
    depth >= 1 || throw(ArgumentError("depth must be at least 1"))

    vars = Symbol[variables...]
    terminals = vcat(Symbol[:const1], vars)
    node_count = 2^depth - 1
    nodes = TreeNodeSpec[]

    for node_id in 1:node_count
        left_child = 2 * node_id <= node_count ? 2 * node_id : nothing
        right_child = 2 * node_id + 1 <= node_count ? 2 * node_id + 1 : nothing

        left_candidates = copy(terminals)
        right_candidates = copy(terminals)
        if !isnothing(left_child)
            push!(left_candidates, Symbol("node_$(left_child)"))
        end
        if !isnothing(right_child)
            push!(right_candidates, Symbol("node_$(right_child)"))
        end

        push!(nodes, TreeNodeSpec(node_id, left_candidates, right_candidates, left_child, right_child))
    end

    return MasterTree(depth, vars, 1, terminals, nodes)
end
