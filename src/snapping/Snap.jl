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

"""
    search_recovered_tree(layer, ps, master, xs, ys; top_k=3, beam_width=24, max_passes=6, loss_atol=1e-8)

active node の top-k 候補を beam search で探索し、validation loss が最も小さい離散木を返します。
"""
function search_recovered_tree(layer::EMLTreeLayer, ps, master::MasterTree, xs, ys; top_k::Int=3, beam_width::Int=24, max_passes::Int=6, loss_atol::Float64=1.0e-8)
    initial = _canonicalize_recovered_tree(snap_model(layer, ps), master)
    beam = [_candidate_entry(initial, master, xs, ys)]
    best = only(beam)

    for _ in 1:max_passes
        candidates = Dict{String,NamedTuple}()
        for entry in beam
            _insert_candidate!(candidates, entry)
            for neighbor in _top_k_neighbors(entry.tree, layer, ps; top_k=top_k)
                canonical = _canonicalize_recovered_tree(neighbor, master)
                _insert_candidate!(candidates, _candidate_entry(canonical, master, xs, ys))
            end
        end

        ranked = sort(collect(values(candidates)); lt=(left, right) -> _candidate_before(left, right; loss_atol=loss_atol))
        new_beam = ranked[1:min(length(ranked), beam_width)]
        new_best = first(new_beam)
        previous_formulas = Set(entry.formula for entry in beam)
        next_formulas = Set(entry.formula for entry in new_beam)
        if previous_formulas == next_formulas
            best = _candidate_before(new_best, best; loss_atol=loss_atol) ? new_best : best
            break
        end
        beam = new_beam
        best = _candidate_before(new_best, best; loss_atol=loss_atol) ? new_best : best
    end

    return refine_recovered_tree(best.tree, master, xs, ys; loss_atol=loss_atol)
end

"""
    refine_recovered_tree(tree, master, xs, ys; max_passes=4, loss_atol=1e-8)

argmax で得た離散木を、数値一致を保ちながらより単純な構造へ貪欲に寄せます。
"""
function refine_recovered_tree(tree::RecoveredTree, master::MasterTree, xs, ys; max_passes::Int=4, loss_atol::Float64=1.0e-8)
    best_tree = _canonicalize_recovered_tree(tree, master)
    best_loss = _mse(evaluate_recovered(best_tree, master, xs), ys)
    best_complexity = _tree_complexity(best_tree)

    for _ in 1:max_passes
        improved = false
        for node_id in sort!(collect(_active_node_ids(best_tree)))
            node = master.nodes[node_id]
            for (side, candidates) in ((:left, node.left_candidates), (:right, node.right_candidates))
                for symbol in candidates
                    candidate_tree = _with_symbol_choice(best_tree, node_id, side, symbol)
                    candidate_loss = _mse(evaluate_recovered(candidate_tree, master, xs), ys)
                    candidate_complexity = _tree_complexity(candidate_tree)
                    if _prefer_recovered_candidate(candidate_tree, candidate_loss, candidate_complexity, best_tree, best_loss, best_complexity; loss_atol=loss_atol)
                        best_tree = candidate_tree
                        best_loss = candidate_loss
                        best_complexity = candidate_complexity
                        improved = true
                    end
                end
            end
        end
        canonical_tree = _canonicalize_recovered_tree(best_tree, master)
        canonical_loss = _mse(evaluate_recovered(canonical_tree, master, xs), ys)
        canonical_complexity = _tree_complexity(canonical_tree)
        if _prefer_recovered_candidate(canonical_tree, canonical_loss, canonical_complexity, best_tree, best_loss, best_complexity; loss_atol=loss_atol)
            best_tree = canonical_tree
            best_loss = canonical_loss
            best_complexity = canonical_complexity
            improved = true
        end
        improved || break
    end

    return best_tree
end

function snap_status(layer::EMLTreeLayer, ps; margin_threshold=0.0)
    active_node_ids = _active_node_ids(snap_model(layer, ps; margin_threshold=margin_threshold))
    for node in layer.tree.nodes
        node.id in active_node_ids || continue
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
    active_node_ids = _active_node_ids(snap_model(layer, ps; margin_threshold=margin_threshold))
    for node in layer.tree.nodes
        node.id in active_node_ids || continue
        if _choice_margin(ps.nodes[node.id].left_logits) < margin_threshold || _choice_margin(ps.nodes[node.id].right_logits) < margin_threshold
            ambiguous += 1
        end
    end
    return ambiguous
end

function structure_match(expected::RecoveredTree, actual::RecoveredTree)
    return _active_choices(expected) == _active_choices(actual)
end

function _active_choices(tree::RecoveredTree)
    active = Dict{Int,Tuple{Symbol,Symbol}}()
    visited = Set{Int}()
    _collect_active_choices!(active, visited, tree, 1)
    return active
end

function _active_node_ids(tree::RecoveredTree)
    return Set(keys(_active_choices(tree)))
end

function _collect_active_choices!(active, visited, tree::RecoveredTree, node_id::Int)
    node_id in visited && return active
    push!(visited, node_id)
    choice = tree.choices[node_id]
    active[node_id] = choice
    for symbol in choice
        if startswith(String(symbol), "node_")
            child_id = parse(Int, split(String(symbol), "_")[2])
            _collect_active_choices!(active, visited, tree, child_id)
        end
    end
    return active
end

function _snap_choice(candidates, logits; margin_threshold=0.0)
    max_index = argmax(logits)
    return candidates[max_index]
end

function _choice_margin(logits)
    sorted = sort(collect(logits); rev=true)
    return length(sorted) < 2 ? Inf : sorted[1] - sorted[2]
end

function _with_symbol_choice(tree::RecoveredTree, node_id::Int, side::Symbol, symbol::Symbol)
    current_left, current_right = tree.choices[node_id]
    updated = side === :left ? (symbol, current_right) : (current_left, symbol)
    current = tree.choices[node_id]
    updated == current && return tree
    choices = copy(tree.choices)
    choices[node_id] = updated
    return RecoveredTree(choices, copy(tree.terminals), copy(tree.weights))
end

function _top_k_neighbors(tree::RecoveredTree, layer::EMLTreeLayer, ps; top_k::Int)
    neighbors = RecoveredTree[]
    for node_id in sort!(collect(_active_node_ids(tree)))
        node = layer.tree.nodes[node_id]
        node_ps = ps.nodes[node_id]
        for (side, symbols) in (
            (:left, _top_k_symbols(node.left_candidates, node_ps.left_logits, top_k)),
            (:right, _top_k_symbols(node.right_candidates, node_ps.right_logits, top_k)),
        )
            current_symbol = side === :left ? tree.choices[node_id][1] : tree.choices[node_id][2]
            for symbol in symbols
                symbol == current_symbol && continue
                push!(neighbors, _with_symbol_choice(tree, node_id, side, symbol))
            end
        end
    end
    return neighbors
end

function _top_k_symbols(candidates, logits, top_k::Int)
    ranked = sort(collect(zip(candidates, logits)); by=last, rev=true)
    limit = min(length(ranked), max(top_k, 1))
    return [ranked[i][1] for i in 1:limit]
end

function _canonicalize_recovered_tree(tree::RecoveredTree, master::MasterTree)
    canonical = _left_packed_tree(tree, master)
    for node_id in sort!(collect(_active_node_ids(tree)))
        canonical = _rewrite_log_identity(canonical, master, node_id)
    end
    canonical = _left_packed_tree(canonical, master)
    return canonical
end

function _left_packed_tree(tree::RecoveredTree, master::MasterTree)
    packed = copy(tree.choices)
    _repack_subtree!(packed, tree.choices, master, master.root_id, master.root_id)
    return RecoveredTree(packed, copy(tree.terminals), copy(tree.weights))
end

function _repack_subtree!(packed, source_choices, master::MasterTree, source_id::Int, packed_id::Int)
    haskey(source_choices, source_id) || return packed

    left_symbol, right_symbol = source_choices[source_id]
    left_is_subtree = startswith(String(left_symbol), "node_")
    right_is_subtree = startswith(String(right_symbol), "node_")
    node = master.nodes[packed_id]

    new_left = left_symbol
    new_right = right_symbol

    if left_is_subtree && right_is_subtree
        left_child_id = something(node.left_child)
        right_child_id = something(node.right_child)
        new_left = Symbol("node_$(left_child_id)")
        new_right = Symbol("node_$(right_child_id)")
        _repack_subtree!(packed, source_choices, master, _symbol_node_id(left_symbol), left_child_id)
        _repack_subtree!(packed, source_choices, master, _symbol_node_id(right_symbol), right_child_id)
    elseif left_is_subtree || right_is_subtree
        if left_is_subtree
            child_id = right_symbol === :const1 ? something(node.left_child) : something(node.left_child)
            new_left = Symbol("node_$(child_id)")
            _repack_subtree!(packed, source_choices, master, _symbol_node_id(left_symbol), child_id)
        else
            child_id = left_symbol === :const1 ? something(node.left_child) : something(node.right_child)
            new_right = Symbol("node_$(child_id)")
            _repack_subtree!(packed, source_choices, master, _symbol_node_id(right_symbol), child_id)
        end
    end

    packed[packed_id] = (new_left, new_right)
    return packed
end

function _rewrite_log_identity(tree::RecoveredTree, master::MasterTree, node_id::Int)
    outer_left, outer_right = tree.choices[node_id]
    startswith(String(outer_right), "node_") || return tree

    mid_id = parse(Int, split(String(outer_right), "_")[2])
    haskey(tree.choices, mid_id) || return tree
    mid_left, mid_right = tree.choices[mid_id]
    mid_right === :const1 || return tree
    startswith(String(mid_left), "node_") || return tree

    inner_id = parse(Int, split(String(mid_left), "_")[2])
    haskey(tree.choices, inner_id) || return tree
    inner_left, inner_right = tree.choices[inner_id]
    inner_left == outer_left || return tree

    node = master.nodes[node_id]
    node.left_child === nothing && return tree
    left_child_id = something(node.left_child)
    left_child = master.nodes[left_child_id]
    left_child.left_child === nothing && return tree
    left_grandchild_id = something(left_child.left_child)

    choices = copy(tree.choices)
    choices[node_id] = (:const1, Symbol("node_$(left_child_id)"))
    choices[left_child_id] = (Symbol("node_$(left_grandchild_id)"), :const1)
    choices[left_grandchild_id] = (:const1, inner_right)
    return RecoveredTree(choices, copy(tree.terminals), copy(tree.weights))
end

function _symbol_node_id(symbol::Symbol)
    return parse(Int, split(String(symbol), "_")[2])
end

function _candidate_entry(tree::RecoveredTree, master::MasterTree, xs, ys)
    loss = _mse(evaluate_recovered(tree, master, xs), ys)
    complexity = _tree_complexity(tree)
    formula = formula_string(tree, master)
    return (tree=tree, loss=loss, complexity=complexity, formula=formula)
end

function _candidate_before(left, right; loss_atol::Float64)
    left.loss + loss_atol < right.loss && return true
    right.loss + loss_atol < left.loss && return false
    left.complexity + 1.0e-12 < right.complexity && return true
    right.complexity + 1.0e-12 < left.complexity && return false
    return left.formula < right.formula
end

function _insert_candidate!(candidates, entry)
    key = entry.formula
    if !haskey(candidates, key) || _candidate_before(entry, candidates[key]; loss_atol=1.0e-8)
        candidates[key] = entry
    end
    return candidates
end

function _prefer_recovered_candidate(candidate_tree, candidate_loss, candidate_complexity, best_tree, best_loss, best_complexity; loss_atol::Float64)
    candidate_loss + loss_atol < best_loss && return true
    if abs(candidate_loss - best_loss) <= loss_atol
        candidate_complexity + 1.0e-12 < best_complexity && return true
        if isapprox(candidate_complexity, best_complexity; atol=1.0e-12)
            return formula_string(candidate_tree) < formula_string(best_tree)
        end
    end
    return false
end

function _tree_complexity(tree::RecoveredTree)
    return sum(
        _candidate_cost(symbol)
        for (_, choice) in _active_choices(tree)
        for symbol in choice
    )
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
