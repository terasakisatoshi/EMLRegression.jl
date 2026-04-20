using ChainRulesCore
using Lux
using StableRNGs: LehmerRNG

"""
    EMLTreeLayer(tree)

`MasterTree` を `Lux.jl` のレイヤとして評価する最小実装です。
"""
struct EMLTreeLayer{T} <: Lux.AbstractLuxLayer
    tree::T
    init_strategy::Symbol
end

const _EML_EXP_REAL_LIMIT = 40.0
const _EML_NODE_ABS_LIMIT = 1.0e6
EMLTreeLayer(tree; init_strategy::Symbol=:small_gaussian) = EMLTreeLayer(tree, init_strategy)

function Lux.initialparameters(rng::LehmerRNG, layer::EMLTreeLayer)
    node_params = Tuple(
        (
            left_logits=_initial_logits(rng, node.left_candidates, layer.init_strategy, node.left_child, node.id, layer.tree.depth),
            right_logits=_initial_logits(rng, node.right_candidates, layer.init_strategy, node.right_child, node.id, layer.tree.depth),
        ) for node in layer.tree.nodes
    )
    return (; nodes=node_params)
end

function _initial_logits(rng::LehmerRNG, candidates, strategy::Symbol, child_id, node_id::Int, tree_depth::Int)
    logits = 0.01 .* randn(rng, Float64, length(candidates))

    if strategy === :small_gaussian
        return logits
    elseif strategy === :target_tree_noise
        return logits
    elseif strategy === :zero_bias_to_inputs
        return logits .+ _terminal_bias(candidates, child_id; terminal_boost=0.75, child_penalty=-0.75)
    elseif strategy === :subtree_favoring
        return logits .+ _terminal_bias(candidates, child_id; terminal_boost=-0.75, child_penalty=0.75)
    elseif strategy === :margin_biased
        preferred = rand(rng, eachindex(candidates))
        logits[preferred] += 1.5
        return logits .+ _terminal_bias(candidates, child_id; terminal_boost=0.25, child_penalty=0.5)
    elseif strategy === :depth5_blind_bias
        return logits .+ _depth5_blind_bias(candidates, child_id, node_id, tree_depth)
    else
        throw(ArgumentError("unsupported init strategy: $(strategy)"))
    end
end

function _depth5_blind_bias(candidates, child_id, node_id::Int, tree_depth::Int)
    isnothing(child_id) && return zeros(Float64, length(candidates))

    node_depth = _node_depth(node_id)
    remaining_levels = max(tree_depth - node_depth, 0)
    scale = tree_depth <= 1 ? 0.0 : remaining_levels / (tree_depth - 1)
    child_boost = 0.65 + 0.65 * scale
    terminal_boost = 0.55 + 0.1 * (1.0 - scale)

    bias = fill(terminal_boost, length(candidates))
    child_symbol = Symbol("node_$(child_id)")
    for (idx, symbol) in pairs(candidates)
        if symbol === child_symbol
            bias[idx] = child_boost
        end
    end
    return bias
end

_node_depth(node_id::Int) = floor(Int, log2(node_id)) + 1

function _terminal_bias(candidates, child_id; terminal_boost::Float64, child_penalty::Float64)
    bias = zeros(Float64, length(candidates))
    for (idx, symbol) in pairs(candidates)
        if symbol === :const1 || symbol === :x || symbol === :y
            bias[idx] = terminal_boost
        elseif !isnothing(child_id) && symbol == Symbol("node_$(child_id)")
            bias[idx] = child_penalty
        end
    end
    return bias
end

Lux.initialstates(::LehmerRNG, ::EMLTreeLayer) = NamedTuple()

function (layer::EMLTreeLayer)(x, ps, st)
    x_complex = _to_complex_input(x)
    return _evaluate_node(layer, layer.tree.root_id, x_complex, ps), st
end

_to_complex_input(x::Tuple) = map(v -> ComplexF64.(v), x)
_to_complex_input(x) = ComplexF64.(x)

_sample_vector(x::Tuple) = x[1]
_sample_vector(x) = x

function _evaluate_node(layer::EMLTreeLayer, node_id::Int, x, ps)
    node = layer.tree.nodes[node_id]
    node_ps = ps.nodes[node_id]
    left_value = _soft_source_value(layer, node.left_candidates, x, ps, node_ps.left_logits)
    right_value = _soft_source_value(layer, node.right_candidates, x, ps, node_ps.right_logits)
    return clamp_complex_magnitude(_layer_eml(left_value, right_value), _EML_NODE_ABS_LIMIT)
end

function _soft_source_value(layer::EMLTreeLayer, candidates, x, ps, logits)
    values = map(candidates) do symbol
        _resolve_candidate_value(layer, symbol, x, ps)
    end

    shifted = logits .- maximum(logits)
    weights = exp.(shifted)
    probabilities = weights ./ sum(weights)
    terms = map(((w, v),) -> w .* v, zip(probabilities, values))
    return reduce((left, right) -> left .+ right, terms)
end

function _resolve_candidate_value(layer::EMLTreeLayer, symbol::Symbol, x, ps)
    if symbol === :const1
        return fill(1.0 + 0.0im, length(_sample_vector(x)))
    elseif symbol === :x
        return x isa Tuple ? x[1] : x
    elseif symbol === :y
        return x isa Tuple ? x[2] : throw(ArgumentError("terminal y requires binary input"))
    else
        child_id = _child_id_from_symbol(symbol)
        return _evaluate_node(layer, child_id, x, ps)
    end
end

function _child_id_from_symbol(symbol::Symbol)
    return parse(Int, split(String(symbol), "_")[2])
end

function _layer_eml(x, y)
    return eml(_clamp_exp_input(x), y)
end

function _clamp_exp_input(values)
    return map(values) do z
        ComplexF64(clamp(real(z), -_EML_EXP_REAL_LIMIT, _EML_EXP_REAL_LIMIT), imag(z))
    end
end

function _inspect_node_outputs(layer::EMLTreeLayer, x, ps)
    x_complex = _to_complex_input(x)
    outputs = ComplexF64[]
    _collect_node_outputs!(outputs, layer, layer.tree.root_id, x_complex, ps)
    return inspect_complex_values(outputs)
end

function _collect_node_outputs!(outputs, layer::EMLTreeLayer, node_id::Int, x, ps)
    value = _evaluate_node(layer, node_id, x, ps)
    append!(outputs, vec(value))

    node = layer.tree.nodes[node_id]
    for symbol in (node.left_candidates[argmax(ps.nodes[node_id].left_logits)], node.right_candidates[argmax(ps.nodes[node_id].right_logits)])
        if symbol !== :const1 && symbol !== :x && symbol !== :y
            child_id = _child_id_from_symbol(symbol)
            _collect_node_outputs!(outputs, layer, child_id, x, ps)
        end
    end
    return outputs
end

ChainRulesCore.@non_differentiable _child_id_from_symbol(::Any...)

struct EMLTree <: Lux.AbstractLuxLayer
    depth::Int
    n_leaves::Int
    n_internal::Int
    eml_clamp::Float64
    init_strategy::Symbol
    init_scale::Float64
end

function EMLTree(; depth::Int, eml_clamp::Float64=1.0e300, init_strategy::Symbol=:biased, init_scale::Float64=1.0)
    depth >= 1 || throw(ArgumentError("depth must be at least 1"))
    n_leaves = 2^depth
    n_internal = n_leaves - 1
    return EMLTree(depth, n_leaves, n_internal, eml_clamp, init_strategy, init_scale)
end

function Lux.initialparameters(rng::LehmerRNG, tree::EMLTree)
    leaf_logits, blend_logits = initialize_logits(rng, tree.depth; strategy=tree.init_strategy, scale=tree.init_scale)
    return (; leaf_logits, blend_logits)
end

function Lux.initialstates(::LehmerRNG, ::EMLTree)
    return (; tau_leaf=1.0, tau_gate=1.0, leaf_probs=nothing, gate_probs=nothing, eml_outputs=nothing)
end

function initialize_logits(rng::LehmerRNG, depth::Int; strategy::Symbol=:biased, scale::Float64=1.0)
    n_leaves = 2^depth
    n_internal = n_leaves - 1
    if strategy === :manual
        leaf_init = zeros(Float64, n_leaves, 3)
        gate_init = zeros(Float64, n_internal, 2)
    elseif strategy === :biased
        leaf_init = randn(rng, Float64, n_leaves, 3) .* scale
        leaf_init[:, 1] .+= 2.0
        gate_init = randn(rng, Float64, n_internal, 2) .* scale .+ 4.0
    elseif strategy === :uniform
        leaf_init = randn(rng, Float64, n_leaves, 3) .* scale
        gate_init = randn(rng, Float64, n_internal, 2) .* scale .+ 4.0
    elseif strategy === :xy_biased
        leaf_init = randn(rng, Float64, n_leaves, 3) .* scale
        leaf_init[:, 2] .+= 1.0
        leaf_init[:, 3] .+= 1.0
        gate_init = randn(rng, Float64, n_internal, 2) .* scale .+ 4.0
    elseif strategy === :random_hot
        leaf_init = randn(rng, Float64, n_leaves, 3) .* scale
        hot_idx = rand(rng, 1:3, n_leaves)
        for i in 1:n_leaves
            leaf_init[i, hot_idx[i]] += 3.0
        end
        gate_init = randn(rng, Float64, n_internal, 2) .* scale .+ 3.0
        open_mask = rand(rng, n_internal, 2) .< 0.25
        gate_init[open_mask] .-= 6.0
    else
        throw(ArgumentError("unsupported init strategy: $(strategy)"))
    end
    return leaf_init, gate_init
end

function (tree::EMLTree)(xy, ps, st)
    tau_leaf = get(st, :tau_leaf, 1.0)
    tau_gate = get(st, :tau_gate, 1.0)
    pred, _ = forward_with_aux(tree, xy, ps; tau_leaf=tau_leaf, tau_gate=tau_gate)
    return pred, st
end

function forward_with_aux(tree::EMLTree, xy, ps; tau_leaf::Real=1.0, tau_gate::Real=1.0)
    x, y = _to_complex_pair(xy)
    leaf_probs = _softmax_rows(ps.leaf_logits, tau_leaf)
    candidates = hcat(
        fill(1.0 + 0.0im, length(x)),
        x,
        y,
    )
    current_level = Matrix{ComplexF64}(candidates * leaf_probs')
    gate_prob_levels = ()
    eml_output_levels = ()
    node_idx = 1

    while size(current_level, 2) > 1
        n_pairs = size(current_level, 2) ÷ 2
        raw = @view ps.blend_logits[node_idx:(node_idx + n_pairs - 1), :]
        gates = _gate_probs(raw, tau_gate)
        left_children = @view current_level[:, 1:2:size(current_level, 2)]
        right_children = @view current_level[:, 2:2:size(current_level, 2)]
        left_input = _complex_blend(left_children, vec(gates[:, 1]), tree.eml_clamp)
        right_input = _complex_blend(right_children, vec(gates[:, 2]), tree.eml_clamp)
        next_level = _sanitize_and_clamp_matrix(_layer_eml(left_input, right_input), tree.eml_clamp)
        current_level = next_level
        gate_prob_levels = (gate_prob_levels..., Matrix(gates))
        eml_output_levels = (eml_output_levels..., vec(next_level))
        node_idx += n_pairs
    end

    gate_probs = isempty(gate_prob_levels) ? zeros(Float64, 0, 2) : reduce(vcat, gate_prob_levels)
    eml_outputs = isempty(eml_output_levels) ? ComplexF64[] : vcat(eml_output_levels...)
    aux = (; tau_leaf, tau_gate, leaf_probs, gate_probs, eml_outputs)
    return vec(current_level[:, 1]), aux
end

clamp_probs(x::Real; lo::Real=1.0e-12, hi::Real=1.0 - 1.0e-12) = clamp(x, lo, hi)
_stable_sigmoid(x::Real) = x >= 0 ? inv(1.0 + exp(-x)) : (exp(x) / (1.0 + exp(x)))
_gate_probs(raw::AbstractArray{<:Real}, tau_gate::Real) = clamp_probs.(_stable_sigmoid.(raw ./ max(tau_gate, 1.0e-6)))

function _to_complex_pair(xy::Tuple)
    length(xy) == 2 || throw(ArgumentError("EMLTree expects (x, y) inputs"))
    return ComplexF64.(xy[1]), ComplexF64.(xy[2])
end

function _softmax_rows(logits::AbstractMatrix{<:Real}, tau::Real)
    scaled = logits ./ max(tau, 1.0e-6)
    shifted = scaled .- maximum(scaled; dims=2)
    weights = exp.(shifted)
    return weights ./ sum(weights; dims=2)
end

function _complex_blend(children::AbstractMatrix{ComplexF64}, gate_column::AbstractVector{<:Real}, limit::Real)
    gates = reshape(gate_column, 1, :)
    oml = 1.0 .- gates
    real_part = _sanitize_scalar.(real.(children), limit)
    imag_part = _sanitize_scalar.(imag.(children), limit)
    return ComplexF64.(gates .+ oml .* real_part, oml .* imag_part)
end

function _complex_blend(child::AbstractVector{ComplexF64}, gate::Real, limit::Real)
    oml = 1.0 - gate
    return map(child) do z
        rz = _sanitize_scalar(real(z), limit)
        iz = _sanitize_scalar(imag(z), limit)
        ComplexF64(gate + oml * rz, oml * iz)
    end
end

function _sanitize_and_clamp_matrix(values::AbstractMatrix{ComplexF64}, limit::Real)
    return map(values) do z
        complex(_sanitize_scalar(real(z), limit), _sanitize_scalar(imag(z), limit))
    end
end

function _sanitize_and_clamp_vector(values::AbstractVector{ComplexF64}, limit::Real)
    return map(values) do z
        complex(_sanitize_scalar(real(z), limit), _sanitize_scalar(imag(z), limit))
    end
end

function _sanitize_scalar(x::Real, limit::Real)
    if isnan(x)
        return 0.0
    elseif isinf(x)
        return signbit(x) ? -limit : limit
    else
        return clamp(x, -limit, limit)
    end
end

function init_from_expr!(ps, expr::AbstractString; k::Float64=32.0)
    parsed = parse_eml_expr(expr)
    depth = _expr_depth(parsed)
    size(ps.leaf_logits, 1) == 2^depth || throw(ArgumentError("expression depth $(depth) does not match parameter tree"))

    fill!(ps.leaf_logits, -k)
    ps.leaf_logits[:, 1] .= k
    fill!(ps.blend_logits, k)

    recurse(node, level, pos) = _init_expr_recurse!(ps, node, depth, level, pos, k)
    recurse(parsed, 0, 1)
    return parsed
end

function parse_eml_expr(s::AbstractString)
    stripped = strip(s)
    stripped in ("1", "x", "y") && return stripped
    if startswith(stripped, "EML[") && endswith(stripped, "]")
        inner = stripped[5:end-1]
        depth = 0
        for (idx, ch) in enumerate(inner)
            if ch == '['
                depth += 1
            elseif ch == ']'
                depth -= 1
            elseif ch == ',' && depth == 0
                return ("EML", parse_eml_expr(inner[1:idx-1]), parse_eml_expr(inner[idx+1:end]))
            end
        end
    end
    throw(ArgumentError("Cannot parse EML expression: $(s)"))
end

function _expr_depth(node)
    node isa AbstractString && return 0
    return 1 + max(_expr_depth(node[2]), _expr_depth(node[3]))
end

function _flat_node_idx(tree_depth::Int, level_from_top::Int, pos_in_level::Int)
    return 2^tree_depth - 2^(level_from_top + 1) + pos_in_level
end

function _init_expr_recurse!(ps, node, tree_depth::Int, level::Int, pos::Int, k::Float64)
    if node isa AbstractString
        if level == tree_depth
            choice_map = Dict("1" => 1, "x" => 2, "y" => 3)
            ps.leaf_logits[pos, :] .= -k
            ps.leaf_logits[pos, choice_map[node]] = k
        end
        return
    end

    _, left, right = node
    node_idx = _flat_node_idx(tree_depth, level, pos)
    left_is_one = left isa AbstractString && left == "1"
    right_is_one = right isa AbstractString && right == "1"
    ps.blend_logits[node_idx, 1] = left_is_one ? k : -k
    ps.blend_logits[node_idx, 2] = right_is_one ? k : -k

    left_is_one || _init_expr_recurse!(ps, left, tree_depth, level + 1, 2 * pos - 1, k)
    right_is_one || _init_expr_recurse!(ps, right, tree_depth, level + 1, 2 * pos, k)
    return
end
