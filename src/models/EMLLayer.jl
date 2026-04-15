using ChainRulesCore
using Lux
using StableRNGs: LehmerRNG

"""
    EMLTreeLayer(tree)

`MasterTree` を `Lux.jl` のレイヤとして評価する最小実装です。
"""
struct EMLTreeLayer{T} <: Lux.AbstractLuxLayer
    tree::T
end

const _EML_EXP_REAL_LIMIT = 40.0

function Lux.initialparameters(rng::LehmerRNG, layer::EMLTreeLayer)
    node_params = Tuple(
        (
            left_logits=0.01 .* randn(rng, Float64, length(node.left_candidates)),
            right_logits=0.01 .* randn(rng, Float64, length(node.right_candidates)),
        ) for node in layer.tree.nodes
    )
    return (; nodes=node_params)
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
    return _layer_eml(left_value, right_value)
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
