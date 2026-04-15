using Lux
using StableRNGs: LehmerRNG

"""
    EMLTreeLayer(tree)

`MasterTree` を `Lux.jl` のレイヤとして評価する最小実装です。
"""
struct EMLTreeLayer{T} <: Lux.AbstractLuxLayer
    tree::T
end

function Lux.initialparameters(rng::LehmerRNG, layer::EMLTreeLayer)
    node_params = Tuple(
        (
            left_logits=zeros(Float64, length(node.left_candidates)),
            right_logits=zeros(Float64, length(node.right_candidates)),
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
    return eml(left_value, right_value)
end

function _soft_source_value(layer::EMLTreeLayer, candidates, x, ps, logits)
    values = map(candidates) do symbol
        _resolve_candidate_value(layer, symbol, x, ps)
    end

    shifted = logits .- maximum(logits)
    weights = exp.(shifted)
    weights ./= sum(weights)

    total = zero(values[1])
    for (w, v) in zip(weights, values)
        total .+= w .* v
    end
    return total
end

function _resolve_candidate_value(layer::EMLTreeLayer, symbol::Symbol, x, ps)
    if symbol === :const1
        return fill(1.0 + 0.0im, length(_sample_vector(x)))
    elseif symbol === :x
        return x isa Tuple ? x[1] : x
    elseif symbol === :y
        return x isa Tuple ? x[2] : throw(ArgumentError("terminal y requires binary input"))
    elseif startswith(String(symbol), "node_")
        child_id = parse(Int, split(String(symbol), "_")[2])
        return _evaluate_node(layer, child_id, x, ps)
    else
        throw(ArgumentError("unsupported terminal $(symbol)"))
    end
end
