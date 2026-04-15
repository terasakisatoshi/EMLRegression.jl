using Lux
using ComponentArrays
using StableRNGs: LehmerRNG

"""
    EMLTreeLayer(tree)

`MasterTree` を `Lux.jl` のレイヤとして評価する最小実装です。
"""
struct EMLTreeLayer{T} <: Lux.AbstractLuxLayer
    tree::T
end

function Lux.initialparameters(rng::LehmerRNG, layer::EMLTreeLayer)
    logits = zeros(Float64, length(layer.tree.terminals))
    return ComponentArray(logits=logits)
end

Lux.initialstates(::LehmerRNG, ::EMLTreeLayer) = NamedTuple()

function (layer::EMLTreeLayer)(x, ps, st)
    x_complex = _to_complex_input(x)
    weighted = _weighted_terminal_choice(layer.tree.terminals, x_complex, ps.logits)
    sample_vector = _sample_vector(x_complex)
    ones_input = fill(1.0 + 0.0im, length(sample_vector))
    return eml(weighted, ones_input), st
end

_to_complex_input(x::Tuple) = map(v -> ComplexF64.(v), x)
_to_complex_input(x) = ComplexF64.(x)

_sample_vector(x::Tuple) = x[1]
_sample_vector(x) = x

function _weighted_terminal_choice(terminals, x, logits)
    values = map(terminals) do symbol
        if symbol === :const1
            fill(1.0 + 0.0im, length(_sample_vector(x)))
        elseif symbol === :x
            x isa Tuple ? x[1] : x
        elseif symbol === :y
            x isa Tuple ? x[2] : throw(ArgumentError("terminal y requires binary input"))
        else
            throw(ArgumentError("unsupported terminal $(symbol)"))
        end
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
