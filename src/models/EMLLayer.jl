using Lux
using ComponentArrays
using StableRNGs: LehmerRNG

struct EMLTreeLayer{T} <: Lux.AbstractLuxLayer
    tree::T
end

function Lux.initialparameters(rng::LehmerRNG, layer::EMLTreeLayer)
    logits = zeros(Float64, length(layer.tree.terminals))
    return ComponentArray(logits=logits)
end

Lux.initialstates(::LehmerRNG, ::EMLTreeLayer) = NamedTuple()

function (layer::EMLTreeLayer)(x, ps, st)
    x_complex = ComplexF64.(x)
    weighted = _weighted_terminal_choice(layer.tree.terminals, x_complex, ps.logits)
    ones_input = fill(1.0 + 0.0im, length(x_complex))
    return eml(weighted, ones_input), st
end

function _weighted_terminal_choice(terminals, x, logits)
    values = map(terminals) do symbol
        if symbol === :const1
            fill(1.0 + 0.0im, length(x))
        elseif symbol === :x
            x
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
