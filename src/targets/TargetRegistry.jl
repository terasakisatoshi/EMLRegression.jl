using StableRNGs: StableRNG

struct TargetSpec
    name::Symbol
    arity::Int
    tier::Symbol
    depth::Int
    fn::Function
    description::String
end

function _target_eml_depth2(x, y)
    return exp(1.0) - log(exp(y) - log(x))
end

function _target_eml_depth3(x, y)
    return exp(exp(1.0)) / (exp(y) - log(x))
end

function _target_eml_depth4(x, y)
    return log(exp(x) - log(y))
end

function _target_eml_depth5(x, y)
    return log(exp(1.0) - log(exp(x) - log(y)))
end

function _target_eml_depth6(x, y)
    return exp(1.0) - y * exp(exp(1.0) - exp(x))
end

const TARGETS = Dict{Symbol,TargetSpec}(
    :eml_depth2 => TargetSpec(:eml_depth2, 2, :must_pass, 2, _target_eml_depth2, "e - log(exp(y) - log(x))"),
    :eml_depth3 => TargetSpec(:eml_depth3, 2, :must_pass, 3, _target_eml_depth3, "e^e/(e^y - log(x))"),
    :eml_depth4 => TargetSpec(:eml_depth4, 2, :challenge, 4, _target_eml_depth4, "log(e^x - log(y))"),
    :eml_depth5 => TargetSpec(:eml_depth5, 2, :challenge, 5, _target_eml_depth5, "log(e-log(e^x - log(y)))"),
    :eml_depth6 => TargetSpec(:eml_depth6, 2, :challenge, 6, _target_eml_depth6, "e - y*exp(e - exp(x))"),
)

get_target(name::Symbol) = TARGETS[name]

function filter_real_domain(x, y, target_fn; imag_tol::Float64=1.0e-12)
    xc = ComplexF64.(x)
    yc = ComplexF64.(y)
    values = map(target_fn, xc, yc)
    mask = map(values) do z
        isfinite(real(z)) && abs(imag(z)) < imag_tol
    end
    return x[mask], y[mask], ComplexF64.(real.(values[mask]))
end

function make_grid_data(target_fn; lo::Float64=1.0, hi::Float64=3.0, step::Float64=0.1)
    xs = collect(lo:step:hi)
    ys = collect(lo:step:hi)
    xx = Float64[]
    yy = Float64[]
    for x in xs, y in ys
        push!(xx, x)
        push!(yy, y)
    end
    return filter_real_domain(xx, yy, target_fn)
end

function make_generalization_data(target_fn; lo::Float64=0.5, hi::Float64=5.0, n::Int=4000, seed::Int=12345)
    rng = StableRNG(seed)
    x = lo .+ (hi - lo) .* rand(rng, n)
    y = lo .+ (hi - lo) .* rand(rng, n)
    return filter_real_domain(x, y, target_fn)
end

function sample_domain(target::TargetSpec, n::Integer; rng_seed::Union{Integer,Nothing}=1, rng=nothing)
    if !isnothing(rng)
        x = 1.0 .+ 2.0 .* rand(rng, n)
        y = 1.0 .+ 2.0 .* rand(rng, n)
        return x, y
    end
    seeded_rng = StableRNG(something(rng_seed, 1))
    x = 1.0 .+ 2.0 .* rand(seeded_rng, n)
    y = 1.0 .+ 2.0 .* rand(seeded_rng, n)
    return x, y
end

function evaluate_target(target::TargetSpec, xs)
    x, y = xs
    return ComplexF64.(map(target.fn, ComplexF64.(x), ComplexF64.(y)))
end
