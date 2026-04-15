using StableRNGs: StableRNG

struct TargetSpec
    name::Symbol
    arity::Int
    tier::Symbol
    sampler::Function
    evaluator::Function
end

function _sample_positive(rng, n)
    return ComplexF64.(0.25 .+ rand(rng, n) .* 2.0)
end

function _sample_centered(rng, n)
    return ComplexF64.(-1.0 .+ rand(rng, n) .* 2.0)
end

const TARGETS = Dict{Symbol,TargetSpec}(
    :exp => TargetSpec(:exp, 1, :must_pass, (rng, n) -> _sample_centered(rng, n), xs -> exp.(xs)),
    :ln => TargetSpec(:ln, 1, :must_pass, (rng, n) -> _sample_positive(rng, n), xs -> log.(xs)),
    :neg => TargetSpec(:neg, 1, :must_pass, (rng, n) -> _sample_centered(rng, n), xs -> .-xs),
    :inv => TargetSpec(:inv, 1, :must_pass, (rng, n) -> _sample_positive(rng, n), xs -> inv.(xs)),
    :add => TargetSpec(:add, 2, :must_pass, (rng, n) -> (_sample_centered(rng, n), _sample_centered(rng, n)), xy -> xy[1] .+ xy[2]),
    :mul => TargetSpec(:mul, 2, :must_pass, (rng, n) -> (_sample_centered(rng, n), _sample_centered(rng, n)), xy -> xy[1] .* xy[2]),
)

function get_target(name::Symbol)
    return TARGETS[name]
end

function sample_domain(target::TargetSpec, n::Integer; rng_seed::Integer=1)
    rng = StableRNG(rng_seed)
    return target.sampler(rng, n)
end

evaluate_target(target::TargetSpec, xs) = target.evaluator(xs)
