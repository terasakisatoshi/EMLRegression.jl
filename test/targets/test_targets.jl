using Test
using EMLRegression

@testset "target registry" begin
    target = get_target(:ln)
    xs = sample_domain(target, 16; rng_seed=1)
    ys = evaluate_target(target, xs)
    @test length(xs) == 16
    @test length(ys) == 16
    @test eltype(ys) == ComplexF64
end

@testset "challenge targets" begin
    for name in (:div, :square, :sqrt, :sin, :cos, :tan, :logxy)
        target = get_target(name)
        xs = sample_domain(target, 8; rng_seed=2)
        ys = evaluate_target(target, xs)
        @test !isempty(ys)
    end
end
