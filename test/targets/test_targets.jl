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
