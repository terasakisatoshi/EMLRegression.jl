using Test
using StableRNGs
using EMLRegression

@testset "target registry" begin
    target = get_target(:depth2_exp)
    xs = sample_domain(target, 16; rng_seed=1)
    ys = evaluate_target(target, xs)
    @test length(xs) == 16
    @test length(ys) == 16
    @test eltype(ys) == ComplexF64
    @test target.depth == 2
    @test !isempty(target.tree.choices)
end

@testset "paper-aligned benchmark targets" begin
    for name in (:depth2_exp, :depth2_double_exp, :depth3_log, :depth4_nested)
        target = get_target(name)
        xs = sample_domain(target, 8; rng_seed=2)
        ys = evaluate_target(target, xs)
        @test !isempty(ys)
        @test target.tier in (:must_pass, :challenge)
    end
end

@testset "target sampling can use an explicit rng" begin
    target = get_target(:depth3_log)
    xs1 = sample_domain(target, 8; rng=StableRNG(1))
    xs2 = sample_domain(target, 8; rng=StableRNG(2))
    @test xs1 != xs2
end
