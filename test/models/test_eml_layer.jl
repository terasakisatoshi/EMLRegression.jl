using Test
using Lux
using StableRNGs
using EMLRegression

@testset "EML layer forward pass" begin
    layer = EMLTreeLayer(build_master_tree(depth=2, variables=(:x,)))
    rng = StableRNG(1)
    ps, st = Lux.setup(rng, layer)
    x = ComplexF64[1.0 + 0im, 2.0 + 0im]
    y, st2 = Lux.apply(layer, x, ps, st)
    @test length(y) == 2
    @test eltype(y) == ComplexF64
    @test st2 == st
end
