using Test
using Lux
using StableRNGs
using EMLRegression

@testset "EML layer parameters are node-wise" begin
    layer = EMLTreeLayer(build_master_tree(depth=2, variables=(:x,)))
    rng = StableRNG(1)
    ps, st = Lux.setup(rng, layer)
    @test haskey(ps, :nodes)
    @test haskey(ps.nodes[1], :left_logits)
    @test haskey(ps.nodes[1], :right_logits)
    @test st == NamedTuple()
end

@testset "EML layer evaluates recursively" begin
    layer = EMLTreeLayer(build_master_tree(depth=2, variables=(:x,)))
    rng = StableRNG(1)
    ps, st = Lux.setup(rng, layer)
    x = ComplexF64[1.0 + 0im, 2.0 + 0im]
    y, st2 = Lux.apply(layer, x, ps, st)
    @test length(y) == 2
    @test eltype(y) == ComplexF64
    @test st2 == st
end

@testset "EML layer initialization depends on rng" begin
    layer = EMLTreeLayer(build_master_tree(depth=2, variables=(:x,)))
    ps1, _ = Lux.setup(StableRNG(1), layer)
    ps2, _ = Lux.setup(StableRNG(2), layer)
    @test ps1 != ps2
    @test any(!iszero, ps1.nodes[1].left_logits)
end

@testset "EML layer clamps exp inputs to avoid overflow" begin
    layer = EMLTreeLayer(build_master_tree(depth=2, variables=(:x,)))
    ps = (
        nodes=(
            (left_logits=[12.0, -12.0, -12.0], right_logits=[12.0, -12.0, -12.0]),
            (left_logits=[0.0, 0.0], right_logits=[0.0, 0.0]),
            (left_logits=[0.0, 0.0], right_logits=[0.0, 0.0]),
        ),
    )
    st = NamedTuple()
    x = ComplexF64[1000.0 + 0.0im]
    y, _ = Lux.apply(layer, x, ps, st)
    @test isfinite(real(only(y)))
    @test isfinite(imag(only(y)))
end

@testset "EML layer can inspect node outputs" begin
    layer = EMLTreeLayer(build_master_tree(depth=2, variables=(:x,)))
    ps = (
        nodes=(
            (left_logits=[12.0, -12.0, -12.0], right_logits=[12.0, -12.0, -12.0]),
            (left_logits=[0.0, 0.0], right_logits=[0.0, 0.0]),
            (left_logits=[0.0, 0.0], right_logits=[0.0, 0.0]),
        ),
    )
    report = EMLRegression._inspect_node_outputs(layer, ComplexF64[1000.0 + 0.0im], ps)
    @test !report.has_nan
    @test !report.has_inf
    @test isfinite(report.max_abs)
end
