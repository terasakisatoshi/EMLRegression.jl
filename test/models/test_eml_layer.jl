using Test
using Lux
using StableRNGs
using EMLRegression

@testset "paper-faithful tree parameter shapes" begin
    tree = EMLTree(depth=3)
    rng = StableRNG(1)
    ps, _ = Lux.setup(rng, tree)
    @test size(ps.leaf_logits) == (8, 3)
    @test size(ps.blend_logits) == (7, 2)
end

@testset "paper-faithful forward returns root prediction and diagnostics" begin
    tree = EMLTree(depth=2)
    rng = StableRNG(1)
    ps, st = Lux.setup(rng, tree)
    x = ComplexF64[1.0 + 0.0im, 2.0 + 0.0im]
    y = ComplexF64[1.5 + 0.0im, 2.5 + 0.0im]
    pred, aux = Lux.apply(tree, (x, y), ps, st)
    @test length(pred) == 2
    @test haskey(aux, :leaf_probs)
    @test haskey(aux, :gate_probs)
    @test haskey(aux, :eml_outputs)
end

@testset "manual init helper validates depth and populates logits" begin
    tree = EMLTree(depth=2, init_strategy=:manual)
    rng = StableRNG(1)
    ps, _ = Lux.setup(rng, tree)
    EMLRegression.init_from_expr!(ps, "EML[1, EML[x, y]]"; k=32.0)
    @test maximum(ps.leaf_logits) == 32.0
    @test minimum(ps.leaf_logits) == -32.0
end

@testset "complex blend bypasses exact one gate to avoid nan cross terms" begin
    child = ComplexF64[Inf + 2.0im, 3.0 + 4.0im]
    blended = EMLRegression._complex_blend(child, 1.0)
    @test blended[1] == 1.0 + 0.0im
    @test blended[2] == 1.0 + 0.0im
end
