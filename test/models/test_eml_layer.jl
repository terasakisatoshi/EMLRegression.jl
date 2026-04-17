using Test
using Lux
using StableRNGs
using Zygote
using EMLRegression

function saturated_gate_fixture(tree)
    rng = StableRNG(1)
    ps, st = Lux.setup(rng, tree)
    fill!(ps.leaf_logits, 1.0e6)
    fill!(ps.blend_logits, 1.0e6)
    return ps, st
end

fixture_inputs() = (
    ComplexF64[1.0 + 0.0im, 2.0 + 0.0im],
    ComplexF64[1.5 + 0.0im, 2.5 + 0.0im],
)

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
    ps, _ = Lux.setup(rng, tree)
    x = ComplexF64[1.0 + 0.0im, 2.0 + 0.0im]
    y = ComplexF64[1.5 + 0.0im, 2.5 + 0.0im]
    pred, aux = EMLRegression.forward_with_aux(tree, (x, y), ps)
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

@testset "complex blend clamps large child values before blending" begin
    child = ComplexF64[Inf + 2.0im, 3.0 + 4.0im]
    blended = EMLRegression._complex_blend(child, 1.0, 1.0e6)
    @test blended[1] == 1.0 + 0.0im
    @test blended[2] == 1.0 + 0.0im
end

@testset "saturated gates still produce bounded forward outputs" begin
    tree = EMLTree(depth=3, eml_clamp=1.0e6)
    ps, _ = saturated_gate_fixture(tree)
    ŷ, _ = EMLRegression.forward_with_regularizers(tree, fixture_inputs(), ps)
    @test all(isfinite, real.(ŷ))
    @test all(isfinite, imag.(ŷ))
end

@testset "paper regularizers default to linear uncertainty weighting" begin
    tree = EMLTree(depth=2, eml_clamp=1.0e6)
    ps = (
        leaf_logits=[
            1.2 0.4 -0.6;
            -0.2 1.0 0.3;
            0.7 -0.1 0.2;
            0.0 0.5 -0.4
        ],
        blend_logits=[
            1.4 -0.6;
            0.8 0.2;
            -0.3 1.1
        ],
    )
    _, regs_default = EMLRegression.forward_with_regularizers(tree, fixture_inputs(), ps)
    _, regs_linear = EMLRegression.forward_with_regularizers(tree, fixture_inputs(), ps; uncertainty_power=1.0)
    @test regs_default.entropy ≈ regs_linear.entropy atol=1.0e-12
    @test regs_default.binarity ≈ regs_linear.binarity atol=1.0e-12
    @test regs_default.ambiguity ≈ regs_linear.ambiguity atol=1.0e-12
end

@testset "extreme saturated gates keep finite gradients in hardening tail" begin
    tree = EMLTree(depth=2, eml_clamp=1.0e6)
    ps = (
        leaf_logits=[
            6.0 -6.0 -6.0;
            6.0 -6.0 -6.0;
            6.0 -6.0 -6.0;
            6.0 -6.0 -6.0
        ],
        blend_logits=[
            0.0 -12.0;
            0.0 0.0;
            0.0 0.0
        ],
    )
    xy = (collect(1.0:0.5:2.0), collect(1.0:0.5:2.0))
    target = ComplexF64.(fill(1.5, length(xy[1])))
    tau = 0.016604154655958138

    loss(ps_current) = begin
        pred, _ = EMLRegression.forward_with_aux(tree, xy, ps_current; tau_leaf=tau, tau_gate=tau)
        sum(abs2, pred .- target) / length(target)
    end

    grads = only(Zygote.gradient(loss, ps))
    @test all(isfinite, grads.blend_logits)
end

@testset "depth4 random_hot initialization keeps paper-budget forward losses finite" begin
    cfg = TrainConfig(target=:eml_depth4, depth=4, init_strategy=:random_hot, seed=137)
    rng = StableRNG(cfg.seed)
    target = get_target(cfg.target)
    x_train, y_train, t_train = make_grid_data(target.fn; lo=cfg.data_lo, hi=cfg.data_hi, step=cfg.data_step)
    tree = EMLTree(depth=cfg.depth, init_strategy=cfg.init_strategy, init_scale=cfg.init_scale, eml_clamp=cfg.eml_clamp)
    ps, _ = Lux.setup(rng, tree)
    EMLRegression._apply_initialization!(rng, ps, cfg)

    pred, regs = EMLRegression.forward_with_regularizers(
        tree,
        (x_train, y_train),
        ps;
        tau_leaf=cfg.tau_search,
        tau_gate=cfg.tau_search,
        inter_threshold=cfg.inter_threshold,
    )
    data_loss = EMLRegression._data_loss(pred, t_train)

    @test isfinite(data_loss)
    @test isfinite(regs.inter_penalty)
end
