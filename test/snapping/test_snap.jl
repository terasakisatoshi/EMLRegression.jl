using Test
using EMLRegression

@testset "analyze_snap reports uncertain leaves and gates" begin
    ps = (
        leaf_logits=[
            8.0 0.0 0.0
            0.4 0.2 0.1
        ],
        blend_logits=[
            5.0 -5.0
            0.0 5.0
        ],
    )

    info = analyze_snap(ps; snap_threshold=0.01)

    @test info.n_uncertain == 2
    @test length(info.uncertain_leaves) == 1
    @test length(info.uncertain_gates) == 1
    @test occursin("leaf[1]", info.uncertain_leaves[1])
    @test occursin("gate[1].left", info.uncertain_gates[1])
end

@testset "hard_project snaps logits to paper-style hard values" begin
    ps = (
        leaf_logits=[
            4.0 0.0 0.0
            0.1 2.0 -1.0
        ],
        blend_logits=[
            0.5 -0.5
            -1.0 3.0
        ],
    )

    snapped = hard_project(ps; k=24.0)

    @test snapped.leaf_logits == [
        24.0 -24.0 -24.0
        -24.0 24.0 -24.0
    ]
    @test snapped.blend_logits == [
        24.0 -24.0
        -24.0 24.0
    ]
    @test analyze_snap(snapped; snap_threshold=0.01).n_uncertain == 0
end

@testset "analyze_snap stays finite for overflow-prone gate logits" begin
    ps = (
        leaf_logits=fill(6.0, 4, 3),
        blend_logits=[
            0.0 -1000.0
            -1000.0 1000.0
            0.5 -0.5
        ],
    )

    info = analyze_snap(ps; snap_threshold=0.01)

    @test info.n_uncertain >= 0
    @test isempty(filter(msg -> occursin("NaN", msg) || occursin("Inf", msg), info.uncertain_gates))
end
