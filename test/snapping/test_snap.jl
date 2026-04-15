using Test
using EMLRegression

@testset "snap trained logits" begin
    snapped = snap_logits([10.0, -10.0, -10.0])
    @test snapped == [1.0, 0.0, 0.0]
end

@testset "snap_model returns node-wise choices" begin
    tree = build_master_tree(depth=2, variables=(:x,))
    layer = EMLTreeLayer(tree)
    ps = (
        nodes=(
            (left_logits=[0.0, 0.0, 3.0], right_logits=[3.0, 0.0, 0.0]),
            (left_logits=[0.0, 2.0], right_logits=[2.0, 0.0]),
            (left_logits=[2.0, 0.0], right_logits=[0.0, 2.0]),
        ),
    )
    recovered = snap_model(layer, ps; margin_threshold=0.1)
    @test recovered.choices[1] == (:node_2, :const1)
    @test recovered.choices[2] == (:x, :const1)
end

@testset "recovered tree semantics" begin
    depth1 = build_master_tree(depth=1, variables=(:x,))
    recovered_x = RecoveredTree(Dict(1 => (:x, :const1)))
    xs = ComplexF64[0.0 + 0.0im, 1.0 + 0.0im]
    @test evaluate_recovered(recovered_x, depth1, xs) ≈ exp.(xs)
    @test formula_string(recovered_x, depth1) == "eml(x, 1)"

    depth2 = build_master_tree(depth=2, variables=(:x,))
    recovered_nested = RecoveredTree(Dict(1 => (:node_2, :const1), 2 => (:x, :const1), 3 => (:const1, :x)))
    @test evaluate_recovered(recovered_nested, depth2, xs) ≈ exp.(exp.(xs))
    @test formula_string(recovered_nested, depth2) == "eml(eml(x, 1), 1)"
end

@testset "structure match only compares active nodes" begin
    expected = RecoveredTree(Dict(
        1 => (:node_2, :const1),
        2 => (:x, :const1),
        3 => (:const1, :const1),
    ))
    actual = RecoveredTree(Dict(
        1 => (:node_2, :const1),
        2 => (:x, :const1),
        3 => (:x, :const1),
    ))
    @test EMLRegression.structure_match(expected, actual)
end

@testset "snap diagnostics ignore inactive ambiguous nodes" begin
    tree = build_master_tree(depth=2, variables=(:x,))
    layer = EMLTreeLayer(tree)
    ps = (
        nodes=(
            (left_logits=[0.0, 0.0, 3.0], right_logits=[3.0, 0.0, 0.0]),
            (left_logits=[0.0, 2.0], right_logits=[2.0, 0.0]),
            (left_logits=[0.0, 0.0], right_logits=[0.0, 0.0]),
        ),
    )

    @test EMLRegression.snap_status(layer, ps; margin_threshold=0.5) == :ok
    @test EMLRegression.ambiguous_node_count(layer, ps; margin_threshold=0.5) == 0
end

@testset "refine_recovered_tree prefers lower-complexity equivalent structure" begin
    tree = build_master_tree(depth=3, variables=(:x,))
    target = get_target(:depth3_log)
    xs = ComplexF64[0.25 + 0.0im, 0.5 + 0.0im, 1.0 + 0.0im, 2.0 + 0.0im]
    ys = evaluate_target(target, xs)

    initial = RecoveredTree(Dict(
        1 => (:x, :node_3),
        2 => (:const1, :const1),
        3 => (:node_6, :const1),
        4 => (:const1, :const1),
        5 => (:const1, :const1),
        6 => (:x, :x),
        7 => (:const1, :const1),
    ))

    refined = EMLRegression.refine_recovered_tree(initial, tree, xs, ys)
    @test EMLRegression.structure_match(target.tree, refined)
end

@testset "refine_recovered_tree left-packs equivalent subtrees for depth4 target" begin
    tree = build_master_tree(depth=4, variables=(:x, :y))
    target = get_target(:depth4_nested)
    xs = sample_domain(target, 64; rng_seed=1)
    ys = evaluate_target(target, xs)

    initial = RecoveredTree(Dict(
        1 => (:node_2, :node_3),
        2 => (:x, :const1),
        3 => (:const1, :node_7),
        4 => (:const1, :const1),
        5 => (:const1, :const1),
        6 => (:const1, :const1),
        7 => (:node_14, :y),
        8 => (:const1, :const1),
        9 => (:const1, :const1),
        10 => (:const1, :const1),
        11 => (:const1, :const1),
        12 => (:const1, :const1),
        13 => (:const1, :const1),
        14 => (:x, :const1),
        15 => (:const1, :const1),
    ))

    refined = EMLRegression.refine_recovered_tree(initial, tree, xs, ys)
    @test EMLRegression.structure_match(target.tree, refined)
end
