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
