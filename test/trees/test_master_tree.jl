using Test
using EMLRegression

@testset "master tree structure" begin
    tree = build_master_tree(depth=2, variables=(:x,))
    @test tree.depth == 2
    @test length(tree.nodes) == 3
    @test tree.root_id == 1
    @test :const1 in tree.terminals
    @test :x in tree.terminals
    @test :const1 in tree.nodes[2].left_candidates
    @test :x in tree.nodes[2].left_candidates
    @test :node_2 in tree.nodes[1].left_candidates
    @test :node_3 in tree.nodes[1].right_candidates
end

@testset "recovered tree stores node-wise choices" begin
    recovered = RecoveredTree(Dict(1 => (:node_2, :node_3), 2 => (:x, :const1), 3 => (:const1, :x)))
    @test recovered.choices[2] == (:x, :const1)
end
