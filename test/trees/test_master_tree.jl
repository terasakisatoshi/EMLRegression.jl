using Test
using EMLRegression

@testset "master tree structure" begin
    tree = build_master_tree(depth=2, variables=(:x,))
    @test tree.depth == 2
    @test length(tree.nodes) == 3
    @test :const1 in tree.terminals
    @test :x in tree.terminals
end
