using Test
using EMLRegression

@testset "snap trained logits" begin
    snapped = snap_logits([10.0, -10.0, -10.0])
    @test snapped == [1.0, 0.0, 0.0]
end

@testset "recovered tree semantics" begin
    recovered_x = RecoveredTree([:const1, :x], [0.0, 1.0])
    xs = ComplexF64[0.0 + 0.0im, 1.0 + 0.0im]

    @test evaluate_recovered(recovered_x, xs) ≈ exp.(xs)
    @test formula_string(recovered_x) == "eml(x, 1)"

    recovered_const = RecoveredTree([:const1, :x], [1.0, 0.0])
    @test evaluate_recovered(recovered_const, xs) == fill(exp(1.0 + 0.0im), length(xs))
    @test formula_string(recovered_const) == "eml(1, 1)"

    recovered_y = RecoveredTree([:const1, :x, :y], [0.0, 0.0, 1.0])
    xys = (
        ComplexF64[0.5 + 0.0im, 1.0 + 0.0im],
        ComplexF64[1.5 + 0.0im, 2.0 + 0.0im],
    )
    @test evaluate_recovered(recovered_y, xys) ≈ exp.(xys[2])
    @test formula_string(recovered_y) == "eml(y, 1)"
end
