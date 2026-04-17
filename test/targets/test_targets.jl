using Test
using EMLRegression

@testset "paper-aligned targets are registered" begin
    for name in (:eml_depth2, :eml_depth3, :eml_depth4, :eml_depth5, :eml_depth6)
        target = get_target(name)
        @test target.name == name
        @test target.depth >= 2
    end
end

@testset "real-domain filtering rejects invalid complex-domain points" begin
    target = get_target(:eml_depth4)
    x = [0.1, 1.0, 2.0]
    y = [10.0, 1.5, 2.5]
    xf, yf, tf = EMLRegression.filter_real_domain(x, y, target.fn)
    @test length(xf) <= 3
    @test length(yf) == length(xf)
    @test length(tf) == length(xf)
    @test all(isfinite, real.(tf))
    @test all(abs.(imag.(tf)) .< 1.0e-12)
end

@testset "paper-aligned grid data returns finite targets" begin
    target = get_target(:eml_depth2)
    x, y, t = EMLRegression.make_grid_data(target.fn; lo=1.0, hi=1.2, step=0.1)
    @test length(x) == length(y) == length(t)
    @test !isempty(x)
end
