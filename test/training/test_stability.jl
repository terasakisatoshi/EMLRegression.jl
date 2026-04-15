using Test
using EMLRegression

@testset "stability helpers" begin
    z = ComplexF64[1 + 0im, Inf + 0im]
    report = inspect_complex_values(z)
    @test report.has_inf
    @test !report.has_nan
end
