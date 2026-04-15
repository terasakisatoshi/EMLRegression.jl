using Test

@testset "project loads" begin
    using EMLRegression
    @test isdefined(EMLRegression, :eml)
end
