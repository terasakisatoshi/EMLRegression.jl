using Test
using EMLRegression

@testset "recovery judgment" begin
    verdict = recovery_verdict(
        train_loss=1e-12,
        validation_loss=1e-12,
        snap_status=:ok,
        numerical_match=true,
    )
    @test verdict.success
end
