using Test
using EMLRegression

@testset "recovery judgment" begin
    verdict = recovery_verdict(
        train_loss=1e-12,
        validation_loss=1e-12,
        snap_status=:ok,
        structure_match=true,
        ambiguous_nodes=0,
        numerical_match=true,
    )
    @test verdict.success
    @test verdict.structure_match
    @test verdict.ambiguous_nodes == 0
    @test verdict.validation_loss == 1e-12
end

@testset "recovery does not succeed on numerical match when snap is ambiguous" begin
    verdict = recovery_verdict(
        train_loss=1e-12,
        validation_loss=1e-12,
        snap_status=:ambiguous,
        structure_match=false,
        ambiguous_nodes=1,
        numerical_match=true,
    )
    @test !verdict.success
    @test verdict.reason == :snap_failed
    @test verdict.snap_status == :ambiguous
end

@testset "recovery does not succeed on numerical match when structure is wrong" begin
    verdict = recovery_verdict(
        train_loss=1e-12,
        validation_loss=1e-12,
        snap_status=:ok,
        structure_match=false,
        ambiguous_nodes=0,
        numerical_match=true,
    )
    @test !verdict.success
    @test verdict.reason == :mismatch
end
