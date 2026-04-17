using Test

@testset "parallel test runner is configured" begin
    repo_root = normpath(joinpath(@__DIR__, ".."))
    runtests_text = read(joinpath(repo_root, "test", "runtests.jl"), String)
    project_text = read(joinpath(repo_root, "Project.toml"), String)

    @test occursin("using ParallelTestRunner", runtests_text)
    @test occursin("runtests(EMLRegression, ARGS)", runtests_text)
    @test occursin("ParallelTestRunner = ", project_text)
    @test occursin("test = [\"Test\", \"ParallelTestRunner\"]", project_text)
end
