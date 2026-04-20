using Test
using JSON3

@testset "paper sweep forwards lbfgs override into raw results" begin
    repo_root = normpath(joinpath(@__DIR__, "..", ".."))
    script = joinpath(repo_root, "scripts", "run_paper_headless_sweep.jl")

    mktempdir() do dir
        cd(dir) do
            cmd = `$(Base.julia_cmd()) --project=$(repo_root) $(script) --jobs depth2 --seed0 137 --seeds 1 --search-iters 0 --hardening-iters 0 --eval-every 1 --lbfgs-steps 25`
            run(cmd)

            raw_path = joinpath(dir, "results", "raw", "pnas_d2_random-biased-eml_depth2-seed137.json")
            @test isfile(raw_path)

            row = JSON3.read(read(raw_path, String))
            @test haskey(row, :lbfgs_steps)
            @test row[:lbfgs_steps] == 25
        end
    end
end
