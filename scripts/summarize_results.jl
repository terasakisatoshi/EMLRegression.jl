using JSON3
using CSV
using DataFrames

mkpath("results/summaries")
paths = filter(p -> endswith(p, ".json"), readdir("results/raw"; join=true))
rows = Dict{Symbol,Any}[]
for path in paths
    row = JSON3.read(read(path, String))
    push!(rows, Dict(
        :config_name => String(row[:config_name]),
        :depth => Int(row[:depth]),
        :target => String(row[:target]),
        :seed => Int(row[:seed]),
        :success => Bool(row[:success]),
        :reason => String(row[:reason]),
        :formula => String(row[:formula]),
    ))
end
df = isempty(rows) ? DataFrame(config_name=String[], depth=Int[], target=String[], seed=Int[], success=Bool[], reason=String[], formula=String[]) : DataFrame(rows)
if !isempty(df)
    summary = combine(groupby(df, [:config_name, :depth, :target]), :success => sum => :success_count, nrow => :runs)
else
    summary = DataFrame(config_name=String[], depth=Int[], target=String[], success_count=Int[], runs=Int[])
end
CSV.write("results/summaries/summary.csv", summary)
println("aggregated $(length(paths)) raw result file(s) from results/raw/")
println("results/summaries/summary.csv")
