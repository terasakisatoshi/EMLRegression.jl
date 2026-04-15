using JSON3
using CSV
using DataFrames

mkpath("results/summaries")
paths = filter(p -> endswith(p, ".json"), readdir("results/raw"; join=true))
rows = Dict{Symbol,Any}[]
for path in paths
    row = JSON3.read(read(path, String), Dict{String,Any})
    push!(rows, Dict(
        :config_name => row["config_name"],
        :depth => row["depth"],
        :target => row["target"],
        :seed => row["seed"],
        :success => row["success"],
        :reason => row["reason"],
        :formula => row["formula"],
    ))
end
df = isempty(rows) ? DataFrame(config_name=String[], depth=Int[], target=String[], seed=Int[], success=Bool[], reason=String[], formula=String[]) : DataFrame(rows)
if !isempty(df)
    summary = combine(groupby(df, [:config_name, :depth, :target]), :success => sum => :success_count, nrow => :runs)
else
    summary = DataFrame(config_name=String[], depth=Int[], target=String[], success_count=Int[], runs=Int[])
end
CSV.write("results/summaries/summary.csv", summary)
println("results/summaries/summary.csv")
