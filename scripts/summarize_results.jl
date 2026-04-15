using JSON3
using CSV
using DataFrames

_mean(xs) = sum(xs) / length(xs)

mkpath("results/summaries")
paths = filter(p -> endswith(p, ".json"), readdir("results/raw"; join=true))
rows = Dict{Symbol,Any}[]
for path in paths
    row = JSON3.read(read(path, String))
    push!(rows, Dict(
        :config_name => String(row[:config_name]),
        :tier => String(get(row, :tier, "unknown")),
        :depth => Int(row[:depth]),
        :target => String(row[:target]),
        :seed => Int(row[:seed]),
        :init_strategy => String(get(row, :init_strategy, "small_gaussian")),
        :target_noise_std => Float64(get(row, :target_noise_std, 0.0)),
        :success => Bool(row[:success]),
        :reason => String(row[:reason]),
        :snap_status => String(get(row, :snap_status, "unknown")),
        :structure_match => Bool(get(row, :structure_match, false)),
        :ambiguous_nodes => Int(get(row, :ambiguous_nodes, 0)),
        :validation_loss => Float64(get(row, :validation_loss, Inf)),
        :numerical_match => Bool(get(row, :numerical_match, false)),
        :training_failure_reason => String(get(row, :training_failure_reason, "no_failure")),
        :max_output_abs => Float64(get(row, :max_output_abs, Inf)),
        :max_node_abs => Float64(get(row, :max_node_abs, Inf)),
        :formula => String(row[:formula]),
    ))
end
df = isempty(rows) ? DataFrame(
    config_name=String[],
    tier=String[],
    depth=Int[],
    target=String[],
    seed=Int[],
    init_strategy=String[],
    target_noise_std=Float64[],
    success=Bool[],
    reason=String[],
    snap_status=String[],
    structure_match=Bool[],
    ambiguous_nodes=Int[],
    validation_loss=Float64[],
    numerical_match=Bool[],
    training_failure_reason=String[],
    max_output_abs=Float64[],
    max_node_abs=Float64[],
    formula=String[],
) : DataFrame(rows)
if !isempty(df)
    summary = combine(
        groupby(df, [:config_name, :tier, :depth, :target]),
        :success => sum => :success_count,
        :success => (x -> _mean(Float64.(x))) => :success_rate,
        :numerical_match => (x -> _mean(Float64.(x))) => :numerical_match_rate,
        :snap_status => (x -> _mean(Float64.(x .== "ok"))) => :snap_ok_rate,
        :snap_status => (x -> _mean(Float64.(x .== "ambiguous"))) => :ambiguous_rate,
        :structure_match => (x -> _mean(Float64.(x))) => :structure_match_rate,
        :training_failure_reason => (x -> _mean(Float64.(x .!= "no_failure"))) => :training_failure_rate,
        :validation_loss => _mean => :mean_validation_loss,
        :ambiguous_nodes => _mean => :mean_ambiguous_nodes,
        :max_output_abs => _mean => :mean_max_output_abs,
        :max_node_abs => _mean => :mean_max_node_abs,
        :init_strategy => (x -> join(sort!(unique(collect(x))), ",")) => :init_strategies,
        :target_noise_std => (x -> join(string.(sort!(unique(collect(x)))), ",")) => :target_noise_stds,
        nrow => :runs,
    )
else
    summary = DataFrame(
        config_name=String[],
        tier=String[],
        depth=Int[],
        target=String[],
        success_count=Int[],
        success_rate=Float64[],
        numerical_match_rate=Float64[],
        snap_ok_rate=Float64[],
        ambiguous_rate=Float64[],
        structure_match_rate=Float64[],
        training_failure_rate=Float64[],
        mean_validation_loss=Float64[],
        mean_ambiguous_nodes=Float64[],
        mean_max_output_abs=Float64[],
        mean_max_node_abs=Float64[],
        init_strategies=String[],
        target_noise_stds=String[],
        runs=Int[],
    )
end
CSV.write("results/summaries/summary.csv", summary)
println("aggregated $(length(paths)) raw result file(s) from results/raw/")
println("results/summaries/summary.csv")
