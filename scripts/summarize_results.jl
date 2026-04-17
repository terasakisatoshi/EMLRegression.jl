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
        :init_strategy => String(get(row, :init_strategy, "biased")),
        :fit_success => Bool(get(row, :fit_success, false)),
        :symbol_success => Bool(get(row, :symbol_success, false)),
        :stable_symbol_success => Bool(get(row, :stable_symbol_success, false)),
        :success => Bool(get(row, :success, false)),
        :n_uncertain => Int(get(row, :n_uncertain, 0)),
        :snap_mse => Float64(get(row, :snap_mse, Inf)),
        :snap_rmse => Float64(get(row, :snap_rmse, Inf)),
        :snap_max_real => Float64(get(row, :snap_max_real, Inf)),
        :snap_max_imag => Float64(get(row, :snap_max_imag, Inf)),
        :hardening_iter => Int(get(row, :hardening_iter, 0)),
    ))
end

df = isempty(rows) ? DataFrame(
    config_name=String[],
    tier=String[],
    depth=Int[],
    target=String[],
    seed=Int[],
    init_strategy=String[],
    fit_success=Bool[],
    symbol_success=Bool[],
    stable_symbol_success=Bool[],
    success=Bool[],
    n_uncertain=Int[],
    snap_mse=Float64[],
    snap_rmse=Float64[],
    snap_max_real=Float64[],
    snap_max_imag=Float64[],
    hardening_iter=Int[],
) : DataFrame(rows)

if !isempty(df)
    summary = combine(
        groupby(df, [:config_name, :tier, :depth, :target]),
        :fit_success => (x -> _mean(Float64.(x))) => :fit_success_rate,
        :symbol_success => (x -> _mean(Float64.(x))) => :symbol_success_rate,
        :stable_symbol_success => (x -> _mean(Float64.(x))) => :stable_symbol_success_rate,
        :success => (x -> _mean(Float64.(x))) => :success_rate,
        :n_uncertain => _mean => :mean_n_uncertain,
        :snap_mse => _mean => :mean_snap_mse,
        :snap_rmse => _mean => :mean_snap_rmse,
        :snap_max_real => _mean => :mean_snap_max_real,
        :snap_max_imag => _mean => :mean_snap_max_imag,
        :hardening_iter => _mean => :mean_hardening_iter,
        :init_strategy => (x -> join(sort!(unique(collect(x))), ",")) => :init_strategies,
        nrow => :runs,
    )
else
    summary = DataFrame(
        config_name=String[],
        tier=String[],
        depth=Int[],
        target=String[],
        fit_success_rate=Float64[],
        symbol_success_rate=Float64[],
        stable_symbol_success_rate=Float64[],
        success_rate=Float64[],
        mean_n_uncertain=Float64[],
        mean_snap_mse=Float64[],
        mean_snap_rmse=Float64[],
        mean_snap_max_real=Float64[],
        mean_snap_max_imag=Float64[],
        mean_hardening_iter=Float64[],
        init_strategies=String[],
        runs=Int[],
    )
end

CSV.write("results/summaries/summary.csv", summary)
println("aggregated $(length(paths)) raw result file(s) from results/raw/")
println("results/summaries/summary.csv")
