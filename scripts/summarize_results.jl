using JSON3
using CSV
using DataFrames

_mean(xs) = sum(xs) / length(xs)
_mean_or_nan(xs) = isempty(xs) ? NaN : _mean(xs)
_mean_skipmissing(xs) = _mean_or_nan(Float64.(collect(skipmissing(xs))))
_mean_bool_skipmissing(xs) = _mean_or_nan(Float64.(Int.(collect(skipmissing(xs)))))
_json_int_or_missing(x) = isnothing(x) ? missing : Int(x)

const PAPER_FAMILIES = (
    "depth2",
    "depth3",
    "depth4",
    "depth5_random",
    "depth6_random",
    "depth5_manual_noise12",
    "depth6_manual_noise12",
)

function empty_raw_results()
    return DataFrame(
        config_name=String[],
        family=String[],
        tier=String[],
        depth=Int[],
        target=String[],
        seed=Int[],
        init_strategy=String[],
        fit_success=Bool[],
        symbol_success=Bool[],
        stable_symbol_success=Bool[],
        extern_style_stable_symbol_success=Union{Missing,Bool}[],
        success=Bool[],
        pre_recovery_n_uncertain=Union{Missing,Int}[],
        n_uncertain=Int[],
        nonfinite_steps=Int[],
        nonfinite_grad_steps=Int[],
        nan_restarts=Int[],
        snap_mse=Float64[],
        snap_rmse=Float64[],
        snap_max_real=Float64[],
        snap_max_imag=Float64[],
        hardening_iter=Union{Missing,Int}[],
    )
end

function paper_family(config_name::AbstractString)
    startswith(config_name, "pnas_d2_random") && return "depth2"
    startswith(config_name, "pnas_d3_random") && return "depth3"
    startswith(config_name, "pnas_d4_random") && return "depth4"
    startswith(config_name, "pnas_d5_random64") && return "depth5_random"
    startswith(config_name, "pnas_d6_random64") && return "depth6_random"
    startswith(config_name, "pnas_d5_manual_noise12") && return "depth5_manual_noise12"
    startswith(config_name, "pnas_d6_manual_noise12") && return "depth6_manual_noise12"
    return "other"
end

function load_raw_results(raw_dir)
    isdir(raw_dir) || return empty_raw_results()
    paths = filter(p -> endswith(p, ".json"), readdir(raw_dir; join=true))
    rows = Dict{Symbol,Any}[]
    for path in paths
        row = JSON3.read(read(path, String))
        config_name = String(row[:config_name])
        push!(rows, Dict(
            :config_name => config_name,
            :family => String(get(row, :family, paper_family(config_name))),
            :tier => String(get(row, :tier, "unknown")),
            :depth => Int(row[:depth]),
            :target => String(row[:target]),
            :seed => Int(row[:seed]),
            :init_strategy => String(get(row, :init_strategy, "biased")),
            :fit_success => Bool(get(row, :fit_success, false)),
            :symbol_success => Bool(get(row, :symbol_success, false)),
            :stable_symbol_success => Bool(get(row, :stable_symbol_success, false)),
            :extern_style_stable_symbol_success => get(row, :extern_style_stable_symbol_success, missing),
            :success => Bool(get(row, :success, false)),
            :pre_recovery_n_uncertain => haskey(row, :pre_recovery_n_uncertain) ? Int(row[:pre_recovery_n_uncertain]) : missing,
            :n_uncertain => Int(get(row, :n_uncertain, 0)),
            :nonfinite_steps => Int(get(row, :nonfinite_steps, 0)),
            :nonfinite_grad_steps => Int(get(row, :nonfinite_grad_steps, 0)),
            :nan_restarts => Int(get(row, :nan_restarts, 0)),
            :snap_mse => Float64(get(row, :snap_mse, Inf)),
            :snap_rmse => Float64(get(row, :snap_rmse, Inf)),
            :snap_max_real => Float64(get(row, :snap_max_real, Inf)),
            :snap_max_imag => Float64(get(row, :snap_max_imag, Inf)),
            :hardening_iter => _json_int_or_missing(get(row, :hardening_iter, nothing)),
        ))
    end

    return isempty(rows) ? empty_raw_results() : DataFrame(rows)
end

function summarize_section_4_3(df::DataFrame)
    section_df = subset(df, :config_name => ByRow(config_name -> paper_family(config_name) in PAPER_FAMILIES))
    if isempty(section_df)
        return DataFrame(
            job_name=String[],
            family=String[],
            init_strategy=String[],
            fit_count=Int[],
            symbol_count=Int[],
            stable_symbol_count=Int[],
            extern_style_stable_symbol_count=Int[],
            nonfinite_steps=Int[],
            nonfinite_grad_steps=Int[],
            nan_restarts=Int[],
            runs=Int[],
        )
    end
    section_df = transform(section_df, :config_name => ByRow(paper_family) => :paper_family)

    summary = combine(
        groupby(section_df, [:paper_family, :init_strategy]),
        :fit_success => sum => :fit_count,
        :symbol_success => sum => :symbol_count,
        :stable_symbol_success => sum => :stable_symbol_count,
        :extern_style_stable_symbol_success => (x -> sum(Int.(collect(skipmissing(x))))) => :extern_style_stable_symbol_count,
        :nonfinite_steps => sum => :nonfinite_steps,
        :nonfinite_grad_steps => sum => :nonfinite_grad_steps,
        :nan_restarts => sum => :nan_restarts,
        nrow => :runs,
    )
    rename!(summary, :paper_family => :family)
    insertcols!(summary, 1, :job_name => summary.family)
    return summary
end

summarize_section_4_3(raw_dir) = summarize_section_4_3(load_raw_results(raw_dir))

function summarize_all_results(raw_dir="results/raw")
    df = load_raw_results(raw_dir)
    if isempty(df)
        return DataFrame(
            config_name=String[],
            tier=String[],
            depth=Int[],
            target=String[],
            fit_success_rate=Float64[],
            symbol_success_rate=Float64[],
            stable_symbol_success_rate=Float64[],
            extern_style_stable_symbol_success_rate=Float64[],
            success_rate=Float64[],
            mean_pre_recovery_n_uncertain=Float64[],
            mean_n_uncertain=Float64[],
            mean_snap_mse=Float64[],
            mean_snap_rmse=Float64[],
            mean_snap_max_real=Float64[],
            mean_snap_max_imag=Float64[],
            mean_nonfinite_grad_steps=Float64[],
            mean_hardening_iter=Float64[],
            init_strategies=String[],
            runs=Int[],
        ), summarize_section_4_3(df), nrow(df)
    end

    summary = combine(
        groupby(df, [:config_name, :tier, :depth, :target]),
        :fit_success => (x -> _mean(Float64.(x))) => :fit_success_rate,
        :symbol_success => (x -> _mean(Float64.(x))) => :symbol_success_rate,
        :stable_symbol_success => (x -> _mean(Float64.(x))) => :stable_symbol_success_rate,
        :extern_style_stable_symbol_success => _mean_bool_skipmissing => :extern_style_stable_symbol_success_rate,
        :success => (x -> _mean(Float64.(x))) => :success_rate,
        :pre_recovery_n_uncertain => _mean_skipmissing => :mean_pre_recovery_n_uncertain,
        :n_uncertain => _mean => :mean_n_uncertain,
        :snap_mse => _mean => :mean_snap_mse,
        :snap_rmse => _mean => :mean_snap_rmse,
        :snap_max_real => _mean => :mean_snap_max_real,
        :snap_max_imag => _mean => :mean_snap_max_imag,
        :nonfinite_grad_steps => _mean => :mean_nonfinite_grad_steps,
        :hardening_iter => _mean_skipmissing => :mean_hardening_iter,
        :init_strategy => (x -> join(sort!(unique(collect(x))), ",")) => :init_strategies,
        nrow => :runs,
    )
    return summary, summarize_section_4_3(df), nrow(df)
end

function write_summaries(raw_dir="results/raw")
    mkpath("results/summaries")
    summary, section_4_3, raw_count = summarize_all_results(raw_dir)
    CSV.write("results/summaries/summary.csv", summary)
    CSV.write("results/summaries/section_4_3_summary.csv", section_4_3)
    println("aggregated $(raw_count) raw result file(s) from $(raw_dir)")
    println("results/summaries/summary.csv")
    println("results/summaries/section_4_3_summary.csv")
    return summary, section_4_3
end

if abspath(PROGRAM_FILE) == @__FILE__
    write_summaries()
end
