using TOML
using JSON3
using StableRNGs
using EMLRegression

function parse_args(args)
    parsed = Dict{String,String}()
    i = 1
    while i <= length(args)
        parsed[args[i]] = args[i + 1]
        i += 2
    end
    return parsed
end

_override_float(parsed, flag, fallback) = haskey(parsed, flag) ? parse(Float64, parsed[flag]) : fallback
_override_int(parsed, flag, fallback) = haskey(parsed, flag) ? parse(Int, parsed[flag]) : fallback
_override_symbol(parsed, flag, fallback) = haskey(parsed, flag) ? Symbol(parsed[flag]) : fallback
_json_float(x) = isfinite(x) ? x : nothing

function load_cfg(path, target_name, parsed=Dict{String,String}())
    cfg = TOML.parsefile(path)
    target = get_target(Symbol(target_name))
    target.depth == cfg["depth"] || error("config depth $(cfg["depth"]) does not match target $(target_name) depth $(target.depth)")
    config_name = get(parsed, "--config-name", cfg["name"])

    return TrainConfig(
        depth=target.depth,
        target=Symbol(target_name),
        batch_size=_override_int(parsed, "--batch-size", get(cfg, "batch_size", 0)),
        init_strategy=_override_symbol(parsed, "--init-strategy", Symbol(get(cfg, "init_strategy", "biased"))),
        init_expr=get(parsed, "--init-expr", get(cfg, "init_expr", nothing)),
        init_scale=_override_float(parsed, "--init-scale", get(cfg, "init_scale", 1.0)),
        init_noise=_override_float(parsed, "--init-noise", get(cfg, "init_noise", 0.0)),
        search_iters=_override_int(parsed, "--search-iters", get(cfg, "search_iters", 6000)),
        hardening_iters=_override_int(parsed, "--hardening-iters", get(cfg, "hardening_iters", 2000)),
        lr=_override_float(parsed, "--lr", get(cfg, "lr", 1.0e-2)),
        tau_search=_override_float(parsed, "--tau-search", get(cfg, "tau_search", 2.5)),
        tau_hard=_override_float(parsed, "--tau-hard", get(cfg, "tau_hard", 0.01)),
        lam_ent_hard=_override_float(parsed, "--lam-ent-hard", get(cfg, "lam_ent_hard", 2.0e-2)),
        lam_bin_hard=_override_float(parsed, "--lam-bin-hard", get(cfg, "lam_bin_hard", 2.0e-2)),
        lam_inter=_override_float(parsed, "--lam-inter", get(cfg, "lam_inter", 1.0e-4)),
        inter_threshold=_override_float(parsed, "--inter-threshold", get(cfg, "inter_threshold", 50.0)),
        eval_every=_override_int(parsed, "--eval-every", get(cfg, "eval_every", 200)),
        tail_eval_every=_override_int(parsed, "--tail-eval-every", get(cfg, "tail_eval_every", 50)),
        fit_success_thr=_override_float(parsed, "--fit-success-thr", get(cfg, "fit_success_thr", 1.0e-6)),
        success_thr=_override_float(parsed, "--success-thr", get(cfg, "success_thr", 1.0e-20)),
        snap_threshold=_override_float(parsed, "--snap-threshold", get(cfg, "snap_threshold", 0.01)),
        max_uncertain_success=_override_int(parsed, "--max-uncertain-success", get(cfg, "max_uncertain_success", 0)),
        data_lo=_override_float(parsed, "--data-lo", get(cfg, "data_lo", 1.0)),
        data_hi=_override_float(parsed, "--data-hi", get(cfg, "data_hi", 3.0)),
        data_step=_override_float(parsed, "--data-step", get(cfg, "data_step", 0.1)),
        gen_lo=_override_float(parsed, "--gen-lo", get(cfg, "gen_lo", 0.5)),
        gen_hi=_override_float(parsed, "--gen-hi", get(cfg, "gen_hi", 5.0)),
        generalization_points=_override_int(parsed, "--generalization-points", get(cfg, "generalization_points", 4000)),
    ), Dict("name" => config_name)
end

function write_raw_result(config_meta, cfg, seed, outcome)
    mkpath("results/raw")
    result = Dict(
        "config_name" => config_meta["name"],
        "depth" => cfg.depth,
        "target" => String(cfg.target),
        "tier" => String(get_target(cfg.target).tier),
        "seed" => seed,
        "init_strategy" => String(cfg.init_strategy),
        "fit_success" => outcome.recovery.fit_success,
        "symbol_success" => outcome.recovery.symbol_success,
        "stable_symbol_success" => outcome.recovery.stable_symbol_success,
        "success" => outcome.recovery.success,
        "n_uncertain" => outcome.recovery.n_uncertain,
        "failure_reason" => string(outcome.training.failure_reason),
        "nan_restarts" => get(outcome.training.summary, :nan_restarts, 0),
        "nonfinite_steps" => get(outcome.training.summary, :nonfinite_steps, 0),
        "snap_mse" => _json_float(outcome.recovery.snap_mse),
        "snap_rmse" => _json_float(outcome.recovery.snap_rmse),
        "snap_max_real" => _json_float(outcome.recovery.snap_max_real),
        "snap_max_imag" => _json_float(outcome.recovery.snap_max_imag),
        "hardening_iter" => outcome.recovery.hardening_iter,
        "soft_rmse_last" => _json_float(isempty(outcome.training.metrics[:soft_rmse]) ? NaN : outcome.training.metrics[:soft_rmse][end]),
        "hard_rmse_last" => _json_float(isempty(outcome.training.metrics[:hard_rmse]) ? NaN : outcome.training.metrics[:hard_rmse][end]),
        "snap_info" => Dict(
            "uncertain_leaves" => outcome.snap.uncertain_leaves,
            "uncertain_gates" => outcome.snap.uncertain_gates,
            "n_uncertain" => outcome.snap.n_uncertain,
        ),
    )
    path = joinpath("results/raw", "$(config_meta["name"])-$(cfg.target)-seed$(seed).json")
    open(path, "w") do io
        JSON3.write(io, result)
    end
    return path
end

parsed = parse_args(ARGS)
cfg, meta = load_cfg(parsed["--config"], parsed["--target"], parsed)
seed = parse(Int, parsed["--seed"])
outcome = run_experiment(cfg; rng=StableRNG(seed))
path = write_raw_result(meta, cfg, seed, outcome)
println(path)
