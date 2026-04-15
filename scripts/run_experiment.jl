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

function _override_float(parsed, flag, fallback)
    return haskey(parsed, flag) ? parse(Float64, parsed[flag]) : fallback
end

function _override_int(parsed, flag, fallback)
    return haskey(parsed, flag) ? parse(Int, parsed[flag]) : fallback
end

function _override_symbol(parsed, flag, fallback)
    return haskey(parsed, flag) ? Symbol(parsed[flag]) : fallback
end

function load_cfg(path, target_name, parsed=Dict{String,String}())
    cfg = TOML.parsefile(path)
    target = get_target(Symbol(target_name))
    if target.depth != cfg["depth"]
        error("config depth $(cfg["depth"]) does not match target $(target_name) depth $(target.depth)")
    end
    config_name = get(parsed, "--config-name", cfg["name"])
    return TrainConfig(
        depth=target.depth,
        target=Symbol(target_name),
        batch_size=_override_int(parsed, "--batch-size", cfg["batch_size"]),
        steps=_override_int(parsed, "--steps", cfg["steps"]),
        init_strategy=_override_symbol(parsed, "--init-strategy", Symbol(get(cfg, "init_strategy", "small_gaussian"))),
        target_noise_std=_override_float(parsed, "--target-noise-std", get(cfg, "target_noise_std", 0.0)),
        learning_rate=_override_float(parsed, "--learning-rate", get(cfg, "learning_rate", 1.0e-2)),
        complexity_weight=_override_float(parsed, "--complexity-weight", get(cfg, "complexity_weight", 0.0)),
        hardening_steps=_override_int(parsed, "--hardening-steps", get(cfg, "hardening_steps", 0)),
        hardening_start=_override_int(parsed, "--hardening-start", get(cfg, "hardening_start", typemax(Int))),
        hardening_weight=_override_float(parsed, "--hardening-weight", get(cfg, "hardening_weight", 0.1)),
        temperature=_override_float(parsed, "--temperature", get(cfg, "temperature", 1.0)),
        margin_threshold=_override_float(parsed, "--margin-threshold", get(cfg, "margin_threshold", 0.05)),
        stability_limit=_override_float(parsed, "--stability-limit", get(cfg, "stability_limit", 1.0e6)),
    ), Dict("name" => config_name)
end

function write_raw_result(config_meta, cfg, seed, outcome)
    mkpath("results/raw")
    max_output_abs = isempty(outcome.training.metrics[:max_output_abs]) ? Inf : maximum(outcome.training.metrics[:max_output_abs])
    max_node_abs = isempty(outcome.training.metrics[:max_node_abs]) ? Inf : maximum(outcome.training.metrics[:max_node_abs])
    result = Dict(
        "config_name" => config_meta["name"],
        "depth" => cfg.depth,
        "target" => String(cfg.target),
        "tier" => String(get_target(cfg.target).tier),
        "seed" => seed,
        "init_strategy" => String(cfg.init_strategy),
        "target_noise_std" => cfg.target_noise_std,
        "success" => outcome.recovery.success,
        "reason" => String(outcome.recovery.reason),
        "snap_status" => String(outcome.recovery.snap_status),
        "structure_match" => outcome.recovery.structure_match,
        "ambiguous_nodes" => outcome.recovery.ambiguous_nodes,
        "validation_loss" => outcome.recovery.validation_loss,
        "numerical_match" => outcome.recovery.numerical_match,
        "training_failure_reason" => String(outcome.recovery.training_failure_reason),
        "max_output_abs" => max_output_abs,
        "max_node_abs" => max_node_abs,
        "formula" => formula_string(outcome.recovered_tree, outcome.master_tree),
        "train_loss" => outcome.training.metrics[:train_loss],
        "hardening_loss" => outcome.training.metrics[:hardening_loss],
    )
    path = joinpath("results/raw", "$(config_meta["name"])-$(cfg.target)-seed$(seed).json")
    if isfile(path)
        println("overwriting existing raw result: " * path)
    end
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
