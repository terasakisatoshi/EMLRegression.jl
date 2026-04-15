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

function load_cfg(path, target_name)
    cfg = TOML.parsefile(path)
    target = get_target(Symbol(target_name))
    if target.depth != cfg["depth"]
        error("config depth $(cfg["depth"]) does not match target $(target_name) depth $(target.depth)")
    end
    return TrainConfig(
        depth=target.depth,
        target=Symbol(target_name),
        batch_size=cfg["batch_size"],
        steps=cfg["steps"],
        learning_rate=get(cfg, "learning_rate", 1.0e-2),
        hardening_steps=get(cfg, "hardening_steps", 0),
        hardening_start=get(cfg, "hardening_start", typemax(Int)),
        hardening_weight=get(cfg, "hardening_weight", 0.1),
        temperature=get(cfg, "temperature", 1.0),
        margin_threshold=get(cfg, "margin_threshold", 0.05),
    ), cfg
end

function write_raw_result(config_meta, cfg, seed, outcome)
    mkpath("results/raw")
    result = Dict(
        "config_name" => config_meta["name"],
        "depth" => cfg.depth,
        "target" => String(cfg.target),
        "tier" => String(get_target(cfg.target).tier),
        "seed" => seed,
        "success" => outcome.recovery.success,
        "reason" => String(outcome.recovery.reason),
        "snap_status" => String(outcome.recovery.snap_status),
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
cfg, meta = load_cfg(parsed["--config"], parsed["--target"])
seed = parse(Int, parsed["--seed"])
outcome = run_experiment(cfg; rng=StableRNG(seed))
path = write_raw_result(meta, cfg, seed, outcome)
println(path)
