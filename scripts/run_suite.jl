using TOML
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

function suite_variants(cfg)
    raw_variants = get(cfg, "variants", Any[])
    if isempty(raw_variants)
        return [Dict{String,Any}()]
    end
    return [Dict{String,Any}(variant) for variant in raw_variants]
end

function variant_name(base_name, variant)
    return haskey(variant, "name") ? "$(base_name)-$(variant["name"])" : base_name
end

function variant_args(base_name, variant)
    args = String["--config-name", variant_name(base_name, variant)]
    flag_map = Dict(
        "batch_size" => "--batch-size",
        "init_strategy" => "--init-strategy",
        "init_scale" => "--init-scale",
        "init_noise" => "--init-noise",
        "search_iters" => "--search-iters",
        "hardening_iters" => "--hardening-iters",
        "lr" => "--lr",
        "tau_search" => "--tau-search",
        "tau_hard" => "--tau-hard",
        "lam_ent_hard" => "--lam-ent-hard",
        "lam_bin_hard" => "--lam-bin-hard",
        "lam_inter" => "--lam-inter",
        "inter_threshold" => "--inter-threshold",
        "eval_every" => "--eval-every",
        "tail_eval_every" => "--tail-eval-every",
        "fit_success_thr" => "--fit-success-thr",
        "success_thr" => "--success-thr",
        "snap_threshold" => "--snap-threshold",
        "max_uncertain_success" => "--max-uncertain-success",
    )
    for (key, flag) in flag_map
        haskey(variant, key) || continue
        push!(args, flag, string(variant[key]))
    end
    return args
end

function validate_suite_config(cfg)
    suite_depth = cfg["depth"]
    targets = Symbol.(cfg["targets"])
    for target_name in targets
        target = get_target(target_name)
        if target.depth != suite_depth
            error("suite depth $(suite_depth) does not match target $(target_name) depth $(target.depth)")
        end
    end
    return targets
end

parsed = parse_args(ARGS)
cfg = TOML.parsefile(parsed["--config"])
targets = validate_suite_config(cfg)
variants = suite_variants(cfg)
base_name = cfg["name"]
run_experiment_script = joinpath(@__DIR__, "run_experiment.jl")

for variant in variants, target in targets, seed in cfg["seeds"]
    extra_args = variant_args(base_name, variant)
    run(`$(Base.julia_cmd()) --project=. $(run_experiment_script) --config $(parsed["--config"]) --target $(target) --seed $(seed) $(extra_args)`)
end
