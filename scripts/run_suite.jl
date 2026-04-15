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

for target in targets, seed in cfg["seeds"]
    run(`$(Base.julia_cmd()) --project=. scripts/run_experiment.jl --config $(parsed["--config"]) --target $(target) --seed $(seed)`)
end
