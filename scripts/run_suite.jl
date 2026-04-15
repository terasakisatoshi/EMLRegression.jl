using TOML

function parse_args(args)
    parsed = Dict{String,String}()
    i = 1
    while i <= length(args)
        parsed[args[i]] = args[i + 1]
        i += 2
    end
    return parsed
end

parsed = parse_args(ARGS)
cfg = TOML.parsefile(parsed["--config"])

for target in cfg["targets"], seed in cfg["seeds"]
    run(`$(Base.julia_cmd()) --project=. scripts/run_experiment.jl --config $(parsed["--config"]) --target $(target) --seed $(seed)`)
end
