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

_override_int(parsed, flag, fallback) = haskey(parsed, flag) ? parse(Int, parsed[flag]) : fallback
_override_float(parsed, flag, fallback) = haskey(parsed, flag) ? parse(Float64, parsed[flag]) : fallback

const PAPER_JOB_SPECS = Dict(
    "depth2" => [
        (; name="pnas_d2_random", target="eml_depth2", depth=2, init_families=[:biased, :uniform, :xy_biased, :random_hot], seed0=137, seeds=8, search_iters=6000, hardening_iters=2000, eval_every=200, tail_eval_every=50, data_lo=1.0, data_hi=3.0, data_step=0.1, gen_lo=0.5, gen_hi=5.0),
    ],
    "depth3" => [
        (; name="pnas_d3_random", target="eml_depth3", depth=3, init_families=[:biased, :uniform, :xy_biased, :random_hot], seed0=137, seeds=16, search_iters=6000, hardening_iters=2000, eval_every=200, tail_eval_every=50, data_lo=1.0, data_hi=3.0, data_step=0.1, gen_lo=0.5, gen_hi=5.0),
    ],
    "depth4" => [
        (; name="pnas_d4_random", target="eml_depth4", depth=4, init_families=[:biased, :uniform, :xy_biased, :random_hot], seed0=137, seeds=16, search_iters=6000, hardening_iters=2000, eval_every=200, tail_eval_every=50, data_lo=1.0, data_hi=3.0, data_step=0.1, gen_lo=0.5, gen_hi=5.0),
    ],
    "depth5_random" => [
        (; name="pnas_d5_random64_random_hot", target="eml_depth5", depth=5, init_families=[:random_hot], seed0=137, seeds=64, search_iters=9000, hardening_iters=3000, eval_every=200, tail_eval_every=50, data_lo=1.0, data_hi=3.0, data_step=0.1, gen_lo=0.5, gen_hi=5.0),
        (; name="pnas_d5_random64_uniform", target="eml_depth5", depth=5, init_families=[:uniform], seed0=1234, seeds=64, search_iters=9000, hardening_iters=3000, eval_every=200, tail_eval_every=50, data_lo=1.0, data_hi=3.0, data_step=0.1, gen_lo=0.5, gen_hi=5.0),
    ],
    "depth6_random" => [
        (; name="pnas_d6_random64_random_hot", target="eml_depth6", depth=6, init_families=[:random_hot], seed0=137, seeds=64, search_iters=9000, hardening_iters=3000, eval_every=200, tail_eval_every=50, data_lo=1.0, data_hi=3.0, data_step=0.1, gen_lo=0.5, gen_hi=5.0),
        (; name="pnas_d6_random64_uniform", target="eml_depth6", depth=6, init_families=[:uniform], seed0=1234, seeds=64, search_iters=9000, hardening_iters=3000, eval_every=200, tail_eval_every=50, data_lo=1.0, data_hi=3.0, data_step=0.1, gen_lo=0.5, gen_hi=5.0),
    ],
    "depth5_manual_noise12" => [
        (; name="pnas_d5_manual_noise12", target="eml_depth5", depth=5, init_families=[:manual], init_expr="EML[1, EML[EML[1, EML[1, EML[x, y]]], 1]]", init_noise=12.0, seed0=2048, seeds=4, search_iters=3000, hardening_iters=1200, eval_every=100, tail_eval_every=25, data_lo=1.0, data_hi=3.0, data_step=0.1, gen_lo=0.5, gen_hi=5.0),
    ],
    "depth6_manual_noise12" => [
        (; name="pnas_d6_manual_noise12", target="eml_depth6", depth=6, init_families=[:manual], init_expr="EML[1, EML[EML[EML[1, EML[EML[x, y], 1]], 1], 1]]", init_noise=12.0, seed0=2048, seeds=4, search_iters=3000, hardening_iters=1200, eval_every=100, tail_eval_every=25, data_lo=1.0, data_hi=3.0, data_step=0.1, gen_lo=0.5, gen_hi=5.0),
    ],
)

function selected_job_keys(parsed)
    haskey(parsed, "--jobs") || error("--jobs is required")
    keys = split(parsed["--jobs"], ',')
    return [strip(key) for key in keys if !isempty(strip(key))]
end

function concrete_specs(job_key, parsed)
    haskey(PAPER_JOB_SPECS, job_key) || error("unknown paper job: $(job_key)")
    specs = NamedTuple[]
    for spec in PAPER_JOB_SPECS[job_key], strategy in spec.init_families
        config_name = length(spec.init_families) == 1 ? spec.name : "$(spec.name)-$(strategy)"
        push!(specs, merge(spec, (
            family=job_key,
            config_name=config_name,
            init_strategy=String(strategy),
            seed0=_override_int(parsed, "--seed0", spec.seed0),
            seeds=_override_int(parsed, "--seeds", spec.seeds),
            search_iters=_override_int(parsed, "--search-iters", spec.search_iters),
            hardening_iters=_override_int(parsed, "--hardening-iters", spec.hardening_iters),
            eval_every=_override_int(parsed, "--eval-every", spec.eval_every),
            tail_eval_every=_override_int(parsed, "--tail-eval-every", spec.tail_eval_every),
            data_lo=_override_float(parsed, "--data-lo", spec.data_lo),
            data_hi=_override_float(parsed, "--data-hi", spec.data_hi),
            data_step=_override_float(parsed, "--data-step", spec.data_step),
            gen_lo=_override_float(parsed, "--gen-lo", spec.gen_lo),
            gen_hi=_override_float(parsed, "--gen-hi", spec.gen_hi),
        )))
    end
    return specs
end

function passthrough_args(parsed)
    passthrough_flags = (
        "--lr",
        "--tau-search",
        "--tau-hard",
        "--hardening-tau-power",
        "--hardening-lr-floor",
        "--patience",
        "--patience-threshold",
        "--plateau-rtol",
        "--lam-ent-hard",
        "--lam-bin-hard",
        "--lam-inter",
        "--inter-threshold",
        "--eml-clamp",
        "--tail-eval-tau",
        "--early-stop-count",
        "--hard-trigger-mse",
        "--hard-trigger-count",
        "--nan-restart-patience",
        "--max-nan-restarts",
        "--fit-success-thr",
        "--success-thr",
        "--snap-threshold",
        "--max-uncertain-success",
        "--lbfgs-steps",
        "--lbfgs-lr",
        "--grad-clip-norm",
    )
    args = String[]
    for flag in passthrough_flags
        if haskey(parsed, flag)
            push!(args, flag, parsed[flag])
        end
    end
    return args
end

function write_temp_config(path, spec)
    open(path, "w") do io
        println(io, "name = \"$(spec.config_name)\"")
        println(io, "family = \"$(spec.family)\"")
        println(io, "depth = $(spec.depth)")
        println(io, "init_strategy = \"$(spec.init_strategy)\"")
        println(io, "search_iters = $(spec.search_iters)")
        println(io, "hardening_iters = $(spec.hardening_iters)")
        println(io, "eval_every = $(spec.eval_every)")
        println(io, "tail_eval_every = $(spec.tail_eval_every)")
        println(io, "data_lo = $(spec.data_lo)")
        println(io, "data_hi = $(spec.data_hi)")
        println(io, "data_step = $(spec.data_step)")
        println(io, "gen_lo = $(spec.gen_lo)")
        println(io, "gen_hi = $(spec.gen_hi)")
        hasproperty(spec, :init_noise) && println(io, "init_noise = $(spec.init_noise)")
        if hasproperty(spec, :init_expr)
            escaped = replace(spec.init_expr, "\"" => "\\\"")
            println(io, "init_expr = \"$(escaped)\"")
        end
    end
    return path
end

function run_spec(spec, parsed, repo_root)
    run_experiment_script = joinpath(repo_root, "scripts", "run_experiment.jl")
    mktempdir() do dir
        cfg_path = write_temp_config(joinpath(dir, "$(spec.config_name).toml"), spec)
        extra_args = passthrough_args(parsed)
        for seed in spec.seed0:(spec.seed0 + spec.seeds - 1)
            argv = vcat(
                collect(Base.julia_cmd().exec),
                ["--project=$(repo_root)", run_experiment_script, "--config", cfg_path, "--target", spec.target, "--seed", string(seed)],
                extra_args,
            )
            cmd = Cmd(Cmd(argv); dir=pwd())
            run(cmd)
        end
    end
end

parsed = parse_args(ARGS)
repo_root = normpath(joinpath(@__DIR__, ".."))

for job_key in selected_job_keys(parsed)
    for spec in concrete_specs(job_key, parsed)
        run_spec(spec, parsed, repo_root)
    end
end
