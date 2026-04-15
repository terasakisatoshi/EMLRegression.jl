using Test

selected = isempty(ARGS) ? nothing : Set(ARGS)

should_run(name) = selected === nothing || name in selected

include("test_project_load.jl")

if should_run("trees")
    include("trees/test_master_tree.jl")
end

if should_run("models")
    include("models/test_eml_layer.jl")
end

if should_run("training")
    include("training/test_stability.jl")
    include("training/test_train_loop.jl")
end

if should_run("targets")
    include("targets/test_targets.jl")
end

if should_run("snapping")
    include("snapping/test_snap.jl")
end

if should_run("eval")
    include("eval/test_recovery.jl")
end

if should_run("integration")
    include("integration/test_smoke_ln.jl")
end
