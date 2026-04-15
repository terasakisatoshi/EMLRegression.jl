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
