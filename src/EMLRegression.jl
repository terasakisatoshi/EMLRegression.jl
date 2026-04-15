module EMLRegression

include("trees/TreeTypes.jl")
include("trees/MasterTree.jl")
include("models/EMLLayer.jl")
include("training/Stability.jl")
include("targets/TargetRegistry.jl")

export eml
export MasterTree
export build_master_tree
export EMLTreeLayer
export FailureReason
export StabilityReport
export inspect_complex_values
export finite_or_flag
export clamp_complex_magnitude
export TargetSpec
export get_target
export sample_domain
export evaluate_target

eml(x, y) = exp.(x) .- log.(y)

end
