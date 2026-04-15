module EMLRegression

include("trees/TreeTypes.jl")
include("trees/MasterTree.jl")
include("models/EMLLayer.jl")
include("training/Stability.jl")
include("training/TrainConfig.jl")
include("training/TrainLoop.jl")
include("targets/TargetRegistry.jl")
include("snapping/Snap.jl")
include("symbolics/Export.jl")

export eml
export MasterTree
export build_master_tree
export EMLTreeLayer
export FailureReason
export StabilityReport
export inspect_complex_values
export finite_or_flag
export clamp_complex_magnitude
export TrainConfig
export TrainingResult
export run_training
export TargetSpec
export get_target
export sample_domain
export evaluate_target
export RecoveredTree
export snap_logits
export snap_model
export formula_string

eml(x, y) = exp.(x) .- log.(y)

end
