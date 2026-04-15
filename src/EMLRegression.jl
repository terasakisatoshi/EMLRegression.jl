module EMLRegression

"""
    eml(x, y)

論文で中心となる 2 項演算子です。
要素ごとに `exp(x) - log(y)` を計算します。
"""
eml

include("trees/TreeTypes.jl")
include("trees/MasterTree.jl")
include("models/EMLLayer.jl")
include("training/Stability.jl")
include("training/TrainConfig.jl")
include("training/TrainLoop.jl")
include("targets/TargetRegistry.jl")
include("snapping/Snap.jl")
include("symbolics/Export.jl")
include("eval/Metrics.jl")
include("eval/Recovery.jl")

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
export evaluate_recovered
export formula_string
export RecoveryVerdict
export recovery_verdict
export numerical_match
export run_experiment

eml(x, y) = exp.(x) .- log.(y)

end
