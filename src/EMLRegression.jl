module EMLRegression

using ChainRulesCore

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
export EMLTree
export EMLTreeLayer
export init_from_expr!
export _inspect_node_outputs
export FailureReason
export StabilityReport
export inspect_complex_values
export finite_or_flag
export clamp_complex_magnitude
export TrainConfig
export TrainingResult
export run_training
export _autodiff_gradient
export TargetSpec
export get_target
export filter_real_domain
export make_grid_data
export make_generalization_data
export sample_domain
export evaluate_target
export RecoveredTree
export snap_logits
export analyze_snap
export hard_project
export hard_project!
export formula_string
export RecoveryVerdict
export recovery_verdict
export run_experiment

eml(x, y) = exp.(x) .- log.(y)

function ChainRulesCore.rrule(::typeof(eml), x::AbstractArray{<:Complex}, y::AbstractArray{<:Complex})
    output = eml(x, y)
    exp_x = exp.(x)
    conj_y = conj.(y)

    function eml_pullback(Δ)
        Δc = _coerce_eml_tangent(Δ, output)
        return NoTangent(), Δc .* conj.(exp_x), -Δc ./ conj_y
    end

    return output, eml_pullback
end

function _coerce_eml_tangent(::AbstractZero, output)
    return zero(output)
end

function _coerce_eml_tangent(Δ::AbstractArray, output)
    return map(Δ, output) do δ, ref
        isnothing(δ) ? zero(ref) : δ
    end
end

function _coerce_eml_tangent(Δ, output)
    return isnothing(Δ) ? zero(output) : Δ
end

end
