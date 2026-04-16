"""
    RecoveryVerdict

blind recovery 判定の結果です。
"""
struct RecoveryVerdict
    success::Bool
    reason::Symbol
    snap_status::Symbol
    structure_match::Bool
    ambiguous_nodes::Int
    validation_loss::Float64
    numerical_match::Bool
    training_failure_reason::Symbol
    snap_diagnostics
end

"""
    recovery_verdict(; train_loss, validation_loss, snap_status, structure_match, ambiguous_nodes, numerical_match)

学習結果と snapping 結果から、snapped tree が厳密に回復できたかを判定します。
"""
function recovery_verdict(; train_loss, validation_loss, snap_status, structure_match, ambiguous_nodes, numerical_match, training_failure_reason=no_failure, snap_diagnostics=NamedTuple[])
    success = training_failure_reason == no_failure && snap_status == :ok && structure_match && numerical_match
    reason = if training_failure_reason != no_failure
        Symbol(string(training_failure_reason))
    elseif success
        :recovered
    elseif snap_status != :ok
        :snap_failed
    else
        :mismatch
    end
    return RecoveryVerdict(
        success,
        reason,
        snap_status,
        structure_match,
        ambiguous_nodes,
        validation_loss,
        numerical_match,
        Symbol(string(training_failure_reason)),
        snap_diagnostics,
    )
end

"""
    numerical_match(preds, ys; atol=1e-6)

予測値と正解値が許容誤差以内で一致するかを返します。
"""
function numerical_match(preds, ys; atol=1e-6)
    return all(abs.(preds .- ys) .<= atol)
end
