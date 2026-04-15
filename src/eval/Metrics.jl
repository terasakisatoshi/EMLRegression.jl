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
end

"""
    recovery_verdict(; train_loss, validation_loss, snap_status, numerical_match)

学習結果と snapping 結果から回復成功かどうかを判定します。
"""
function recovery_verdict(; train_loss, validation_loss, snap_status, structure_match, ambiguous_nodes, numerical_match)
    success = numerical_match
    reason = success ? :recovered : snap_status != :ok ? :snap_failed : :mismatch
    return RecoveryVerdict(success, reason, snap_status, structure_match, ambiguous_nodes, validation_loss)
end

"""
    numerical_match(preds, ys; atol=1e-6)

予測値と正解値が許容誤差以内で一致するかを返します。
"""
function numerical_match(preds, ys; atol=1e-6)
    return all(abs.(preds .- ys) .<= atol)
end
