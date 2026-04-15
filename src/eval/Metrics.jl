struct RecoveryVerdict
    success::Bool
    reason::Symbol
end

function recovery_verdict(; train_loss, validation_loss, snap_status, numerical_match)
    success = snap_status == :ok && numerical_match
    reason = success ? :recovered : snap_status != :ok ? :snap_failed : :mismatch
    return RecoveryVerdict(success, reason)
end

function numerical_match(preds, ys; atol=1e-6)
    return all(abs.(preds .- ys) .<= atol)
end
