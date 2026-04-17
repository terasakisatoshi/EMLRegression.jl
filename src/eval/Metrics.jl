"""
    RecoveryVerdict

PyTorch v16 相当の snapping 後判定を保持します。
"""
struct RecoveryVerdict
    snap_mse::Float64
    snap_rmse::Float64
    snap_max_real::Float64
    snap_max_imag::Float64
    fit_success::Bool
    symbol_success::Bool
    stable_symbol_success::Bool
    success::Bool
    n_uncertain::Int
    hardening_iter
end

"""
    recovery_verdict(; snap_mse, snap_max_real, snap_max_imag, n_uncertain, ...)

論文寄りの `fit_success / symbol_success / stable_symbol_success` を計算します。
"""
function recovery_verdict(;
    snap_mse,
    snap_max_real,
    snap_max_imag,
    n_uncertain,
    fit_success_thr::Float64=1.0e-6,
    success_thr::Float64=1.0e-20,
    max_uncertain_success::Int=0,
    hardening_iter=nothing,
)
    snap_rmse = isfinite(snap_mse) ? sqrt(max(snap_mse, 0.0)) : NaN
    fit_success = isfinite(snap_mse) && snap_mse < fit_success_thr
    symbol_success = isfinite(snap_mse) && snap_mse < success_thr
    stable_symbol_success = symbol_success && n_uncertain <= max_uncertain_success
    return RecoveryVerdict(
        snap_mse,
        snap_rmse,
        snap_max_real,
        snap_max_imag,
        fit_success,
        symbol_success,
        stable_symbol_success,
        fit_success,
        n_uncertain,
        hardening_iter,
    )
end
