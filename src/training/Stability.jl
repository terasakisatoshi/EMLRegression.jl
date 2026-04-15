"""
    FailureReason

学習中または評価中に観測した数値異常の種類です。
"""
@enum FailureReason begin
    no_failure
    nan_detected
    inf_detected
    nonfinite_detected
end

"""
    StabilityReport

複素値テンソルに対する NaN / Inf / 最大絶対値の検査結果です。
"""
struct StabilityReport
    has_nan::Bool
    has_inf::Bool
    max_abs::Float64
    failure_reason::FailureReason
end

"""
    inspect_complex_values(values)

複素値配列の健全性を検査し、`StabilityReport` を返します。
"""
function inspect_complex_values(values)
    flattened = vec(values)
    has_nan = any(z -> isnan(real(z)) || isnan(imag(z)), flattened)
    has_inf = any(z -> isinf(real(z)) || isinf(imag(z)), flattened)
    max_abs = isempty(flattened) ? 0.0 : maximum(abs, flattened)
    reason = has_nan ? nan_detected : has_inf ? inf_detected : no_failure
    return StabilityReport(has_nan, has_inf, max_abs, reason)
end

"""
    finite_or_flag(values)

値が有限かどうかと、その検査レポートを返します。
"""
function finite_or_flag(values)
    report = inspect_complex_values(values)
    return !report.has_nan && !report.has_inf, report
end

"""
    clamp_complex_magnitude(values, limit)

複素値の絶対値を上限 `limit` に収めます。
"""
function clamp_complex_magnitude(values, limit::Real)
    return map(values) do z
        magnitude = abs(z)
        if magnitude <= limit || magnitude == 0
            z
        else
            z / magnitude * limit
        end
    end
end
