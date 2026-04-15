@enum FailureReason begin
    no_failure
    nan_detected
    inf_detected
    nonfinite_detected
end

struct StabilityReport
    has_nan::Bool
    has_inf::Bool
    max_abs::Float64
    failure_reason::FailureReason
end

function inspect_complex_values(values)
    flattened = vec(values)
    has_nan = any(z -> isnan(real(z)) || isnan(imag(z)), flattened)
    has_inf = any(z -> isinf(real(z)) || isinf(imag(z)), flattened)
    max_abs = isempty(flattened) ? 0.0 : maximum(abs, flattened)
    reason = has_nan ? nan_detected : has_inf ? inf_detected : no_failure
    return StabilityReport(has_nan, has_inf, max_abs, reason)
end

function finite_or_flag(values)
    report = inspect_complex_values(values)
    return !report.has_nan && !report.has_inf, report
end

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
