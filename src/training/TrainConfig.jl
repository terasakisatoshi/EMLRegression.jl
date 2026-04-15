Base.@kwdef struct TrainConfig
    depth::Int
    target::Symbol
    batch_size::Int
    steps::Int
    hardening_steps::Int = 0
    hardening_weight::Float64 = 0.1
end

struct TrainingResult
    config::TrainConfig
    metrics::Dict{Symbol,Vector{Float64}}
    failure_reason::FailureReason
    params
    state
end
