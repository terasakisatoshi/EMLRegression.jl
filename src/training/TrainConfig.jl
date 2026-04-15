"""
    TrainConfig

Section 4.3 の実験を動かすための最小設定です。
"""
Base.@kwdef struct TrainConfig
    depth::Int
    target::Symbol
    batch_size::Int
    steps::Int
    init_strategy::Symbol = :small_gaussian
    target_noise_std::Float64 = 0.0
    learning_rate::Float64 = 1e-2
    complexity_weight::Float64 = 0.0
    margin_penalty_weight::Float64 = 0.0
    margin_target::Float64 = 1.0
    hardening_steps::Int = 0
    hardening_start::Int = typemax(Int)
    hardening_weight::Float64 = 0.1
    temperature::Float64 = 1.0
    margin_threshold::Float64 = 0.05
    stability_limit::Float64 = 1.0e6
end

"""
    TrainingResult

学習ループの結果と、収集したメトリクスをまとめた構造体です。
"""
struct TrainingResult
    config::TrainConfig
    metrics::Dict{Symbol,Vector{Float64}}
    failure_reason::FailureReason
    params
    state
end
