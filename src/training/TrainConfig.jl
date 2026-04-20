Base.@kwdef struct TrainConfig
    depth::Int
    target::Symbol
    seed::Int = 1
    batch_size::Int = 0
    init_strategy::Symbol = :biased
    init_expr::Union{Nothing,String} = nothing
    init_scale::Float64 = 1.0
    init_noise::Float64 = 0.0
    search_iters::Int = 6000
    hardening_iters::Int = 2000
    lr::Float64 = 1.0e-2
    tau_search::Float64 = 2.5
    tau_hard::Float64 = 0.01
    hardening_tau_power::Float64 = 2.0
    hardening_lr_floor::Float64 = 0.01
    patience::Int = 4200
    patience_threshold::Float64 = 1.0e-2
    plateau_rtol::Float64 = 1.0e-3
    lam_ent_hard::Float64 = 2.0e-2
    lam_bin_hard::Float64 = 2.0e-2
    lam_inter::Float64 = 1.0e-4
    inter_threshold::Float64 = 50.0
    eml_clamp::Float64 = 1.0e300
    eval_every::Int = 200
    tail_eval_every::Int = 50
    tail_eval_tau::Float64 = 0.2
    early_stop_count::Int = 10
    hard_trigger_mse::Float64 = 1.0e-20
    hard_trigger_count::Int = 3
    nan_restart_patience::Int = 50
    max_nan_restarts::Int = 100
    fit_success_thr::Float64 = 1.0e-6
    success_thr::Float64 = 1.0e-20
    snap_threshold::Float64 = 0.01
    max_uncertain_success::Int = 0
    lbfgs_steps::Int = 0
    lbfgs_lr::Float64 = 0.6
    grad_clip_norm::Float64 = 1.0
    diagnostics_limit::Int = 64
    data_lo::Float64 = 1.0
    data_hi::Float64 = 3.0
    data_step::Float64 = 0.1
    gen_lo::Float64 = 0.5
    gen_hi::Float64 = 5.0
    generalization_points::Int = 4000
end

struct TrainingResult
    config::TrainConfig
    metrics::Dict{Symbol,Vector{Float64}}
    failure_reason::FailureReason
    params
    state
    summary::Dict{Symbol,Any}
    nonfinite_steps::Int
    nan_restarts::Int
    diagnostics::Dict{Symbol,Vector{Any}}
end
