using StableRNGs: StableRNG

"""
    TargetSpec

paper benchmark target の定義です。離散 EML tree を source of truth とし、
数値評価はその tree から導出します。
"""
struct TargetSpec
    name::Symbol
    arity::Int
    tier::Symbol
    depth::Int
    tree::RecoveredTree
    sampler::Function
end

function _sample_positive(rng, n)
    return ComplexF64.(0.25 .+ rand(rng, n) .* 2.0)
end

function _sample_centered(rng, n)
    return ComplexF64.(-1.0 .+ rand(rng, n) .* 2.0)
end

function _sample_pair(rng, n)
    return (_sample_centered(rng, n), _sample_positive(rng, n))
end

function _paper_target(name, tier, depth, tree, sampler, arity)
    return TargetSpec(name, arity, tier, depth, tree, sampler)
end

const PAPER_TARGETS = Dict{Symbol,TargetSpec}(
    :depth2_exp => _paper_target(
        :depth2_exp,
        :must_pass,
        2,
        RecoveredTree(Dict(1 => (:x, :const1), 2 => (:const1, :const1), 3 => (:const1, :const1))),
        (rng, n) -> _sample_centered(rng, n),
        1,
    ),
    :depth2_double_exp => _paper_target(
        :depth2_double_exp,
        :must_pass,
        2,
        RecoveredTree(Dict(1 => (:node_2, :const1), 2 => (:x, :const1), 3 => (:const1, :const1))),
        (rng, n) -> _sample_centered(rng, n),
        1,
    ),
    :depth3_log => _paper_target(
        :depth3_log,
        :must_pass,
        3,
        RecoveredTree(Dict(
            1 => (:const1, :node_2),
            2 => (:node_4, :const1),
            3 => (:const1, :const1),
            4 => (:const1, :x),
            5 => (:const1, :const1),
            6 => (:const1, :const1),
            7 => (:const1, :const1),
        )),
        (rng, n) -> _sample_positive(rng, n),
        1,
    ),
    :depth4_nested => _paper_target(
        :depth4_nested,
        :challenge,
        4,
        RecoveredTree(Dict(
            1 => (:node_2, :node_3),
            2 => (:x, :const1),
            3 => (:const1, :node_6),
            4 => (:const1, :const1),
            5 => (:const1, :const1),
            6 => (:node_12, :y),
            7 => (:const1, :const1),
            8 => (:const1, :const1),
            9 => (:const1, :const1),
            10 => (:const1, :const1),
            11 => (:const1, :const1),
            12 => (:x, :const1),
            13 => (:const1, :const1),
            14 => (:const1, :const1),
            15 => (:const1, :const1),
        )),
        (rng, n) -> _sample_pair(rng, n),
        2,
    ),
)

const TARGETS = let targets = copy(PAPER_TARGETS)
    merge!(
        targets,
        Dict(
            :ln => PAPER_TARGETS[:depth3_log],
            :exp => PAPER_TARGETS[:depth2_exp],
            :mul => PAPER_TARGETS[:depth4_nested],
        ),
    )
    targets
end

"""
    get_target(name)

登録済みターゲットを取得します。
"""
get_target(name::Symbol) = TARGETS[name]

"""
    sample_domain(target, n; rng_seed=1)

ターゲットに対応する入力サンプルを生成します。
"""
function sample_domain(target::TargetSpec, n::Integer; rng_seed::Integer=1)
    rng = StableRNG(rng_seed)
    return target.sampler(rng, n)
end

"""
    evaluate_target(target, xs)

target tree を independent input 上で評価します。
"""
function evaluate_target(target::TargetSpec, xs)
    master = build_master_tree(depth=target.depth, variables=target.arity == 1 ? (:x,) : (:x, :y))
    return evaluate_recovered(target.tree, master, xs)
end
