"""
    TreeNodeSpec

EML master tree の内部ノード定義です。
左右入力に許される候補集合と、直下の子ノードを持ちます。
"""
struct TreeNodeSpec
    id::Int
    left_candidates::Vector{Symbol}
    right_candidates::Vector{Symbol}
    left_child::Union{Int,Nothing}
    right_child::Union{Int,Nothing}
end

"""
    MasterTree

EML 木の探索空間を表す構造体です。
"""
struct MasterTree
    depth::Int
    variables::Vector{Symbol}
    root_id::Int
    terminals::Vector{Symbol}
    nodes::Vector{TreeNodeSpec}
end

"""
    RecoveredTree

snapping 後の離散木を表します。
`choices` は各ノードの左右入力選択を保持します。
`terminals` と `weights` は旧実装互換のため当面残します。
"""
struct RecoveredTree
    choices::Dict{Int,Tuple{Symbol,Symbol}}
    terminals::Vector{Symbol}
    weights::Vector{Float64}
end

RecoveredTree(choices::Dict{Int,Tuple{Symbol,Symbol}}) = RecoveredTree(choices, Symbol[], Float64[])
RecoveredTree(terminals::Vector{Symbol}, weights::Vector{Float64}) = RecoveredTree(Dict{Int,Tuple{Symbol,Symbol}}(), terminals, weights)
