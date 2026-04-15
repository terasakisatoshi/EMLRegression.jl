"""
    TerminalChoice(symbol)

終端記号を表す単純なラッパです。
"""
struct TerminalChoice
    symbol::Symbol
end

"""
    NodeChoiceSpace(left, right)

各ノードの左右入力に許される候補集合です。
"""
struct NodeChoiceSpace
    left::Vector{Symbol}
    right::Vector{Symbol}
end

"""
    MasterTree

EML 木の探索空間を表す構造体です。
"""
struct MasterTree
    depth::Int
    variables::Vector{Symbol}
    terminals::Vector{Symbol}
    nodes::Vector{NodeChoiceSpace}
end
