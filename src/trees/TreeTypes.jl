struct TerminalChoice
    symbol::Symbol
end

struct NodeChoiceSpace
    left::Vector{Symbol}
    right::Vector{Symbol}
end

struct MasterTree
    depth::Int
    variables::Vector{Symbol}
    terminals::Vector{Symbol}
    nodes::Vector{NodeChoiceSpace}
end
