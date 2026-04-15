module EMLRegression

include("trees/TreeTypes.jl")
include("trees/MasterTree.jl")
include("models/EMLLayer.jl")

export eml
export MasterTree
export build_master_tree
export EMLTreeLayer

eml(x, y) = exp.(x) .- log.(y)

end
