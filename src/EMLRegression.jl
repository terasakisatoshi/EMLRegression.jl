module EMLRegression

include("trees/TreeTypes.jl")
include("trees/MasterTree.jl")

export eml
export MasterTree
export build_master_tree

eml(x, y) = exp(x) - log(y)

end
