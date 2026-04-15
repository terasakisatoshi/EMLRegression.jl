using JSON3

paths = filter(p -> endswith(p, ".json"), readdir("results/raw"; join=true))
for path in sort(paths)
    row = JSON3.read(read(path, String), Dict{String,Any})
    println("$(row["target"])\tseed=$(row["seed"])\t$(row["formula"])")
end
