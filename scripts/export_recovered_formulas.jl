using JSON3

paths = filter(p -> endswith(p, ".json"), readdir("results/raw"; join=true))
for path in sort(paths)
    row = JSON3.read(read(path, String), Dict{String,Any})
    snap_status = get(row, "snap_status", "unknown")
    println("$(row["config_name"])\tdepth=$(row["depth"])\t$(row["target"])\tseed=$(row["seed"])\tsnap=$(snap_status)\t$(row["formula"])")
end
