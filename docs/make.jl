using Pkg

Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Documenter
using EMLRegression

makedocs(;
    sitename="EMLRegression.jl",
    authors="OpenAI Codex",
    modules=[EMLRegression],
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link=nothing,
        repolink="",
    ),
    source="src",
    build="build",
    repo="",
    remotes=nothing,
    checkdocs=:exports,
    pages=[
        "はじめに" => "index.md",
        "使い方マニュアル" => "manual.md",
        "チュートリアル" => "tutorial.md",
        "学習方法" => "training.md",
        "学習チュートリアル" => "training-tutorial.md",
        "発展: 低レベル学習 API" => "advanced-training.md",
        "論文の要点と解説" => "paper.md",
        "API リファレンス" => "api.md",
    ],
)
