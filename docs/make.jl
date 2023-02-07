using Attention
using Documenter

DocMeta.setdocmeta!(Attention, :DocTestSetup, :(using Attention); recursive=true)

makedocs(;
    modules=[Attention],
    authors="Christoph Ortner <christophortner@gmail.com> and contributors",
    repo="https://github.com/ACEsuit/Attention.jl/blob/{commit}{path}#{line}",
    sitename="Attention.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ACEsuit.github.io/Attention.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ACEsuit/Attention.jl",
    devbranch="main",
)
