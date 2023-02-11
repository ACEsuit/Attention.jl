using Attention
using Test

@testset "Attention.jl" begin
    @testset "SingleheadAttention" begin
        # Set state
        rng = Random.default_rng()
        Random.seed!(rng, 0)

        # Paramters for attention layer
        n_embed = 128
        x = rand(Float64, n_embed, 1)
        attnt_layer = SingleheadAttention(n_embed, 0.01)
        ps, st = Lux.setup(rng, attnt_layer)
        out = attnt_layer(x, x, x, ps, st)
    end
end