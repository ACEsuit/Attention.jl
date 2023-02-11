module layers
export SingleheadAttention

include("activations.jl")
using .Activations: native_softmax, matrix_dropout
using Lux, Random, NNlib, Zygote

# Define the attention layer in Lux syntaxis
struct SingleheadAttention <: Lux.AbstractExplicitLayer
    n_x
    n_z
    n_latent
    d_out
    p_drop
    w_k # key weights
    w_v # value weights
    w_q # query weights
    scale # scaling factor for attention
end

# initialization
function SingleheadAttention(n_x::Integer,
    n_latent::Integer,
    d_out::Integer,
    p_drop::AbstractFloat)

    w_q = randn(Float32, n_x, n_latent)
    w_k = randn(Float32, n_x, n_latent)
    w_v = randn(Float32, n_x, d_out)
    return SingleheadAttention(
        () -> copy(n_x), # This is the embedding dimmension of the input
        () -> copy(n_x), # This is the embedding dimmension of the input
        () -> copy(n_latent), # This is the latent dimmension of the input
        () -> copy(d_out), # This is the output dimmension of the input
        () -> copy(p_drop), # dropout probability
        () -> copy(w_k), # key weights
        () -> copy(w_v), # value weights
        () -> copy(w_q), # query weights
        () -> copy(1 / sqrt(n_latent)), # scaling factor for attention
    )
end

function SingleheadAttention(n_x::Integer, p_drop::AbstractFloat)
    return SingleheadAttention(n_x, n_x, n_x, p_drop)
end

# Recall states are not trainable while parameters are. So:
# `n_x' is the Embedding dimmension, a state
Lux.initialstates(rng::AbstractRNG, layer::SingleheadAttention) = (
    n_x=layer.n_x(),
    n_latent=layer.n_latent(),
    p_drop=0.1,
    scale=layer.scale(),
)

# While all the other weights are trainable
Lux.initialparameters(rng::AbstractRNG, layer::SingleheadAttention) = (
    w_k=layer.w_k(),
    w_v=layer.w_v(),
    w_q=layer.w_q(),
)

###############################################################################
# Forward pass
###############################################################################
"""
    Single head attention layer
    x: input (BATCH x n_latent, n_x)
    z: input (BATCH x n_latent, n_x)
    returns: output (BATCH x n_latent, d_out)
"""

function _batch_singlehead_attention(x, z, w_q, w_k, w_v, batch_size, st)
    # Repeat the weights for each batch
    w_q = repeat(w_q, 1, 1, batch_size) # (d_x, n_latent, bs)
    w_k = repeat(w_k, 1, 1, batch_size) # (d_x, n_latent, bs)
    w_v = repeat(w_v, 1, 1, batch_size) # (d_x, d_out, bs)
    q = batched_mul(permutedims(x, [2, 3, 1]), w_q)
    k = batched_mul(permutedims(z, [2, 3, 1]), w_k)
    v = batched_mul(permutedims(z, [2, 3, 1]), w_v)
    attmat = batched_mul(q, permutedims(k, [2, 1, 3])) / st.scale
    attmat = softmax(attmat, dims=2)
    out = batched_mul(attmat, v)
    permutedims(out, [3, 1, 2])
end

function _no_batch_attention(x, z, w_q, w_k, w_v, st)
    q = x * w_q # (d_x, l_x)
    k = z * w_k # (d_z, l_z)
    v = z * w_v # (d_out, l_z)
    # Attention
    kt = k' / st.scale # (l_z, d_z)
    attmat = q * kt # (l_x, l_z)
    attmat = softmax(attmat, dims=2) # (l_x, l_z)
    attmat * v # (l_x, d_out)
end

"""
If the input is a 3d tensor, computes the attention with _batch_first_attention
Thus, the first dimension is the batch size, the second the latent dimension and

"""
@inline function (layer::SingleheadAttention)(x::AbstractVecOrMat, z::AbstractVecOrMat, ps, st)
    # Getting the sizes
    size_xs = size(x)
    # If the initial shape of x is (embedding_dim, batch_size)
    # Reshape it to the form: (1, embedding_dim, batch_size)
    x = reshape(x, size_xs[2], 1, size_xs[1])
    z = reshape(z, size_xs[2], 1, size_xs[1])
    out = reshape(_batch_singlehead_attention(x, z, ps.w_q, ps.w_k, ps.w_v, size_xs[2], st), size_xs[1], size_xs[2])
    out
end

"""
If the input is a matrix, then we assume that we compute the
self attention
"""
@inline function (layer::SingleheadAttention)(x, ps, st)
    @show x
    layer(x, x, ps, st)
end

@inline function (layer::SingleheadAttention)(x, N::Any, ps, st)
    layer(x, x, ps, st)
end

end # module