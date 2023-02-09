module layers
include("activations.jl")
using .Activations: softmax, dropout

# Define the attention layer
struct SingleHeadAttentionLayer
    n_emb::Int
    p_drop::Float64
    ck_weights::Array{Float64,2}
    cv_weights::Array{Float64,2}
    cq_weights::Array{Float64,2}
    c_proj_Weighs::Array{Float64,2}
end

function SingleHeadAttentionLayer(n_embed, dropout)
    ck_weights = randn(Float32, n_embed, n_embed)
    cv_weights = randn(Float32, n_embed, n_embed)
    cq_weights = randn(Float32, n_embed, n_embed)
    c_proj_Weighs = randn(Float32, n_embed, n_embed)
    SingleHeadAttentionLayer(n_embed, dropout, ck_weights, cv_weights, cq_weights, c_proj_Weighs)
end

function (self::SingleHeadAttentionLayer)(x::Matrix{Float64})
    # x: Single head attention input
    # c_att_Weights: (n_emb, n_emb * 3)
    # c_proj_Weights: (n_emb, n_emb)
    # attn_ou

    seq_len, n_emb = size(x) # batch_size, seq_len, n_emb

    k = x * self.ck_weights
    v = x * self.cv_weights
    q = x * self.cq_weights

    k = reshape(k, seq_len, n_emb)
    v = reshape(v, seq_len, n_emb)
    q = reshape(q, seq_len, n_emb)

    k = permutedims(k, [2, 1])
    v = permutedims(v, [2, 1])
    q = permutedims(q, [2, 1])

    att = q * permutedims(k, [2, 1]) / sqrt(n_emb)
    att = softmax(att, 2)
    att = dropout(att, self.p_drop)
    att = att * v

    att = reshape(permutedims(att, [2, 1]), seq_len, n_emb)

    att = att * self.c_proj_Weighs
    att = dropout(att, self.p_drop)
end

function (self::SingleHeadAttentionLayer)(k, v, q)
    # key, value, query
    # c_att_Weights: (n_emb, n_emb * 3)
    # c_proj_Weights: (n_emb, n_emb)

    seq_len, n_emb = size(k) # batch_size, seq_len, n_emb

    k = k * self.ck_weights
    v = v * self.cv_weights
    q = q * self.cq_weights

    k = reshape(k, seq_len, n_emb)
    v = reshape(v, seq_len, n_emb)
    q = reshape(q, seq_len, n_emb)

    k = permutedims(k, [2, 1])
    v = permutedims(v, [2, 1])
    q = permutedims(q, [2, 1])

    att = q * permutedims(k, [2, 1]) / sqrt(n_emb)
    att = softmax(att, 2)
    att = dropout(att, self.p_drop)
    att = att * v

    att = reshape(permutedims(att, [2, 1]), seq_len, n_emb)

    att = att * self.c_proj_Weighs
    att = dropout(att, self.p_drop)
end


export SingleHeadAttentionLayer


end