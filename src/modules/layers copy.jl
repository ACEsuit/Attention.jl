module layers
include("activations.jl")
using .Activations: softmax, dropout

struct MultiHeadAttentionLayer
    W_q::Matrix{Float32}
    W_k::Matrix{Float32}
    W_v::Matrix{Float32}
    W_o::Matrix{Float32}
    d_k::Int64
    heads::Int64
    dropout::Float32
end

function MultiHeadAttentionLayer(d_model, d_k, heads, dropout=0.0)
    W_q = randn(Float32, d_model, d_k * heads)
    W_k = randn(Float32, d_model, d_k * heads)
    W_v = randn(Float32, d_model, d_k * heads)
    W_o = randn(Float32, d_model, d_model)
    MultiHeadAttentionLayer(W_q, W_k, W_v, W_o, d_k, heads, dropout)
end

function (self::MultiHeadAttentionLayer)(query, key, value)
    batch_size, q_len, d_model = size(query)
    k_len = size(key, 2)

    query = reshape(query, (batch_size, q_len, self.head, self.d_k))
    key = reshape(key, (batch_size, k_len, self.head, self.d_k))
    value = reshape(value, (batch_size, k_len, self.head, self.d_k))

    q = query * self.W_q
    k = key * self.W_k
    v = value * self.W_v



    q = permutedims(q, [1, 2, 4, 3])
    k = permutedims(k, [1, 2, 4, 3])
    v = permutedims(v, [1, 2, 4, 3])

    q = reshape(q, (batch_size * q_len, self.d_k))
    k = reshape(k, (batch_size * k_len, self.d_k))
    v = reshape(v, (batch_size * k_len, self.d_k))

    logits = (q * transpose(k)) / sqrt(self.d_k)
    attn = softmax(logits, dims=2)
    attn = dropout(attn, self.dropout)

    x = (attn * v)
    x = reshape(x, (batch_size, q_len, self.head * self.d_k))
    x = (x * self.W_o)

    x
end

export MultiHeadAttentionLayer

end