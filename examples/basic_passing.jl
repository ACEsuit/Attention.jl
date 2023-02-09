using Revise
using Attention: SingleHeadAttentionLayer

seq_len = 3 # Number of examples in a batch
n_embeds = 4 # Embedding size
n_heads = 2 # Number of heads
attn_layer = SingleHeadAttentionLayer(n_embeds, 0.1)

x = rand(Float64, seq_len, n_embeds)


attn_output = attn_layer(x)

@show attn_output