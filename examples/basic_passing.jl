
using Lux, Random
using Attention: SingleheadAttention

# Set state
rng = Random.default_rng()
Random.seed!(rng, 0)


### Batched Attention
b = 16
l_x = 3
d_x = 5
d_out = 2
d_attn = 8

x = rand(b, l_x, d_x)
z = rand(b, l_x, d_x)

layer = SingleheadAttention(d_x, d_attn, d_out, 0.1)

ps, st = Lux.setup(rng, layer)
outb_batched = layer(x, z, ps, st)


# Unbatched
x = rand(l_x, d_x)
z = rand(l_x, d_x)

# Output should be the (3, 2)

layer = SingleheadAttention(d_x, d_attn, d_out, 0.1)
layer(x, z, ps, st)