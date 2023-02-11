
using Lux, Random, Zygote, Optimisers
using Revise, NNlib
using Attention: SingleheadAttention

# Set state
rng = Random.default_rng()
Random.seed!(rng, 0)


### Batched Attention

BATCH_SIZE = 16
N_IN = 3
n_embed = 64
N_OUT = 2

x = randn(rng, Float32, N_IN, BATCH_SIZE)

reshape(x, 1, size(x, 1), size(x, 2))

# Define a model
model = Chain(BatchNorm(N_IN), Dense(N_IN, n_embed, tanh), SingleheadAttention(n_embed, 0.1))

ps, st = Lux.setup(rng, model)
# Run the model
y, st = Lux.apply(model, x, ps, st)
# Gradients
## Pullback API to capture change in state
(l, st_), pb = pullback(p -> Lux.apply(model, x, p, st), ps)
gs = pb((one.(l), nothing))[1]

# Optimization
st_opt = Optimisers.setup(Optimisers.ADAM(0.0001), ps)
st_opt, ps = Optimisers.update(st_opt, ps, gs)

ps, st = Lux.setup(rng, model)