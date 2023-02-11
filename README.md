# Attention.jl

Native attention mechanism in Julia

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ACEsuit.github.io/Attention.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ACEsuit.github.io/Attention.jl/dev/)
[![Build Status](https://github.com/ACEsuit/Attention.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ACEsuit/Attention.jl/actions/workflows/CI.yml?query=branch%3Amain) -->


This repository contains a very flexible implementation of the attention mechanism in Julia based on the Lux deep learning framework. The attention mechanism is a powerful tool for dealing with sequential data in natural language processing and other fields. The implementation is based on the paper "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Vaswani et al., 2017).

## Usage

To use the SingleheadAttention layer, you need to specify the input embedding dimension (n_x), the latent dimension (n_latent), the output dimension (d_out), and the dropout probability (p_drop). You can do this using the following code:


```julia
layer = SingleheadAttention(n_x, n_latent, d_out, p_drop)
```

where n_x is the input embedding dimension, n_latent is the latent dimension, d_out is the output dimension, and p_drop is the dropout probability.

Alternatively, if you want to use the default latent dimension (n_latent=n_x), you can use the following code:

```julia
layer = SingleheadAttention(n_x, p_drop)
```

The layer is designed to be used with Lux, so you will need to install and include Lux in your project before using the SingleheadAttention layer.

How it works
The SingleheadAttention layer calculates the attention scores between the input and the query, and applies the attention scores to the value. This is done using a series of matrix multiplications, where the input and query are first multiplied by the query, key, and value weights, respectively. The attention scores are then calculated by dividing the dot product of the query and key matrices by the square root of the latent dimension. The attention scores are then passed through a softmax activation to ensure that they sum up to 1, and are finally multiplied by the value matrix to obtain the output.

The layer also includes dropout, which can be applied to the input and query by specifying the dropout probability (p_drop).

### Example

```julia


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
```

## Limitations

The current implementation is a basic almost direct implementation of the attention mechanism. It is not optimized for speed and is not GPU compatible. The current implementation is also not compatible with the Transformers.jl package. We will very soon add support for the MultiHeadAttentionLayer and the TransformerEncoderLayer.

## Contributing

Contributions are welcome! Please open an issue or a pull request if you have any suggestions or if you find any bugs.

