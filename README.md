# Attention.jl

Native attention mechanism in Julia

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ACEsuit.github.io/Attention.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ACEsuit.github.io/Attention.jl/dev/)
[![Build Status](https://github.com/ACEsuit/Attention.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ACEsuit/Attention.jl/actions/workflows/CI.yml?query=branch%3Amain) -->


This repository contains a very flexible implementation of the attention mechanism in Julia with basically no external requirements. The attention mechanism is a powerful tool for dealing with sequential data in natural language processing and other fields. The implementation is based on the paper "[Attention is All You Need](https://arxiv.org/abs/1706.03762)" (Vaswani et al., 2017).

## Usage

To use the attention mechanism, simply include the Attention packahe in your project and create an instance of the SingleHeadAttentionLayer structure. The SingleHeadAttentionLayer structure has the following fields:

`n_emb`: The size of the input vector
`p_drop`: The dropout probability
`ck_weights`: The weights for the key matrix
`cv_weights`: The weights for the value matrix
`cq_weights`: The weights for the query matrix
`c_proj_Weighs`: The weights for the output matrix

The layer can be initialized with random weights using the `SingleHeadAttentionLayer` constructor or with pre-trained weights using the `SingleHeadAttentionLayer` constructor. The `SingleHeadAttentionLayer` constructor can be created as follows:

```julia
layer = SingleHeadAttentionLayer(input_size::Int, dropout::Float64)
```

Where input_size is the size of the input vector, and dropout is the dropout probability. The `SingleHeadAttentionLayer` constructor can also be created as follows:

Once you have created an instance of the SingleHeadAttentionLayer, you can compute the attention scores, weighted sum of the values, and output using the forward function:

```julia
output = layer(query::Matrix, key::Matrix, value::Matrix)
```

or in case you want to use the self attention mechanism:

```julia
output = layer(x::Matrix)
```

Where query, key, and value are matrices of the same number of rows, and x is a matrix of shape (n, m). The output is a matrix of shape (n, m).

## Example

Here's a simple example of how to use the complete attention mechanism:

```julia
using Attention: SingleHeadAttentionLayer

seq_len = 3 # Number of examples in a batch
n_embeds = 4 # Embedding size
n_heads = 2 # Number of heads
attn_layer = SingleHeadAttentionLayer(n_embeds, 0.1)

x = rand(Float64, seq_len, n_embeds)

attn_output = attn_layer(x)

@show attn_output
```

## Limitations

The current implementation is a basic almost direct implementation of the attention mechanism. It is not optimized for speed and is not GPU compatible. The current implementation is also not compatible with the Transformers.jl package. We will very soon add support for the MultiHeadAttentionLayer and the TransformerEncoderLayer.

## Contributing

Contributions are welcome! Please open an issue or a pull request if you have any suggestions or if you find any bugs.

