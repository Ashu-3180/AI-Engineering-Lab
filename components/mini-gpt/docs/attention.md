# Attention experiments

M3 moves from a fixed causal summary to learned, content-dependent summaries. Inputs use
`x: [B, T, C]`, where `B` is batch size, `T` sequence length, and `C` channels.

1. Explicit averaging constructs each prefix average directly: `y_t = mean(x_0 ... x_t)`.
2. Matrix averaging expresses that same operation as `W @ x`, where `W` is lower triangular
   and every row sums to one.
3. Softmax aggregation replaces fixed `W` with normalized causal weights derived from scores.
4. Q/K/V attention projects each token into a query (what it seeks), key (what it offers), and
   value (what it contributes). Scores are `QK^T`; softmax turns each causal score row into a
   distribution that aggregates values.

The reusable stages use scaled causal attention:

`softmax((QK^T / sqrt(D)) + M) V`

`D` is the head dimension; scaling prevents dot products from becoming excessively large.
`M` masks future positions before softmax, preserving decoder-only causality. Single-head
attention returns `[B, T, C]`. Multi-head attention reshapes projections to `[B, H, T, D]`,
attends independently per head, concatenates back to `[B, T, C]`, and applies an output
projection. These modules remain independent of tokenizer, corpus, and training code for M4.
