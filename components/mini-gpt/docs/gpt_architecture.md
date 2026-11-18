# Decoder-only GPT architecture

M4 maps token IDs `[B,T]` to logits `[B,T,V]`. A token embedding and learned position
embedding each produce `[B,T,C]`; their sum enters `n_layer` Pre-LayerNorm blocks.

Each block computes `x = x + attention(LN1(x))` and then
`x = x + feed_forward(LN2(x))`. Attention is the existing M3 custom multi-head causal
module, preserving its `[B,H,T,T]` causal score structure. The feed-forward network is
`C -> 4C -> ReLU -> C`. A final LayerNorm and `C -> V` vocabulary head produce logits.

Learned positions distinguish otherwise identical token IDs at different sequence locations.
Causal masking means logits at time `t` cannot depend on input tokens after `t`. Token and
output projection weights are intentionally separate. M4 provides only the architecture and
forward loss; training, checkpointing, and generation remain M5 responsibilities.
