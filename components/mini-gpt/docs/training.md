# M5 training workflow

`config/training.toml` contains `smoke` and `experiment` CPU presets. The GPT model preset
is loaded from `config/gpt.toml`; the verified Tiny Shakespeare corpus, tokenizer, and
sequential split remain the authoritative data source.

Training uses AdamW, configured gradient clipping, explicit Python/PyTorch/batch-generator
seeds, and mean cross-entropy evaluation on the held-out validation split. Perplexity is
`exp(mean_loss)`. Checkpoints store model and optimizer state, iteration, model/training
configuration, corpus checksum, tokenizer vocabulary size, and RNG states. Checkpoint writes
are atomic; loading rejects mismatched corpus, vocabulary, or architecture.

Generated artifacts live under ignored `outputs/checkpoints/` and `outputs/experiments/`.
Generation crops context to the GPT block size, disables gradients, and supports deterministic
seeded sampling or greedy decoding. The bigram comparison is trained from random initialization
on the identical split using the same batch/context, AdamW settings, iteration budget, and
mean-loss validation procedure.
