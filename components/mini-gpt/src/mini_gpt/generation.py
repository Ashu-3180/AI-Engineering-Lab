"""Basic seeded autoregressive generation for the bigram baseline."""

from __future__ import annotations

import torch

from .batching import make_seeded_generator
from .models.bigram import BigramLanguageModel
from .tokenizer import CharacterTokenizer


def generate_text(
    model: BigramLanguageModel,
    tokenizer: CharacterTokenizer,
    *,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    context_length: int,
) -> str:
    """Sample new characters; empty prompts start from token zero but omit it from output."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    token_ids = tokenizer.encode(prompt)
    generated_ids = list(token_ids)
    context_ids = token_ids[-context_length:] or [0]
    generator = make_seeded_generator(seed)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = torch.tensor([context_ids[-context_length:]], dtype=torch.long)
            logits, _ = model(context)
            probabilities = torch.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1, generator=generator).item()
            generated_ids.append(next_id)
            context_ids.append(next_id)
    model.train(was_training)
    return tokenizer.decode(generated_ids)
