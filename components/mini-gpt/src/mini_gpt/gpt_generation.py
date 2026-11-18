"""Autoregressive inference for trained M5 GPT models."""

from __future__ import annotations

import torch

from .batching import make_seeded_generator
from .models.gpt import GPTLanguageModel
from .tokenizer import CharacterTokenizer


def generate_gpt_text(
    model: GPTLanguageModel,
    tokenizer: CharacterTokenizer,
    *,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    greedy: bool = False,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    ids = tokenizer.encode(prompt)
    generated = list(ids)
    context = ids[-model.config.block_size :] or [0]
    generator = make_seeded_generator(seed)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(torch.tensor([context[-model.config.block_size :]], dtype=torch.long))
            probabilities = torch.softmax(logits[:, -1], dim=-1)
            next_id = (
                probabilities.argmax(dim=-1).item()
                if greedy
                else torch.multinomial(probabilities, 1, generator=generator).item()
            )
            generated.append(next_id)
            context.append(next_id)
    model.train(was_training)
    return tokenizer.decode(generated)
