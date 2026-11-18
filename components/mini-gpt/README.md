# Mini GPT

An educational implementation of a character-level language-model data pipeline. This
milestone provides corpus acquisition, provenance, tokenization, deterministic dataset
splitting, and model-ready batch generation. It deliberately contains no model or
training loop.

## Setup

Use CPython 3.15.2 and [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --group dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

The Tiny Shakespeare corpus is intentionally excluded from version control. Acquire it
with `python -m mini_gpt.acquire`; its provenance is retained in
`data/provenance/tinyshakespeare.json` after the first successful download.

The corpus is obtained from the original char-rnn repository. Its licence/status must
be reviewed independently; this project makes no claim about its licensing.
