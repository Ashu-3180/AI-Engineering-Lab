"""A deterministic, standalone character-level tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CharacterTokenizer:
    characters: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.characters or tuple(sorted(set(self.characters))) != self.characters:
            raise ValueError("Tokenizer characters must be a non-empty sorted unique sequence")

    @classmethod
    def from_text(cls, text: str) -> CharacterTokenizer:
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text")
        return cls(tuple(sorted(set(text))))

    @property
    def vocabulary_size(self) -> int:
        return len(self.characters)

    @property
    def char_to_id(self) -> dict[str, int]:
        return {character: index for index, character in enumerate(self.characters)}

    @property
    def id_to_char(self) -> dict[int, str]:
        return dict(enumerate(self.characters))

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Text to encode must be a string")
        mapping = self.char_to_id
        try:
            return [mapping[character] for character in text]
        except KeyError as error:
            raise ValueError(f"Unknown character: {error.args[0]!r}") from error

    def decode(self, token_ids: list[int]) -> str:
        mapping = self.id_to_char
        decoded: list[str] = []
        for token_id in token_ids:
            if (
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id not in mapping
            ):
                raise ValueError(f"Invalid token id: {token_id!r}")
            decoded.append(mapping[token_id])
        return "".join(decoded)

    def to_metadata(self) -> dict[str, object]:
        return {"type": "character", "characters": list(self.characters)}

    @classmethod
    def from_metadata(cls, metadata: dict[str, object]) -> CharacterTokenizer:
        if metadata.get("type") != "character" or not isinstance(metadata.get("characters"), list):
            raise ValueError("Invalid character tokenizer metadata")
        characters = metadata["characters"]
        if not all(isinstance(character, str) and len(character) == 1 for character in characters):
            raise ValueError("Tokenizer metadata characters must be individual strings")
        return cls(tuple(characters))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_metadata(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> CharacterTokenizer:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"Tokenizer metadata does not exist: {path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Tokenizer metadata is invalid JSON: {path}") from error
        if not isinstance(metadata, dict):
            raise ValueError("Tokenizer metadata must be an object")
        return cls.from_metadata(metadata)
