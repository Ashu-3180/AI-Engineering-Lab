import pytest

from mini_gpt.tokenizer import CharacterTokenizer


def test_vocabulary_is_deterministic_and_round_trips():
    first = CharacterTokenizer.from_text("cabca")
    second = CharacterTokenizer.from_text("abc")
    assert first.characters == ("a", "b", "c") == second.characters
    encoded = first.encode("cab")
    assert encoded == [2, 0, 1]
    assert first.decode(encoded) == "cab"


def test_unknown_characters_and_invalid_ids_are_rejected():
    tokenizer = CharacterTokenizer.from_text("ab")
    with pytest.raises(ValueError, match="Unknown character"):
        tokenizer.encode("c")
    with pytest.raises(ValueError, match="Invalid token id"):
        tokenizer.decode([2])
    with pytest.raises(ValueError, match="Invalid token id"):
        tokenizer.decode([True])
    with pytest.raises(ValueError, match="Invalid token id"):
        tokenizer.decode(["0"])
    with pytest.raises(TypeError, match="string"):
        tokenizer.encode(["a"])


def test_saved_tokenizer_restores_exact_mappings(tmp_path):
    tokenizer = CharacterTokenizer.from_text("bca")
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = CharacterTokenizer.load(path)
    assert restored.characters == tokenizer.characters
    assert restored.char_to_id == tokenizer.char_to_id
    assert restored.id_to_char == tokenizer.id_to_char


def test_invalid_or_corrupt_tokenizer_metadata_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid character tokenizer metadata"):
        CharacterTokenizer.from_metadata({"type": "wrong", "characters": ["a"]})
    path = tmp_path / "tokenizer.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        CharacterTokenizer.load(path)
