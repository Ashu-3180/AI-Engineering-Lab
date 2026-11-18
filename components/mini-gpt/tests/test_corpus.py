import hashlib
from dataclasses import replace
from urllib.error import URLError

import pytest

from mini_gpt.corpus import (
    ChecksumMismatchError,
    CorpusError,
    MissingCorpusError,
    acquire_corpus,
    calculate_sha256,
    load_provenance,
    make_provenance,
    read_corpus,
    write_provenance,
)


def write_verified_corpus(tmp_path, text="To be"):
    corpus = tmp_path / "input.txt"
    provenance_path = tmp_path / "provenance.json"
    data = text.encode("utf-8")
    corpus.write_bytes(data)
    write_provenance(
        provenance_path,
        make_provenance("tinyshakespeare", "https://example.test/input.txt", data, text),
    )
    return corpus, provenance_path


def test_valid_corpus_loads_and_checksum_is_correct(tmp_path):
    corpus, provenance_path = write_verified_corpus(tmp_path)
    assert read_corpus(corpus, provenance_path, expected_dataset_id="tinyshakespeare") == "To be"
    assert calculate_sha256(b"To be") == hashlib.sha256(b"To be").hexdigest()


@pytest.mark.parametrize("content", [b"", b"\xff\xfe"])
def test_empty_or_invalid_utf8_corpus_is_rejected(tmp_path, content):
    corpus, provenance_path = write_verified_corpus(tmp_path)
    corpus.write_bytes(content)
    with pytest.raises(CorpusError):
        read_corpus(corpus, provenance_path, expected_dataset_id="tinyshakespeare")


def test_missing_corpus_or_provenance_is_handled(tmp_path):
    with pytest.raises(MissingCorpusError):
        read_corpus(
            tmp_path / "missing.txt",
            tmp_path / "provenance.json",
            expected_dataset_id="tinyshakespeare",
        )
    corpus = tmp_path / "input.txt"
    corpus.write_text("text", encoding="utf-8")
    with pytest.raises(CorpusError, match="Provenance file does not exist"):
        read_corpus(corpus, tmp_path / "missing.json", expected_dataset_id="tinyshakespeare")


def test_local_utf8_modification_with_changed_size_is_rejected(tmp_path):
    corpus, provenance_path = write_verified_corpus(tmp_path)
    corpus.write_text("To bee", encoding="utf-8")
    with pytest.raises(CorpusError, match="file size"):
        read_corpus(corpus, provenance_path, expected_dataset_id="tinyshakespeare")


def test_character_count_and_checksum_mismatches_are_rejected(tmp_path):
    corpus, provenance_path = write_verified_corpus(tmp_path, "é")
    provenance = load_provenance(provenance_path)
    write_provenance(provenance_path, replace(provenance, character_count=2))
    with pytest.raises(CorpusError, match="character count"):
        read_corpus(corpus, provenance_path, expected_dataset_id="tinyshakespeare")
    write_provenance(provenance_path, replace(provenance, sha256="0" * 64))
    with pytest.raises(ChecksumMismatchError, match="checksum"):
        read_corpus(corpus, provenance_path, expected_dataset_id="tinyshakespeare")


def test_provenance_round_trip_is_valid_and_malformed_is_rejected(tmp_path):
    data = b"hello"
    provenance = make_provenance("tinyshakespeare", "https://example.test/input.txt", data, "hello")
    path = tmp_path / "provenance.json"
    write_provenance(path, provenance)
    assert load_provenance(path) == provenance
    path.write_text('{"acquired_at_utc": "not valid"}', encoding="utf-8")
    with pytest.raises(CorpusError, match="invalid"):
        load_provenance(path)


@pytest.mark.parametrize("timestamp", ["2026-01-01T00:00:00", "2026-01-01T01:00:00+01:00"])
def test_non_utc_provenance_timestamp_is_rejected(tmp_path, timestamp):
    _, provenance_path = write_verified_corpus(tmp_path)
    provenance = load_provenance(provenance_path)
    provenance_path.write_text(
        provenance_path.read_text(encoding="utf-8").replace(provenance.acquired_at_utc, timestamp),
        encoding="utf-8",
    )
    with pytest.raises(CorpusError, match="UTC"):
        load_provenance(provenance_path)


def test_provenance_write_preserves_existing_record_when_write_fails(tmp_path, monkeypatch):
    _, provenance_path = write_verified_corpus(tmp_path)
    original = provenance_path.read_bytes()
    replacement = make_provenance(
        "tinyshakespeare", "https://example.test/input.txt", b"new", "new"
    )
    monkeypatch.setattr(
        "mini_gpt.corpus.atomic_write_bytes",
        lambda *args: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        write_provenance(provenance_path, replacement)
    assert provenance_path.read_bytes() == original


class Response:
    def __init__(self, status=200, data=b"new"):
        self.status = status
        self.data = data

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_successful_acquisition_writes_verified_corpus_and_provenance(tmp_path, monkeypatch):
    destination = tmp_path / "input.txt"
    provenance_path = tmp_path / "provenance.json"
    monkeypatch.setattr("mini_gpt.corpus.urlopen", lambda *args, **kwargs: Response(data=b"new"))
    provenance = acquire_corpus(
        dataset_id="tinyshakespeare",
        source_url="https://example.test/input.txt",
        destination=destination,
        provenance_path=provenance_path,
    )
    assert destination.read_bytes() == b"new"
    assert load_provenance(provenance_path) == provenance
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("response", [Response(status=404), Response(data=b"\xff")])
def test_invalid_download_response_is_rejected(tmp_path, monkeypatch, response):
    monkeypatch.setattr("mini_gpt.corpus.urlopen", lambda *args, **kwargs: response)
    with pytest.raises(CorpusError):
        acquire_corpus(
            dataset_id="tinyshakespeare",
            source_url="https://example.test/input.txt",
            destination=tmp_path / "input.txt",
            provenance_path=tmp_path / "provenance.json",
        )


def test_url_failure_and_timeout_are_handled(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mini_gpt.corpus.urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")),
    )
    with pytest.raises(CorpusError, match="download failed"):
        acquire_corpus(
            dataset_id="tinyshakespeare",
            source_url="https://example.test/input.txt",
            destination=tmp_path / "input.txt",
            provenance_path=tmp_path / "provenance.json",
        )
    with pytest.raises(ValueError, match="timeout"):
        acquire_corpus(
            dataset_id="tinyshakespeare",
            source_url="https://example.test/input.txt",
            destination=tmp_path / "input.txt",
            provenance_path=tmp_path / "provenance.json",
            timeout_seconds=0,
        )


def test_changed_upstream_content_is_not_accepted(tmp_path, monkeypatch):
    destination = tmp_path / "input.txt"
    provenance_path = tmp_path / "provenance.json"
    write_provenance(
        provenance_path,
        make_provenance("tinyshakespeare", "https://example.test/input.txt", b"old", "old"),
    )
    monkeypatch.setattr("mini_gpt.corpus.urlopen", lambda *args, **kwargs: Response(data=b"new"))
    with pytest.raises(ChecksumMismatchError):
        acquire_corpus(
            dataset_id="tinyshakespeare",
            source_url="https://example.test/input.txt",
            destination=destination,
            provenance_path=provenance_path,
        )
    assert not destination.exists()
