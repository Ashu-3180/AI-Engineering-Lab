"""Tiny Shakespeare acquisition, validation, and provenance handling."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.error import URLError
from urllib.request import urlopen


class CorpusError(RuntimeError):
    """Base error for corpus acquisition and validation failures."""


class MissingCorpusError(CorpusError):
    pass


class ChecksumMismatchError(CorpusError):
    pass


@dataclass(frozen=True)
class DatasetPaths:
    raw_file: Path
    provenance_file: Path


@dataclass(frozen=True)
class Provenance:
    dataset_id: str
    source_url: str
    acquired_at_utc: str
    sha256: str
    file_size_bytes: int
    encoding: str
    character_count: int
    licensing_status_notes: str


def resolve_dataset_paths(
    project_root: Path,
    dataset_id: str,
    *,
    raw_data_dir: Path = Path("data/raw"),
    provenance_dir: Path = Path("data/provenance"),
) -> DatasetPaths:
    """Resolve the only supported Tiny Shakespeare storage locations."""
    if dataset_id != "tinyshakespeare":
        raise ValueError(f"Unsupported dataset id: {dataset_id}")
    from .config import resolve_component_path

    return DatasetPaths(
        raw_file=resolve_component_path(project_root, raw_data_dir)
        / "tinyshakespeare"
        / "input.txt",
        provenance_file=resolve_component_path(project_root, provenance_dir)
        / "tinyshakespeare.json",
    )


def calculate_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_text(text: str) -> None:
    if not text:
        raise CorpusError("Corpus content is empty")


def read_corpus(path: Path, provenance_path: Path, *, expected_dataset_id: str) -> str:
    """Read a locally stored corpus only after validating established provenance."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise MissingCorpusError(f"Corpus file does not exist: {path}") from error
    if not raw:
        raise CorpusError(f"Corpus file is empty: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CorpusError(f"Corpus is not valid UTF-8: {path}") from error
    validate_text(text)
    validate_local_corpus(raw, text, load_provenance(provenance_path), expected_dataset_id)
    return text


def make_provenance(dataset_id: str, source_url: str, data: bytes, text: str) -> Provenance:
    validate_text(text)
    return Provenance(
        dataset_id=dataset_id,
        source_url=source_url,
        acquired_at_utc=datetime.now(UTC).isoformat(),
        sha256=calculate_sha256(data),
        file_size_bytes=len(data),
        encoding="utf-8",
        character_count=len(text),
        licensing_status_notes=(
            "Original Tiny Shakespeare corpus from karpathy/char-rnn; licensing/status "
            "not independently verified by this project."
        ),
    )


def validate_provenance(provenance: Provenance) -> None:
    if not provenance.dataset_id or not provenance.source_url.startswith("https://"):
        raise CorpusError("Provenance has invalid dataset identity or source URL")
    if len(provenance.sha256) != 64 or any(
        char not in "0123456789abcdef" for char in provenance.sha256
    ):
        raise CorpusError("Provenance has an invalid SHA-256 checksum")
    if provenance.file_size_bytes <= 0 or provenance.character_count <= 0:
        raise CorpusError("Provenance file size and character count must be positive")
    if provenance.encoding.lower() != "utf-8" or not provenance.licensing_status_notes:
        raise CorpusError("Provenance encoding or licensing/status notes are invalid")
    try:
        acquired_at = datetime.fromisoformat(provenance.acquired_at_utc)
    except ValueError as error:
        raise CorpusError("Provenance acquisition timestamp is invalid") from error
    if acquired_at.tzinfo is None or acquired_at.utcoffset() != timedelta(0):
        raise CorpusError("Provenance acquisition timestamp must be timezone-aware UTC")


def load_provenance(path: Path) -> Provenance:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        provenance = Provenance(**values)
    except FileNotFoundError as error:
        raise CorpusError(f"Provenance file does not exist: {path}") from error
    except (json.JSONDecodeError, TypeError) as error:
        raise CorpusError(f"Provenance metadata is invalid: {path}") from error
    validate_provenance(provenance)
    return provenance


def write_provenance(path: Path, provenance: Provenance) -> None:
    validate_provenance(provenance)
    atomic_write_bytes(
        path, (json.dumps(asdict(provenance), indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def validate_local_corpus(
    data: bytes, text: str, provenance: Provenance, expected_dataset_id: str
) -> None:
    """Ensure local bytes are exactly the corpus described by trusted provenance."""
    if provenance.dataset_id != expected_dataset_id:
        raise CorpusError(
            "Provenance dataset id "
            f"{provenance.dataset_id!r} does not match {expected_dataset_id!r}"
        )
    if provenance.encoding.lower() != "utf-8":
        raise CorpusError(f"Provenance encoding must be UTF-8, not {provenance.encoding!r}")
    if len(data) != provenance.file_size_bytes:
        raise CorpusError("Local corpus file size differs from established provenance")
    if len(text) != provenance.character_count:
        raise CorpusError("Local corpus character count differs from established provenance")
    if calculate_sha256(data) != provenance.sha256:
        raise ChecksumMismatchError("Local corpus checksum differs from established provenance")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace a destination after successfully writing a sibling temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp", delete=False
    ) as temporary:
        temporary.write(data)
        temporary_path = Path(temporary.name)
    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def acquire_corpus(
    *,
    dataset_id: str,
    source_url: str,
    destination: Path,
    provenance_path: Path,
    timeout_seconds: float = 30.0,
) -> Provenance:
    """Download and atomically store a UTF-8 corpus, preserving checksum identity."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    try:
        with urlopen(source_url, timeout=timeout_seconds) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
            if status != 200:
                raise CorpusError(f"Download returned HTTP status {status}")
            data = response.read()
    except URLError as error:
        raise CorpusError(f"Corpus download failed: {error.reason}") from error
    if not data:
        raise CorpusError("Downloaded corpus is empty")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CorpusError("Downloaded corpus is not valid UTF-8") from error
    validate_text(text)
    provenance = make_provenance(dataset_id, source_url, data, text)
    if provenance_path.exists():
        established = load_provenance(provenance_path)
        if established.sha256 != provenance.sha256:
            raise ChecksumMismatchError(
                "Downloaded corpus checksum differs from the established provenance record"
            )

    atomic_write_bytes(destination, data)
    if not provenance_path.exists():
        write_provenance(provenance_path, provenance)
    return provenance
