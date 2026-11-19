
#!/usr/bin/env python3
"""Audit normalized official documentation pages T01-T20.

Run from the repository root:
python docs/rag/source-curation/scripts/audit_official_docs.py

Outputs:
data/interim/rag/official-documentation/quality_audit.json
data/interim/rag/official-documentation/quality_audit.txt

This is deterministic triage, not proof of semantic correctness.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

TEXT_DIR = (
    ROOT / "data/interim/rag/official-documentation/pages"
)

REPORT = (
    ROOT / "data/raw/rag/official-documentation/acquisition_report.json"
)

OUT = (
    ROOT / "data/interim/rag/official-documentation/quality_audit.json"
)

TXT = (
    ROOT / "data/interim/rag/official-documentation/quality_audit.txt"
)


# Topic anchors are source-specific indicators.
# They test whether representative concepts are present,
# not whether the document is semantically complete.

ANCHORS = {
    "T01": ["data model", "__hash__", "__getitem__", "mutable"],
    "T02": ["try", "except", "finally", "raise"],
    "T03": ["venv", "virtual environment", "activate"],
    "T04": ["ON CONFLICT", "DO UPDATE", "DO NOTHING"],
    "T05": ["COMMIT", "ROLLBACK", "BEGIN"],
    "T06": ["B-tree", "index"],
    "T07": ["autograd", "backward", "gradient"],
    "T08": ["nn.Module", "parameters", "forward"],
    "T09": ["DataLoader", "Dataset", "batch_size"],
    "T10": ["no_grad", "gradient", "inference"],
    "T11": ["Collection", "query", "get", "add"],
    "T12": [
        "K.DOCUMENT",
        ".is_in(",
        ".not_in(",
        "Document operators",
    ],
    "T13": ["PersistentClient", "path", "Client"],
    "T14": ["DoclingDocument", "provenance", "tables"],
    "T15": ["export", "table", "dataframe", "CSV"],
    "T16": ["tools/list", "tools/call", "inputSchema"],
    "T17": ["host", "client", "server"],
    "T18": ["stdio", "stdin", "stdout", "stderr", "JSON-RPC"],
    "T19": ["Link", "per_page", "page", "pagination"],
    "T20": ["rate limit", "x-ratelimit", "429", "403"],
}


def sha256(path: Path) -> str:
    """Return SHA-256 of the exact file bytes."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def canonicalize_text(raw_text: str) -> str:
    """Normalize Windows and Unix line endings for auditing."""
    return (
        raw_text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def main() -> int:
    if not REPORT.is_file():
        raise SystemExit(
            f"Missing acquisition report: {REPORT}"
        )

    if not TEXT_DIR.is_dir():
        raise SystemExit(
            f"Missing text directory: {TEXT_DIR}"
        )

    acquisition = json.loads(
        REPORT.read_text(encoding="utf-8")
    )

    records = {
        item["source_id"]: item
        for item in acquisition.get("records", [])
    }

    expected = {
        f"T{i:02d}" for i in range(1, 21)
    }

    actual = {
        path.stem for path in TEXT_DIR.glob("T*.txt")
    }

    documents = []
    errors = []

    for source_id in sorted(expected & actual):
        path = TEXT_DIR / f"{source_id}.txt"

        raw_bytes = path.read_bytes()

        decoded_text = raw_bytes.decode(
            "utf-8",
            errors="replace"
        )

        canonical_text = canonicalize_text(decoded_text)

        # The acquisition script writes body + one newline.
        # Acquisition metrics count body characters, excluding
        # that final file-terminating newline.
        normalized_body = (
            canonical_text[:-1]
            if canonical_text.endswith("\n")
            else canonical_text
        )

        words = normalized_body.split()

        lines = [
            line.strip()
            for line in normalized_body.splitlines()
            if line.strip()
        ]

        normalized_lines = [
            re.sub(
                r"\d+",
                "#",
                re.sub(r"\s+", " ", line).strip().lower()
            )
            for line in lines
        ]

        counts = Counter(
            line
            for line in normalized_lines
            if len(line) >= 12
        )

        repeated_count = sum(
            count - 1
            for count in counts.values()
            if count > 1
        )

        repeated_lines = [
            {"line": line, "count": count}
            for line, count in counts.most_common(8)
            if count >= 3
        ]

        terms = ANCHORS[source_id]

        found = [
            term
            for term in terms
            if term.casefold() in normalized_body.casefold()
        ]

        missing_anchors = [
            term
            for term in terms
            if term not in found
        ]

        control_characters = sum(
            unicodedata.category(char) == "Cc"
            and char not in "\n\r\t"
            for char in normalized_body
        )

        replacement_characters = normalized_body.count("\ufffd")

        invisible_characters = sum(
            char in "\u200b\u200c\u200d\ufeff"
            for char in normalized_body
        )

        long_tokens = [
            word
            for word in words
            if len(word) > 100
        ]

        long_lines = [
            line
            for line in lines
            if len(line) > 800
        ]

        acquisition_record = records.get(source_id, {})

        flags = []

        if len(words) < 200:
            flags.append("low_word_count")

        if replacement_characters:
            flags.append("replacement_unicode")

        if control_characters:
            flags.append("control_characters")

        if invisible_characters > 10:
            flags.append("invisible_format_characters")

        if long_tokens:
            flags.append("very_long_tokens")

        if len(long_lines) > 10:
            flags.append("many_very_long_lines")

        repeated_line_ratio = (
            repeated_count / max(1, len(lines))
        )

        if lines and repeated_line_ratio > 0.20:
            flags.append("repeated_line_noise")

        if missing_anchors:
            flags.append("topic_anchor_terms_missing")

        normalized_characters = len(normalized_body)
        normalized_words = len(words)

        reported_words = acquisition_record.get(
            "normalized_words"
        )

        reported_characters = acquisition_record.get(
            "normalized_characters"
        )

        word_count_matches = (
            normalized_words == reported_words
            if acquisition_record
            else None
        )

        character_count_matches = (
            normalized_characters == reported_characters
            if acquisition_record
            else None
        )

        documents.append({
            "source_id": source_id,
            "path": path.relative_to(ROOT).as_posix(),

            # Hash of exact stored bytes, preserving file integrity.
            "normalized_text_sha256": sha256(path),

            # Metrics use canonicalized text and exclude the final
            # file-terminating newline.
            "characters": normalized_characters,
            "words": normalized_words,

            "nonempty_lines": len(lines),
            "repeated_line_ratio": round(
                repeated_line_ratio, 4
            ),

            "top_repeated_lines": repeated_lines,

            "long_token_count": len(long_tokens),
            "long_token_examples": long_tokens[:5],
            "long_line_count": len(long_lines),

            "replacement_unicode_count": replacement_characters,
            "unexpected_control_characters": control_characters,
            "invisible_format_characters": invisible_characters,

            "topic_anchor_terms": terms,
            "topic_anchor_terms_found": found,
            "topic_anchor_terms_missing": missing_anchors,

            "reported_words": reported_words,
            "reported_characters": reported_characters,

            "word_count_matches_report": word_count_matches,
            "character_count_matches_report": character_count_matches,

            "flags": flags,
            "preview": normalized_body[:400].replace(
                "\n", " ⏎ "
            ),
        })

    missing_files = sorted(expected - actual)
    unexpected_files = sorted(actual - expected)

    missing_records = sorted(
        expected - set(records)
    )

    if missing_files:
        errors.append(
            "Missing text files: "
            + ", ".join(missing_files)
        )

    if unexpected_files:
        errors.append(
            "Unexpected text files: "
            + ", ".join(unexpected_files)
        )

    if missing_records:
        errors.append(
            "Missing acquisition report entries: "
            + ", ".join(missing_records)
        )

    metric_mismatches = [
        document["source_id"]
        for document in documents
        if (
            document["word_count_matches_report"] is False
            or document["character_count_matches_report"] is False
        )
    ]

    if metric_mismatches:
        errors.append(
            "Normalized text metrics differ from acquisition report: "
            + ", ".join(metric_mismatches)
        )

    result = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "audit_version": "0.2.0",

        "scope": (
            "official documentation T01-T20 normalized text"
        ),

        "audit_type": (
            "deterministic structural and topic-anchor triage"
        ),

        "expected_documents": 20,
        "found_documents": len(documents),

        "missing_files": missing_files,
        "unexpected_files": unexpected_files,
        "missing_report_records": missing_records,

        "documents_flagged": sum(
            bool(document["flags"])
            for document in documents
        ),

        "total_words": sum(
            document["words"]
            for document in documents
        ),

        "total_characters": sum(
            document["characters"]
            for document in documents
        ),

        "documents": documents,
        "errors": errors,

        "limitations": [
            (
                "Topic anchors show term presence only, "
                "not correctness or complete answer support."
            ),
            (
                "Repeated lines may be legitimate in API references "
                "and code examples."
            ),
            (
                "Very long tokens may be legitimate API identifiers; "
                "review before removing them."
            ),
            (
                "Flattened HTML can lose table structure, code formatting "
                "and heading hierarchy."
            ),
            (
                "Manual review of representative source pages is required "
                "before indexing approval."
            ),
        ],
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ) + "\n",
        encoding="utf-8"
    )

    report_lines = [
        f"Documents: {len(documents)}/20",
        f"Flagged: {result['documents_flagged']}",
        f"Total words: {result['total_words']:,}",
        f"Total characters: {result['total_characters']:,}",
        "",
    ]

    for document in documents:
        label = (
            "REVIEW"
            if document["flags"]
            else "OK"
        )

        report_lines.append(
            f"[{label}] {document['source_id']}: "
            f"{document['words']:,} words; "
            f"repeated-line ratio="
            f"{document['repeated_line_ratio']:.1%}"
        )

        if document["flags"]:
            report_lines.append(
                "  " + ", ".join(document["flags"])
            )

        if document["topic_anchor_terms_missing"]:
            report_lines.append(
                "  missing anchors: "
                + ", ".join(
                    document["topic_anchor_terms_missing"]
                )
            )

    if errors:
        report_lines.extend(
            ["", "ERRORS:"] + errors
        )

    report_lines.extend([
        "",
        "JSON: " + OUT.relative_to(ROOT).as_posix(),
    ])

    TXT.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8"
    )

    print("\n".join(report_lines))

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())