
#!/usr/bin/env python3
"""Acquire or reprocess official documentation sources.

Run from AI-Engineering-Lab root:

Acquire sources:
python docs/rag/source-curation/scripts/acquire_official_docs.py

Reprocess selected existing HTML without downloading:
python docs/rag/source-curation/scripts/acquire_official_docs.py --reprocess-existing T11 T12 T13

Raw HTML:
data/raw/rag/official-documentation/html/

Normalized text:
data/interim/rag/official-documentation/pages/

Acquisition report:
data/raw/rag/official-documentation/acquisition_report.json

Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time

from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[4]

REG = (
    ROOT / "docs/rag/source-curation/"
    "official_documentation_register_v0_1_0.json"
)

RAW = (
    ROOT / "data/raw/rag/official-documentation/html"
)

TXT = (
    ROOT / "data/interim/rag/official-documentation/pages"
)

REPORT = (
    ROOT / "data/raw/rag/official-documentation/"
    "acquisition_report.json"
)

UA = "AI-Engineering-Lab-RAG-Documentation-Acquirer/0.2"

PARSER_VERSION = "0.2.0"

SKIP_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "template",
}

BLOCK_TAGS = {
    "p", "div", "section", "article", "main",
    "li", "ul", "ol", "h1", "h2", "h3", "h4",
    "tr", "td", "th", "pre", "blockquote", "br",
}


class Parser(HTMLParser):
    """Extract visible HTML text while excluding non-visible payloads."""

    def __init__(self):
        super().__init__(convert_charrefs=True)

        self.parts = []
        self.title = []
        self.in_title = False
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()

        if tag in SKIP_TAGS:
            self.skip_depth += 1
            return

        if self.skip_depth:
            return

        if tag == "title":
            self.in_title = True

        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag in SKIP_TAGS:
            if self.skip_depth:
                self.skip_depth -= 1
            return

        if self.skip_depth:
            return

        if tag == "title":
            self.in_title = False

        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            return

        if self.in_title:
            self.title.append(data)

        self.parts.append(data)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def norm(value: str) -> str:
    value = (
        value
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x00", "")
    )

    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()


def extract_document(raw: bytes) -> tuple[str, str]:
    """Return normalized visible text and document title."""

    parser = Parser()

    parser.feed(
        raw.decode("utf-8", errors="replace")
    )

    body = norm("".join(parser.parts))
    title = norm(" ".join(parser.title))

    if len(body) < 250:
        raise RuntimeError(
            f"Suspiciously short extracted body: {len(body)} chars"
        )

    return body, title


def write_text(path: Path, content: str) -> None:
    """Write text using the platform's normal text encoding behavior."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        content,
        encoding="utf-8"
    )


def get(item: dict) -> dict:
    """Download one registered document and record its provenance."""

    request = Request(
        item["url"],
        headers={
            "User-Agent": UA,
            "Accept": (
                "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Encoding": "identity",
        },
    )

    with urlopen(request, timeout=60) as response:
        raw = response.read()
        status = getattr(response, "status", 200)
        final_url = response.geturl()
        headers = response.headers

    if not raw or not 200 <= status < 300:
        raise RuntimeError(
            f"HTTP status={status}, bytes={len(raw)}"
        )

    body, title = extract_document(raw)

    RAW.mkdir(
        parents=True,
        exist_ok=True
    )

    TXT.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_path = RAW / f"{item['source_id']}.html"
    text_path = TXT / f"{item['source_id']}.txt"

    raw_path.write_bytes(raw)
    write_text(text_path, body + "\n")

    return {
        "source_id": item["source_id"],
        "title": item["title"],
        "page_title": title,

        "requested_url": item["url"],
        "final_url": final_url,
        "version_or_snapshot": item["version_or_snapshot"],
        "licence_classification": item["licence_classification"],

        "http_status": status,
        "content_type": headers.get("Content-Type"),
        "last_modified": headers.get("Last-Modified"),
        "etag": headers.get("ETag"),

        "acquired_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "raw_html_path": raw_path.relative_to(
            ROOT
        ).as_posix(),

        "normalized_text_path": text_path.relative_to(
            ROOT
        ).as_posix(),

        "raw_bytes": len(raw),
        "raw_html_sha256": sha(raw),

        "normalized_characters": len(body),
        "normalized_words": len(body.split()),

        "acquisition_status": "downloaded_hash_recorded",
    }


def acquire_all() -> int:
    """Perform the normal registered-source acquisition."""

    if not REG.is_file():
        raise SystemExit(
            f"Missing source register: {REG}"
        )

    register = json.loads(
        REG.read_text(encoding="utf-8")
    )

    records = []
    failures = []

    for index, item in enumerate(register["sources"]):
        try:
            record = get(item)
            records.append(record)

            print(
                f"[OK] {item['source_id']}: "
                f"{record['raw_bytes']:,} bytes, "
                f"{record['normalized_words']:,} words"
            )

        except Exception as error:
            failures.append({
                "source_id": item["source_id"],
                "url": item["url"],
                "error": str(error),
            })

            print(
                f"[FAIL] {item['source_id']}: {error}"
            )

        if index < len(register["sources"]) - 1:
            time.sleep(1)

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "normalization": {
            "parser_version": PARSER_VERSION,
            "excluded_html_elements": sorted(SKIP_TAGS),
        },

        "records": records,
        "failures": failures,

        "totals": {
            "expected": len(register["sources"]),
            "downloaded": len(records),
            "failed": len(failures),

            "raw_bytes": sum(
                record["raw_bytes"]
                for record in records
            ),

            "normalized_words": sum(
                record["normalized_words"]
                for record in records
            ),

            "normalized_characters": sum(
                record["normalized_characters"]
                for record in records
            ),
        },

        "warning": (
            "Independently recompute hashes from local raw HTML "
            "before updating the G1 manifest. Extraction requires "
            "quality review."
        ),
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ) + "\n",
        encoding="utf-8"
    )

    print(
        f"Downloaded: {len(records)}/{len(register['sources'])}; "
        f"failures: {len(failures)}"
    )

    print(
        f"Report: {REPORT.relative_to(ROOT)}"
    )

    return 1 if failures else 0


def reprocess_existing(source_ids: list[str]) -> int:
    """Re-extract selected sources from verified local raw HTML."""

    if not REPORT.is_file():
        raise SystemExit(
            f"Missing acquisition report: {REPORT}"
        )

    if not source_ids:
        raise SystemExit(
            "Provide at least one source ID to reprocess."
        )

    if len(source_ids) != len(set(source_ids)):
        raise SystemExit(
            "Duplicate source IDs supplied."
        )

    report = json.loads(
        REPORT.read_text(encoding="utf-8")
    )

    records = report.get("records", [])

    records_by_id = {
        record["source_id"]: record
        for record in records
    }

    unknown = sorted(
        set(source_ids) - set(records_by_id)
    )

    if unknown:
        raise SystemExit(
            "Source IDs missing from acquisition report: "
            + ", ".join(unknown)
        )

    # Prepare and verify every requested document before
    # writing any normalized output.

    prepared = []

    for source_id in source_ids:
        record = records_by_id[source_id]

        raw_path = ROOT / record["raw_html_path"]
        text_path = ROOT / record["normalized_text_path"]

        if not raw_path.is_file():
            raise SystemExit(
                f"Missing raw HTML for {source_id}: {raw_path}"
            )

        raw = raw_path.read_bytes()

        expected_bytes = record.get("raw_bytes")
        expected_hash = record.get("raw_html_sha256")

        if expected_bytes is None or expected_hash is None:
            raise SystemExit(
                f"Missing original integrity metadata for {source_id}. "
                "Refusing to reprocess."
            )

        actual_bytes = len(raw)
        actual_hash = sha(raw)

        if actual_bytes != expected_bytes:
            raise SystemExit(
                f"Raw byte-count mismatch for {source_id}: "
                f"expected {expected_bytes}, got {actual_bytes}. "
                "No normalized files were changed."
            )

        if actual_hash != expected_hash:
            raise SystemExit(
                f"Raw SHA-256 mismatch for {source_id}. "
                "No normalized files were changed."
            )

        body, title = extract_document(raw)

        prepared.append({
            "source_id": source_id,
            "record": record,
            "text_path": text_path,
            "body": body,
            "title": title,
            "characters": len(body),
            "words": len(body.split()),
        })

        print(
            f"[RAW VERIFIED] {source_id}: "
            f"{actual_bytes:,} bytes; SHA-256 unchanged"
        )

    # All source files passed integrity checks.
    # Only normalized text and its derived metrics are updated.

    for item in prepared:
        write_text(
            item["text_path"],
            item["body"] + "\n"
        )

        record = item["record"]

        record["page_title"] = item["title"]
        record["normalized_characters"] = item["characters"]
        record["normalized_words"] = item["words"]

        print(
            f"[REPROCESSED] {item['source_id']}: "
            f"{item['words']:,} words, "
            f"{item['characters']:,} characters"
        )

    report["normalization"] = {
        "parser_version": PARSER_VERSION,
        "excluded_html_elements": sorted(SKIP_TAGS),
        "last_reprocessed_source_ids": source_ids,
        "last_reprocessed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "raw_html_modified": False,
    }

    report["updated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat()

    totals = report.setdefault("totals", {})

    totals["normalized_words"] = sum(
        record.get("normalized_words", 0)
        for record in records
    )

    totals["normalized_characters"] = sum(
        record.get("normalized_characters", 0)
        for record in records
    )

    report["records"] = records

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ) + "\n",
        encoding="utf-8"
    )

    print(
        f"\nReprocessed {len(prepared)} existing source(s)."
    )

    print(
        "Original raw HTML files and SHA-256 records preserved."
    )

    print(
        f"Updated report: {REPORT.relative_to(ROOT)}"
    )

    return 0


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Acquire official documentation or reprocess "
            "previously verified local HTML."
        )
    )

    argument_parser.add_argument(
        "--reprocess-existing",
        nargs="+",
        metavar="SOURCE_ID",
        help=(
            "Reprocess selected existing sources without downloading. "
            "Example: --reprocess-existing T11 T12 T13"
        ),
    )

    args = argument_parser.parse_args()

    if args.reprocess_existing is not None:
        return reprocess_existing(
            args.reprocess_existing
        )

    return acquire_all()


if __name__ == "__main__":
    sys.exit(main())