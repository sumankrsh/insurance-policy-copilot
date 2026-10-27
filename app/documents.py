"""Load the policy PDFs and split them into one chunk per section per page.

Section-level chunks keep each citation pointing at a heading a person can find in
the PDF ("Term Life Insurance Policy, page 1, Section 3: Surrender value and charges").
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

BULLETS = "\x7f•▪●"  # pypdf renders the PDFs' bullet glyph as \x7f
SECTION_RE = re.compile(r"^SECTION(?:\s+(\d+))?\s*:\s*(.+)$", re.IGNORECASE)
SUBHEADING_RE = re.compile(r"^\d+\.\d+\s")
ACRONYMS = {"ulip": "ULIP", "kyc": "KYC", "irdai": "IRDAI"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_file: str
    doc_title: str
    page: int
    section: str
    lines: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def label(self) -> str:
        return f"{self.doc_title}, page {self.page}, {self.section}"


def _section_title(number: str | None, raw_title: str) -> str:
    words = raw_title.strip().lower().split()
    words = [ACRONYMS.get(w, w) for w in words]
    title = " ".join(words)
    title = title[0].upper() + title[1:]
    return f"Section {number}: {title}" if number else title


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _page_lines(page) -> list[str]:
    """Text lines of a page. Layout mode keeps table rows on one line;
    cells are joined with ' | '. Wrapped sentences are re-joined."""
    out: list[str] = []
    for raw in page.extract_text(extraction_mode="layout").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        is_bullet = stripped[0] in BULLETS
        line = stripped.lstrip(BULLETS).strip()
        line = re.sub(r"\s{3,}", " | ", line)
        line = re.sub(r"\s+", " ", line)
        if not line:
            continue
        continuation = out and not is_bullet and line[0].islower() and not out[-1].endswith((".", ":"))
        if continuation:
            out[-1] = f"{out[-1]} {line}"
        else:
            out.append(line)
    return out


def load_chunks(docs_dir: Path) -> list[Chunk]:
    meta = json.loads((docs_dir / "metadata.json").read_text())
    chunks: list[Chunk] = []
    for doc in meta["documents"]:
        path = docs_dir / doc["filename"]
        section: str | None = None
        for page_no, page in enumerate(PdfReader(path).pages, start=1):
            # (section, lines) groups on this page; a section that continues from the
            # previous page starts a new group with the same heading
            groups: list[tuple[str | None, list[str]]] = [(section, [])]
            for line in _page_lines(page):
                match = SECTION_RE.match(line)
                if match:
                    section = _section_title(match.group(1), match.group(2))
                    groups.append((section, []))
                else:
                    groups[-1][1].append(line)
            for group_section, lines in groups:
                if group_section and lines:  # lines before the first heading are the title
                    chunks.append(
                        Chunk(
                            chunk_id=f"{path.stem}:p{page_no}:{_slug(group_section)}",
                            source_file=doc["filename"],
                            doc_title=doc["title"],
                            page=page_no,
                            section=group_section,
                            lines=tuple(lines),
                        )
                    )
    return chunks
