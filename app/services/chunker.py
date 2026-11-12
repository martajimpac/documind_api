"""PDF text extraction and splitting into chunks the model reads whole (SPEC RF-01, RF-07)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Protocol

from app.core.config import MAX_MODEL_TOKENS, MAX_PAGES

MAX_CHUNK_PIECES = MAX_MODEL_TOKENS - 2  # room for the two special tokens


class FileErrorReason(str, Enum):
    """Errors caused by the file itself: retrying would fail again (SPEC RF-06)."""

    NO_TEXT = "no_text"
    TOO_MANY_PAGES = "too_many_pages"


class DocumentFileError(Exception):
    def __init__(self, reason: FileErrorReason) -> None:
        super().__init__(reason.value)
        self.reason = reason


@dataclass(frozen=True)
class Chunk:
    index: int
    page_number: int  # 1-based
    content: str


class PieceTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def starts_word(self, piece_id: int) -> bool: ...


def read_pages(path: Path, max_pages: int = MAX_PAGES) -> Iterator[str]:
    """Yields the text of each page. Raises DocumentFileError for files that cannot be used."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")  # PDFs protected only against editing open with an empty password
        page_count = len(reader.pages)
    except (PdfReadError, ValueError, NotImplementedError) as e:
        raise DocumentFileError(FileErrorReason.NO_TEXT) from e
    if page_count > max_pages:
        raise DocumentFileError(FileErrorReason.TOO_MANY_PAGES)
    for page in reader.pages:
        try:
            yield page.extract_text() or ""
        except (PdfReadError, ValueError, KeyError):
            yield ""  # one unreadable page should not lose the rest of the document


def normalize(text: str) -> str:
    return " ".join(text.split())


def split_text(text: str, tokenizer: PieceTokenizer, max_pieces: int = MAX_CHUNK_PIECES) -> list[str]:
    """Splits [text] in order, without overlap, into pieces of at most [max_pieces] tokens.

    Cuts at the start of a word when possible. Each result is checked again, because
    decoding and encoding a cut that falls inside a word may give more pieces.
    """
    ids = tokenizer.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(ids):
        end = min(start + max_pieces, len(ids))
        if end < len(ids):
            boundary = next((k for k in range(end, start, -1) if tokenizer.starts_word(ids[k])), None)
            if boundary is not None:
                end = boundary
        while True:
            chunk = tokenizer.decode(ids[start:end]).strip()
            if end - start <= 1 or len(tokenizer.encode(chunk)) <= max_pieces:
                break
            end -= 1
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def build_chunks(pages: Iterator[str], tokenizer: PieceTokenizer) -> list[Chunk]:
    """Chunks of every page in document order. A chunk never spans two pages."""
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        for content in split_text(normalize(page_text), tokenizer):
            chunks.append(Chunk(index=len(chunks), page_number=page_number, content=content))
    if not chunks:
        raise DocumentFileError(FileErrorReason.NO_TEXT)
    return chunks
