"""Turns one PDF file into chunks with embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import EMBED_BATCH_SIZE, MAX_PAGES
from app.repositories.supabase_repository import StoredChunk
from app.services.chunker import PieceTokenizer, build_chunks, read_pages


class TextEmbedder(Protocol):
    tokenizer: PieceTokenizer

    def embed(self, texts: list[str]) -> np.ndarray: ...


class DocumentProcessor:
    def __init__(self, embedder: TextEmbedder, max_pages: int = MAX_PAGES, batch_size: int = EMBED_BATCH_SIZE) -> None:
        self._embedder = embedder
        self._max_pages = max_pages
        self._batch_size = batch_size

    def process(self, pdf_path: Path) -> list[StoredChunk]:
        """Raises DocumentFileError when the file has no text or too many pages."""
        chunks = build_chunks(read_pages(pdf_path, self._max_pages), self._embedder.tokenizer)
        result: list[StoredChunk] = []
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            vectors = self._embedder.embed([c.content for c in batch])
            result.extend(
                StoredChunk(index=c.index, page_number=c.page_number, content=c.content, embedding=v.tolist())
                for c, v in zip(batch, vectors)
            )
        return result
