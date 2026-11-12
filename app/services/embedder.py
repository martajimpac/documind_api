"""Local embeddings with paraphrase-multilingual-MiniLM-L12-v2 (ONNX int8).

Low-memory setup measured on Render Free limits (SPEC.md, 2026-11-05): `sentencepiece`
instead of `tokenizers`, no onnxruntime memory arena, one thread and `malloc_trim`.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.config import MAX_MODEL_TOKENS, MODEL_FILE, SENTENCEPIECE_FILE

# XLM-RoBERTa special token ids, as in the Hugging Face tokenizer.
BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
WORD_START = "▁"  # sentencepiece marks the start of a word with "▁"


def make_trim() -> Callable[[], None]:
    """Hands freed memory back to the system (glibc only; a no-op elsewhere)."""
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return lambda: None
    return lambda: libc.malloc_trim(0)


class Tokenizer:
    """XLM-RoBERTa tokenization with `sentencepiece`. One piece is one model token."""

    def __init__(self, model_dir: Path) -> None:
        import sentencepiece

        self._sp = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / SENTENCEPIECE_FILE))

    def encode(self, text: str) -> list[int]:
        return self._sp.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)

    def starts_word(self, piece_id: int) -> bool:
        return self._sp.id_to_piece(piece_id).startswith(WORD_START)

    def count_model_tokens(self, text: str) -> int:
        """Tokens the model sees for [text], including the two special tokens."""
        return len(self.encode(text)) + 2

    def to_model_ids(self, texts: list[str]) -> list[list[int]]:
        rows = []
        for ids in self._sp.encode(texts):
            # Same mapping as the Hugging Face tokenizer: ids shift by one, unknown -> 3.
            body = [UNK_ID if i == 0 else i + 1 for i in ids][: MAX_MODEL_TOKENS - 2]
            rows.append([BOS_ID, *body, EOS_ID])
        return rows


class Embedder:
    def __init__(self, model_dir: Path, tokenizer: Tokenizer | None = None) -> None:
        import onnxruntime as ort

        self._trim = make_trim()
        self.tokenizer = tokenizer or Tokenizer(model_dir)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        self._session = ort.InferenceSession(
            str(model_dir / MODEL_FILE), options, providers=["CPUExecutionProvider"]
        )
        self._needs_token_types = "token_type_ids" in {i.name for i in self._session.get_inputs()}
        self._trim()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns one 384-dimensional vector per text, with the model's mean pooling."""
        rows = self.tokenizer.to_model_ids(texts)
        width = max(len(r) for r in rows)
        input_ids = np.full((len(rows), width), PAD_ID, dtype=np.int64)
        attention_mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = row
            attention_mask[i, : len(row)] = 1
        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self._needs_token_types:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        token_embeddings = self._session.run(None, feeds)[0]
        # Mean pooling over real tokens, as in 1_Pooling/config.json. Not normalized:
        # searches use cosine distance.
        mask = attention_mask[..., None].astype(np.float32)
        vectors = (token_embeddings * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        del token_embeddings, feeds
        self._trim()
        return vectors
