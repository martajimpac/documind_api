"""Backend settings, read once from environment variables (never from source code)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# Model files downloaded at build time (see Dockerfile).
MODEL_FILE = "onnx/model_quint8_avx2.onnx"
SENTENCEPIECE_FILE = "sentencepiece.bpe.model"

# sentence_bert_config.json -> max_seq_length. Includes the two special tokens.
MAX_MODEL_TOKENS = 128
EMBEDDING_DIMENSIONS = 384

# SPEC RF-07.
MAX_PAGES = 100

# One text per run. With the int8 model, padding in a batch changes the vectors slightly
# (cosine 0.995 against the text alone), and on 0.1 CPU one at a time was also faster
# (11 chunks: 8.1 s against 12.5 s in batches of 16; measured 2026-11-07).
EMBED_BATCH_SIZE = 1
STORAGE_BUCKET = "documents"
JWT_AUDIENCE = "authenticated"


@dataclass(frozen=True)
class Settings:
    supabase_url: str
    supabase_secret_key: str
    model_dir: Path


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing environment variable {name}")
    return value


@lru_cache
def get_settings() -> Settings:
    return Settings(
        supabase_url=_required("SUPABASE_URL").rstrip("/"),
        supabase_secret_key=_required("SUPABASE_SECRET_KEY"),
        model_dir=Path(os.environ.get("MODEL_DIR", "/app/model")),
    )
