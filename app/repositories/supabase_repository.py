"""Access to Supabase (PostgREST and Storage) with the secret key.

The secret key bypasses RLS, so every user-facing check happens before calling this
repository (see dependencies/auth.py) or inside the SQL functions it calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Protocol, Sequence

import httpx

from app.core.config import STORAGE_BUCKET

TIMEOUT = httpx.Timeout(30.0, read=120.0)


@dataclass(frozen=True)
class ClaimedDocument:
    id: str
    user_id: str
    storage_path: str
    attempt: str


@dataclass(frozen=True)
class StoredChunk:
    index: int
    page_number: int
    content: str
    embedding: Sequence[float]


class ProcessingRepository(Protocol):
    def claim_next_document(self) -> ClaimedDocument | None: ...
    def download_pdf(self, storage_path: str, target: BinaryIO) -> None: ...
    def complete_document(self, document_id: str, attempt: str, chunks: list[StoredChunk]) -> bool: ...
    def fail_document(self, document_id: str, attempt: str, reason: str) -> bool: ...
    def reset_interrupted_processing(self) -> int: ...
    def request_retry(self, document_id: str, user_id: str) -> str: ...


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"


class SupabaseRepository:
    def __init__(self, supabase_url: str, secret_key: str, client: httpx.Client | None = None) -> None:
        headers = {"apikey": secret_key}
        if secret_key.startswith("eyJ"):
            # Legacy service_role JWT. The new sb_secret_ keys must only go in the apikey header.
            headers["Authorization"] = f"Bearer {secret_key}"
        self._client = client or httpx.Client(base_url=supabase_url, headers=headers, timeout=TIMEOUT)

    def close(self) -> None:
        self._client.close()

    def _rpc(self, name: str, payload: dict | None = None):
        response = self._client.post(f"/rest/v1/rpc/{name}", json=payload or {})
        response.raise_for_status()
        return response.json()

    def claim_next_document(self) -> ClaimedDocument | None:
        rows = self._rpc("claim_next_document")
        if not rows:
            return None
        row = rows[0]
        return ClaimedDocument(
            id=row["id"], user_id=row["user_id"], storage_path=row["storage_path"], attempt=row["processing_attempt"]
        )

    def download_pdf(self, storage_path: str, target: BinaryIO) -> None:
        with self._client.stream("GET", f"/storage/v1/object/{STORAGE_BUCKET}/{storage_path}") as response:
            response.raise_for_status()
            for block in response.iter_bytes(64 * 1024):
                target.write(block)

    def complete_document(self, document_id: str, attempt: str, chunks: list[StoredChunk]) -> bool:
        payload = [
            {
                "chunk_index": c.index,
                "page_number": c.page_number,
                "content": c.content,
                "embedding": _vector_literal(c.embedding),
            }
            for c in chunks
        ]
        return bool(
            self._rpc(
                "complete_document_processing",
                {"p_document_id": document_id, "p_attempt": attempt, "p_chunks": payload},
            )
        )

    def fail_document(self, document_id: str, attempt: str, reason: str) -> bool:
        return bool(
            self._rpc(
                "fail_document_processing",
                {"p_document_id": document_id, "p_attempt": attempt, "p_reason": reason},
            )
        )

    def reset_interrupted_processing(self) -> int:
        return int(self._rpc("reset_interrupted_processing"))

    def request_retry(self, document_id: str, user_id: str) -> str:
        return str(self._rpc("request_document_retry", {"p_document_id": document_id, "p_user_id": user_id}))

