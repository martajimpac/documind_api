"""Single background thread that processes pending sources one at a time.

Requests only wake it up. The queue lives in the database (documents.processing_status),
so nothing is lost if the process stops: the next start puts interrupted work back to pending.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Protocol

from app.repositories.supabase_repository import ClaimedDocument, ProcessingRepository, StoredChunk
from app.services.chunker import DocumentFileError

TEMPORARY_ERROR = "temporary"

logger = logging.getLogger(__name__)


class Processor(Protocol):
    def process(self, pdf_path: Path) -> list[StoredChunk]: ...


class ProcessingWorker:
    def __init__(self, repository: ProcessingRepository, processor: Processor) -> None:
        self._repository = repository
        self._processor = processor
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="processing-worker", daemon=True)
        self._thread.start()
        self.wake()  # process whatever is pending as soon as the service starts

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def wake(self) -> None:
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            self.process_pending()

    def process_pending(self) -> int:
        """Processes sources until none is pending. Returns how many were taken."""
        taken = 0
        while not self._stop.is_set():
            try:
                document = self._repository.claim_next_document()
            except Exception:
                logger.exception("Could not claim the next document")
                return taken
            if document is None:
                return taken
            taken += 1
            self._process(document)
        return taken

    def _process(self, document: ClaimedDocument) -> None:
        logger.info("Processing document %s", document.id)
        try:
            with tempfile.TemporaryDirectory() as directory:
                pdf_path = Path(directory) / "source.pdf"
                with pdf_path.open("wb") as target:
                    self._repository.download_pdf(document.storage_path, target)
                chunks = self._processor.process(pdf_path)
            completed = self._repository.complete_document(document.id, document.attempt, chunks)
            logger.info("Document %s: %s chunks, completed=%s", document.id, len(chunks), completed)
        except DocumentFileError as e:
            logger.info("Document %s cannot be processed: %s", document.id, e.reason.value)
            self._fail(document, e.reason.value)
        except Exception:
            # Never log the document text or credentials, only the failure.
            logger.exception("Document %s failed", document.id)
            self._fail(document, TEMPORARY_ERROR)

    def _fail(self, document: ClaimedDocument, reason: str) -> None:
        try:
            self._repository.fail_document(document.id, document.attempt, reason)
        except Exception:
            # It stays in processing; the next start or the 30-minute rule takes it again.
            logger.exception("Could not mark document %s as failed", document.id)
