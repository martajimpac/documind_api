"""Worker behaviour with a fake repository (SPEC CA-01, CA-04, CA-06, CA-07)."""

from __future__ import annotations

from pathlib import Path

from app.repositories.supabase_repository import ClaimedDocument, StoredChunk
from app.services.chunker import DocumentFileError, FileErrorReason
from app.services.processing_worker import ProcessingWorker

PDF_BYTES = b"%PDF-1.4 fake"


class FakeRepository:
    def __init__(self, documents: list[ClaimedDocument]) -> None:
        self.queue = list(documents)
        self.completed: dict[str, list[StoredChunk]] = {}
        self.failed: dict[str, str] = {}
        self.claim_error: Exception | None = None
        self.complete_error: Exception | None = None

    def claim_next_document(self):
        if self.claim_error:
            raise self.claim_error
        return self.queue.pop(0) if self.queue else None

    def download_pdf(self, storage_path, target):
        target.write(PDF_BYTES)

    def complete_document(self, document_id, attempt, chunks):
        if self.complete_error:
            raise self.complete_error
        self.completed[document_id] = chunks
        return True

    def fail_document(self, document_id, attempt, reason):
        self.failed[document_id] = reason
        return True

    def reset_interrupted_processing(self):
        return 0

    def request_retry(self, document_id, user_id):
        return "accepted"


class FakeProcessor:
    def __init__(self, outcomes: dict[str, object]) -> None:
        self.outcomes = outcomes
        self.seen_bytes: list[bytes] = []
        self.current: str | None = None

    def process(self, pdf_path: Path):
        self.seen_bytes.append(pdf_path.read_bytes())
        outcome = self.outcomes[self.current]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def doc(document_id: str) -> ClaimedDocument:
    return ClaimedDocument(id=document_id, user_id="u1", storage_path=f"u1/n1/{document_id}.pdf", attempt="a1")


def make_worker(repository, outcomes):
    processor = FakeProcessor(outcomes)
    original_claim = repository.claim_next_document

    def claim():
        document = original_claim()
        processor.current = document.id if document else None
        return document

    repository.claim_next_document = claim
    return ProcessingWorker(repository, processor), processor


CHUNKS = [StoredChunk(index=0, page_number=1, content="hola", embedding=[0.1] * 384)]


def test_processes_every_pending_document_one_after_another():
    repository = FakeRepository([doc("d1"), doc("d2")])
    worker, processor = make_worker(repository, {"d1": CHUNKS, "d2": CHUNKS})

    taken = worker.process_pending()

    assert taken == 2
    assert set(repository.completed) == {"d1", "d2"}
    assert processor.seen_bytes == [PDF_BYTES, PDF_BYTES]


def test_file_errors_are_stored_with_their_reason():
    repository = FakeRepository([doc("scan"), doc("long")])
    worker, _ = make_worker(
        repository,
        {
            "scan": DocumentFileError(FileErrorReason.NO_TEXT),
            "long": DocumentFileError(FileErrorReason.TOO_MANY_PAGES),
        },
    )

    worker.process_pending()

    assert repository.failed == {"scan": "no_text", "long": "too_many_pages"}


def test_any_other_failure_is_temporary_and_the_queue_continues():
    repository = FakeRepository([doc("bad"), doc("good")])
    worker, _ = make_worker(repository, {"bad": RuntimeError("boom"), "good": CHUNKS})

    worker.process_pending()

    assert repository.failed == {"bad": "temporary"}
    assert set(repository.completed) == {"good"}


def test_a_failure_while_saving_is_temporary():
    repository = FakeRepository([doc("d1")])
    repository.complete_error = ConnectionError("network")
    worker, _ = make_worker(repository, {"d1": CHUNKS})

    worker.process_pending()

    assert repository.failed == {"d1": "temporary"}


def test_stops_quietly_when_the_queue_cannot_be_read():
    repository = FakeRepository([doc("d1")])
    repository.claim_error = ConnectionError("offline")
    worker, _ = make_worker(repository, {})

    assert worker.process_pending() == 0


def test_wake_processes_pending_documents_in_the_background():
    repository = FakeRepository([doc("d1")])
    worker, _ = make_worker(repository, {"d1": CHUNKS})

    worker.start()
    try:
        for _ in range(100):
            if repository.completed:
                break
            import time

            time.sleep(0.02)
    finally:
        worker.stop(timeout=2)

    assert set(repository.completed) == {"d1"}
