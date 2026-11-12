"""Endpoints the app calls to start processing (SPEC RF-02, RF-03, RF-06)."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.dependencies.auth import current_user_id
from app.repositories.supabase_repository import ProcessingRepository
from app.schemas.processing import AcceptedResponse, HealthResponse
from app.services.processing_worker import ProcessingWorker

router = APIRouter()


def get_worker(request: Request) -> ProcessingWorker:
    return request.app.state.worker


def get_repository(request: Request) -> ProcessingRepository:
    return request.app.state.repository


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.post("/v1/processing/wake", status_code=status.HTTP_202_ACCEPTED, response_model=AcceptedResponse)
def wake(
    _user_id: str = Depends(current_user_id),
    worker: ProcessingWorker = Depends(get_worker),
) -> AcceptedResponse:
    """Processes every pending source. It only wakes the worker: the queue is in the database."""
    worker.wake()
    return AcceptedResponse()


@router.post(
    "/v1/documents/{document_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=AcceptedResponse,
)
def retry(
    document_id: UUID,
    user_id: str = Depends(current_user_id),
    repository: ProcessingRepository = Depends(get_repository),
    worker: ProcessingWorker = Depends(get_worker),
) -> AcceptedResponse:
    """Puts a source with a temporary error back to pending. Only its owner may do it."""
    result = repository.request_retry(str(document_id), user_id)
    if result == "not_found":
        # Same answer for a missing source and another user's, so ids cannot be probed.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="document_not_found")
    if result != "accepted":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="not_retryable")
    worker.wake()
    return AcceptedResponse()
