"""DocuMind backend: processes uploaded PDFs into chunks with embeddings (feature pdf-processing)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routers import processing
from app.core.config import get_settings
from app.repositories.supabase_repository import SupabaseRepository
from app.services.document_processor import DocumentProcessor
from app.services.embedder import Embedder
from app.services.processing_worker import ProcessingWorker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# httpx logs every URL at INFO; keep it quiet so storage paths do not fill the logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("documind")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    repository = SupabaseRepository(settings.supabase_url, settings.supabase_secret_key)
    # The model is in the image (downloaded at build time), never fetched at startup.
    worker = ProcessingWorker(repository, DocumentProcessor(Embedder(settings.model_dir)))
    try:
        reset = repository.reset_interrupted_processing()
        logger.info("Sources put back to pending after a restart: %s", reset)
    except Exception:
        logger.exception("Could not reset interrupted processing")
    app.state.repository = repository
    app.state.worker = worker
    worker.start()
    try:
        yield
    finally:
        worker.stop(timeout=5)
        repository.close()


app = FastAPI(title="DocuMind backend", lifespan=lifespan)
app.include_router(processing.router)
