import os
from pathlib import Path

import pytest

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))


@pytest.fixture(scope="session")
def tokenizer():
    if not MODEL_DIR.exists():
        pytest.skip("Model files are only available in the Docker test image")
    from app.services.embedder import Tokenizer

    return Tokenizer(MODEL_DIR)


@pytest.fixture(scope="session")
def embedder(tokenizer):
    from app.services.embedder import Embedder

    return Embedder(MODEL_DIR, tokenizer)
