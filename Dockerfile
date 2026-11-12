# DocuMind backend. Python 3.12: onnxruntime 1.20.1 has no wheels for newer versions.
#   Tests:   docker build --target test -t documind-backend-test . && docker run --rm documind-backend-test
#   Service: docker build -t documind-backend . && docker run --rm --env-file .env -p 8000:8000 documind-backend
#   Render Free limits: add --memory=512m --memory-swap=512m --cpus=0.1 to docker run.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/app/model
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The model is downloaded at build time, never on a cold start (SPEC.md, Deployment).
# Pinned to a revision of the Hugging Face repository so every build gets the same files.
ARG MODEL_REPO=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ARG MODEL_REVISION=e8f8c211226b894fcb81acc59f3b34ba3efd5f42
RUN pip install --no-cache-dir huggingface_hub==0.27.1 \
 && python -c "import os; from huggingface_hub import hf_hub_download as d; \
[d(os.environ['MODEL_REPO'], f, revision=os.environ['MODEL_REVISION'], local_dir=os.environ['MODEL_DIR']) \
 for f in ('onnx/model_quint8_avx2.onnx', 'sentencepiece.bpe.model')]" \
 && rm -rf "$MODEL_DIR/.cache" \
 && pip uninstall -y huggingface_hub

FROM base AS test
RUN pip install --no-cache-dir pytest==8.3.4
COPY app app
COPY tests tests
CMD ["python", "-m", "pytest", "-q", "tests"]

FROM base AS runtime
COPY app app
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
# One worker process: the processing thread and the model must exist only once (512 MB).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
