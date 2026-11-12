# Reproduces the Render Free limits when run with:
#   docker run --rm --memory=512m --memory-swap=512m --cpus=0.1 documind-embedding-benchmark
FROM python:3.12-slim

WORKDIR /benchmark
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/benchmark/model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The model is downloaded at build time, as the spec requires for Render.
RUN python -c "from huggingface_hub import hf_hub_download as d; \
[d('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', f, local_dir='/benchmark/model') \
 for f in ('onnx/model_quint8_avx2.onnx', 'sentencepiece.bpe.model', 'tokenizer.json')]"

COPY measure_embedding_memory.py .
ENTRYPOINT ["python", "measure_embedding_memory.py"]
