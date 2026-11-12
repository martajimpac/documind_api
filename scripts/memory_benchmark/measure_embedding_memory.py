"""Measure RAM and CPU time of the ONNX int8 embeddings model.

Checks whether `paraphrase-multilingual-MiniLM-L12-v2` (ONNX, int8) fits in the
Render Free plan (512 MB RAM, 0.1 CPU) before the RAG pipeline is built.
See SPEC.md, "Resource Estimate for the Embeddings Model on Render Free".

What it does:
  1. Imports the libraries the backend will use (FastAPI, onnxruntime, pypdf...).
  2. Loads the tokenizer and the model with one CPU thread.
  3. Reads a PDF (or generates sample text), splits it into chunks of at most
     128 tokens and embeds every chunk.
  4. Prints the memory after each step, the peak memory and the timings.

Defaults are the low-memory setup: `sentencepiece` tokenizer, no onnxruntime
memory arena and freed memory returned to the system after each step.
`--tokenizer tokenizers --arena --no-trim` measures the default setup instead.

Run it inside Docker with the Render Free limits (see Dockerfile):
  docker build -t documind-embedding-benchmark backend/scripts/memory_benchmark
  docker run --rm --memory=512m --memory-swap=512m --cpus=0.1 documind-embedding-benchmark
  docker run --rm --memory=512m --memory-swap=512m --cpus=0.1 \
      -v "<folder with the PDF>:/data:ro" documind-embedding-benchmark --pdf /data/file.pdf
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

import psutil

MODEL_REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# x86 int8 build. Render runs on x86 machines with AVX2.
MODEL_FILE = "onnx/model_quint8_avx2.onnx"
SENTENCEPIECE_FILE = "sentencepiece.bpe.model"
TOKENIZER_FILE = "tokenizer.json"
MAX_TOKENS = 128  # sentence_bert_config.json -> max_seq_length
RENDER_FREE_LIMIT_MB = 512

# XLM-RoBERTa special token ids.
BOS_ID, PAD_ID, EOS_ID, UNK_ID = 0, 1, 2, 3

_process = psutil.Process()


def rss_mb() -> float:
    return _process.memory_info().rss / 1024 / 1024


def cgroup_peak_mb() -> float | None:
    """Peak memory of the whole container, as the platform's memory limit sees it."""
    for path in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return int(Path(path).read_text().strip()) / 1024 / 1024
        except (OSError, ValueError):
            continue
    return None


def make_trim(enabled: bool):
    """Returns a function that hands freed memory back to the system (glibc only)."""
    if not enabled:
        return lambda: None
    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        return lambda: None
    return lambda: libc.malloc_trim(0)


class PeakSampler:
    """Samples the process memory in the background to catch short peaks."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self.peak_mb = rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, rss_mb())
            time.sleep(self.interval_s)

    def __enter__(self) -> "PeakSampler":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, rss_mb())


class SentencePieceTokenizer:
    """XLM-RoBERTa tokenization with `sentencepiece` (about 45 MB of RAM)."""

    def __init__(self, model_dir: Path) -> None:
        import sentencepiece

        self._sp = sentencepiece.SentencePieceProcessor(model_file=str(model_dir / SENTENCEPIECE_FILE))

    def _to_model_ids(self, piece_ids: list[int]) -> list[int]:
        # Same mapping as the Hugging Face tokenizer: ids shift by one, unknown -> 3.
        body = [UNK_ID if i == 0 else i + 1 for i in piece_ids][: MAX_TOKENS - 2]
        return [BOS_ID, *body, EOS_ID]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self._to_model_ids(ids) for ids in self._sp.encode(texts)]

    def split(self, text: str) -> list[str]:
        ids = self._sp.encode(text)
        window = MAX_TOKENS - 2
        chunks = (self._sp.decode(ids[i:i + window]).strip() for i in range(0, len(ids), window))
        return [c for c in chunks if c]


class HuggingFaceTokenizer:
    """`tokenizers` with tokenizer.json (about 240 MB of RAM for this vocabulary)."""

    def __init__(self, model_dir: Path) -> None:
        from tokenizers import Tokenizer

        self._tokenizer = Tokenizer.from_file(str(model_dir / TOKENIZER_FILE))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        self._tokenizer.enable_truncation(MAX_TOKENS)
        return [e.ids for e in self._tokenizer.encode_batch(texts)]

    def split(self, text: str) -> list[str]:
        self._tokenizer.no_truncation()  # the whole text, not only the first 128 tokens
        offsets = self._tokenizer.encode(text, add_special_tokens=False).offsets
        window = MAX_TOKENS - 2
        chunks = []
        for start in range(0, len(offsets), window):
            piece = offsets[start:start + window]
            chunk = text[piece[0][0]:piece[-1][1]].strip()
            if chunk:
                chunks.append(chunk)
        return chunks


TOKENIZERS = {"sentencepiece": SentencePieceTokenizer, "tokenizers": HuggingFaceTokenizer}
TOKENIZER_FILES = {"sentencepiece": SENTENCEPIECE_FILE, "tokenizers": TOKENIZER_FILE}


def embed(session, tokenizer, texts: list[str]):
    import numpy as np

    rows = tokenizer.encode_batch(texts)
    width = max(len(r) for r in rows)
    input_ids = np.full((len(rows), width), PAD_ID, dtype=np.int64)
    attention_mask = np.zeros((len(rows), width), dtype=np.int64)
    for i, row in enumerate(rows):
        input_ids[i, :len(row)] = row
        attention_mask[i, :len(row)] = 1
    feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in {i.name for i in session.get_inputs()}:
        feeds["token_type_ids"] = np.zeros_like(input_ids)
    token_embeddings = session.run(None, feeds)[0]
    # Mean pooling over real tokens, as in 1_Pooling/config.json.
    mask = attention_mask[..., None].astype(np.float32)
    return (token_embeddings * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)


def cosine(a, b) -> float:
    import numpy as np

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sample_text(pages: int) -> str:
    """Spanish sample text of about 400 words per page."""
    paragraph = (
        "La empresa presentó su informe anual con los resultados del ejercicio. "
        "Los ingresos crecieron gracias a la expansión internacional y a la mejora "
        "de los márgenes en la división de servicios. El consejo propone mantener "
        "la política de dividendos y aumentar la inversión en investigación. "
    )
    return "\n".join(f"Página {n}. {paragraph * 7}" for n in range(1, pages + 1))


def read_pdf(path: Path) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages), len(reader.pages)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, help="PDF to embed. Without it, sample text is used.")
    parser.add_argument("--pages", type=int, default=100, help="Pages of sample text when no PDF is given.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1, help="onnxruntime threads (Render Free has 0.1 CPU).")
    parser.add_argument("--tokenizer", choices=sorted(TOKENIZERS), default="sentencepiece")
    parser.add_argument("--arena", action="store_true", help="Enable the onnxruntime memory arena (uses more RAM).")
    parser.add_argument("--no-trim", action="store_true", help="Do not return freed memory to the system.")
    parser.add_argument("--model-dir", type=Path, default=Path(os.environ.get("MODEL_DIR", "model")))
    parser.add_argument("--json", action="store_true", help="Print the results as JSON only.")
    args = parser.parse_args()

    trim = make_trim(not args.no_trim)
    log = (lambda *_: None) if args.json else print
    results: dict[str, object] = {
        "tokenizer": args.tokenizer, "arena": args.arena, "trim": not args.no_trim,
        "threads": args.threads, "batch_size": args.batch_size,
    }

    # 1. Libraries the backend will load.
    start = time.perf_counter()
    import fastapi  # noqa: F401  (counts the web stack in the baseline)
    import numpy  # noqa: F401
    import onnxruntime as ort
    import pypdf  # noqa: F401
    import uvicorn  # noqa: F401

    results["imports_s"] = time.perf_counter() - start
    results["rss_after_imports_mb"] = rss_mb()
    log(f"After imports:     {results['rss_after_imports_mb']:.0f} MB")

    # 2. Tokenizer and model.
    model_path = args.model_dir / MODEL_FILE
    needed = (MODEL_FILE, TOKENIZER_FILES[args.tokenizer])
    if not all((args.model_dir / name).exists() for name in needed):
        from huggingface_hub import hf_hub_download

        log(f"Downloading model to {args.model_dir} ...")
        for name in needed:
            hf_hub_download(MODEL_REPO, name, local_dir=args.model_dir)

    with PeakSampler() as load_peak:
        start = time.perf_counter()
        tokenizer = TOKENIZERS[args.tokenizer](args.model_dir)
        trim()
        results["rss_after_tokenizer_mb"] = rss_mb()
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = args.arena
        session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        trim()
        results["model_load_s"] = time.perf_counter() - start
    results["rss_after_model_mb"] = rss_mb()
    results["model_file_mb"] = model_path.stat().st_size / 1024 / 1024
    log(f"After tokenizer:   {results['rss_after_tokenizer_mb']:.0f} MB ({args.tokenizer})")
    log(f"After model load:  {results['rss_after_model_mb']:.0f} MB "
        f"(file {results['model_file_mb']:.0f} MB, {results['model_load_s']:.1f} s)")

    # Quick sanity check: a paraphrase must be closer than an unrelated sentence.
    probe = embed(session, tokenizer, [
        "¿Cuánto crecieron los ingresos este año?",
        "How much did revenue grow this year?",
        "El gato duerme en el sofá.",
    ])
    results["embedding_dimensions"] = int(probe.shape[1])
    results["similarity_paraphrase"] = cosine(probe[0], probe[1])
    results["similarity_unrelated"] = cosine(probe[0], probe[2])
    log(f"Sanity check:      paraphrase {results['similarity_paraphrase']:.2f} "
        f"vs unrelated {results['similarity_unrelated']:.2f}, {probe.shape[1]} dims")

    # 3. Document.
    with PeakSampler() as doc_peak:
        start = time.perf_counter()
        if args.pdf:
            text, pages = read_pdf(args.pdf)
            results["source"] = f"pdf:{args.pdf.name}"
        else:
            text, pages = sample_text(args.pages), args.pages
            results["source"] = "sample text"
        chunks = tokenizer.split(text)
        del text
        trim()
        results["read_and_split_s"] = time.perf_counter() - start

        start = time.perf_counter()
        for i in range(0, len(chunks), args.batch_size):
            embed(session, tokenizer, chunks[i:i + args.batch_size])
            trim()
        results["embed_s"] = time.perf_counter() - start

    results["pages"] = pages
    results["chunks"] = len(chunks)
    results["chunks_per_s"] = len(chunks) / results["embed_s"] if results["embed_s"] else None
    results["rss_peak_model_load_mb"] = load_peak.peak_mb
    results["rss_peak_document_mb"] = doc_peak.peak_mb
    results["rss_peak_mb"] = max(load_peak.peak_mb, doc_peak.peak_mb)
    results["container_peak_mb"] = cgroup_peak_mb()
    results["cpu_time_s"] = sum(_process.cpu_times()[:2])
    peak = results["container_peak_mb"] or results["rss_peak_mb"]
    results["fits_render_free"] = peak < RENDER_FREE_LIMIT_MB

    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    log(f"Document:          {results['source']}, {pages} pages, {len(chunks)} chunks")
    log(f"Read + split:      {results['read_and_split_s']:.1f} s")
    log(f"Embedding:         {results['embed_s']:.1f} s ({results['chunks_per_s']:.2f} chunks/s)")
    log(f"Peak (process):    {results['rss_peak_mb']:.0f} MB")
    if results["container_peak_mb"] is not None:
        log(f"Peak (container):  {results['container_peak_mb']:.0f} MB")
    log(f"CPU time used:     {results['cpu_time_s']:.1f} s")
    verdict = "FITS" if results["fits_render_free"] else "DOES NOT FIT"
    log(f"Render Free ({RENDER_FREE_LIMIT_MB} MB): {verdict} (peak {peak:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
