import numpy as np

from app.core.config import EMBEDDING_DIMENSIONS


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_one_vector_of_384_per_text(embedder):
    vectors = embedder.embed(["uno", "dos frases más largas que la primera"])

    assert vectors.shape == (2, EMBEDDING_DIMENSIONS)
    assert np.all(np.isfinite(vectors))


def test_paraphrases_in_two_languages_are_closer_than_unrelated_text(embedder):
    spanish, english, unrelated = embedder.embed(
        [
            "¿Cuánto crecieron los ingresos este año?",
            "How much did revenue grow this year?",
            "El gato duerme encima del sofá.",
        ]
    )

    assert cosine(spanish, english) > 0.7
    assert cosine(spanish, english) > cosine(spanish, unrelated) + 0.3


def test_a_pdf_becomes_chunks_embedded_one_at_a_time(tmp_path, embedder):
    from app.services.document_processor import DocumentProcessor
    from tests.pdf_factory import write_pdf

    calls = []

    class Recorder:
        tokenizer = embedder.tokenizer

        def embed(self, texts):
            calls.append(list(texts))
            return embedder.embed(texts)

    pdf = write_pdf(tmp_path / "doc.pdf", [["Short first page"], ["The second page talks about revenue " * 8]])

    chunks = DocumentProcessor(Recorder()).process(pdf)

    # With the int8 model a batch changes the vectors; one at a time each chunk always gets the same one.
    assert all(len(batch) == 1 for batch in calls)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].page_number == 1 and chunks[-1].page_number == 2
    assert all(len(c.embedding) == EMBEDDING_DIMENSIONS for c in chunks)
    assert cosine(chunks[0].embedding, embedder.embed(["Short first page"])[0]) > 0.9999
