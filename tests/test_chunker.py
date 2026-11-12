"""SPEC CA-02 (chunks fit the model and keep the text in order) and CA-06 (file errors)."""

import pytest

from app.core.config import MAX_MODEL_TOKENS
from app.services.chunker import (
    DocumentFileError,
    FileErrorReason,
    build_chunks,
    normalize,
    read_pages,
    split_text,
)
from tests.pdf_factory import write_pdf

SPANISH = (
    "La empresa presentó su informe anual con los resultados del ejercicio. Los ingresos crecieron "
    "gracias a la expansión internacional y a la mejora de los márgenes en la división de servicios. "
)


def test_chunks_fit_the_model_and_cover_the_text_in_order(tokenizer):
    text = normalize(SPANISH * 40)

    chunks = split_text(text, tokenizer)

    assert len(chunks) > 1
    assert all(tokenizer.count_model_tokens(c) <= MAX_MODEL_TOKENS for c in chunks)
    assert " ".join(chunks) == text


def test_cuts_between_words_when_possible(tokenizer):
    text = normalize(SPANISH * 10)
    words = set(text.split())

    chunks = split_text(text, tokenizer)

    for chunk in chunks:
        assert chunk.split()[0] in words
        assert chunk.split()[-1] in words


def test_a_word_longer_than_a_chunk_is_still_split_within_the_limit(tokenizer):
    text = "x" * 2000

    chunks = split_text(text, tokenizer)

    assert all(tokenizer.count_model_tokens(c) <= MAX_MODEL_TOKENS for c in chunks)
    assert "".join(chunks) == text


def test_chunks_keep_their_page_and_never_span_two_pages(tokenizer):
    pages = iter(["Primera página. " * 60, "", "Tercera página."])

    chunks = build_chunks(pages, tokenizer)

    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} == {1, 3}
    assert chunks[-1].page_number == 3
    assert chunks[-1].content == "Tercera página."


def test_text_is_read_page_by_page(tmp_path, tokenizer):
    pdf = write_pdf(tmp_path / "doc.pdf", [["Hello from page one"], ["Second page text"]])

    chunks = build_chunks(read_pages(pdf), tokenizer)

    assert [(c.page_number, c.content) for c in chunks] == [(1, "Hello from page one"), (2, "Second page text")]


def test_a_pdf_without_text_is_a_file_error(tmp_path, tokenizer):
    pdf = write_pdf(tmp_path / "scan.pdf", [[], []])

    with pytest.raises(DocumentFileError) as error:
        build_chunks(read_pages(pdf), tokenizer)

    assert error.value.reason == FileErrorReason.NO_TEXT


def test_more_than_100_pages_is_a_file_error(tmp_path):
    pdf = write_pdf(tmp_path / "long.pdf", [["page"]] * 101)

    with pytest.raises(DocumentFileError) as error:
        list(read_pages(pdf))

    assert error.value.reason == FileErrorReason.TOO_MANY_PAGES


def test_exactly_100_pages_is_accepted(tmp_path):
    pdf = write_pdf(tmp_path / "limit.pdf", [["page"]] * 100)

    assert len(list(read_pages(pdf))) == 100


def test_an_unreadable_file_is_a_file_error(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nnot really a pdf")

    with pytest.raises(DocumentFileError) as error:
        list(read_pages(broken))

    assert error.value.reason == FileErrorReason.NO_TEXT
