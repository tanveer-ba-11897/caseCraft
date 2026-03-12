import pytest

from core import chunking


def test_chunk_text_basic():
    text = "x" * 2000
    chunks = chunking.chunk_text(text, chunk_size=800, overlap=100)

    # every chunk must be at most chunk_size
    assert all(len(c) <= 800 for c in chunks)
    # there should be multiple chunks for long text
    assert len(chunks) >= 2


def test_chunk_text_invalid_overlap():
    with pytest.raises(ValueError):
        chunking.chunk_text("text", chunk_size=100, overlap=100)

    with pytest.raises(ValueError):
        chunking.chunk_text("text", chunk_size=0, overlap=0)

    with pytest.raises(ValueError):
        chunking.chunk_text("text", chunk_size=100, overlap=-1)


def test_chunk_preserves_word_boundaries():
    # Repeating short words; chunking should split at spaces so non-final
    # chunks end with whitespace
    text = "word " * 100
    chunks = chunking.chunk_text(text, chunk_size=30, overlap=5)
    assert len(chunks) >= 2

    for c in chunks[:-1]:
        assert c and c[-1].isspace()


def test_chunk_handles_long_token():
    # A single long token with no whitespace should be split by size
    long_word = "A" * 1000
    chunks = chunking.chunk_text(long_word, chunk_size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) >= 10
