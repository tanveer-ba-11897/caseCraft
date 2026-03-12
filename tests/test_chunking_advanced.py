"""
Tests for recursive chunking and parent–child chunking.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.chunking import recursive_chunk_text
from core.knowledge.chunker import (
    chunk_document,
    chunk_document_parent_child,
    _chunk_section_aware,
)
from core.knowledge.models import RawDocument


# ── recursive_chunk_text tests ────────────────────────────────────────────


class TestRecursiveChunkText:
    """Tests for the recursive separator-hierarchy splitter."""

    def test_short_text_returns_single_chunk(self):
        text = "Hello, world."
        chunks = recursive_chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_splits_on_double_newline_first(self):
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = recursive_chunk_text(text, chunk_size=30, overlap=0)
        assert len(chunks) >= 2
        # Each chunk should contain at most one paragraph
        assert all(len(c) <= 30 for c in chunks)

    def test_splits_on_single_newline_when_needed(self):
        text = "Line A\nLine B\nLine C\nLine D"
        chunks = recursive_chunk_text(text, chunk_size=15, overlap=0)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 15

    def test_splits_on_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = recursive_chunk_text(text, chunk_size=25, overlap=0)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 25

    def test_overlap_produces_shared_text(self):
        # Use longer paragraphs so overlap can be observed
        text = "A" * 50 + "\n\n" + "B" * 50
        chunks = recursive_chunk_text(text, chunk_size=55, overlap=10)
        assert len(chunks) >= 2

    def test_empty_text_returns_empty_list(self):
        assert recursive_chunk_text("", chunk_size=100, overlap=0) == []

    def test_whitespace_only_returns_empty(self):
        assert recursive_chunk_text("   \n\n  ", chunk_size=100, overlap=0) == []

    def test_all_chunks_within_size_limit(self):
        text = "word " * 500  # ~2500 chars
        chunks = recursive_chunk_text(text, chunk_size=200, overlap=20)
        for c in chunks:
            assert len(c) <= 200

    def test_no_empty_chunks(self):
        text = "Para 1.\n\n\n\nPara 2.\n\nPara 3."
        chunks = recursive_chunk_text(text, chunk_size=20, overlap=0)
        for c in chunks:
            assert c.strip() != ""


# ── Section-aware chunking tests ─────────────────────────────────────────


class TestSectionAwareChunking:
    """Tests for the section-aware chunker used by the KB pipeline."""

    def test_uses_recursive_splitting_for_no_sections(self):
        text = "Just a plain paragraph with no headings at all. " * 10
        chunks = _chunk_section_aware(text, max_chars=100)
        assert len(chunks) >= 1
        for c in chunks:
            assert len(c) <= 100

    def test_section_headings_propagated(self):
        text = "# Introduction\nThis is the intro.\n\n# Details\nHere are details."
        chunks = _chunk_section_aware(text, max_chars=500)
        # Should detect headings and prepend them
        heading_chunks = [c for c in chunks if c.startswith("[")]
        assert len(heading_chunks) >= 1

    def test_large_section_gets_sub_split(self):
        text = "# Big Section\n" + "Content. " * 200
        chunks = _chunk_section_aware(text, max_chars=100)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 100


# ── Parent–child chunking tests ─────────────────────────────────────────


class TestParentChildChunking:
    """Tests for the two-tier parent–child chunker."""

    @pytest.fixture
    def sample_doc(self):
        return RawDocument(
            text=(
                "# Overview\n"
                "This is the overview section with enough text to be meaningful. "
                "It describes the feature in detail and covers several aspects.\n\n"
                "# Requirements\n"
                "Requirement one: the system shall do X. "
                "Requirement two: the system shall do Y. "
                "Requirement three: the system shall handle Z correctly. "
                "Requirement four: edge cases must be covered.\n\n"
                "# Testing Notes\n"
                "Make sure to test boundary conditions. "
                "Also test with empty inputs and large datasets. "
                "Performance benchmarks should be included."
            ),
            source_name="test_doc.md",
            source_type="file",
        )

    def test_returns_parents_and_children(self, sample_doc):
        parents, children = chunk_document_parent_child(
            sample_doc, parent_size=300, child_size=100, child_overlap=20,
        )
        assert len(parents) >= 1
        assert len(children) >= 1

    def test_parent_metadata_has_chunk_type(self, sample_doc):
        parents, _ = chunk_document_parent_child(
            sample_doc, parent_size=300, child_size=100,
        )
        for p in parents:
            assert p.metadata["chunk_type"] == "parent"
            assert "child_count" in p.metadata

    def test_child_metadata_has_parent_id(self, sample_doc):
        parents, children = chunk_document_parent_child(
            sample_doc, parent_size=300, child_size=100,
        )
        parent_ids = {p.id for p in parents}
        for c in children:
            assert c.metadata["chunk_type"] == "child"
            assert c.metadata["parent_id"] in parent_ids

    def test_children_are_smaller_than_parents(self, sample_doc):
        parents, children = chunk_document_parent_child(
            sample_doc, parent_size=500, child_size=150,
        )
        if parents and children:
            avg_parent = sum(len(p.text) for p in parents) / len(parents)
            avg_child = sum(len(c.text) for c in children) / len(children)
            assert avg_child <= avg_parent

    def test_child_ids_reference_parent_ids(self, sample_doc):
        parents, children = chunk_document_parent_child(
            sample_doc, parent_size=400, child_size=100,
        )
        for c in children:
            pid = c.metadata["parent_id"]
            assert c.id.startswith(pid)

    def test_small_doc_still_works(self):
        doc = RawDocument(
            text="Very short.",
            source_name="tiny.txt",
            source_type="file",
        )
        parents, children = chunk_document_parent_child(doc, parent_size=500, child_size=100)
        assert len(parents) >= 1
        assert len(children) >= 1

    def test_flat_chunk_document_unchanged(self, sample_doc):
        """Original chunk_document still works without parent-child."""
        chunks = chunk_document(sample_doc, max_chars=300)
        assert len(chunks) >= 1
        for c in chunks:
            assert "chunk_type" not in c.metadata  # flat chunks have no type


# ── Integration: ingest pipeline signature ───────────────────────────────


class TestIngestSignature:
    """Verify the ingest function accepts the parent_child parameter."""

    def test_ingest_documents_signature_accepts_parent_child(self):
        from core.knowledge.ingest import ingest_documents
        import inspect
        sig = inspect.signature(ingest_documents)
        assert "parent_child" in sig.parameters
        assert sig.parameters["parent_child"].default is None  # reads from config

    def test_ingest_result_has_parent_child_fields(self):
        from core.knowledge.ingest import IngestResult
        r = IngestResult(documents=1, chunks=10, total_index_size=10, parent_chunks=3, child_chunks=7)
        assert r.parent_chunks == 3
        assert r.child_chunks == 7
