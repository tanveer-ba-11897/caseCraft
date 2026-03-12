"""
Tests for the lightweight knowledge graph layer.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.knowledge.graph import (
    KnowledgeGraph,
    _extract_entities,
    _normalise,
    DEFAULT_GRAPH_PATH,
)
from core.knowledge.models import KnowledgeChunk


# ── Entity extraction tests ──────────────────────────────────────────────


class TestEntityExtraction:
    """Tests for lightweight noun-phrase / key-term extraction."""

    def test_extracts_capitalised_phrases(self):
        entities = _extract_entities("The Payment Gateway handles all transactions.")
        assert "payment gateway" in entities

    def test_extracts_acronyms(self):
        entities = _extract_entities("The API uses OAuth for SSO integration.")
        assert "api" in entities
        assert "sso" in entities

    def test_extracts_quoted_terms(self):
        entities = _extract_entities('Click the "Submit Order" button.')
        assert "submit order" in entities

    def test_ignores_short_terms(self):
        entities = _extract_entities("A is B.")
        # Single-letter words should not appear
        assert not any(len(e) < 3 for e in entities)

    def test_normalise_strips_special_chars(self):
        assert _normalise("Hello-World!") == "helloworld"
        assert _normalise("API_KEY") == "apikey"

    def test_empty_text(self):
        assert _extract_entities("") == set()


# ── Graph construction tests ─────────────────────────────────────────────


def _make_chunk(cid: str, text: str, source: str = "doc.md", **extra_meta) -> KnowledgeChunk:
    meta = {"source_name": source, "source_type": "file", **extra_meta}
    return KnowledgeChunk(id=cid, text=text, metadata=meta)


class TestGraphConstruction:
    """Tests for building the knowledge graph from chunks."""

    def test_parent_child_edges(self):
        parent = _make_chunk("p1", "Parent text.", chunk_type="parent")
        child1 = _make_chunk("c1", "Child one.", chunk_type="child", parent_id="p1")
        child2 = _make_chunk("c2", "Child two.", chunk_type="child", parent_id="p1")

        kg = KnowledgeGraph()
        kg.build_from_chunks([parent, child1, child2])

        assert kg.graph.has_edge("p1", "c1")
        assert kg.graph.has_edge("p1", "c2")
        assert kg.graph["p1"]["c1"]["relation"] == "parent_of"

    def test_same_source_edges(self):
        c1 = _make_chunk("a1", "First section.", source="feature.md")
        c2 = _make_chunk("a2", "Second section.", source="feature.md")
        c3 = _make_chunk("a3", "Third section.", source="feature.md")

        kg = KnowledgeGraph()
        kg.build_from_chunks([c1, c2, c3])

        assert kg.graph.has_edge("a1", "a2")
        assert kg.graph.has_edge("a2", "a3")
        assert kg.graph["a1"]["a2"]["relation"] == "same_source"

    def test_cross_reference_edges(self):
        c1 = _make_chunk("x1", "See the login_spec for authentication details.", source="readme.md")
        c2 = _make_chunk("x2", "Login requirements listed here.", source="login_spec.md")

        kg = KnowledgeGraph()
        kg.build_from_chunks([c1, c2])

        # x1 mentions "login_spec" which is the stem of x2's source
        assert kg.graph.has_edge("x1", "x2")
        assert kg.graph["x1"]["x2"]["relation"] == "cross_reference"

    def test_shared_entity_edges(self):
        c1 = _make_chunk(
            "e1",
            "The Payment Gateway processes Credit Card transactions via the Checkout Flow.",
            source="payments.md",
        )
        c2 = _make_chunk(
            "e2",
            "Credit Card validation occurs in the Payment Gateway before the Checkout Flow completes.",
            source="validation.md",
        )

        kg = KnowledgeGraph()
        kg.build_from_chunks([c1, c2], min_shared_entities=2)

        assert kg.graph.has_edge("e1", "e2")
        assert kg.graph["e1"]["e2"]["relation"] == "shared_entities"

    def test_empty_chunks_produce_empty_graph(self):
        kg = KnowledgeGraph()
        kg.build_from_chunks([])
        assert kg.graph.number_of_nodes() == 0
        assert kg.graph.number_of_edges() == 0

    def test_statistics(self):
        c1 = _make_chunk("s1", "Text A.", source="doc.md")
        c2 = _make_chunk("s2", "Text B.", source="doc.md")

        kg = KnowledgeGraph()
        kg.build_from_chunks([c1, c2])

        stats = kg.get_statistics()
        assert stats["nodes"] == 2
        assert stats["edges"] >= 2  # bidirectional same_source
        assert "same_source" in stats["relations"]


# ── Graph traversal tests ────────────────────────────────────────────────


class TestGraphTraversal:
    """Tests for get_related_ids BFS expansion."""

    @pytest.fixture
    def linear_graph(self):
        """A → B → C → D (all same_source)."""
        chunks = [
            _make_chunk("a", "Section A.", source="doc.md"),
            _make_chunk("b", "Section B.", source="doc.md"),
            _make_chunk("c", "Section C.", source="doc.md"),
            _make_chunk("d", "Section D.", source="doc.md"),
        ]
        kg = KnowledgeGraph(max_hops=2)
        kg.build_from_chunks(chunks)
        return kg

    def test_direct_neighbours(self, linear_graph):
        related = linear_graph.get_related_ids(["a"], max_hops=1)
        assert "b" in related
        # "c" should be reachable since same_source is bidirectional
        # and a→b, b→c are both same_source edges within 1 hop from "a" only reaches "b"

    def test_two_hop_expansion(self, linear_graph):
        related = linear_graph.get_related_ids(["a"], max_hops=2)
        assert "b" in related
        assert "c" in related

    def test_seed_excluded_from_results(self, linear_graph):
        related = linear_graph.get_related_ids(["b"])
        assert "b" not in related

    def test_unknown_id_returns_empty(self, linear_graph):
        assert linear_graph.get_related_ids(["nonexistent"]) == []

    def test_relation_filter(self, linear_graph):
        # Add a cross_reference edge manually
        linear_graph.graph.add_edge("a", "d", relation="cross_reference")
        related = linear_graph.get_related_ids(
            ["a"], max_hops=1, relation_filter={"cross_reference"},
        )
        assert "d" in related
        # same_source neighbours should be excluded by filter
        assert "b" not in related


# ── Persistence tests ────────────────────────────────────────────────────


class TestGraphPersistence:
    """Tests for save/load round-trip."""

    def test_save_and_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_graph.json"

            chunks = [
                _make_chunk("p1", "Parent chunk.", chunk_type="parent", source="a.md"),
                _make_chunk("c1", "Child chunk.", chunk_type="child", parent_id="p1", source="a.md"),
                _make_chunk("x1", "Another document.", source="b.md"),
            ]

            kg = KnowledgeGraph(graph_path=path)
            kg.build_from_chunks(chunks)
            kg.save()

            assert path.exists()

            loaded = KnowledgeGraph.load(path)
            assert loaded is not None
            assert loaded.graph.number_of_nodes() == kg.graph.number_of_nodes()
            assert loaded.graph.number_of_edges() == kg.graph.number_of_edges()

    def test_load_nonexistent_returns_none(self):
        result = KnowledgeGraph.load("/nonexistent/path/graph.json")
        assert result is None

    def test_reset_clears_graph_and_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_graph.json"
            kg = KnowledgeGraph(graph_path=path)
            kg.build_from_chunks([_make_chunk("n1", "Some text.")])
            kg.save()
            assert path.exists()

            kg.reset()
            assert kg.graph.number_of_nodes() == 0
            assert not path.exists()

    def test_saved_json_is_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_graph.json"
            kg = KnowledgeGraph(graph_path=path)
            kg.build_from_chunks([
                _make_chunk("j1", "Hello.", source="doc.md"),
                _make_chunk("j2", "World.", source="doc.md"),
            ])
            kg.save()

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert "nodes" in data
            # networkx 3.x uses "edges", 2.x uses "links"
            assert "edges" in data or "links" in data


# ── Config integration tests ─────────────────────────────────────────────


class TestGraphConfig:
    """Verify knowledge graph config fields are present."""

    def test_config_has_knowledge_graph_fields(self):
        from core.config import KnowledgeSettings
        fields = KnowledgeSettings.model_fields
        assert "knowledge_graph" in fields
        assert "graph_path" in fields
        assert "graph_max_hops" in fields
        assert "graph_max_expansion" in fields

    def test_defaults_are_sensible(self):
        from core.config import KnowledgeSettings
        ks = KnowledgeSettings()
        assert ks.knowledge_graph is False
        assert ks.graph_max_hops == 2
        assert ks.graph_max_expansion == 3

    def test_ingest_result_has_graph_fields(self):
        from core.knowledge.ingest import IngestResult
        r = IngestResult(
            documents=1, chunks=10, total_index_size=10,
            graph_nodes=5, graph_edges=8,
        )
        assert r.graph_nodes == 5
        assert r.graph_edges == 8
