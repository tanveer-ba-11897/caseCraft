"""
Lightweight in-memory knowledge graph for CaseCraft.

Builds a directed graph of relationships between knowledge chunks at
ingestion time and persists it as a JSON adjacency file alongside
ChromaDB.  The retriever uses graph traversal to expand results with
structurally related chunks that embedding similarity might miss.

Relationship types
------------------
- **parent_of** : structural parent → child chunk edge
- **same_source** : chunks originating from the same document
- **cross_reference** : chunk A explicitly mentions the source of chunk B
- **shared_entities** : chunks share significant noun-phrase entities

Dependencies: ``networkx`` (pure Python, ~3 MB, zero native deps).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from networkx.readwrite import json_graph

from core.knowledge.models import KnowledgeChunk

logger = logging.getLogger("casecraft.knowledge_graph")

# ── Defaults ──────────────────────────────────────────────────────────────

DEFAULT_GRAPH_PATH = "knowledge_base/knowledge_graph.json"
DEFAULT_MAX_HOPS = 2

# Minimum entity length to be considered meaningful
_MIN_ENTITY_LEN = 3
# Minimum shared entities to create an edge
_MIN_SHARED_ENTITIES = 2

# Regex for lightweight noun-phrase / key-term extraction (no spaCy needed)
_ENTITY_PATTERN = re.compile(
    r"""
    (?:                         # Group: multi-word capitalised phrases
        [A-Z][a-z]+             # First capitalised word
        (?:\s+[A-Z][a-z]+)+    # Followed by one or more capitalised words
    )
    |                           # OR
    (?:                         # Quoted terms
        "([^"]{3,60})"
    )
    |                           # OR
    (?:                         # ALL-CAPS acronyms (2-8 letters)
        \b[A-Z]{2,8}\b
    )
    """,
    re.VERBOSE,
)

# Normalise extracted entities for comparison
_NORMALISE_RE = re.compile(r"[^a-z0-9 ]+")
_LEADING_ARTICLES = re.compile(r"^(?:the|a|an)\s+", re.IGNORECASE)


def _normalise(entity: str) -> str:
    return _NORMALISE_RE.sub("", entity.lower()).strip()


def _extract_entities(text: str) -> Set[str]:
    """Extract significant noun phrases / acronyms / quoted terms."""
    entities: Set[str] = set()
    for match in _ENTITY_PATTERN.finditer(text):
        term = match.group(0).strip().strip('"')
        # Strip leading articles ("The Payment Gateway" → "Payment Gateway")
        term = _LEADING_ARTICLES.sub("", term)
        normed = _normalise(term)
        if len(normed) >= _MIN_ENTITY_LEN:
            entities.add(normed)
    return entities


# ── KnowledgeGraph class ─────────────────────────────────────────────────


class KnowledgeGraph:
    """
    Lightweight ``networkx``-based relation graph over knowledge chunks.

    Nodes are chunk IDs; edges carry a ``relation`` label.

    The graph is designed to be:

    * **Built at ingest time** (no LLM calls required).
    * **Persisted** as a JSON adjacency file next to ChromaDB.
    * **Loaded lazily** by the retriever to expand results.

    Parameters
    ----------
    graph_path:
        Where to persist / load the graph JSON.
    max_hops:
        Default BFS hop limit when expanding a chunk ID.
    """

    def __init__(
        self,
        graph_path: str | Path = DEFAULT_GRAPH_PATH,
        max_hops: int = DEFAULT_MAX_HOPS,
    ):
        self.graph_path = Path(graph_path)
        self.max_hops = max_hops
        self.graph: nx.DiGraph = nx.DiGraph()

    # ── Construction ──────────────────────────────────────────────────

    def build_from_chunks(
        self,
        chunks: List[KnowledgeChunk],
        *,
        enable_cross_reference: bool = True,
        enable_shared_entities: bool = True,
        min_shared_entities: int = _MIN_SHARED_ENTITIES,
    ) -> "KnowledgeGraph":
        """
        Build the graph from a list of ingested knowledge chunks.

        Extracts four relationship types without any LLM calls:

        1. **parent_of** — from ``chunk_type`` / ``parent_id`` metadata.
        2. **same_source** — chunks sharing the same ``source_name``.
        3. **cross_reference** — chunk text mentions another document's
           source name.
        4. **shared_entities** — chunks sharing ≥ *min_shared_entities*
           extracted noun phrases.

        Returns *self* for chaining.
        """
        t0 = time.time()
        self.graph.clear()

        # Index helpers
        source_groups: Dict[str, List[str]] = defaultdict(list)
        source_names: Set[str] = set()
        chunk_lookup: Dict[str, KnowledgeChunk] = {}
        entity_index: Dict[str, Set[str]] = {}  # chunk_id → entities

        # ── Pass 1: add nodes + collect metadata ──────────────────────
        for chunk in chunks:
            cid = chunk.id
            meta = chunk.metadata or {}
            self.graph.add_node(cid, **meta)
            chunk_lookup[cid] = chunk

            src = meta.get("source_name", "")
            if src:
                source_groups[src].append(cid)
                source_names.add(src)

            # Parent → child edges
            if meta.get("chunk_type") == "child" and "parent_id" in meta:
                pid = meta["parent_id"]
                self.graph.add_edge(pid, cid, relation="parent_of")

        # ── Pass 2: same-source edges ─────────────────────────────────
        for src, ids in source_groups.items():
            if len(ids) < 2:
                continue
            # Connect consecutive chunks from the same source
            for i in range(len(ids) - 1):
                a, b = ids[i], ids[i + 1]
                # Don't overwrite existing parent_of edges
                if not self.graph.has_edge(a, b):
                    self.graph.add_edge(a, b, relation="same_source")
                if not self.graph.has_edge(b, a):
                    self.graph.add_edge(b, a, relation="same_source")

        # ── Pass 3: cross-reference detection ─────────────────────────
        if enable_cross_reference and source_names:
            # Build lookups: normalised name → original source, plus
            # raw name variants for substring matching in text.
            norm_source_map: Dict[str, str] = {}
            raw_source_variants: Dict[str, str] = {}  # lowered variant → original
            for sn in source_names:
                key = _normalise(sn)
                stem = _normalise(Path(sn).stem)
                norm_source_map[key] = sn
                if stem != key:
                    norm_source_map[stem] = sn
                # Also match raw lowered forms (e.g. "login_spec" in text)
                raw_source_variants[sn.lower()] = sn
                raw_stem = Path(sn).stem.lower()
                if raw_stem != sn.lower():
                    raw_source_variants[raw_stem] = sn

            for cid, chunk in chunk_lookup.items():
                chunk_src = (chunk.metadata or {}).get("source_name", "")
                text_lower = chunk.text.lower()
                # Check both normalised and raw variants
                matched_sources: Set[str] = set()
                for variant, orig_name in raw_source_variants.items():
                    if orig_name == chunk_src:
                        continue
                    if len(variant) >= 4 and variant in text_lower:
                        matched_sources.add(orig_name)
                for norm_name, orig_name in norm_source_map.items():
                    if orig_name == chunk_src:
                        continue
                    if len(norm_name) >= 4 and norm_name in text_lower:
                        matched_sources.add(orig_name)

                for orig_name in matched_sources:
                    for target_id in source_groups.get(orig_name, []):
                        if target_id != cid:
                            self.graph.add_edge(
                                cid, target_id, relation="cross_reference",
                            )

        # ── Pass 4: shared entity co-occurrence ───────────────────────
        if enable_shared_entities:
            # Extract entities for all chunks
            for cid, chunk in chunk_lookup.items():
                entity_index[cid] = _extract_entities(chunk.text)

            # Inverted index: entity → chunk IDs
            entity_to_chunks: Dict[str, List[str]] = defaultdict(list)
            for cid, entities in entity_index.items():
                for ent in entities:
                    entity_to_chunks[ent].append(cid)

            # Find pairs sharing enough entities
            pair_shared: Dict[Tuple[str, str], int] = defaultdict(int)
            for ent, cids in entity_to_chunks.items():
                if len(cids) > 50:
                    # Skip very common entities (noise)
                    continue
                for i in range(len(cids)):
                    for j in range(i + 1, len(cids)):
                        a, b = cids[i], cids[j]
                        if a == b:
                            continue
                        key = (min(a, b), max(a, b))
                        pair_shared[key] += 1

            for (a, b), count in pair_shared.items():
                if count >= min_shared_entities:
                    if not self.graph.has_edge(a, b):
                        self.graph.add_edge(
                            a, b, relation="shared_entities", weight=count,
                        )
                    if not self.graph.has_edge(b, a):
                        self.graph.add_edge(
                            b, a, relation="shared_entities", weight=count,
                        )

        elapsed = time.time() - t0
        logger.info(
            "Knowledge graph built: %d nodes, %d edges in %.2fs",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            elapsed,
        )
        return self

    # ── Retrieval expansion ───────────────────────────────────────────

    def get_related_ids(
        self,
        chunk_ids: List[str],
        max_hops: int | None = None,
        relation_filter: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Return chunk IDs reachable within *max_hops* from the given IDs.

        Parameters
        ----------
        chunk_ids:
            Seed chunk IDs to expand from.
        max_hops:
            BFS hop limit (default: ``self.max_hops``).
        relation_filter:
            If provided, only traverse edges whose ``relation`` is in
            this set.  ``None`` = traverse all edges.

        Returns
        -------
        List[str]
            Related chunk IDs **excluding** the input seeds, ordered by
            shortest distance first.
        """
        if max_hops is None:
            max_hops = self.max_hops

        seed_set = set(chunk_ids)
        related: Dict[str, int] = {}  # chunk_id → shortest distance

        for cid in chunk_ids:
            if cid not in self.graph:
                continue

            if relation_filter is not None:
                # Manual BFS with edge filtering
                visited: Set[str] = {cid}
                frontier: List[Tuple[str, int]] = [(cid, 0)]
                while frontier:
                    node, depth = frontier.pop(0)
                    if depth >= max_hops:
                        continue
                    for _, neighbor, data in self.graph.edges(node, data=True):
                        if neighbor in visited:
                            continue
                        if data.get("relation") not in relation_filter:
                            continue
                        visited.add(neighbor)
                        new_depth = depth + 1
                        if neighbor not in seed_set:
                            if neighbor not in related or new_depth < related[neighbor]:
                                related[neighbor] = new_depth
                        frontier.append((neighbor, new_depth))
            else:
                lengths = nx.single_source_shortest_path_length(
                    self.graph, cid, cutoff=max_hops,
                )
                for nid, dist in lengths.items():
                    if nid in seed_set:
                        continue
                    if nid not in related or dist < related[nid]:
                        related[nid] = dist

        # Sort by distance (closest first), then alphabetically for stability
        return sorted(related.keys(), key=lambda x: (related[x], x))

    def get_statistics(self) -> Dict[str, Any]:
        """Return summary statistics about the graph."""
        relation_counts: Dict[str, int] = defaultdict(int)
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "unknown")
            relation_counts[rel] += 1

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "relations": dict(relation_counts),
            "connected_components": nx.number_weakly_connected_components(self.graph),
        }

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str | Path | None = None) -> Path:
        """Serialise the graph to a JSON adjacency file."""
        save_path = Path(path) if path else self.graph_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = json_graph.node_link_data(self.graph)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Knowledge graph saved: %s (%d nodes)", save_path, len(self.graph))
        return save_path

    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_GRAPH_PATH,
        max_hops: int = DEFAULT_MAX_HOPS,
    ) -> Optional["KnowledgeGraph"]:
        """
        Load a previously saved graph.  Returns ``None`` if the file
        does not exist.
        """
        p = Path(path)
        if not p.exists():
            logger.debug("No knowledge graph found at %s", p)
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            kg = cls(graph_path=p, max_hops=max_hops)
            kg.graph = json_graph.node_link_graph(data, directed=True)
            logger.info(
                "Knowledge graph loaded: %d nodes, %d edges from %s",
                kg.graph.number_of_nodes(),
                kg.graph.number_of_edges(),
                p,
            )
            return kg
        except Exception as exc:
            logger.warning("Failed to load knowledge graph from %s: %s", p, exc)
            return None

    def reset(self) -> None:
        """Clear the graph in memory and delete the persisted file."""
        self.graph.clear()
        if self.graph_path.exists():
            self.graph_path.unlink()
            logger.info("Knowledge graph deleted: %s", self.graph_path)
