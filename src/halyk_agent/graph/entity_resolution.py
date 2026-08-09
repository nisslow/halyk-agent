"""
Entity Resolution and Knowledge Graph using KuzuDB.
Builds bi-temporal graph of entities, documents, and transactions.
"""
from __future__ import annotations
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import kuzu
from loguru import logger
from rapidfuzz import fuzz, process

from halyk_agent.config import settings
from halyk_agent.models import Entity, Transaction, DocumentMetadata, ExtractedTable

# Try to import LLM for entity resolution
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("OpenAI not available for entity resolution")


class KuzuGraph:
    """KuzuDB graph database wrapper."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.graph_db.path
        self.db = kuzu.Database(self.db_path)
        self.conn = kuzu.Connection(self.db)
        self._init_schema()

    def _init_schema(self):
        """Initialize graph schema."""
        # Node tables
        for entity_type in settings.graph_db.entity_types:
            self.conn.execute(f"""
                CREATE NODE TABLE IF NOT EXISTS {entity_type} (
                    entity_id STRING,
                    canonical_name STRING,
                    entity_type STRING,
                    aliases STRING[],
                    attributes STRING,
                    source_docs STRING[],
                    source_txns STRING[],
                    confidence DOUBLE,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    PRIMARY KEY (entity_id)
                )
            """)

        # Document node table
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Document (
                doc_id STRING,
                title STRING,
                doc_type STRING,
                valid_from TIMESTAMP,
                valid_to TIMESTAMP,
                version STRING,
                supersedes STRING,
                file_hash STRING,
                page_count INT64,
                PRIMARY KEY (doc_id)
            )
        """)

        # Transaction node table
        self.conn.execute("""
            CREATE NODE TABLE IF NOT EXISTS Transaction (
                txn_id STRING,
                date TIMESTAMP,
                amount DOUBLE,
                currency STRING,
                sender STRING,
                receiver STRING,
                sender_bin STRING,
                receiver_bin STRING,
                purpose STRING,
                category STRING,
                PRIMARY KEY (txn_id)
            )
        """)

        # Relationship tables
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS APPEARS_IN (FROM Entity TO Document, confidence DOUBLE, created_at TIMESTAMP)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS HAS_TXN (FROM Entity TO Transaction, confidence DOUBLE, created_at TIMESTAMP)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS SUPERSEDES (FROM Document TO Document, properties STRING, confidence DOUBLE, created_at TIMESTAMP)")
        self.conn.execute("CREATE REL TABLE IF NOT EXISTS AMENDS (FROM Document TO Document, properties STRING, confidence DOUBLE, created_at TIMESTAMP)")
        
        # Add the others just in case
        for rel_type in ["APPENDIX_OF", "ISSUED_BY", "GOVERNS"]:
            if rel_type in settings.graph_db.relation_types:
                self.conn.execute(f"CREATE REL TABLE IF NOT EXISTS {rel_type} (FROM Document TO Document, properties STRING, confidence DOUBLE, created_at TIMESTAMP)")

        logger.info("KuzuDB schema initialized")

    def execute(self, query: str, params: Optional[dict] = None):
        """Execute Cypher query."""
        return self.conn.execute(query, params or {})

    def close(self):
        """Close connection."""
        self.conn.close()
        self.db.close()


class EntityResolver:
    """Resolves entities across documents and transactions."""

    def __init__(self, graph: KuzuGraph):
        self.graph = graph
        self.llm_client = None
        if LLM_AVAILABLE:
            from halyk_agent.config import settings
            if settings.llm.api_key:
                self.llm_client = OpenAI(api_key=settings.llm.api_key)

        # Thresholds
        self.fuzzy_threshold = 85
        self.llm_threshold = 0.8

    def resolve_entities(
        self,
        documents: list[DocumentMetadata],
        transactions: list[Transaction],
        tables: list[ExtractedTable],
    ) -> list[Entity]:
        """Main entity resolution pipeline."""
        logger.info("Starting entity resolution")

        # 1. Extract candidate entities from all sources
        candidates = self._extract_candidates(documents, transactions, tables)

        # 2. Cluster similar entities (fuzzy matching)
        clusters = self._cluster_entities(candidates)

        # 3. LLM verification for ambiguous clusters
        verified_clusters = self._verify_clusters(clusters)

        # 4. Create canonical entities
        entities = self._create_entities(verified_clusters)

        # 5. Persist to graph
        self._persist_entities(entities)

        # 6. Build relationships
        self._build_relationships(entities, documents, transactions, tables)

        logger.info(f"Resolved {len(entities)} unique entities")
        return entities

    def _extract_candidates(
        self,
        documents: list[DocumentMetadata],
        transactions: list[Transaction],
        tables: list[ExtractedTable],
    ) -> list[dict]:
        """Extract entity candidates from all sources."""
        candidates = []

        # From documents
        for doc in documents:
            for org in doc.organizations:
                candidates.append({
                    "name": org,
                    "type": "Organization",
                    "source": "document",
                    "source_id": doc.doc_id,
                    "context": f"Document: {doc.title}",
                })
            for person in doc.persons:
                candidates.append({
                    "name": person,
                    "type": "Person",
                    "source": "document",
                    "source_id": doc.doc_id,
                    "context": f"Document: {doc.title}",
                })

        # From transactions
        for txn in transactions:
            candidates.append({
                "name": txn.sender,
                "type": "Organization",
                "source": "transaction",
                "source_id": txn.txn_id,
                "bin": txn.sender_bin,
                "context": f"Transaction {txn.txn_id}: sender",
            })
            candidates.append({
                "name": txn.receiver,
                "type": "Organization",
                "source": "transaction",
                "source_id": txn.txn_id,
                "bin": txn.receiver_bin,
                "context": f"Transaction {txn.txn_id}: receiver",
            })

        # From tables (extract org names from table cells)
        for table in tables:
            for row in table.rows:
                for cell in row:
                    # Heuristic: look for BIN patterns, org suffixes
                    if self._looks_like_org(cell.text):
                        candidates.append({
                            "name": cell.text,
                            "type": "Organization",
                            "source": "table",
                            "source_id": table.table_id,
                            "context": f"Table {table.table_id} row {cell.row}",
                        })

        return candidates

    def _looks_like_org(self, text: str) -> bool:
        """Heuristic to detect organization names."""
        text_lower = text.lower()
        org_suffixes = [
            "тоо", "ооо", "АО", "JSC", "LLC", "LTD", "Inc",
            "банк", "bank", "компания", "company", "группа", "group",
            "холдинг", "holding", "корпорация", "corporation",
        ]
        # Check for BIN pattern (12 digits)
        if re.search(r"\b\d{12}\b", text):
            return True
        return any(suffix in text_lower for suffix in org_suffixes)

    def _cluster_entities(self, candidates: list[dict]) -> list[list[dict]]:
        """Cluster similar entities using fuzzy matching."""
        clusters = []
        used = set()

        for i, cand in enumerate(candidates):
            if i in used:
                continue

            cluster = [cand]
            used.add(i)

            for j, other in enumerate(candidates[i+1:], i+1):
                if j in used:
                    continue

                # Compare names
                score = fuzz.ratio(cand["name"].lower(), other["name"].lower())
                if score >= self.fuzzy_threshold:
                    # Same type or both org/person
                    if cand["type"] == other["type"] or \
                       (cand["type"] in ["Organization", "Person"] and
                        other["type"] in ["Organization", "Person"]):
                        cluster.append(other)
                        used.add(j)

                # Also check BIN match
                if cand.get("bin") and other.get("bin") and cand["bin"] == other["bin"]:
                    if j not in used:
                        cluster.append(other)
                        used.add(j)

            clusters.append(cluster)

        return clusters

    def _verify_clusters(self, clusters: list[list[dict]]) -> list[list[dict]]:
        """Verify ambiguous clusters using LLM."""
        if not self.llm_client:
            return clusters

        verified = []
        for cluster in clusters:
            if len(cluster) <= 1:
                verified.append(cluster)
                continue

            # Check if names are very different
            names = [c["name"] for c in cluster]
            min_similarity = min(
                fuzz.ratio(a.lower(), b.lower())
                for i, a in enumerate(names)
                for b in names[i+1:]
            )

            if min_similarity > 90:
                verified.append(cluster)
                continue

            # Ask LLM to verify
            prompt = f"""Are these the same entity? Answer YES or NO with brief reason.
Entities: {', '.join(names)}
Contexts: {'; '.join(c['context'] for c in cluster)}"""

            try:
                response = self.llm_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=100,
                )
                answer = response.choices[0].message.content.strip().upper()
                if answer.startswith("YES"):
                    verified.append(cluster)
                else:
                    # Split into singletons
                    verified.extend([[c] for c in cluster])
            except Exception as e:
                logger.warning(f"LLM verification failed: {e}")
                verified.append(cluster)

        return verified

    def _create_entities(self, clusters: list[list[dict]]) -> list[Entity]:
        """Create canonical entities from verified clusters."""
        entities = []

        for cluster in clusters:
            # Choose canonical name (longest/most complete)
            canonical = max(cluster, key=lambda c: len(c["name"]))["name"]
            entity_type = cluster[0]["type"]

            # Collect all aliases
            aliases = list(set(c["name"] for c in cluster))
            aliases.remove(canonical) if canonical in aliases else None

            # Collect sources
            source_docs = list(set(
                c["source_id"] for c in cluster if c["source"] == "document"
            ))
            source_txns = list(set(
                c["source_id"] for c in cluster if c["source"] == "transaction"
            ))

            # BIN if available
            bins = [c.get("bin") for c in cluster if c.get("bin")]
            attributes = {"bins": bins} if bins else {}

            entity = Entity(
                canonical_name=canonical,
                entity_type=entity_type,
                aliases=aliases,
                attributes=attributes,
                source_docs=source_docs,
                source_txns=source_txns,
                confidence=0.9 if len(cluster) == 1 else 0.7,
            )
            entities.append(entity)

        return entities

    def _persist_entities(self, entities: list[Entity]):
        """Persist entities to KuzuDB."""
        for entity in entities:
            self.graph.execute(
                """
                MERGE (e:Entity {entity_id: $entity_id})
                SET e.canonical_name = $canonical_name,
                    e.entity_type = $entity_type,
                    e.aliases = $aliases,
                    e.attributes = $attributes,
                    e.source_docs = $source_docs,
                    e.source_txns = $source_txns,
                    e.confidence = $confidence,
                    e.updated_at = $updated_at
                """,
                {
                    "entity_id": entity.entity_id,
                    "canonical_name": entity.canonical_name,
                    "entity_type": entity.entity_type,
                    "aliases": entity.aliases,
                    "attributes": str(entity.attributes),
                    "source_docs": entity.source_docs,
                    "source_txns": entity.source_txns,
                    "confidence": entity.confidence,
                    "updated_at": datetime.utcnow(),
                }
            )

    def _build_relationships(
        self,
        entities: list[Entity],
        documents: list[DocumentMetadata],
        transactions: list[Transaction],
        tables: list[ExtractedTable],
    ):
        """Build relationships in the graph."""
        # Entity -> Document (APPEARS_IN)
        for entity in entities:
            for doc_id in entity.source_docs:
                self.graph.execute(
                    """
                    MATCH (e:Entity {entity_id: $entity_id}), (d:Document {doc_id: $doc_id})
                    MERGE (e)-[:APPEARS_IN]->(d)
                    """,
                    {"entity_id": entity.entity_id, "doc_id": doc_id}
                )

        # Entity -> Transaction (HAS_TXN)
        for entity in entities:
            for txn_id in entity.source_txns:
                self.graph.execute(
                    """
                    MATCH (e:Entity {entity_id: $entity_id}), (t:Transaction {txn_id: $txn_id})
                    MERGE (e)-[:HAS_TXN]->(t)
                    """,
                    {"entity_id": entity.entity_id, "txn_id": txn_id}
                )

        # Document version relationships
        for doc in documents:
            if doc.supersedes:
                self.graph.execute(
                    """
                    MATCH (d1:Document {doc_id: $doc_id}), (d2:Document {doc_id: $supersedes})
                    MERGE (d1)-[:SUPERSEDES]->(d2)
                    """,
                    {"doc_id": doc.doc_id, "supersedes": doc.supersedes}
                )
            if doc.amends:
                self.graph.execute(
                    """
                    MATCH (d1:Document {doc_id: $doc_id}), (d2:Document {doc_id: $amends})
                    MERGE (d1)-[:AMENDS]->(d2)
                    """,
                    {"doc_id": doc.doc_id, "amends": doc.amends}
                )

        # Persist documents and transactions
        self._persist_documents(documents)
        self._persist_transactions(transactions)

    def _persist_documents(self, documents: list[DocumentMetadata]):
        """Persist documents to graph."""
        for doc in documents:
            self.graph.execute(
                """
                MERGE (d:Document {doc_id: $doc_id})
                SET d.title = $title,
                    d.doc_type = $doc_type,
                    d.valid_from = $valid_from,
                    d.valid_to = $valid_to,
                    d.version = $version,
                    d.supersedes = $supersedes,
                    d.file_hash = $file_hash,
                    d.page_count = $page_count
                """,
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "doc_type": doc.doc_type.value,
                    "valid_from": doc.valid_from,
                    "valid_to": doc.valid_to,
                    "version": doc.version,
                    "supersedes": doc.supersedes,
                    "file_hash": doc.file_hash,
                    "page_count": doc.page_count,
                }
            )

    def _persist_transactions(self, transactions: list[Transaction]):
        """Persist transactions to graph."""
        for txn in transactions:
            self.graph.execute(
                """
                MERGE (t:Transaction {txn_id: $txn_id})
                SET t.date = $date,
                    t.amount = $amount,
                    t.currency = $currency,
                    t.sender = $sender,
                    t.receiver = $receiver,
                    t.sender_bin = $sender_bin,
                    t.receiver_bin = $receiver_bin,
                    t.purpose = $purpose,
                    t.category = $category
                """,
                {
                    "txn_id": txn.txn_id,
                    "date": txn.date,
                    "amount": txn.amount,
                    "currency": txn.currency,
                    "sender": txn.sender,
                    "receiver": txn.receiver,
                    "sender_bin": txn.sender_bin,
                    "receiver_bin": txn.receiver_bin,
                    "purpose": txn.purpose,
                    "category": txn.category,
                }
            )

    def query_entities(
        self,
        name: str,
        entity_type: Optional[str] = None,
    ) -> list[dict]:
        """Query entities by name."""
        query = """
        MATCH (e:Entity)
        WHERE e.canonical_name CONTAINS $name
        """
        params = {"name": name}
        if entity_type:
            query += " AND e.entity_type = $entity_type"
            params["entity_type"] = entity_type
        query += " RETURN e"
        return self.graph.execute(query, params).get_as_pl()

    def get_entity_timeline(self, entity_id: str) -> list[dict]:
        """Get bi-temporal timeline for an entity."""
        query = """
        MATCH (e:Entity {entity_id: $entity_id})-[:APPEARS_IN]->(d:Document)
        RETURN d.doc_id, d.title, d.valid_from, d.valid_to, d.version
        ORDER BY d.valid_from
        """
        return self.graph.execute(query, {"entity_id": entity_id}).get_as_pl()

    def get_transaction_network(self, entity_id: str, hops: int = 2) -> list[dict]:
        """Get transaction network around an entity."""
        query = f"""
        MATCH (e:Entity {{entity_id: $entity_id}})-[:HAS_TXN*1..{hops}]-(t:Transaction)
        RETURN t
        """
        return self.graph.execute(query, {"entity_id": entity_id}).get_as_pl()


def create_graph(db_path: Optional[str] = None) -> KuzuGraph:
    """Factory function to create graph."""
    return KuzuGraph(db_path)


def create_resolver(graph: KuzuGraph) -> EntityResolver:
    """Factory function to create resolver."""
    return EntityResolver(graph)