"""
Mneme — DuckDB/DuckLake-backed fact warehouse for agent memory.

Schema:
  facts        — typed content nodes (fact, decision, project, feedback, session_summary)
  embeddings   — nomic-embed vectors, kept separate so SELECT on facts never loads BLOBs
  relationships — typed edges for DuckPGQ graph traversal
  session_items — conversation turn history (implements the Session protocol)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

MNEME_DIR = Path.home() / ".mneme"
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
EMBEDDING_DIM = 768
DEFAULT_AGENT = "default"

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True, device="cpu")
    return _embed_model


def _embed(text: str) -> list[float]:
    model = _get_embed_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


def _db_path(agent_id: str) -> Path:
    MNEME_DIR.mkdir(parents=True, exist_ok=True)
    return MNEME_DIR / f"{agent_id}.db"


def _get_conn(agent_id: str = DEFAULT_AGENT) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(_db_path(agent_id)))
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id         VARCHAR PRIMARY KEY,
            content    VARCHAR NOT NULL,
            type       VARCHAR NOT NULL,
            agent_id   VARCHAR NOT NULL,
            session_id VARCHAR DEFAULT '',
            source     VARCHAR DEFAULT '',
            created_at TIMESTAMP NOT NULL
        )
    """)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS embeddings (
            fact_id VARCHAR PRIMARY KEY REFERENCES facts(id),
            vector  FLOAT[{EMBEDDING_DIM}] NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS relationships (
            from_id   VARCHAR NOT NULL,
            to_id     VARCHAR NOT NULL,
            edge_type VARCHAR NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_items (
            id         VARCHAR PRIMARY KEY,
            session_id VARCHAR NOT NULL,
            item_json  VARCHAR NOT NULL,
            position   INTEGER NOT NULL,
            created_at TIMESTAMP NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_type      ON facts(type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_agent     ON facts(agent_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_session   ON facts(session_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session_items   ON session_items(session_id, position)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_from        ON relationships(from_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_to          ON relationships(to_id)")


# ---------------------------------------------------------------------------
# Facts / KG
# ---------------------------------------------------------------------------


def write_fact(
    content: str,
    type: str,
    agent_id: str = DEFAULT_AGENT,
    session_id: str = "",
    source: str = "",
    relationships: list[dict[str, str]] | None = None,
) -> str:
    """Write a fact and its embedding. Returns the new fact id."""
    fact_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    vector = _embed(content)

    conn = _get_conn(agent_id)
    conn.execute(
        "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?)",
        [fact_id, content, type, agent_id, session_id, source, now],
    )
    conn.execute(
        "INSERT INTO embeddings VALUES (?, ?)",
        [fact_id, vector],
    )
    if relationships:
        for rel in relationships:
            conn.execute(
                "INSERT INTO relationships VALUES (?, ?, ?)",
                [fact_id, rel["to_id"], rel["edge_type"]],
            )
    conn.close()
    return fact_id


def search_facts(
    query: str,
    agent_id: str = DEFAULT_AGENT,
    top_k: int = 5,
    type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over facts using DuckDB array_cosine_similarity."""
    vector = _embed(query)
    conn = _get_conn(agent_id)

    type_clause = "AND f.type = ?" if type_filter else ""
    params: list[Any] = [vector, top_k]
    if type_filter:
        params = [vector, type_filter, top_k]

    rows = conn.execute(
        f"""
        SELECT
            f.id,
            f.content,
            f.type,
            f.session_id,
            f.source,
            f.created_at,
            array_cosine_similarity(e.vector, ?::FLOAT[{EMBEDDING_DIM}]) AS score
        FROM facts f
        JOIN embeddings e ON f.id = e.fact_id
        WHERE f.agent_id = '{agent_id}'
        {type_clause}
        ORDER BY score DESC
        LIMIT ?
    """,
        params,
    ).fetchall()

    conn.close()
    return [
        {
            "id": r[0],
            "content": r[1],
            "type": r[2],
            "session_id": r[3],
            "source": r[4],
            "created_at": str(r[5]),
            "score": r[6],
        }
        for r in rows
    ]


def graph_related(
    fact_id: str,
    edge_type: str | None = None,
    agent_id: str = DEFAULT_AGENT,
) -> list[dict[str, Any]]:
    """Return facts reachable from fact_id via relationship edges."""
    conn = _get_conn(agent_id)
    edge_clause = "AND r.edge_type = ?" if edge_type else ""
    params: list[Any] = [fact_id]
    if edge_type:
        params.append(edge_type)

    # depth=1 single hop for now; DuckPGQ recursive traversal can extend this
    rows = conn.execute(
        f"""
        SELECT DISTINCT f.id, f.content, f.type, f.created_at, r.edge_type
        FROM relationships r
        JOIN facts f ON f.id = r.to_id
        WHERE r.from_id = ?
        {edge_clause}
    """,
        params,
    ).fetchall()

    conn.close()
    return [
        {"id": r[0], "content": r[1], "type": r[2], "created_at": str(r[3]), "edge_type": r[4]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Session items (conversation history)
# ---------------------------------------------------------------------------


def session_get_items(
    session_id: str,
    agent_id: str = DEFAULT_AGENT,
    limit: int | None = None,
) -> list[str]:
    """Return serialized session items in chronological order."""
    conn = _get_conn(agent_id)
    limit_clause = f"LIMIT {limit}" if limit else ""
    rows = conn.execute(
        f"""
        SELECT item_json FROM session_items
        WHERE session_id = ?
        ORDER BY position ASC
        {limit_clause}
    """,
        [session_id],
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def session_add_items(
    session_id: str,
    items_json: list[str],
    agent_id: str = DEFAULT_AGENT,
) -> None:
    conn = _get_conn(agent_id)
    row = conn.execute(
        "SELECT COALESCE(MAX(position), -1) FROM session_items WHERE session_id = ?",
        [session_id],
    ).fetchone()
    next_pos = (row[0] + 1) if row else 0
    now = datetime.now(timezone.utc)
    for i, item_json in enumerate(items_json):
        conn.execute(
            "INSERT INTO session_items VALUES (?, ?, ?, ?, ?)",
            [str(uuid.uuid4()), session_id, item_json, next_pos + i, now],
        )
    conn.close()


def session_pop_item(
    session_id: str,
    agent_id: str = DEFAULT_AGENT,
) -> str | None:
    conn = _get_conn(agent_id)
    row = conn.execute(
        """
        SELECT id, item_json FROM session_items
        WHERE session_id = ?
        ORDER BY position DESC
        LIMIT 1
    """,
        [session_id],
    ).fetchone()
    if not row:
        conn.close()
        return None
    conn.execute("DELETE FROM session_items WHERE id = ?", [row[0]])
    conn.close()
    return row[1]


def session_clear(session_id: str, agent_id: str = DEFAULT_AGENT) -> None:
    conn = _get_conn(agent_id)
    conn.execute("DELETE FROM session_items WHERE session_id = ?", [session_id])
    conn.close()
