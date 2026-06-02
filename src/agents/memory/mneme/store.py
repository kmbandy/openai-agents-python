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
import logging
import time

MNEME_DIR = Path.home() / ".mneme"
EMBEDDING_MODEL = "qwen3-embedding-0.6b"
EMBEDDING_DIM = 768
EMBED_URL = "http://127.0.0.1:8082/v1/embeddings"
DEFAULT_AGENT = "default"

log = logging.getLogger("mneme.store")

# DuckDB lock errors (single-writer per file). Version-robust tuple.
_DB_LOCK_ERRORS = (getattr(duckdb, "IOException", getattr(duckdb, "Error", Exception)),)


def _with_retry(fn, *, exceptions, label, tries=4, base=0.25):
    """Run fn(), retrying transient failures with exponential backoff.

    The embed server can briefly queue (batch syncs) and DuckDB is single-writer
    per file, so a connect can collide with a concurrent sync/aging job. Retrying
    lets these self-heal instead of surfacing as a hard timeout to the caller.
    Re-raises the real error if the service is genuinely down (never hangs).
    """
    last_exc = None
    for attempt in range(tries):
        try:
            return fn()
        except exceptions as exc:
            last_exc = exc
            if attempt == tries - 1:
                break
            log.warning(
                "mneme %s failed (attempt %d/%d): %s", label, attempt + 1, tries, exc
            )
            time.sleep(base * (2 ** attempt))
    raise last_exc


def _embed(text: str) -> list[float]:
    import httpx, math

    def _call() -> list[float]:
        resp = httpx.post(
            EMBED_URL, json={"input": text, "encoding_format": "float"}, timeout=30.0
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"][:EMBEDDING_DIM]

    vec = _with_retry(_call, exceptions=(httpx.HTTPError,), label="embed")
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _db_path(agent_id: str) -> Path:
    MNEME_DIR.mkdir(parents=True, exist_ok=True)
    return MNEME_DIR / f"{agent_id}.db"


def _get_conn(agent_id: str = DEFAULT_AGENT) -> duckdb.DuckDBPyConnection:
    # connect raises duckdb.IOException when another process holds the
    # single-writer lock; be patient (lock windows can be a few seconds).
    def _connect() -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(str(_db_path(agent_id)))
        _ensure_schema(conn)
        return conn

    return _with_retry(
        _connect, exceptions=_DB_LOCK_ERRORS, label="duckdb connect", tries=6, base=0.3
    )


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
