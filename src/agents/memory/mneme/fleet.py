"""
Mneme fleet KB — MotherDuck-backed shared knowledge base.

All agents on all machines share a single cloud fleet DB at md:mneme_fleet.
Bonsai (or any caller) can promote high-signal facts from individual agent
DBs into the fleet KB, making them searchable by any agent in the fleet.

Usage:
    from agents.memory.mneme.fleet import FleetStore

    fleet = FleetStore()                        # connects via MOTHERDUCK_TOKEN env var
    fleet.promote_from(agent_id="my-agent")     # push today's decisions/projects up
    results = fleet.search("ROCm FP8 support")  # search across all agents
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import duckdb

from . import store

FLEET_DB = "md:mneme_fleet"
PROMOTE_TYPES = {"decision", "project", "feedback"}


class FleetStore:
    """
    Cloud fleet knowledge base backed by MotherDuck.

    Wraps a persistent md:mneme_fleet connection and mirrors the Mneme
    facts/embeddings/relationships schema so local store queries work
    identically against the fleet DB.
    """

    def __init__(self, token: str | None = None) -> None:
        tok = token or os.environ.get("MOTHERDUCK_TOKEN", "")
        if not tok:
            raise RuntimeError(
                "MOTHERDUCK_TOKEN not set. Export it or pass token= to FleetStore()."
            )
        os.environ["MOTHERDUCK_TOKEN"] = tok
        # Create the fleet DB in MotherDuck if it doesn't exist yet.
        bootstrap = duckdb.connect("md:")
        bootstrap.execute("CREATE DATABASE IF NOT EXISTS mneme_fleet")
        bootstrap.close()
        self._conn = duckdb.connect(FLEET_DB)
        store._ensure_schema(self._conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> FleetStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_from(
        self,
        agent_id: str,
        types: set[str] = PROMOTE_TYPES,
        since_hours: int = 25,
    ) -> int:
        """
        Copy high-signal facts from a local agent DB into the fleet KB.

        Only promotes facts whose type is in `types` and were created
        within the last `since_hours` hours. Skips facts already present.

        Returns:
            Number of new facts promoted.
        """
        local_path = store._db_path(agent_id)
        if not local_path.exists():
            return 0

        type_list = ", ".join(f"'{t}'" for t in types)

        self._conn.execute(f"ATTACH '{local_path}' AS src (READ_ONLY)")
        try:
            before = (self._conn.execute("SELECT COUNT(*) FROM facts").fetchone() or (0,))[0]

            self._conn.execute(f"""
                INSERT INTO facts
                SELECT id, content, type, agent_id, session_id, source, created_at
                FROM src.facts
                WHERE type IN ({type_list})
                AND created_at >= now() - INTERVAL '{since_hours} hours'
                ON CONFLICT (id) DO NOTHING
            """)

            promoted_ids = self._conn.execute(f"""
                SELECT id FROM src.facts
                WHERE type IN ({type_list})
                AND created_at >= now() - INTERVAL '{since_hours} hours'
            """).fetchall()

            if promoted_ids:
                id_list = ", ".join(f"'{r[0]}'" for r in promoted_ids)
                self._conn.execute(f"""
                    INSERT INTO embeddings
                    SELECT fact_id, vector
                    FROM src.embeddings
                    WHERE fact_id IN ({id_list})
                    ON CONFLICT (fact_id) DO NOTHING
                """)
                self._conn.execute(f"""
                    INSERT INTO relationships
                    SELECT from_id, to_id, edge_type
                    FROM src.relationships
                    WHERE from_id IN ({id_list})
                """)

            after = (self._conn.execute("SELECT COUNT(*) FROM facts").fetchone() or (0,))[0]
            return after - before
        finally:
            self._conn.execute("DETACH src")

    def promote_all(self, since_hours: int = 25) -> dict[str, int]:
        """
        Promote from every local agent DB in ~/.mneme/.

        Returns a dict of {agent_id: facts_promoted}.
        """
        results: dict[str, int] = {}
        for db_file in sorted(Path(store.MNEME_DIR).glob("*.db")):
            agent_id = db_file.stem
            if agent_id == "fleet":
                continue
            n = self.promote_from(agent_id=agent_id, since_hours=since_hours)
            if n > 0:
                results[agent_id] = n
        return results

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        type_filter: str | None = None,
        agent_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Semantic search across the entire fleet KB.

        Args:
            query: Natural language query.
            top_k: Max results (capped at 50).
            type_filter: Restrict to a specific fact type.
            agent_filter: Restrict to a specific agent_id.

        Returns:
            List of facts with id, content, type, agent_id, score.
        """
        top_k = min(top_k, 50)
        vector = store._embed(query)

        clauses: list[str] = []
        params: list[Any] = [vector, top_k]

        if type_filter:
            clauses.append("f.type = ?")
            params.insert(1, type_filter)
        if agent_filter:
            clauses.append("f.agent_id = ?")
            params.insert(1 + int(bool(type_filter)), agent_filter)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params = [vector] + [p for p in params[1:] if isinstance(p, str)] + [top_k]

        rows = self._conn.execute(f"""
            SELECT
                f.id, f.content, f.type, f.agent_id, f.created_at,
                array_cosine_similarity(e.vector, ?::FLOAT[{store.EMBEDDING_DIM}]) AS score
            FROM facts f
            JOIN embeddings e ON f.id = e.fact_id
            {where}
            ORDER BY score DESC
            LIMIT ?
        """, params).fetchall()

        return [
            {
                "id": r[0], "content": r[1], "type": r[2],
                "agent_id": r[3], "created_at": str(r[4]), "score": r[5],
            }
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        """Return basic fleet KB stats."""
        total = (self._conn.execute("SELECT COUNT(*) FROM facts").fetchone() or (0,))[0]
        by_agent = self._conn.execute(
            "SELECT agent_id, COUNT(*) FROM facts GROUP BY agent_id ORDER BY 2 DESC"
        ).fetchall()
        by_type = self._conn.execute(
            "SELECT type, COUNT(*) FROM facts GROUP BY type ORDER BY 2 DESC"
        ).fetchall()
        return {
            "total_facts": total,
            "by_agent": dict(by_agent),
            "by_type": dict(by_type),
        }
