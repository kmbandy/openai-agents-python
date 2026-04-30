"""
Mneme agent tools — memory_search, memory_write, memory_graph.

Attach these to any OpenAI Agents SDK agent to give it persistent
semantic memory backed by the Mneme DuckDB fact warehouse.

Usage:
    from agents.memory.mneme.tools import build_memory_tools

    agent = Agent(
        name="my-agent",
        tools=build_memory_tools(agent_id="my-agent"),
    )
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from . import store


class Relationship(BaseModel):
    to_id: str
    edge_type: str


def build_memory_tools(agent_id: str = store.DEFAULT_AGENT):
    """Return the three Mneme memory tools scoped to agent_id."""
    from agents import function_tool

    @function_tool
    def memory_search(query: str, top_k: int = 5, type_filter: str = "") -> list[dict[str, Any]]:
        """
        Search long-term memory for facts semantically related to the query.

        Args:
            query: Natural language search query.
            top_k: Number of results to return (default 5, max 20).
            type_filter: Optional node type to restrict results.
                         One of: fact, decision, project, feedback, session_summary.
                         Leave empty to search all types.

        Returns:
            List of matching facts with id, content, type, score, and created_at.
        """
        top_k = min(top_k, 20)
        return store.search_facts(
            query,
            agent_id=agent_id,
            top_k=top_k,
            type_filter=type_filter or None,
        )

    @function_tool
    def memory_write(
        content: str,
        type: str = "fact",
        session_id: str = "",
        relationships: list[Relationship] | None = None,
    ) -> str:
        """
        Write a new fact to long-term memory.

        Args:
            content: The text content to remember.
            type: Node type — one of: fact, decision, project, feedback, session_summary.
            session_id: Optional session this memory is associated with.
            relationships: Optional list of edges to other facts.
                           Each entry: {"to_id": "<fact_id>", "edge_type": "<type>"}.
                           Common edge types: references, caused_by, related_to, same_session.

        Returns:
            The id of the newly created fact.
        """
        rels = [r.model_dump() for r in relationships] if relationships else None
        return store.write_fact(
            content=content,
            type=type,
            agent_id=agent_id,
            session_id=session_id,
            source="agent",
            relationships=rels,
        )

    @function_tool
    def memory_graph(
        fact_id: str,
        edge_type: str = "",
    ) -> list[dict[str, Any]]:
        """
        Traverse the memory graph from a given fact to find related facts.

        Args:
            fact_id: The id of the fact to start traversal from.
            edge_type: Optional edge type filter (e.g. "references", "caused_by").
                       Leave empty to return all connected facts.

        Returns:
            List of related facts with their edge types.
        """
        return store.graph_related(
            fact_id=fact_id,
            edge_type=edge_type or None,
            agent_id=agent_id,
        )

    return [memory_search, memory_write, memory_graph]
