"""MnemeSession — DuckDB-backed drop-in for SQLiteSession."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ..session import SessionABC
from ..session_settings import SessionSettings
from . import store

if TYPE_CHECKING:
    from ...items import TResponseInputItem


class MnemeSession(SessionABC):
    """
    Conversation history backed by the Mneme DuckDB fact warehouse.

    Drop-in replacement for SQLiteSession. Pass agent_id to namespace
    the store per agent (defaults to "default").
    """

    def __init__(
        self,
        session_id: str,
        agent_id: str = store.DEFAULT_AGENT,
        session_settings: SessionSettings | None = None,
    ) -> None:
        self.session_id = session_id
        self.agent_id = agent_id
        self.session_settings = session_settings

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        raw = store.session_get_items(self.session_id, self.agent_id, limit)
        return [json.loads(r) for r in raw]

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        store.session_add_items(
            self.session_id,
            [json.dumps(item) for item in items],
            self.agent_id,
        )

    async def pop_item(self) -> TResponseInputItem | None:
        raw = store.session_pop_item(self.session_id, self.agent_id)
        return json.loads(raw) if raw is not None else None

    async def clear_session(self) -> None:
        store.session_clear(self.session_id, self.agent_id)
