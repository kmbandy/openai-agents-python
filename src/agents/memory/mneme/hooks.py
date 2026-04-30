"""
Mneme lifecycle hooks — automatic memory injection and session summarization.

make_memory_filter  — CallModelInputFilter that prepends relevant memory facts
                      to the system prompt before every LLM call.
MnemeRunHooks       — RunHooks subclass that auto-writes a session_summary fact
                      after each agent turn completes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...lifecycle import RunHooksBase
from ...run_context import AgentHookContext
from . import store

if TYPE_CHECKING:
    from ...run_config import CallModelData, ModelInputData


def make_memory_filter(
    agent_id: str = store.DEFAULT_AGENT,
    top_k: int = 5,
    score_threshold: float = 0.4,
):
    """
    Return a CallModelInputFilter that injects relevant memory into the system prompt.

    On every model call the filter:
      1. Extracts the last user message from the input list.
      2. Searches Mneme for semantically related facts (top_k, score >= threshold).
      3. Appends a <memory> block to the existing instructions string.

    If no relevant facts are found, the instructions are returned unchanged.
    """

    async def _filter(data: CallModelData[Any]) -> ModelInputData:
        from ...run_config import ModelInputData

        query = _last_user_text(data.model_data.input)
        if not query:
            return data.model_data

        hits = [
            h
            for h in store.search_facts(query, agent_id=agent_id, top_k=top_k)
            if h["score"] >= score_threshold
        ]
        if not hits:
            return data.model_data

        memory_block = "\n\n<memory>\n" + "\n".join(
            f"[{h['type']}] {h['content']}" for h in hits
        ) + "\n</memory>"

        base = data.model_data.instructions or ""
        return ModelInputData(
            input=data.model_data.input,
            instructions=base + memory_block,
        )

    return _filter


def _last_user_text(items: list[Any]) -> str:
    """Extract the text of the most recent user message from an input list."""
    for item in reversed(items):
        if not isinstance(item, dict) or item.get("role") != "user":
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
    return ""


class MnemeRunHooks(RunHooksBase[Any]):
    """
    RunHooks that auto-writes a session_summary fact after each agent turn.

    Pair with make_memory_filter so the summary is searchable in future turns.
    """

    def __init__(
        self,
        agent_id: str = store.DEFAULT_AGENT,
        session_id: str = "",
        max_summary_length: int = 2000,
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.max_summary_length = max_summary_length

    async def on_agent_end(
        self,
        context: AgentHookContext[Any],  # noqa: ARG002
        agent: Any,  # noqa: ARG002
        output: Any,
    ) -> None:
        if not output:
            return
        summary = str(output)[: self.max_summary_length]
        store.write_fact(
            content=summary,
            type="session_summary",
            agent_id=self.agent_id,
            session_id=self.session_id,
            source="auto",
        )
