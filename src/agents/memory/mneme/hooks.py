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
    fleet: bool = True,
):
    """
    Return a CallModelInputFilter that injects relevant memory into the system prompt.

    On every model call the filter:
      1. Extracts the last user message from the input list.
      2. Searches the agent's local Mneme store for semantically related facts.
      3. If fleet=True and MOTHERDUCK_TOKEN is set, also searches the Fleet KB on
         MotherDuck (across all agents) and merges the two result sets by id,
         sorted by score, capped at top_k.
      4. Appends a <memory> block to the existing instructions string.

    Each fact is labelled with its `agent_id` in the injected block when it
    came from a different agent than the caller, so the model can tell
    cross-agent context apart from its own.

    The FleetStore connection is created lazily and cached across the run via
    the closure so we don't pay a fresh MotherDuck connect per LLM call. If
    MotherDuck is unreachable, the filter falls back to local-only and tags the
    closure so subsequent calls skip the fleet path until the next process.
    """
    import os
    state: dict[str, Any] = {"fleet": None, "fleet_disabled": False}

    def _get_fleet():
        if not fleet or state["fleet_disabled"]:
            return None
        if state["fleet"] is not None:
            return state["fleet"]
        if not os.environ.get("MOTHERDUCK_TOKEN", "").strip():
            state["fleet_disabled"] = True
            return None
        try:
            from .fleet import FleetStore
            state["fleet"] = FleetStore()
            return state["fleet"]
        except Exception:
            state["fleet_disabled"] = True
            return None

    async def _filter(data: CallModelData[Any]) -> ModelInputData:
        from ...run_config import ModelInputData

        query = _last_user_text(data.model_data.input)
        if not query:
            return data.model_data

        local_hits = store.search_facts(query, agent_id=agent_id, top_k=top_k)

        fleet_hits: list[dict[str, Any]] = []
        f = _get_fleet()
        if f is not None:
            try:
                fleet_hits = f.search(query, top_k=top_k)
            except Exception:
                state["fleet_disabled"] = True
                fleet_hits = []

        # Merge by id, preferring whichever copy has the higher score (a fact
        # already promoted to fleet may appear in both; keep the better one).
        # Local hits don't include agent_id in their dict by default — inject the
        # caller's agent_id so downstream rendering can tell self vs cross-agent.
        merged: dict[str, dict[str, Any]] = {}
        for h in local_hits:
            h2 = dict(h)
            h2.setdefault("agent_id", agent_id)
            merged[h["id"]] = h2
        for h in fleet_hits:
            existing = merged.get(h["id"])
            if existing is None or (h.get("score", 0) > existing.get("score", 0)):
                merged[h["id"]] = h

        ranked = sorted(merged.values(), key=lambda x: x.get("score", 0), reverse=True)
        hits = [h for h in ranked if h.get("score", 0) >= score_threshold][:top_k]
        if not hits:
            return data.model_data

        def _render(h: dict[str, Any]) -> str:
            origin = h.get("agent_id", agent_id)
            tag = h.get("type", "fact")
            if origin and origin != agent_id:
                return f"[{tag} from {origin}] {h['content']}"
            return f"[{tag}] {h['content']}"

        memory_block = "\n\n<memory>\n" + "\n".join(_render(h) for h in hits) + "\n</memory>"

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
    RunHooks that auto-writes a session_summary fact after each agent turn,
    then opportunistically promotes the agent's recent high-signal facts to the
    MotherDuck FleetStore so cross-agent learning is real-time (not nightly).

    Pair with make_memory_filter so the summary is searchable in future turns.
    """

    def __init__(
        self,
        agent_id: str = store.DEFAULT_AGENT,
        session_id: str = "",
        max_summary_length: int = 2000,
        promote_to_fleet: bool = True,
    ) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.max_summary_length = max_summary_length
        self.promote_to_fleet = promote_to_fleet

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
        # Real-time promotion to MotherDuck Fleet KB. Belt-and-braces with the
        # nightly bonsai-agents.py catch-all. Silent-skip if MOTHERDUCK_TOKEN is
        # not configured, fail-silent if MotherDuck is unreachable — must never
        # break a successful agent run.
        if not self.promote_to_fleet:
            return
        import os
        if not os.environ.get("MOTHERDUCK_TOKEN", "").strip():
            return
        try:
            from .fleet import FleetStore
            fleet = FleetStore()
            try:
                fleet.promote_from(agent_id=self.agent_id, since_hours=1)
            finally:
                fleet.close()
        except Exception:
            return
