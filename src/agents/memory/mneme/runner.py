"""
run_with_memory — convenience wrapper around Runner.run that wires up:
  - MnemeSession for conversation history
  - make_memory_filter for automatic pre-call memory injection
  - MnemeRunHooks for automatic post-turn session_summary writes
"""

from __future__ import annotations

import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from . import store
from .hooks import MnemeRunHooks, make_memory_filter
from .session import MnemeSession

if TYPE_CHECKING:
    from agents import Agent
    from agents.result import RunResult
    from agents.run_config import RunConfig


async def run_with_memory(
    agent: Agent[Any],
    message: str,
    *,
    agent_id: str = store.DEFAULT_AGENT,
    session: MnemeSession | None = None,
    session_id: str | None = None,
    top_k: int = 5,
    score_threshold: float = 0.4,
    auto_summarize: bool = True,
    run_config: RunConfig | None = None,
    **kwargs: Any,
) -> RunResult:
    """
    Run an agent with automatic Mneme memory injection and summarization.

    On every model call, relevant facts are fetched from the store and injected
    into the system prompt as a <memory> block. If auto_summarize is True, the
    agent's final output is written back to the store as a session_summary fact
    so it is searchable in future runs.

    Args:
        agent: The agent to run.
        message: The user message to send.
        agent_id: Mneme store namespace (one DuckDB file per agent_id).
        session: An existing MnemeSession. Created from session_id if not provided.
        session_id: Session ID to use when creating a new MnemeSession.
        top_k: Max facts to inject per model call.
        score_threshold: Minimum cosine similarity score to include a fact.
        auto_summarize: Write a session_summary fact after each turn.
        run_config: Existing RunConfig to extend. Memory filter is merged in;
                    an existing call_model_input_filter is left in place.
        **kwargs: Forwarded to Runner.run (context, max_turns, hooks, etc.).

    Returns:
        RunResult from Runner.run.
    """
    from agents import Runner
    from agents.run_config import RunConfig as _RunConfig

    sid = session_id or (session.session_id if session else str(uuid.uuid4()))
    if session is None:
        session = MnemeSession(session_id=sid, agent_id=agent_id)

    memory_filter = make_memory_filter(
        agent_id=agent_id, top_k=top_k, score_threshold=score_threshold
    )

    if run_config is None:
        config = _RunConfig(call_model_input_filter=memory_filter)
    elif run_config.call_model_input_filter is None:
        config = replace(run_config, call_model_input_filter=memory_filter)
    else:
        config = run_config

    mneme_hooks = MnemeRunHooks(agent_id=agent_id, session_id=sid) if auto_summarize else None

    # Caller may pass their own hooks via kwargs; if so, use theirs and skip ours.
    if "hooks" in kwargs:
        mneme_hooks = None

    return await Runner.run(
        agent,
        message,
        session=session,
        run_config=config,
        hooks=mneme_hooks,
        **kwargs,
    )
