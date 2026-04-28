"""
Chainlit lifecycle hooks that bridge the OpenAI Agents SDK to the UI.

ChainlitRunHooks  — emits cl.Step for tool calls, sends handoff indicators
FleetSyncHooks    — promotes facts to MotherDuck on agent_end
ComposedHooks     — fans lifecycle calls out to a list of RunHooks delegates
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import chainlit as cl

from agents.lifecycle import RunHooksBase
from agents.run_context import AgentHookContext

log = logging.getLogger(__name__)


class ChainlitRunHooks(RunHooksBase):
    """Emit Chainlit Steps for tool calls and a system message for handoffs."""

    async def on_tool_start(
        self,
        context: AgentHookContext,
        agent: Any,
        tool: Any,
    ) -> None:
        tool_name = getattr(tool, "name", str(tool))
        step = cl.Step(name=tool_name, type="tool", show_input=True)
        await step.__aenter__()
        cl.user_session.set(f"_step_{tool_name}", step)

    async def on_tool_end(
        self,
        context: AgentHookContext,
        agent: Any,
        tool: Any,
        result: str,
    ) -> None:
        tool_name = getattr(tool, "name", str(tool))
        step: cl.Step | None = cl.user_session.get(f"_step_{tool_name}")
        if step is not None:
            step.output = result[:2000] if result else "(no output)"
            await step.__aexit__(None, None, None)
            cl.user_session.set(f"_step_{tool_name}", None)

    async def on_handoff(
        self,
        context: AgentHookContext,
        from_agent: Any,
        to_agent: Any,
    ) -> None:
        from_name = getattr(from_agent, "name", "?")
        to_name = getattr(to_agent, "name", "?")
        await cl.Message(
            content=f"↪ **{from_name}** → **{to_name}**",
            author="System",
        ).send()


class FleetSyncHooks(RunHooksBase):
    """Promote high-signal facts to MotherDuck fleet KB after each agent turn."""

    def __init__(self, agent_id: str, enabled: bool = True) -> None:
        self.agent_id = agent_id
        self.enabled = enabled

    async def on_agent_end(
        self,
        context: AgentHookContext,
        agent: Any,
        output: Any,
    ) -> None:
        if not self.enabled:
            return
        # Run in background — don't block the response
        asyncio.create_task(self._promote())

    async def _promote(self) -> None:
        try:
            from agents.memory.mneme.fleet import FleetStore
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._promote_sync)
        except Exception as e:
            log.warning("Fleet sync failed: %s", e)

    def _promote_sync(self) -> None:
        from agents.memory.mneme.fleet import FleetStore
        with FleetStore() as fleet:
            fleet.promote_from(self.agent_id)


class ComposedHooks(RunHooksBase):
    """Fan out all RunHooks lifecycle events to a list of delegates."""

    def __init__(self, delegates: list[RunHooksBase]) -> None:
        self.delegates = delegates

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> None:
        for d in self.delegates:
            fn = getattr(d, method, None)
            if fn is not None:
                try:
                    await fn(*args, **kwargs)
                except Exception as e:
                    log.warning("Hook %s.%s raised: %s", type(d).__name__, method, e)

    async def on_llm_start(self, *a: Any, **kw: Any) -> None:
        await self._call("on_llm_start", *a, **kw)

    async def on_llm_end(self, *a: Any, **kw: Any) -> None:
        await self._call("on_llm_end", *a, **kw)

    async def on_agent_start(self, *a: Any, **kw: Any) -> None:
        await self._call("on_agent_start", *a, **kw)

    async def on_agent_end(self, *a: Any, **kw: Any) -> None:
        await self._call("on_agent_end", *a, **kw)

    async def on_handoff(self, *a: Any, **kw: Any) -> None:
        await self._call("on_handoff", *a, **kw)

    async def on_tool_start(self, *a: Any, **kw: Any) -> None:
        await self._call("on_tool_start", *a, **kw)

    async def on_tool_end(self, *a: Any, **kw: Any) -> None:
        await self._call("on_tool_end", *a, **kw)
