"""
mad-lab Chainlit UI — full-featured interface for openai-agents-python fork.

Profiles:
  - One profile per configured agent (loaded from config.json)
  - "⚙ Admin" profile for agent builder + server settings

Features:
  - Streaming token-by-token output
  - Tool call steps (collapsible, with input/output)
  - Agent handoff indicators
  - Per-session Mneme DuckDB memory with semantic injection
  - Auto fleet sync to MotherDuck on session end
  - In-chat settings panel: endpoint, temperature, max turns, fleet sync toggle
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from pathlib import Path

# Make repo src importable
_repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo_root / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chainlit as cl

from agents import Runner, set_tracing_disabled
from agents.memory.mneme.hooks import MnemeRunHooks, make_memory_filter
from agents.memory.mneme.session import MnemeSession
from agents.model_settings import ModelSettings
from agents.run_config import RunConfig

from mad_agents.registry import build_all_agents
from config import Config
from hooks import ChainlitRunHooks, ComposedHooks, FleetSyncHooks
from pages.agent_builder import show_agent_list
from pages.server_settings import show_settings_menu

set_tracing_disabled(disabled=True)

log = logging.getLogger(__name__)

ADMIN_PROFILE = "⚙ Admin"


# ---------------------------------------------------------------------------
# Profile selection
# ---------------------------------------------------------------------------

@cl.set_chat_profiles
async def chat_profiles() -> list[cl.ChatProfile]:
    config = Config.load()
    profiles = []
    for defn in config.agents:
        desc = f"`{defn.endpoint_name}`\n\n{defn.system_prompt[:120]}{'…' if len(defn.system_prompt) > 120 else ''}"
        profiles.append(cl.ChatProfile(name=defn.name, markdown_description=desc))
    profiles.append(
        cl.ChatProfile(
            name=ADMIN_PROFILE,
            markdown_description="Manage agents, model endpoints, and fleet sync.",
        )
    )
    return profiles


# ---------------------------------------------------------------------------
# Session init
# ---------------------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start() -> None:
    profile = cl.user_session.get("chat_profile") or ""

    if profile == ADMIN_PROFILE:
        await cl.Message(
            content="**mad-lab Admin**\n\nWhat would you like to manage?",
            actions=[
                cl.Action(name="admin_agents", label="🤖 Agents", payload={}),
                cl.Action(name="admin_settings", label="⚙️ Server Settings", payload={}),
            ],
        ).send()
        return

    config = Config.load()
    defn = config.get_agent(profile)
    if not defn:
        await cl.Message(content=f"⚠️ Agent '{profile}' not found in config.").send()
        return

    agents = build_all_agents(config)
    agent = agents.get(profile)
    if not agent:
        await cl.Message(content=f"⚠️ Could not build agent '{profile}'.").send()
        return

    session_id = str(uuid.uuid4())
    session = MnemeSession(session_id=session_id, agent_id=defn.agent_id)

    cl.user_session.set("agent", agent)
    cl.user_session.set("session", session)
    cl.user_session.set("agent_id", defn.agent_id)
    cl.user_session.set("config", config)
    cl.user_session.set("settings", {})

    await cl.ChatSettings([
        cl.input_widget.Select(
            id="endpoint",
            label="Model Endpoint",
            values=[ep.name for ep in config.endpoints],
            initial_value=defn.endpoint_name,
        ),
        cl.input_widget.Slider(
            id="temperature",
            label="Temperature",
            min=0.0,
            max=2.0,
            step=0.05,
            initial=0.7,
        ),
        cl.input_widget.NumberInput(
            id="max_turns",
            label="Max Turns",
            initial=10,
        ),
        cl.input_widget.Switch(
            id="fleet_sync",
            label="Auto-sync to Fleet (MotherDuck)",
            initial=bool(os.environ.get("MOTHERDUCK_TOKEN")),
        ),
    ]).send()

    await cl.Message(
        content=f"**{profile}** ready.\n- Model: `{defn.endpoint_name}`\n- Memory: `~/.mneme/{defn.agent_id}.db`",
        author="System",
    ).send()


# ---------------------------------------------------------------------------
# Admin nav actions
# ---------------------------------------------------------------------------

@cl.action_callback("admin_agents")
async def on_admin_agents(action: cl.Action) -> None:
    await show_agent_list()


@cl.action_callback("admin_settings")
async def on_admin_settings(action: cl.Action) -> None:
    await show_settings_menu()


# ---------------------------------------------------------------------------
# Settings update
# ---------------------------------------------------------------------------

@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    cl.user_session.set("settings", settings)

    # Rebuild agent if endpoint changed
    config = cl.user_session.get("config")
    profile = cl.user_session.get("chat_profile", "")
    if not config or profile == ADMIN_PROFILE:
        return

    defn = config.get_agent(profile)
    if not defn:
        return

    new_endpoint = settings.get("endpoint", defn.endpoint_name)
    if new_endpoint != defn.endpoint_name:
        defn.endpoint_name = new_endpoint
        from mad_agents.registry import build_agent
        agent = build_agent(defn, config)
        cl.user_session.set("agent", agent)
        await cl.Message(
            content=f"🔄 Switched to endpoint **{new_endpoint}**",
            author="System",
        ).send()


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

@cl.on_message
async def on_message(message: cl.Message) -> None:
    profile = cl.user_session.get("chat_profile", "")

    if profile == ADMIN_PROFILE:
        await cl.Message(
            content="Use the buttons above to manage agents and settings.",
            author="System",
        ).send()
        return

    agent = cl.user_session.get("agent")
    session: MnemeSession | None = cl.user_session.get("session")
    agent_id: str = cl.user_session.get("agent_id", "default")
    settings: dict = cl.user_session.get("settings", {})

    if not agent or not session:
        await cl.Message(content="⚠️ Session not initialized. Please refresh.").send()
        return

    temperature = float(settings.get("temperature", 0.7))
    max_turns = int(settings.get("max_turns", 10))
    fleet_sync = bool(settings.get("fleet_sync", False))

    ui_hooks = ChainlitRunHooks()
    mneme_hooks = MnemeRunHooks(agent_id=agent_id, session_id=session.session_id)
    fleet_hooks = FleetSyncHooks(agent_id=agent_id, enabled=fleet_sync)
    combined = ComposedHooks([ui_hooks, mneme_hooks, fleet_hooks])

    memory_filter = make_memory_filter(agent_id=agent_id, top_k=5, score_threshold=0.4)
    run_config = RunConfig(
        call_model_input_filter=memory_filter,
        model_settings=ModelSettings(temperature=temperature),
    )

    msg = cl.Message(content="")
    await msg.send()

    try:
        result = Runner.run_streamed(
            agent,
            message.content,
            session=session,
            hooks=combined,
            run_config=run_config,
            max_turns=max_turns,
        )

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                delta = _extract_text_delta(event)
                if delta:
                    await msg.stream_token(delta)
            elif event.type == "run_item_stream_event":
                if event.name in ("handoff_occured", "handoff_occurred"):
                    # Finalize current message and start a new one for the new agent
                    await msg.update()
                    msg = cl.Message(content="")
                    await msg.send()

    except Exception as e:
        log.exception("Runner error")
        msg.content = f"❌ Error: {e}"

    await msg.update()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_text_delta(event) -> str:
    """Pull a text delta string from a RawResponsesStreamEvent."""
    try:
        data = event.data
        # Responses API streaming
        if hasattr(data, "delta") and isinstance(data.delta, str):
            return data.delta
        # Chat completions streaming
        if hasattr(data, "choices"):
            choice = data.choices[0] if data.choices else None
            if choice and hasattr(choice, "delta"):
                return choice.delta.content or ""
    except Exception:
        pass
    return ""
