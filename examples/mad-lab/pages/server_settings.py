"""
Server settings UI — runs inside the Admin chat profile.

Manages model endpoints and triggers MotherDuck fleet sync.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chainlit as cl

from config import Config, ModelEndpoint


async def show_settings_menu() -> None:
    config = Config.load()

    ep_lines = ["**Model Endpoints:**\n"]
    for ep in config.endpoints:
        ep_lines.append(f"- **{ep.name}** — `{ep.url}` — model: `{ep.model}`")
    await cl.Message(content="\n".join(ep_lines)).send()

    actions = [
        cl.Action(name="add_endpoint", label="➕ Add Endpoint", payload={}),
        cl.Action(name="delete_endpoint", label="🗑️ Delete Endpoint", payload={}),
        cl.Action(name="sync_fleet", label="☁️ Sync All to Fleet", payload={}),
        cl.Action(name="fleet_stats", label="📊 Fleet Stats", payload={}),
    ]
    await cl.Message(content="Server settings:", actions=actions).send()


@cl.action_callback("add_endpoint")
async def on_add_endpoint(action: cl.Action) -> None:
    res = await cl.AskUserMessage(content="**Endpoint name** (e.g. 'Llama-3 Local'):").send()
    if not res or res["output"].strip().lower() == "cancel":
        return

    name = res["output"].strip()

    res = await cl.AskUserMessage(content="**Base URL** (e.g. http://localhost:8080/v1):").send()
    if not res:
        return
    url = res["output"].strip()

    res = await cl.AskUserMessage(content="**Model string** (default: openai/default):").send()
    if not res:
        return
    model = res["output"].strip() or "openai/default"

    res = await cl.AskUserMessage(content="**API key** (default: none):").send()
    if not res:
        return
    api_key = res["output"].strip() or "none"

    config = Config.load()
    config.upsert_endpoint(ModelEndpoint(name=name, url=url, model=model, api_key=api_key))
    config.save()
    await cl.Message(content=f"✅ Endpoint **{name}** saved.").send()


@cl.action_callback("delete_endpoint")
async def on_delete_endpoint(action: cl.Action) -> None:
    config = Config.load()
    names = [e.name for e in config.endpoints]
    res = await cl.AskUserMessage(
        content=f"Which endpoint to delete? Options: {', '.join(names)}"
    ).send()
    if not res:
        return
    name = res["output"].strip()
    config.delete_endpoint(name)
    config.save()
    await cl.Message(content=f"✅ Endpoint **{name}** deleted.").send()


@cl.action_callback("sync_fleet")
async def on_sync_fleet(action: cl.Action) -> None:
    await cl.Message(content="☁️ Syncing all agents to MotherDuck fleet...").send()
    msg = cl.Message(content="")
    await msg.send()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _sync_all_sync)
        msg.content = f"✅ Fleet sync complete:\n{result}"
    except Exception as e:
        msg.content = f"❌ Fleet sync failed: {e}"
    await msg.update()


def _sync_all_sync() -> str:
    from agents.memory.mneme.fleet import FleetStore
    with FleetStore() as fleet:
        fleet.promote_all()
        stats = fleet.stats()
    lines = [f"Total facts: {stats.get('total', 0)}"]
    for agent_id, count in stats.get("by_agent", {}).items():
        lines.append(f"  {agent_id}: {count}")
    return "\n".join(lines)


@cl.action_callback("fleet_stats")
async def on_fleet_stats(action: cl.Action) -> None:
    msg = cl.Message(content="Fetching fleet stats...")
    await msg.send()
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _stats_sync)
        msg.content = result
    except Exception as e:
        msg.content = f"❌ Could not fetch stats: {e}"
    await msg.update()


def _stats_sync() -> str:
    from agents.memory.mneme.fleet import FleetStore
    with FleetStore() as fleet:
        stats = fleet.stats()
    total = stats.get("total", 0)
    lines = [f"**Fleet KB — {total} total facts**\n"]
    for agent_id, count in stats.get("by_agent", {}).items():
        lines.append(f"- `{agent_id}`: {count} facts")
    for fact_type, count in stats.get("by_type", {}).items():
        lines.append(f"- type `{fact_type}`: {count}")
    return "\n".join(lines)
