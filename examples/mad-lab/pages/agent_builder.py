"""
Agent builder UI — runs inside the Admin chat profile.

Exposes cl.Action buttons for Create / Edit / Delete agent.
Uses cl.AskUserMessage flows to collect field values interactively.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chainlit as cl

from config import AgentDef, Config

TOOLS_AVAILABLE = ["memory_search", "memory_write", "memory_graph", "shell"]


async def show_agent_list() -> None:
    config = Config.load()
    if not config.agents:
        await cl.Message(content="No agents configured yet.").send()
    else:
        lines = ["**Configured Agents:**\n"]
        for a in config.agents:
            lines.append(f"- **{a.name}** — `{a.endpoint_name}` — tools: {', '.join(a.tools) or 'none'}")
        await cl.Message(content="\n".join(lines)).send()

    actions = [
        cl.Action(name="create_agent", label="➕ Create Agent", payload={}),
    ]
    if config.agents:
        actions += [
            cl.Action(name="edit_agent", label="✏️ Edit Agent", payload={}),
            cl.Action(name="delete_agent", label="🗑️ Delete Agent", payload={}),
        ]
    await cl.Message(content="What would you like to do?", actions=actions).send()


@cl.action_callback("create_agent")
async def on_create_agent(action: cl.Action) -> None:
    await _create_agent_flow()


@cl.action_callback("edit_agent")
async def on_edit_agent(action: cl.Action) -> None:
    config = Config.load()
    names = [a.name for a in config.agents]
    res = await cl.AskUserMessage(
        content=f"Which agent to edit? Options: {', '.join(names)}"
    ).send()
    if not res:
        return
    name = res["output"].strip()
    defn = config.get_agent(name)
    if not defn:
        await cl.Message(content=f"Agent '{name}' not found.").send()
        return
    await _create_agent_flow(existing=defn)


@cl.action_callback("delete_agent")
async def on_delete_agent(action: cl.Action) -> None:
    config = Config.load()
    names = [a.name for a in config.agents]
    res = await cl.AskUserMessage(
        content=f"Which agent to delete? Options: {', '.join(names)}\n\nType the name to confirm deletion."
    ).send()
    if not res:
        return
    name = res["output"].strip()
    if not config.get_agent(name):
        await cl.Message(content=f"Agent '{name}' not found.").send()
        return
    config.delete_agent(name)
    config.save()
    await cl.Message(content=f"✅ Deleted agent **{name}**. Refresh to update the profile list.").send()


async def _create_agent_flow(existing: AgentDef | None = None) -> None:
    config = Config.load()
    verb = "Editing" if existing else "Creating"
    await cl.Message(content=f"**{verb} agent** — answer each prompt (or type 'cancel' to abort).\n").send()

    # Name
    default_name = f" (current: {existing.name})" if existing else ""
    res = await cl.AskUserMessage(content=f"**Agent name**{default_name}:").send()
    if not res or res["output"].strip().lower() == "cancel":
        await cl.Message(content="Cancelled.").send()
        return
    name = res["output"].strip()

    # System prompt
    default_prompt = f"\n\nCurrent prompt:\n```\n{existing.system_prompt}\n```" if existing else ""
    res = await cl.AskUserMessage(
        content=f"**System prompt** (what the agent is and can do){default_prompt}:"
    ).send()
    if not res or res["output"].strip().lower() == "cancel":
        await cl.Message(content="Cancelled.").send()
        return
    system_prompt = res["output"].strip()

    # Endpoint
    ep_names = [e.name for e in config.endpoints]
    default_ep = f" (current: {existing.endpoint_name})" if existing else f" (default: {ep_names[0]})"
    res = await cl.AskUserMessage(
        content=f"**Model endpoint**{default_ep}\nOptions: {', '.join(ep_names)}"
    ).send()
    if not res or res["output"].strip().lower() == "cancel":
        await cl.Message(content="Cancelled.").send()
        return
    ep_input = res["output"].strip()
    endpoint_name = ep_input if ep_input in ep_names else ep_names[0]

    # Tools
    default_tools = ", ".join(existing.tools) if existing else "memory_search, memory_write, memory_graph"
    res = await cl.AskUserMessage(
        content=f"**Tools** (comma-separated, current: {default_tools})\nAvailable: {', '.join(TOOLS_AVAILABLE)}"
    ).send()
    if not res or res["output"].strip().lower() == "cancel":
        await cl.Message(content="Cancelled.").send()
        return
    tools = [t.strip() for t in res["output"].split(",") if t.strip() in TOOLS_AVAILABLE]

    # Handoffs
    agent_names = [a.name for a in config.agents if a.name != name]
    default_handoffs = ", ".join(existing.handoffs) if existing else "none"
    res = await cl.AskUserMessage(
        content=f"**Handoffs** — agents this agent can pass to (current: {default_handoffs})\n"
                f"Available agents: {', '.join(agent_names) or 'none yet'}\n"
                "Enter comma-separated names or 'none':"
    ).send()
    if not res or res["output"].strip().lower() == "cancel":
        await cl.Message(content="Cancelled.").send()
        return
    raw_handoffs = res["output"].strip()
    handoffs = [] if raw_handoffs.lower() == "none" else [
        h.strip() for h in raw_handoffs.split(",") if h.strip() in agent_names
    ]

    defn = AgentDef(
        name=name,
        system_prompt=system_prompt,
        endpoint_name=endpoint_name,
        tools=tools,
        handoffs=handoffs,
    )
    config.upsert_agent(defn)
    config.save()
    await cl.Message(
        content=f"✅ Agent **{name}** saved.\n\n"
                f"- Endpoint: `{endpoint_name}`\n"
                f"- Tools: {', '.join(tools) or 'none'}\n"
                f"- Handoffs: {', '.join(handoffs) or 'none'}\n\n"
                "**Refresh the page** to see the new agent in the profile selector."
    ).send()
