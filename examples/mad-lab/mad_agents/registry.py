"""
Build live Agent objects from AgentDef config.

Two-pass construction handles circular handoff references:
  pass 1 — create placeholder Agent shells
  pass 2 — fill in tools and handoffs now that all names exist
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Ensure src/agents and examples/mad-lab/ are importable
_here = Path(__file__).resolve().parent.parent  # examples/mad-lab/
_repo_src = _here.parents[1] / "src"
for _p in (_repo_src, _here):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from agents import Agent, handoff
from agents.extensions.models.litellm_model import LitellmModel
from agents.memory.mneme.tools import build_memory_tools
from agents.tool import LocalShellTool

from config import AgentDef, Config, ModelEndpoint


def _make_model(endpoint: ModelEndpoint) -> LitellmModel:
    return LitellmModel(
        model=endpoint.model,
        base_url=endpoint.url,
        api_key=endpoint.api_key,
    )


def _make_tools(defn: AgentDef) -> list[Any]:
    tools: list[Any] = []
    mem_names = {"memory_search", "memory_write", "memory_graph"}
    if mem_names & set(defn.tools):
        mem_map = {t.name: t for t in build_memory_tools(agent_id=defn.agent_id)}
        for name in defn.tools:
            if name in mem_map:
                tools.append(mem_map[name])
    if "shell" in defn.tools:
        tools.append(LocalShellTool())
    return tools


def build_all_agents(config: Config) -> dict[str, Agent]:
    """Build all agents, resolving handoff cross-references by name."""
    # Pass 1: shells for handoff resolution
    shells: dict[str, Agent] = {}
    for defn in config.agents:
        ep = config.get_endpoint(defn.endpoint_name)
        shells[defn.name] = Agent(
            name=defn.name,
            instructions=defn.system_prompt,
            model=_make_model(ep),
        )
    # Pass 2: full agents with tools + handoffs
    result: dict[str, Agent] = {}
    for defn in config.agents:
        ep = config.get_endpoint(defn.endpoint_name)
        result[defn.name] = Agent(
            name=defn.name,
            instructions=defn.system_prompt,
            model=_make_model(ep),
            tools=_make_tools(defn),
            handoffs=[handoff(shells[n]) for n in defn.handoffs if n in shells],
        )
    return result


def build_agent(defn: AgentDef, config: Config) -> Agent:
    """Build a single agent. Handoffs reference lightweight shells of other agents."""
    shells: dict[str, Agent] = {}
    for other in config.agents:
        if other.name != defn.name:
            ep = config.get_endpoint(other.endpoint_name)
            shells[other.name] = Agent(
                name=other.name,
                instructions=other.system_prompt,
                model=_make_model(ep),
            )
    ep = config.get_endpoint(defn.endpoint_name)
    return Agent(
        name=defn.name,
        instructions=defn.system_prompt,
        model=_make_model(ep),
        tools=_make_tools(defn),
        handoffs=[handoff(shells[n]) for n in defn.handoffs if n in shells],
    )
