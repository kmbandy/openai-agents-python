"""
Central configuration for the mad-lab Chainlit UI.

Persisted to ~/.config/mad-lab-agents/config.json.
Two concepts: ModelEndpoint (how to reach an LLM) and AgentDef (agent spec).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_PATH = Path.home() / ".config" / "mad-lab-agents" / "config.json"

_DEFAULT_ENDPOINTS = [
    {
        "name": "Nemotron-9B (GPU)",
        "url": "http://localhost:8091/v1",
        "model": "openai/default",
        "api_key": "none",
    },
    {
        "name": "Bonsai-8B (CPU)",
        "url": "http://localhost:8089/v1",
        "model": "openai/default",
        "api_key": "none",
    },
]

_DEFAULT_AGENTS = [
    {
        "name": "General Assistant",
        "system_prompt": (
            "You are a helpful general-purpose assistant with access to long-term memory. "
            "Search your memory before answering questions about past conversations, projects, "
            "or decisions. Save important facts, decisions, and project context to memory."
        ),
        "endpoint_name": "Nemotron-9B (GPU)",
        "tools": ["memory_search", "memory_write", "memory_graph"],
        "handoffs": [],
        "agent_id": "general_assistant",
    },
    {
        "name": "Code Assistant",
        "system_prompt": (
            "You are a senior software engineer. You write clean, correct code and explain "
            "your reasoning. You have access to a local shell and long-term memory. "
            "Search memory for relevant past decisions before starting new work."
        ),
        "endpoint_name": "Nemotron-9B (GPU)",
        "tools": ["memory_search", "memory_write", "memory_graph", "shell"],
        "handoffs": ["General Assistant"],
        "agent_id": "code_assistant",
    },
]


@dataclass
class ModelEndpoint:
    name: str
    url: str
    model: str = "openai/default"
    api_key: str = "none"


@dataclass
class AgentDef:
    name: str
    system_prompt: str
    endpoint_name: str
    tools: list[str] = field(default_factory=list)
    handoffs: list[str] = field(default_factory=list)
    agent_id: str = ""

    def __post_init__(self) -> None:
        if not self.agent_id:
            self.agent_id = re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")


@dataclass
class Config:
    endpoints: list[ModelEndpoint] = field(default_factory=list)
    agents: list[AgentDef] = field(default_factory=list)

    @classmethod
    def load(cls) -> Config:
        if CONFIG_PATH.exists():
            try:
                raw = json.loads(CONFIG_PATH.read_text())
                endpoints = [ModelEndpoint(**e) for e in raw.get("endpoints", [])]
                agents = [AgentDef(**a) for a in raw.get("agents", [])]
                cfg = cls(endpoints=endpoints, agents=agents)
                cfg._inject_defaults()
                return cfg
            except Exception:
                pass
        cfg = cls()
        cfg._inject_defaults()
        return cfg

    def _inject_defaults(self) -> None:
        existing_ep = {e.name for e in self.endpoints}
        for ep in _DEFAULT_ENDPOINTS:
            if ep["name"] not in existing_ep:
                self.endpoints.append(ModelEndpoint(**ep))

        existing_ag = {a.name for a in self.agents}
        for ag in _DEFAULT_AGENTS:
            if ag["name"] not in existing_ag:
                self.agents.append(AgentDef(**ag))

    def save(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            "endpoints": [asdict(e) for e in self.endpoints],
            "agents": [asdict(a) for a in self.agents],
        }
        CONFIG_PATH.write_text(json.dumps(raw, indent=2))

    def get_endpoint(self, name: str) -> ModelEndpoint:
        for ep in self.endpoints:
            if ep.name == name:
                return ep
        return self.endpoints[0]

    def get_agent(self, name: str) -> AgentDef | None:
        for a in self.agents:
            if a.name == name:
                return a
        return None

    def upsert_agent(self, defn: AgentDef) -> None:
        for i, a in enumerate(self.agents):
            if a.name == defn.name:
                self.agents[i] = defn
                return
        self.agents.append(defn)

    def delete_agent(self, name: str) -> None:
        self.agents = [a for a in self.agents if a.name != name]

    def upsert_endpoint(self, ep: ModelEndpoint) -> None:
        for i, e in enumerate(self.endpoints):
            if e.name == ep.name:
                self.endpoints[i] = ep
                return
        self.endpoints.append(ep)

    def delete_endpoint(self, name: str) -> None:
        self.endpoints = [e for e in self.endpoints if e.name != name]
