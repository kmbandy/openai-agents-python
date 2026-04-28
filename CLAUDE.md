# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a fork of `openai/openai-agents-python` with a custom memory layer (Project Mneme) and a Chainlit UI. The upstream contributor guide lives in `AGENTS.md` — read it for PR, testing, and API compatibility rules. This file covers only the fork-specific additions.

## Commands

```bash
make sync              # install/refresh all deps (uv)
make format            # ruff format + fix
make lint              # ruff check only
make typecheck         # mypy + pyright in parallel
make tests             # full suite (parallel + serial)
uv run pytest -s -k <pattern>   # focused test
make snapshots-fix     # update inline snapshots after intentional changes
make coverage          # run suite + fail if <85% coverage
make serve-docs        # live-preview docs
```

Use `uv run python ...` for all Python invocations — never bare `python3`.

## Running the Chainlit UI

```bash
cd examples/mad-lab
# Use the Python 3.12 venv — Python 3.14 has an anyio incompatibility with static file serving
/home/kmbandy/venvs/agents312/bin/chainlit run chainlit_app.py
```

Requires the local LLM endpoint running at `http://localhost:8080/v1` (llama-server or compatible). The `LitellmModel` wrapper in `src/agents/extensions/models/litellm_model.py` bridges the OpenAI Agents SDK to any LiteLLM-compatible endpoint.

## Architecture: Custom Additions

### Project Mneme — DuckDB-backed semantic memory (`src/agents/memory/mneme/`)

> **Status: commits orphaned.** The three Mneme commits (`64f7e80d`, `421a8f94`, `62a9f776`) are not reachable from `main` — they were lost during an upstream merge. Recover with `git cherry-pick 64f7e80d 421a8f94 62a9f776` before working on Mneme.

Mneme replaces flat `.md` memory files with a DuckDB fact warehouse:

- `store.py` — `write_fact`, `search_facts` (cosine similarity, nomic-embed-text-v1.5 768-dim), `graph_related` (relationship edges)
- `session.py` — `MnemeSession`: drop-in for `SQLiteSession`; stores conversation history in a `session_items` DuckDB table
- `tools.py` — `build_memory_tools(agent_id)`: returns `memory_search`, `memory_write`, `memory_graph` as `@function_tool`, scoped per agent
- `hooks.py` — `MnemeRunHooks`: `RunHooksBase` subclass that auto-writes a `session_summary` fact via `on_agent_end`; `make_memory_filter`: `CallModelInputFilter` that injects relevant facts as a `<memory>` block in the system prompt on every model call
- `runner.py` — `run_with_memory(agent, input, agent_id)`: one-call convenience wrapper that wires `MnemeSession` + memory filter + `MnemeRunHooks`
- `fleet.py` — `FleetStore`: MotherDuck-backed shared KB; `promote_from(agent_id)` copies facts + embeddings + edges into `md:mneme_fleet`; reads `MOTHERDUCK_TOKEN` from env

Per-agent databases live in `~/.mneme/<agent_id>.db`. Fleet DB is `md:mneme_fleet` on MotherDuck.

### Local LLM integration

`src/agents/extensions/models/litellm_model.py` implements the SDK's `Model` protocol on top of LiteLLM. Use it to point any agent at a local llama-server:

```python
from agents.extensions.models.litellm_model import LitellmModel
model = LitellmModel(model="openai/default", base_url="http://localhost:8080/v1", api_key="none")
```

### mad-lab examples (`examples/mad-lab/`)

- `chainlit_app.py` — Chainlit UI stub (minimal; full UI is planned)
- `basic_agent.py` — bare-bones local agent example

## Key upstream internals (non-obvious)

- Adding a new tool type requires coordinated edits across `items.py`, `run_internal/run_steps.py`, `turn_resolution.py`, `tool_execution.py`, `tool_planning.py`, `stream_events.py`, `run_state.py`, and `run_internal/session_persistence.py`. See `AGENTS.md` for the full list.
- When `RunState` schema changes, bump `CURRENT_SCHEMA_VERSION` in `run_state.py` and add an entry to `SCHEMA_VERSION_SUMMARIES`.
- `run.py` is orchestration-only — new runtime logic goes in `run_internal/` modules.
- Input guardrails fire only on the first turn of a fresh run; resuming from `RunState` must not increment the turn counter.
