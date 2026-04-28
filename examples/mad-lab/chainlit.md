# mad-lab Agent UI

Chat with your local AI agents, backed by semantic memory (Mneme / DuckDB).

**Profiles**
- Select an agent from the dropdown at the top
- Each agent has its own memory store in `~/.mneme/<agent_id>.db`
- The **⚙ Admin** profile lets you create agents and manage model endpoints

**Settings** (gear icon)
- Switch model endpoints live without restarting
- Adjust temperature and max turns per session
- Toggle MotherDuck fleet sync on/off

**Memory**
- Agents automatically search past facts before each response
- Important facts, decisions, and project context are saved to long-term memory
- Fleet sync promotes key facts to the shared MotherDuck knowledge base
