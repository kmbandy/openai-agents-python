import asyncio
from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

set_tracing_disabled(disabled=True)

model = LitellmModel(
    model="openai/default",
    base_url="http://localhost:8080/v1",
    api_key="none"
)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=model
)

async def main():
    result = await Runner.run(agent, "Hello! What can you help me with?")
    print(result.final_output)

asyncio.run(main())
