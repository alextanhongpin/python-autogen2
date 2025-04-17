import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Memory and RAG

        https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/memory.html
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## ListMemory Example""")
    return


@app.cell
def _():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.ui import Console
    from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    return (
        AssistantAgent,
        Console,
        ListMemory,
        MemoryContent,
        MemoryMimeType,
        OllamaChatCompletionClient,
    )


@app.cell
async def _(
    AssistantAgent,
    ListMemory,
    MemoryContent,
    MemoryMimeType,
    OllamaChatCompletionClient,
):
    # Initialize user memory.
    user_memory = ListMemory()

    # Add user preferences to memory.
    await user_memory.add(
        MemoryContent(
            content="The weather should be in metric units",
            mime_type=MemoryMimeType.TEXT,
        )
    )

    await user_memory.add(
        MemoryContent(
            content="Meal recipe must be vegan", mime_type=MemoryMimeType.TEXT
        )
    )


    async def get_weather(city: str, units: str = "imperial") -> str:
        if units == "imperial":
            return f"The weather in {city} is 73 °F and Sunny."
        elif units == "metric":
            return f"The weather in {city} is 23 °C and Sunny."
        else:
            return f"Sorry, I don't know the weather in {city}."


    assistant_agent = AssistantAgent(
        name="assistant_agent",
        model_client=OllamaChatCompletionClient(
            model="llama3.2",
        ),
        tools=[get_weather],
        memory=[user_memory],
    )
    return assistant_agent, get_weather, user_memory


@app.cell
async def _(Console, assistant_agent):
    # Run the agent with a task.
    stream = assistant_agent.run_stream(task="What is the weather in New York?")
    await Console(stream)
    return (stream,)


@app.cell
async def _(assistant_agent):
    await assistant_agent._model_context.get_messages()
    return


@app.cell
async def _(Console, assistant_agent):
    stream2 = assistant_agent.run_stream(task="Write brief meal recipe with broth")
    await Console(stream2)
    return (stream2,)


if __name__ == "__main__":
    app.run()
