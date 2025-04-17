import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Custom Memory Stores (Vector DBs, etc.)

        https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/memory.html#custom-memory-stores-vector-dbs-etc
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
async def _():
    import os
    from pathlib import Path

    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.ui import Console
    from autogen_core.memory import MemoryContent, MemoryMimeType
    from autogen_ext.memory.chromadb import (
        ChromaDBVectorMemory,
        PersistentChromaDBVectorMemoryConfig,
    )
    from autogen_ext.models.ollama import OllamaChatCompletionClient

    # Initialize ChromaDB memory with custom config
    chroma_user_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="preferences",
            persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
            k=2,  # Return top  k results
            score_threshold=0.4,  # Minimum similarity score
        )
    )
    # a HttpChromaDBVectorMemoryConfig is also supported for connecting to a remote ChromaDB server

    # Add user preferences to memory
    await chroma_user_memory.add(
        MemoryContent(
            content="The weather should be in metric units",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "preferences", "type": "units"},
        )
    )

    await chroma_user_memory.add(
        MemoryContent(
            content="Meal recipe must be vegan",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "preferences", "type": "dietary"},
        )
    )


    async def get_weather(city: str, units: str = "imperial") -> str:
        if units == "imperial":
            return f"The weather in {city} is 73 °F and Sunny."
        elif units == "metric":
            return f"The weather in {city} is 23 °C and Sunny."
        else:
            return f"Sorry, I don't know the weather in {city}."


    model_client = OllamaChatCompletionClient(
        model="llama3.2",
    )

    # Create assistant agent with ChromaDB memory
    assistant_agent = AssistantAgent(
        name="assistant_agent",
        model_client=model_client,
        tools=[get_weather],
        memory=[chroma_user_memory],
    )

    stream = assistant_agent.run_stream(task="What is the weather in New York?")
    await Console(stream)

    await model_client.close()
    await chroma_user_memory.close()
    return (
        AssistantAgent,
        ChromaDBVectorMemory,
        Console,
        MemoryContent,
        MemoryMimeType,
        OllamaChatCompletionClient,
        Path,
        PersistentChromaDBVectorMemoryConfig,
        assistant_agent,
        chroma_user_memory,
        get_weather,
        model_client,
        os,
        stream,
    )


if __name__ == "__main__":
    app.run()
