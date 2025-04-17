import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        # RAG Agent
        https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/memory.html#rag-agent-putting-it-all-together
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import re
    from typing import List

    import aiofiles
    import aiohttp
    from autogen_core.memory import Memory, MemoryContent, MemoryMimeType


    class SimpleDocumentIndexer:
        """Basic document indexer for AutoGen Memory."""

        def __init__(self, memory: Memory, chunk_size: int = 1500) -> None:
            self.memory = memory
            self.chunk_size = chunk_size

        async def _fetch_content(self, source: str) -> str:
            """Fetch content from URL or file."""
            if source.startswith(("http://", "https://")):
                async with aiohttp.ClientSession() as session:
                    async with session.get(source) as response:
                        return await response.text()
            else:
                async with aiofiles.open(source, "r", encoding="utf-8") as f:
                    return await f.read()

        def _strip_html(self, text: str) -> str:
            """Remove HTML tags and normalize whitespace."""
            text = re.sub(r"<[^>]*>", " ", text)
            text = re.sub(r"\s+", " ", text)
            return text.strip()

        def _split_text(self, text: str) -> List[str]:
            """Split text into fixed-size chunks."""
            chunks: list[str] = []
            # Just split text into fixed-size chunks
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i : i + self.chunk_size]
                chunks.append(chunk.strip())
            return chunks

        async def index_documents(self, sources: List[str]) -> int:
            """Index documents into memory."""
            total_chunks = 0

            for source in sources:
                try:
                    content = await self._fetch_content(source)

                    # Strip HTML if content appears to be HTML
                    if "<" in content and ">" in content:
                        content = self._strip_html(content)

                    chunks = self._split_text(content)

                    for i, chunk in enumerate(chunks):
                        await self.memory.add(
                            MemoryContent(
                                content=chunk,
                                mime_type=MemoryMimeType.TEXT,
                                metadata={"source": source, "chunk_index": i},
                            )
                        )

                    total_chunks += len(chunks)

                except Exception as e:
                    print(f"Error indexing {source}: {str(e)}")

            return total_chunks
    return (
        List,
        Memory,
        MemoryContent,
        MemoryMimeType,
        SimpleDocumentIndexer,
        aiofiles,
        aiohttp,
        re,
    )


@app.cell
async def _(SimpleDocumentIndexer):
    import os
    from pathlib import Path

    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.ui import Console
    from autogen_ext.memory.chromadb import (
        ChromaDBVectorMemory,
        PersistentChromaDBVectorMemoryConfig,
    )

    # Initialize vector memory

    rag_memory = ChromaDBVectorMemory(
        config=PersistentChromaDBVectorMemoryConfig(
            collection_name="autogen_docs",
            persistence_path=os.path.join(str(Path.home()), ".chromadb_autogen"),
            k=3,  # Return top 3 results
            score_threshold=0.4,  # Minimum similarity score
        )
    )

    await rag_memory.clear()  # Clear existing memory


    # Index AutoGen documentation
    async def index_autogen_docs() -> None:
        indexer = SimpleDocumentIndexer(memory=rag_memory)
        sources = [
            "https://raw.githubusercontent.com/microsoft/autogen/main/README.md",
            "https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/agents.html",
            "https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/teams.html",
            "https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/termination.html",
        ]
        chunks: int = await indexer.index_documents(sources)
        print(f"Indexed {chunks} chunks from {len(sources)} AutoGen documents")


    await index_autogen_docs()
    return (
        AssistantAgent,
        ChromaDBVectorMemory,
        Console,
        Path,
        PersistentChromaDBVectorMemoryConfig,
        index_autogen_docs,
        os,
        rag_memory,
    )


@app.cell
async def _(AssistantAgent, Console, rag_memory):
    from autogen_ext.models.ollama import OllamaChatCompletionClient

    # Create our RAG assistant agent
    rag_assistant = AssistantAgent(
        name="rag_assistant",
        model_client=OllamaChatCompletionClient(
            model="llama3.2", options={"num_ctx": 8096}
        ),
        memory=[rag_memory],
    )

    # Ask questions about AutoGen
    stream = rag_assistant.run_stream(task="What is AgentChat?")
    await Console(stream)

    # Remember to close the memory when done
    await rag_memory.close()
    return OllamaChatCompletionClient, rag_assistant, stream


if __name__ == "__main__":
    app.run()
