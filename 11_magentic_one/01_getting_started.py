import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Magentic One

        https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/magentic-one.html

        ```
        pip install "autogen-agentchat" "autogen-ext[magentic-one,openai]"

        # If using the MultimodalWebSurfer, you also need to install playwright dependencies:
        playwright install --with-deps chromium
        ```
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
async def _():
    import asyncio

    from autogen_ext.models.ollama import OllamaChatCompletionClient
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import MagenticOneGroupChat
    from autogen_agentchat.ui import Console


    model_client = OllamaChatCompletionClient(model="llama3.2")

    assistant = AssistantAgent("Assistant", model_client=model_client)

    team = MagenticOneGroupChat([assistant], model_client=model_client)
    await Console(
        team.run_stream(task="Provide a different proof for Fermat's Last Theorem")
    )
    await model_client.close()
    return (
        AssistantAgent,
        Console,
        MagenticOneGroupChat,
        OllamaChatCompletionClient,
        assistant,
        asyncio,
        model_client,
        team,
    )


if __name__ == "__main__":
    app.run()
