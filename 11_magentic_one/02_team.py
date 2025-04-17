import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Team

        https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/magentic-one.html
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
    from autogen_agentchat.teams import MagenticOneGroupChat
    from autogen_agentchat.ui import Console
    from autogen_ext.agents.web_surfer import MultimodalWebSurfer
    # from autogen_ext.agents.file_surfer import FileSurfer
    # from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
    # from autogen_agentchat.agents import CodeExecutorAgent
    # from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor


    model_client = OllamaChatCompletionClient(model="llama3.2")

    surfer = MultimodalWebSurfer(
        "WebSurfer",
        model_client=model_client,
    )

    team = MagenticOneGroupChat([surfer], model_client=model_client)
    await Console(team.run_stream(task="What is the UV index in Melbourne today?"))

    # # Note: you can also use  other agents in the team
    # team = MagenticOneGroupChat([surfer, file_surfer, coder, terminal], model_client=model_client)
    # file_surfer = FileSurfer( "FileSurfer",model_client=model_client)
    # coder = MagenticOneCoderAgent("Coder",model_client=model_client)
    # terminal = CodeExecutorAgent("ComputerTerminal",code_executor=LocalCommandLineCodeExecutor())
    return (
        Console,
        MagenticOneGroupChat,
        MultimodalWebSurfer,
        OllamaChatCompletionClient,
        asyncio,
        model_client,
        surfer,
        team,
    )


if __name__ == "__main__":
    app.run()
