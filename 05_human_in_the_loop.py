import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Human-in-the-Loop""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
async def _():
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.conditions import TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.ui import Console
    from autogen_ext.models.ollama import OllamaChatCompletionClient

    # Create the agents.
    model_client = OllamaChatCompletionClient(model="llama3.2")
    assistant = AssistantAgent("assistant", model_client=model_client)
    user_proxy = UserProxyAgent("user_proxy", input_func=input)

    # Create the termination condition which will end the conversation when the user says 'APPROVE'
    termination = TextMentionTermination("APPROVE")

    # Create the team.
    team = RoundRobinGroupChat(
        [assistant, user_proxy], termination_condition=termination
    )

    # Run the conversation and stream to the console.
    stream = team.run_stream(task="Write a 4-line poem about the ocean.")

    await Console(stream)
    await model_client.close()
    return (
        AssistantAgent,
        Console,
        OllamaChatCompletionClient,
        RoundRobinGroupChat,
        TextMentionTermination,
        UserProxyAgent,
        assistant,
        model_client,
        stream,
        team,
        termination,
        user_proxy,
    )


if __name__ == "__main__":
    app.run()
