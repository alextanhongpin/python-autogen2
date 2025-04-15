import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Saving and Loading Agents""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
async def _():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.ui import Console
    from autogen_core import CancellationToken
    from autogen_ext.models.ollama import OllamaChatCompletionClient

    model_client = OllamaChatCompletionClient(model="llama3.2")

    assistant_agent = AssistantAgent(
        name="assistant_agent",
        system_message="You are a helpful assistant",
        model_client=model_client,
    )

    response = await assistant_agent.on_messages(
        [TextMessage(content="Write a 3 line poem on Singapore", source="user")],
        CancellationToken(),
    )
    print(response.chat_message)
    await model_client.close()
    return (
        AssistantAgent,
        CancellationToken,
        Console,
        MaxMessageTermination,
        OllamaChatCompletionClient,
        RoundRobinGroupChat,
        TextMessage,
        assistant_agent,
        model_client,
        response,
    )


@app.cell
async def _(assistant_agent):
    agent_state = await assistant_agent.save_state()
    print(agent_state)
    return (agent_state,)


@app.cell
async def _(
    AssistantAgent,
    CancellationToken,
    TextMessage,
    agent_state,
    model_client,
):
    new_assistant_agent = AssistantAgent(
        name="assistant_agent",
        system_message="You are a helpful assistant",
        model_client=model_client,
    )
    await new_assistant_agent.load_state(agent_state)

    # Use asyncio.run(...) when running in a script.
    response2 = await new_assistant_agent.on_messages(
        [
            TextMessage(
                content="What was the last line of the previous poem you wrote",
                source="user",
            )
        ],
        CancellationToken(),
    )
    print(response2.chat_message)
    await model_client.close()
    return new_assistant_agent, response2


if __name__ == "__main__":
    app.run()
