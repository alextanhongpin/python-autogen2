import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Saving and Loading Teams""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.ui import Console
    from autogen_core import CancellationToken
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    return (
        AssistantAgent,
        CancellationToken,
        Console,
        MaxMessageTermination,
        OllamaChatCompletionClient,
        RoundRobinGroupChat,
        TextMessage,
    )


@app.cell
async def _(
    AssistantAgent,
    Console,
    MaxMessageTermination,
    OllamaChatCompletionClient,
    RoundRobinGroupChat,
):
    model_client = OllamaChatCompletionClient(model="llama3.2")

    # Define a team.
    assistant_agent = AssistantAgent(
        name="assistant_agent",
        system_message="You are a helpful assistant",
        model_client=model_client,
    )
    agent_team = RoundRobinGroupChat(
        [assistant_agent],
        termination_condition=MaxMessageTermination(max_messages=2),
    )

    # Run the team and stream messages to the console.
    stream = agent_team.run_stream(
        task="Write a beautiful poem 3-line about lake tangayika"
    )

    # Use asyncio.run(...) when running in a script.
    await Console(stream)

    # Save the state of the agent team.
    team_state = await agent_team.save_state()
    return agent_team, assistant_agent, model_client, stream, team_state


@app.cell
async def _(Console, agent_team):
    await agent_team.reset()
    stream2 = agent_team.run_stream(
        task="What was the last line of the poem you wrote?"
    )
    await Console(stream2)
    return (stream2,)


@app.cell
async def _(Console, agent_team, team_state):
    print(team_state)

    # Load team state.
    await agent_team.load_state(team_state)
    stream3 = agent_team.run_stream(
        task="What was the last line of the poem you wrote?"
    )
    await Console(stream3)
    return (stream3,)


@app.cell
def _(mo):
    mo.md(r"""## Persisting State (File or Database)""")
    return


@app.cell
async def _(
    Console,
    MaxMessageTermination,
    RoundRobinGroupChat,
    assistant_agent,
    model_client,
    team_state,
):
    import json

    with open("team_state.json", "w") as f:
        json.dump(team_state, f)

    with open("team_state.json", "r") as f:
        team_state2 = json.load(f)


    new_agent_team = RoundRobinGroupChat(
        [assistant_agent],
        termination_condition=MaxMessageTermination(max_messages=2),
    )
    await new_agent_team.load_state(team_state2)
    stream4 = new_agent_team.run_stream(
        task="What was the last line of the poem you wrote?"
    )
    await Console(stream4)
    await model_client.close()
    return f, json, new_agent_team, stream4, team_state2


if __name__ == "__main__":
    app.run()
