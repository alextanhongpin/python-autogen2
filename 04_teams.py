import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Teams

        How to create multi-agent team using AutoGen.

        A team is a group of agents that work together to achieve a common goal.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    ## Creating a Team
    return


@app.cell
def _():
    import asyncio

    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.base import TaskResult
    from autogen_agentchat.conditions import (
        ExternalTermination,
        TextMentionTermination,
    )
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.ui import Console
    from autogen_core import CancellationToken
    from autogen_ext.models.ollama import OllamaChatCompletionClient


    model_client = OllamaChatCompletionClient(model="llama3.2")

    # Create the primary agent.
    primary_agent = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    # Create the critic agent.
    critic_agent = AssistantAgent(
        "critic",
        model_client=model_client,
        system_message='Provide constructive feedback. Respond with "APPROVE" to when your feedbacks are addressed.',
    )

    # Define a termination condition that stops the task if the critic approves.
    text_termination = TextMentionTermination("APPROVE")

    # Create a team with the primary and critic agents.
    team = RoundRobinGroupChat(
        [primary_agent, critic_agent], termination_condition=text_termination
    )
    return (
        AssistantAgent,
        CancellationToken,
        Console,
        ExternalTermination,
        OllamaChatCompletionClient,
        RoundRobinGroupChat,
        TaskResult,
        TextMentionTermination,
        asyncio,
        critic_agent,
        model_client,
        primary_agent,
        team,
        text_termination,
    )


@app.cell
async def _(team):
    result = await team.run(task="Write a short poem about AI.")
    result
    return (result,)


@app.cell
def _(mo):
    mo.md(r"""## Observing a Team""")
    return


@app.cell
async def _(TaskResult, team):
    await team.reset()  # Reset the team for a new task.
    async for message in team.run_stream(
        task="Write a short poem about the fall season."
    ):  # type: ignore
        if isinstance(message, TaskResult):
            print("Stop Reason:", message.stop_reason)
        else:
            print(message)
    return (message,)


@app.cell
async def _(Console, team):
    await team.reset()  # Reset the team for a new task.
    await Console(
        team.run_stream(task="Write a short poem about the fall season.")
    )  # Stream the messages to the console.
    return


@app.cell
def _(mo):
    mo.md(r"""## Stopping a Team""")
    return


@app.cell
async def _(
    Console,
    ExternalTermination,
    RoundRobinGroupChat,
    asyncio,
    critic_agent,
    primary_agent,
    text_termination,
):
    # Create a new team with an external termination condition.
    external_termination = ExternalTermination()
    team2 = RoundRobinGroupChat(
        [primary_agent, critic_agent],
        termination_condition=external_termination
        | text_termination,  # Use the bitwise OR operator to combine conditions.
    )

    # Run the team in a background task.
    run = asyncio.create_task(
        Console(team2.run_stream(task="Write a short poem about the fall season."))
    )

    # Wait for some time.
    await asyncio.sleep(0.1)

    # Stop the team.
    external_termination.set()

    # Wait for the team to finish.
    await run
    return external_termination, run, team2


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
