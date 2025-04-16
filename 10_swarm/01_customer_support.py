import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Swarm
        https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/swarm.html
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from typing import Any, Dict, List

    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import (
        HandoffTermination,
        TextMentionTermination,
    )
    from autogen_agentchat.messages import HandoffMessage
    from autogen_agentchat.teams import Swarm
    from autogen_agentchat.ui import Console
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    return (
        Any,
        AssistantAgent,
        Console,
        Dict,
        HandoffMessage,
        HandoffTermination,
        List,
        OllamaChatCompletionClient,
        Swarm,
        TextMentionTermination,
    )


@app.cell
def _(mo):
    mo.md(r"""## Tools""")
    return


@app.cell
def refund_flight():
    def refund_flight(flight_id: str) -> str:
        """Refund a flight"""
        return f"Flight {flight_id} refunded"
    return (refund_flight,)


@app.cell
def _(mo):
    mo.md(r"""## Agents""")
    return


@app.cell
def _(AssistantAgent, OllamaChatCompletionClient, refund_flight):
    model_client = OllamaChatCompletionClient(
        model="llama3.2",
    )

    travel_agent = AssistantAgent(
        "travel_agent",
        model_client=model_client,
        handoffs=["flights_refunder", "user"],
        system_message="""You are a travel agent.
        The flights_refunder is in charge of refunding flights.
        If you need information from the user, you must first send your message, then you can handoff to the user.
        Use TERMINATE when the travel planning is complete.""",
    )

    flights_refunder = AssistantAgent(
        "flights_refunder",
        model_client=model_client,
        handoffs=["travel_agent", "user"],
        tools=[refund_flight],
        system_message="""You are an agent specialized in refunding flights.
        You only need flight reference numbers to refund a flight.
        You have the ability to refund a flight using the refund_flight tool.
        If you need information from the user, you must first send your message, then you can handoff to the user.
        When the transaction is complete, handoff to the travel agent to finalize.""",
    )
    return flights_refunder, model_client, travel_agent


@app.cell
def _(
    HandoffTermination,
    Swarm,
    TextMentionTermination,
    flights_refunder,
    travel_agent,
):
    termination = HandoffTermination(target="user") | TextMentionTermination(
        "TERMINATE"
    )
    team = Swarm(
        [travel_agent, flights_refunder], termination_condition=termination
    )
    return team, termination


@app.cell
async def _(Console, HandoffMessage, mo, model_client, team):
    mo.stop(True)

    task = "I need to refund my flight."


    async def run_team_stream() -> None:
        task_result = await Console(team.run_stream(task=task))
        last_message = task_result.messages[-1]

        while (
            isinstance(last_message, HandoffMessage)
            and last_message.target == "user"
        ):
            user_message = input("User: ")

            task_result = await Console(
                team.run_stream(
                    task=HandoffMessage(
                        source="user",
                        target=last_message.source,
                        content=user_message,
                    )
                )
            )
            last_message = task_result.messages[-1]


    # Use asyncio.run(...) if you are running this in a script.
    await run_team_stream()
    await model_client.close()
    return run_team_stream, task


if __name__ == "__main__":
    app.run()
