import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Termination

        https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/tutorial/termination.html
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import (
        MaxMessageTermination,
        TextMentionTermination,
    )
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.ui import Console
    from autogen_ext.models.ollama import OllamaChatCompletionClient

    model_client = OllamaChatCompletionClient(
        model="llama3.2",
    )

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
        system_message="Provide constructive feedback for every message. Respond with 'APPROVE' to when your feedbacks are addressed.",
    )
    return (
        AssistantAgent,
        Console,
        MaxMessageTermination,
        OllamaChatCompletionClient,
        RoundRobinGroupChat,
        TextMentionTermination,
        critic_agent,
        model_client,
        primary_agent,
    )


@app.cell
async def _(
    Console,
    MaxMessageTermination,
    RoundRobinGroupChat,
    critic_agent,
    primary_agent,
):
    max_msg_termination = MaxMessageTermination(max_messages=3)
    round_robin_team = RoundRobinGroupChat(
        [primary_agent, critic_agent], termination_condition=max_msg_termination
    )

    # Use asyncio.run(...) if you are running this script as a standalone script.
    await Console(
        round_robin_team.run_stream(
            task="Write a unique, Haiku about the weather in Paris"
        )
    )
    return max_msg_termination, round_robin_team


@app.cell
async def _(Console, round_robin_team):
    await Console(round_robin_team.run_stream())
    return


@app.cell
def _(mo):
    mo.md(r"""## Combining Termination Conditions""")
    return


@app.cell
async def _(
    Console,
    MaxMessageTermination,
    RoundRobinGroupChat,
    TextMentionTermination,
    critic_agent,
    primary_agent,
):
    combined_termination = MaxMessageTermination(
        max_messages=10
    ) | TextMentionTermination("APPROVE")

    # If we want to terminate when both conditions are met.
    # combined_termination = MaxMessageTermination(max_messages=10) & TextMentionTermination("APPROVE")

    round_robin_team2 = RoundRobinGroupChat(
        [primary_agent, critic_agent], termination_condition=combined_termination
    )

    await Console(
        round_robin_team2.run_stream(
            task="Write a unique, Haiku about the weather in Singapore"
        )
    )
    return combined_termination, round_robin_team2


@app.cell
def _(mo):
    mo.md(r"""## Custom Termination Condition""")
    return


@app.cell
def _():
    from typing import Sequence
    from autogen_agentchat.base import TerminatedException, TerminationCondition
    from autogen_agentchat.messages import (
        BaseAgentEvent,
        BaseChatMessage,
        StopMessage,
        ToolCallExecutionEvent,
    )
    from autogen_core import Component
    from pydantic import BaseModel
    from typing_extensions import Self


    class FunctionCallTerminationConfig(BaseModel):
        """Configuration for the termination condition to allow for serialization and deserialization of the component."""

        function_name: str


    class FunctionCallTermination(
        TerminationCondition, Component[FunctionCallTerminationConfig]
    ):
        """Terminate the conversation if a FunctionExecutionResult with a specific name is received."""

        component_config_schema = FunctionCallTerminationConfig
        component_provider_override = (
            "autogen_agentchat.conditions.FunctionCallTermination"
        )
        """The schema for the component configuration."""

        def __init__(self, function_name: str):
            self._terminated = False
            self._function_name = function_name

        @property
        def terminated(self) -> bool:
            return self._terminated

        async def __call__(
            self,
            messages: Sequence[BaseAgentEvent | BaseChatMessage],
        ) -> StopMessage | None:
            if self._terminated:
                raise TerminatedException(
                    "Termination condition has already been reached"
                )
            for message in messages:
                if isinstance(message, ToolCallExecutionEvent):
                    for execution in message.content:
                        if execution.name == self._function_name:
                            self._terminated = True
                            return StopMessage(
                                content=f"Function '{self._function_name}' was executed.",
                                source="FunctionCallTermination",
                            )

            return None

        async def reset(self) -> None:
            self._terminated = False

        def _to_config(self) -> FunctionCallTerminationConfig:
            return FunctionCallTerminationConfig(function_name=self._function_name)

        @classmethod
        def _from_config(cls, config: FunctionCallTerminationConfig) -> Self:
            return cls(function_name=config.function_name)
    return (
        BaseAgentEvent,
        BaseChatMessage,
        BaseModel,
        Component,
        FunctionCallTermination,
        FunctionCallTerminationConfig,
        Self,
        Sequence,
        StopMessage,
        TerminatedException,
        TerminationCondition,
        ToolCallExecutionEvent,
    )


@app.cell
def approve():
    def approve() -> None:
        """Approve the message when all feedbacks have been addressed."""
        pass
    return (approve,)


@app.cell
def _(AssistantAgent, approve, model_client):
    # Create the primary agent.
    primary_agent2 = AssistantAgent(
        "primary",
        model_client=model_client,
        system_message="You are a helpful AI assistant.",
    )

    # Create the critic agent with the approve function as a tool.
    critic_agent2 = AssistantAgent(
        "critic",
        model_client=model_client,
        tools=[approve],  # Register the approve function as a tool.
        system_message="Provide constructive feedback. Use the approve tool to approve when all feedbacks are addressed.",
    )
    return critic_agent2, primary_agent2


@app.cell
async def _(
    Console,
    FunctionCallTermination,
    RoundRobinGroupChat,
    critic_agent2,
    model_client,
    primary_agent2,
):
    function_call_termination = FunctionCallTermination(function_name="approve")
    round_robin_team3 = RoundRobinGroupChat(
        [primary_agent2, critic_agent2],
        termination_condition=function_call_termination,
    )

    # Use asyncio.run(...) if you are running this script as a standalone script.
    await Console(
        round_robin_team3.run_stream(
            task="Write a unique, Haiku about the weather in Paris"
        )
    )
    await model_client.close()
    return function_call_termination, round_robin_team3


if __name__ == "__main__":
    app.run()
