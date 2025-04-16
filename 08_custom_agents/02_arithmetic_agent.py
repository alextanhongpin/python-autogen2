import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Arithmetic Agent""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(List):
    from typing import Callable, Sequence

    from autogen_agentchat.agents import BaseChatAgent
    from autogen_agentchat.base import Response
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.messages import BaseChatMessage, TextMessage
    from autogen_agentchat.teams import SelectorGroupChat
    from autogen_agentchat.ui import Console
    from autogen_core import CancellationToken
    from autogen_ext.models.ollama import OllamaChatCompletionClient


    class ArithmeticAgent(BaseChatAgent):
        def __init__(
            self, name: str, description: str, operator_func: Callable[[int], int]
        ) -> None:
            super().__init__(name, description=description)
            self._operator_func = operator_func
            self._message_history: List[BaseChatMessage] = []

        @property
        def produced_message_types(self) -> Sequence[type(BaseChatMessage)]:
            return (TextMessage,)

        async def on_messages(
            self,
            messages: Sequence[BaseChatMessage],
            cancellation_token: CancellationToken,
        ) -> Response:
            self._message_history.extend(messages)
            # Parse the number in the last message.
            assert isinstance(self._message_history[-1], TextMessage)
            number = int(self._message_history[-1].content)

            # Apply the operator function to the number.
            result = self._operator_func(number)

            # Create a new message with the result.
            response_message = TextMessage(content=str(result), source=self.name)

            # Update the message history.
            self._message_history.append(response_message)

            # Return the response.
            return Response(chat_message=response_message)

        async def on_reset(self, cancellation_token: CancellationToken) -> None:
            pass
    return (
        ArithmeticAgent,
        BaseChatAgent,
        BaseChatMessage,
        Callable,
        CancellationToken,
        Console,
        MaxMessageTermination,
        OllamaChatCompletionClient,
        Response,
        SelectorGroupChat,
        Sequence,
        TextMessage,
    )


@app.cell
async def _(
    ArithmeticAgent,
    BaseChatMessage,
    Console,
    List,
    MaxMessageTermination,
    OllamaChatCompletionClient,
    SelectorGroupChat,
    TextMessage,
):
    async def run_number_agents() -> None:
        # Create agents for number operations.
        add_agent = ArithmeticAgent(
            "add_agent", "Adds 1 to the number.", lambda x: x + 1
        )
        multiply_agent = ArithmeticAgent(
            "multiply_agent", "Multiplies the number by 2.", lambda x: x * 2
        )
        subtract_agent = ArithmeticAgent(
            "subtract_agent", "Subtracts 1 from the number.", lambda x: x - 1
        )
        divide_agent = ArithmeticAgent(
            "divide_agent",
            "Divides the number by 2 and rounds down.",
            lambda x: x // 2,
        )
        identity_agent = ArithmeticAgent(
            "identity_agent", "Returns the number as is.", lambda x: x
        )

        # The termination condition is to stop after 10 messages.
        termination_condition = MaxMessageTermination(10)

        # Create a selector group chat.
        selector_group_chat = SelectorGroupChat(
            [
                add_agent,
                multiply_agent,
                subtract_agent,
                divide_agent,
                identity_agent,
            ],
            model_client=OllamaChatCompletionClient(model="llama3.2"),
            termination_condition=termination_condition,
            allow_repeated_speaker=True,  # Allow the same agent to speak multiple times, necessary for this task.
            selector_prompt=(
                "Available roles:\n{roles}\nTheir job descriptions:\n{participants}\n"
                "Current conversation history:\n{history}\n"
                "Please select the most appropriate role for the next message, and only return the role name."
            ),
        )

        # Run the selector group chat with a given task and stream the response.
        task: List[BaseChatMessage] = [
            TextMessage(
                content="Apply the operations to turn the given number into 25.",
                source="user",
            ),
            TextMessage(content="10", source="user"),
        ]
        stream = selector_group_chat.run_stream(task=task)
        await Console(stream)


    # Use asyncio.run(run_number_agents()) when running in a script.
    await run_number_agents()
    return (run_number_agents,)


if __name__ == "__main__":
    app.run()
