import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # CountDownAgent

        In this example, we create a simple agent that counts down from a given number to zero, and produces a stream of messages with the current count.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from typing import AsyncGenerator, List, Sequence
    from autogen_agentchat.agents import BaseChatAgent
    from autogen_agentchat.base import Response
    from autogen_agentchat.messages import (
        BaseAgentEvent,
        BaseChatMessage,
        TextMessage,
    )
    from autogen_core import CancellationToken


    class CountDownAgent(BaseChatAgent):
        def __init__(self, name: str, count: int = 3):
            super().__init__(name, "A simple agent that counts down.")
            self._count = count

        @property
        def produced_message_types(self) -> Sequence[type(BaseChatMessage)]:
            return (TextMessage,)

        async def on_messages(
            self,
            messages: Sequence[BaseChatMessage],
            cancellation_token: CancellationToken,
        ) -> Response:
            # Calls the on_messages_stream
            response: Response | None = None
            async for message in self.on_messages_stream(
                messages, cancellation_token
            ):
                if isinstance(message, Response):
                    response = message
            assert response is not None
            return response

        async def on_messages_stream(
            self,
            messages: Sequence[BaseChatMessage],
            cancellation_token: CancellationToken,
        ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
            inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
            for i in range(self._count, 0, -1):
                msg = TextMessage(content=f"{i}...", source=self.name)
                inner_messages.append(msg)
                yield msg

            # The response is returned at the end of the stream.
            # It contains the final message and all the inner messages.
            yield Response(
                chat_message=TextMessage(content="Done!", source=self.name)
            )

        async def on_reset(self, cancellation_token: CancellationToken) -> None:
            pass
    return (
        AsyncGenerator,
        BaseAgentEvent,
        BaseChatAgent,
        BaseChatMessage,
        CancellationToken,
        CountDownAgent,
        List,
        Response,
        Sequence,
        TextMessage,
    )


@app.cell
async def _(CancellationToken, CountDownAgent, Response):
    async def run_countdown_agent() -> None:
        # Create a countdown agent.
        countdown_agent = CountDownAgent("countdown")

        # Run the agnet with a given task and stream the response.
        async for message in countdown_agent.on_messages_stream(
            [], CancellationToken()
        ):
            if isinstance(message, Response):
                print(message.chat_message)
            else:
                print(message)


    await run_countdown_agent()
    return (run_countdown_agent,)


if __name__ == "__main__":
    app.run()
