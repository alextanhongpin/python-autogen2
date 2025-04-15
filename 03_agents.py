import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Agents
        Demonstrates the usage of Assistant Agent
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
    from autogen_agentchat.messages import StructuredMessage, TextMessage
    from autogen_agentchat.ui import Console
    from autogen_core import CancellationToken
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    return (
        AssistantAgent,
        CancellationToken,
        Console,
        OllamaChatCompletionClient,
        StructuredMessage,
        TextMessage,
    )


@app.cell
def _(AssistantAgent, OllamaChatCompletionClient):
    # Define a tool that searches the web for information.
    async def web_search(query: str) -> str:
        """Find information on the web"""
        return "AutoGen is a programming framework for building multi-agent applications."


    # Create an agent that uses the Ollama llama3.2 model.
    model_client = OllamaChatCompletionClient(model="llama3.2")

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[web_search],
        system_message="Use tools to solve tasks.",
    )
    return agent, model_client, web_search


@app.cell
async def _(CancellationToken, TextMessage, agent):
    async def assistant_run() -> None:
        response = await agent.on_messages(
            [TextMessage(content="Find information on AutoGen", source="user")],
            cancellation_token=CancellationToken(),
        )
        print(response.inner_messages)
        print(response.chat_message)


    await assistant_run()
    return (assistant_run,)


@app.cell
def _(mo):
    mo.md(r"""## Multi-Modal Input""")
    return


@app.cell
def _():
    from io import BytesIO

    import PIL
    import requests
    from autogen_agentchat.messages import MultiModalMessage
    from autogen_core import Image

    # Create a multi-modal message with random image and text.
    pil_image = PIL.Image.open(
        BytesIO(requests.get("https://picsum.photos/300/200").content)
    )
    img = Image(pil_image)
    multi_modal_message = MultiModalMessage(
        content=["Can you describe the content of this image?", img], source="user"
    )
    img
    return (
        BytesIO,
        Image,
        MultiModalMessage,
        PIL,
        img,
        multi_modal_message,
        pil_image,
        requests,
    )


@app.cell
async def _(
    AssistantAgent,
    CancellationToken,
    OllamaChatCompletionClient,
    multi_modal_message,
):
    vision_agent = AssistantAgent(
        name="assistant",
        model_client=OllamaChatCompletionClient(model="llama3.2-vision"),
    )

    # Use asyncio.run(...) when running in a script.
    response = await vision_agent.on_messages(
        [multi_modal_message], CancellationToken()
    )
    print(response.chat_message)
    return response, vision_agent


@app.cell
def _(mo):
    mo.md(r"""## Streaming Messages""")
    return


@app.cell
async def _(CancellationToken, Console, TextMessage, agent):
    async def assistant_run_stream() -> None:
        await Console(
            agent.on_messages_stream(
                [
                    TextMessage(
                        content="Find information on AutoGen", source="user"
                    )
                ],
                cancellation_token=CancellationToken(),
            ),
            output_stats=True,
        )


    # Use asyncio.run(assistant_run_stream()) when running in a script.
    await assistant_run_stream()
    return (assistant_run_stream,)


@app.cell
def _(mo):
    mo.md(r"""## Using Tools""")
    return


@app.cell
def _():
    from autogen_core.tools import FunctionTool


    # Define a tool using a Python function.
    async def web_search_func(query: str) -> str:
        """Find information on the web"""
        return "AutoGen is a programming framework for building multi-agent applications."


    # This step is automatically performed inside the AssistantAgent if the tool is a Python function.
    web_search_function_tool = FunctionTool(
        web_search_func, description="Find information on the web"
    )
    # The schema is provided to the model during AssistantAgent's on_messages call.
    web_search_function_tool.schema
    return FunctionTool, web_search_func, web_search_function_tool


@app.cell
def _(mo):
    mo.md(r"""## Model Context Protocol Tools""")
    return


@app.cell
async def _(AssistantAgent, TextMessage, model_client):
    from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

    # Get the fetch tool from mcp-server-fetch.
    fetch_mcp_server = StdioServerParams(command="uvx", args=["mcp-server-fetch"])
    tools = await mcp_server_tools(fetch_mcp_server)

    # Create an agent that can use the fetch tool.
    mcp_agent = AssistantAgent(
        name="fetcher",
        model_client=model_client,
        tools=tools,
        reflect_on_tool_use=True,
    )  # type: ignore

    # Let the agent fetch the content of a URL and summarize it.
    result = await mcp_agent.run(
        task="Summarize the content of https://en.wikipedia.org/wiki/Seattle"
    )
    assert isinstance(result.messages[-1], TextMessage)
    print(result.messages[-1].content)

    # Close the connection to the model client.
    await model_client.close()
    return (
        StdioServerParams,
        fetch_mcp_server,
        mcp_agent,
        mcp_server_tools,
        result,
        tools,
    )


@app.cell
def _(mo):
    mo.md(r"""## Structured Output""")
    return


@app.cell
async def _(AssistantAgent, Console, model_client):
    from typing import Literal
    from pydantic import BaseModel


    class AgentResponse(BaseModel):
        thoughts: str
        response: Literal["happy", "sad", "neutral"]


    json_agent = AssistantAgent(
        "assistant",
        model_client=model_client,
        system_message="Categorize the input as happy, sad or neutral following the JSON format",
        output_content_type=AgentResponse,
    )

    json_result = await Console(json_agent.run_stream(task="I am happy"))
    print(json_result)

    await model_client.close()
    print("Thought:", json_result.messages[-1].content.thoughts)
    print("Response:", json_result.messages[-1].content.response)
    return AgentResponse, BaseModel, Literal, json_agent, json_result


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
