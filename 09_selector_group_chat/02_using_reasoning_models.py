import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Using Reasoning Models""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from typing import List, Sequence

    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.conditions import (
        MaxMessageTermination,
        TextMentionTermination,
    )
    from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
    from autogen_agentchat.teams import SelectorGroupChat
    from autogen_agentchat.ui import Console
    from autogen_ext.models.ollama import OllamaChatCompletionClient
    return (
        AssistantAgent,
        BaseAgentEvent,
        BaseChatMessage,
        Console,
        List,
        MaxMessageTermination,
        OllamaChatCompletionClient,
        SelectorGroupChat,
        Sequence,
        TextMentionTermination,
        UserProxyAgent,
    )


@app.cell
def _(mo):
    mo.md(r"""## Tools""")
    return


@app.cell
def _():
    # Note: This example uses mock tools instead of real APIs for demonstration purposes
    def search_web_tool(query: str) -> str:
        if "2006-2007" in query:
            return """Here are the total points scored by Miami Heat players in the 2006-2007 season:
            Udonis Haslem: 844 points
            Dwayne Wade: 1397 points
            James Posey: 550 points
            ...
            """
        elif "2007-2008" in query:
            return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2007-2008 is 214."
        elif "2008-2009" in query:
            return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2008-2009 is 398."
        return "No data found."


    def percentage_change_tool(start: float, end: float) -> float:
        return ((end - start) / start) * 100
    return percentage_change_tool, search_web_tool


@app.cell
def _(mo):
    mo.md(r"""## Termination Conditions""")
    return


@app.cell
def _(MaxMessageTermination, TextMentionTermination):
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination
    return max_messages_termination, termination, text_mention_termination


@app.cell
def _(mo):
    mo.md(r"""## Task""")
    return


@app.cell
def _():
    task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"
    return (task,)


@app.cell
def _(
    AssistantAgent,
    OllamaChatCompletionClient,
    SelectorGroupChat,
    UserProxyAgent,
    percentage_change_tool,
    search_web_tool,
    termination,
):
    model_client = OllamaChatCompletionClient(model="llama3.2")

    web_search_agent = AssistantAgent(
        "WebSearchAgent",
        description="An agent for searching information on the web.",
        tools=[search_web_tool],
        model_client=model_client,
        system_message="""Use web search tool to find information.""",
    )

    data_analyst_agent = AssistantAgent(
        "DataAnalystAgent",
        description="An agent for performing calculations.",
        model_client=model_client,
        tools=[percentage_change_tool],
        system_message="""Use tool to perform calculation. If you have not seen the data, ask for it.""",
    )

    user_proxy_agent = UserProxyAgent(
        "UserProxyAgent",
        description="A user to approve or disapprove tasks.",
    )

    selector_prompt = """Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    When the task is complete, let the user approve or disapprove the task.
    """

    team = SelectorGroupChat(
        [web_search_agent, data_analyst_agent, user_proxy_agent],
        model_client=model_client,
        termination_condition=termination,  # Use the same termination condition as before.
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,
    )
    return (
        data_analyst_agent,
        model_client,
        selector_prompt,
        team,
        user_proxy_agent,
        web_search_agent,
    )


@app.cell
async def _(Console, task, team):
    await Console(team.run_stream(task=task))
    return


if __name__ == "__main__":
    app.run()
