import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Selector Group Chat


        https://microsoft.github.io/autogen/stable//user-guide/agentchat-user-guide/selector-group-chat.html

        ##  Example: Web Search/Analysis
        """
    )
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
def _(
    AssistantAgent,
    OllamaChatCompletionClient,
    percentage_change_tool,
    search_web_tool,
):
    model_client = OllamaChatCompletionClient(model="llama3.2")

    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            WebSearchAgent: Searches for information
            DataAnalystAgent: Performs calculations

        You only plan and delegate tasks - you do not execute them yourself.

        When assigning tasks, use this format:
        1. <agent> : <task>

        After all tasks are complete, summarize the findings and end with "TERMINATE".
        """,
    )


    web_search_agent = AssistantAgent(
        "WebSearchAgent",
        description="An agent for searching information on the web.",
        tools=[search_web_tool],
        model_client=model_client,
        system_message="""
        You are a web search agent.
        Your only tool is search_tool - use it to find information.
        You make only one search call at a time.
        Once you have the results, you never do calculations based on them.
        """,
    )

    data_analyst_agent = AssistantAgent(
        "DataAnalystAgent",
        description="An agent for performing calculations.",
        model_client=model_client,
        tools=[percentage_change_tool],
        system_message="""
        You are a data analyst.
        Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
        If you have not seen the data, ask for it.
        """,
    )
    return data_analyst_agent, model_client, planning_agent, web_search_agent


@app.cell
def _(mo):
    mo.md(r"""### Termination Conditions""")
    return


@app.cell
def _(MaxMessageTermination, TextMentionTermination):
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination
    return max_messages_termination, termination, text_mention_termination


@app.cell
def _(mo):
    mo.md(r"""### Selector Prompt""")
    return


@app.cell
def _():
    selector_prompt = """Select an agent to perform task.

    {roles}

    Current conversation context:
    {history}

    Read the above conversation, then select an agent from {participants} to perform the next task.
    Make sure the planner agent has assigned tasks before other agents start working.
    Only select one agent.
    """
    return (selector_prompt,)


@app.cell
def _(
    SelectorGroupChat,
    data_analyst_agent,
    model_client,
    planning_agent,
    selector_prompt,
    termination,
    web_search_agent,
):
    team = SelectorGroupChat(
        [planning_agent, web_search_agent, data_analyst_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,  # Allow an agent to speak multiple turns in a row.
    )
    return (team,)


@app.cell
def _():
    task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"
    return (task,)


@app.cell
async def _(Console, task, team):
    # Use asyncio.run(...) if you are running this in a script.
    await Console(team.run_stream(task=task))
    return


@app.cell
def _(mo):
    mo.md(r"""### User Feedback""")
    return


@app.cell
async def _(
    BaseAgentEvent,
    BaseChatMessage,
    Console,
    SelectorGroupChat,
    Sequence,
    UserProxyAgent,
    data_analyst_agent,
    model_client,
    planning_agent,
    selector_prompt,
    task,
    team,
    termination,
    web_search_agent,
):
    user_proxy_agent = UserProxyAgent(
        "UserProxyAgent",
        description="A proxy for the user to approve or disapprove tasks.",
    )


    def selector_func_with_user_proxy(
        messages: Sequence[BaseAgentEvent | BaseChatMessage],
    ) -> str | None:
        if (
            messages[-1].source != planning_agent.name
            and messages[-1].source != user_proxy_agent.name
        ):
            # Planning agent should be the first to engage when given a new task, or check progress.
            return planning_agent.name
        if messages[-1].source == planning_agent.name:
            if (
                messages[-2].source == user_proxy_agent.name
                and "APPROVE" in messages[-1].content.upper()
            ):  # type: ignore
                # User has approved the plan, proceed to the next agent.
                return None
            # Use the user proxy agent to get the user's approval to proceed.
            return user_proxy_agent.name
        if messages[-1].source == user_proxy_agent.name:
            # If the user does not approve, return to the planning agent.
            if "APPROVE" not in messages[-1].content.upper():  # type: ignore
                return planning_agent.name
        return None


    # Reset the previous agents and run the chat again with the user proxy agent and selector function.
    await team.reset()
    team2 = SelectorGroupChat(
        [planning_agent, web_search_agent, data_analyst_agent, user_proxy_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        selector_func=selector_func_with_user_proxy,
        allow_repeated_speaker=True,
    )

    await Console(team2.run_stream(task=task))
    return selector_func_with_user_proxy, team2, user_proxy_agent


if __name__ == "__main__":
    app.run()
