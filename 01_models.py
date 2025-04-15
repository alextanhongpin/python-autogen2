import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Models

        Example on using ollama models with autogen.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
async def _():
    from autogen_core.models import UserMessage
    from autogen_ext.models.ollama import OllamaChatCompletionClient


    ollama_model_client = OllamaChatCompletionClient(model="llama3.2")
    response = await ollama_model_client.create(
        [UserMessage(content="What is the capital of Singapore?", source="user")]
    )
    response
    return (
        OllamaChatCompletionClient,
        UserMessage,
        ollama_model_client,
        response,
    )


@app.cell
async def _(ollama_model_client):
    await ollama_model_client.close()
    return


if __name__ == "__main__":
    app.run()
