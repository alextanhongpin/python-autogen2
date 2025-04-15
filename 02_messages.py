import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Messages""")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""## Agent-Agent Messages""")
    return


@app.cell
def _():
    from autogen_agentchat.messages import TextMessage

    text_message = TextMessage(content="Hello, world!", source="User")
    text_message
    return TextMessage, text_message


@app.cell
def _():
    from io import BytesIO

    import requests
    from autogen_agentchat.messages import MultiModalMessage
    from autogen_core import Image as AGImage
    from PIL import Image

    pil_image = Image.open(
        BytesIO(requests.get("https://picsum.photos/300/200").content)
    )
    img = AGImage(pil_image)
    multi_modal_message = MultiModalMessage(
        content=["Can you describe the content of this image?", img], source="User"
    )
    img
    return (
        AGImage,
        BytesIO,
        Image,
        MultiModalMessage,
        img,
        multi_modal_message,
        pil_image,
        requests,
    )


if __name__ == "__main__":
    app.run()
