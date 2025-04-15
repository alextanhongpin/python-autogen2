marimo:
	uv run marimo edit

lint:
	#uv add mypy
	#uv run mypy --install-types
	@uvx ruff format
	@uvx ruff check --fix --select I
	@uv run mypy . # uvx runs in separate virtual environment.

