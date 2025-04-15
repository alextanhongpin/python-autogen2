marimo:
	uv run marimo edit

lint:
	#uv add mypy
	#uv run mypy --install-types
	@uvx ruff format
	@uvx ruff check --fix --select I
	@uv run mypy . # uvx runs in separate virtual environment.


studio:
	uv run autogenstudio ui --port 8080 --appdir ./myapp

# Use ollama model in autogenstudio
# http://localhost:11434/v1
