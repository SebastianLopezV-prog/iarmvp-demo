# Developer: getting started

## Install for development

- Install the global tools: `uv` and `pre-commit` (`brew install uv pre-commit` or
  `pipx install uv pre-commit`).
- Clone the repo.
- Set up the git hooks: `pre-commit install`.
- Generate the lockfile (first time only): `uv lock`.
- Install dependencies: `uv sync --group dev`.
- Run the tests: `uv run pytest`.

All dependencies are public PyPI packages; there is no private index and no API key.

## Run the app

```bash
uv run streamlit run app/dashboard.py
# or the console entry point (prints how to launch):
uv run run
```

## Dependencies

- `pyproject.toml` declares dependencies; `uv.lock` pins exact versions (commit it).
- Update: `uv lock --upgrade` then `uv sync --group dev`.
- `requirements.txt` is kept as a pinned mirror for the Streamlit Community Cloud deploy
  (that host installs from requirements.txt). Keep it roughly in step with pyproject.

## Code quality

```bash
uv run ruff check .     # lint
uv run ruff format .    # format
pre-commit run --all-files
```

## Tests

The suite is hermetic: `tests/conftest.py` disables network sockets by default
(`pytest-socket`). The demo is synthetic and offline, so no test needs the network.
