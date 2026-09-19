# Contributing

Report reproducible bugs and feature requests through the [issue tracker](https://github.com/steph4n-gh/system1/issues). For vulnerabilities, follow [SECURITY.md](SECURITY.md).

## Set up

Use Python 3.11 or newer:

```bash
git clone https://github.com/steph4n-gh/system1.git
cd system1
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,langchain]'
python -m pytest -q
```

With uv, use `uv sync --locked --extra dev --extra langchain` and `uv run --no-sync pytest -q`. LangChain integration tests use the real callback dispatcher. MLX and emulator coverage requires optional extras and suitable hardware/assets; skipped optional tests are reported by pytest.

## Changes

Keep changes focused and prefer the simplest implementation that addresses a reproduced problem. Add regression coverage for behavior changes, preserve `system1`/`reflex` API parity, and update runnable documentation when the API changes. Do not include private keys, tokens, ledger data, or proprietary ROMs.

Include the problem, resulting behavior, and relevant validation in pull requests. Benchmark claims should include the command, source revision, environment, dataset, overall quality metrics, and raw results. Simulations and cloud measurements must be clearly distinguished.

## Package checks

```bash
uv lock --check
uv build
uvx twine check --strict dist/*
```

CI also installs the wheel into a clean environment and runs the CLI outside the checkout. A release tag must match the version in `pyproject.toml`. PyPI publishing requires the repository's `pypi` environment and trusted-publisher configuration; tagging and publishing are separate release actions.
