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

Keep the [documentation index](docs/README.md), README, architecture and papers in
agreement with the implemented behavior. Link measurements to their exact report;
retain historical results and failed targets as labeled snapshots. Report local
computation, acceptance without review and correctness separately. A teacher can
be a rule, file, API or LLM; teaching a decision head does not fine-tune an LLM.

Run `python scripts/check_documentation.py` from a Git checkout. This offline
check covers local Markdown links/anchors, links to this repository's `main`
branch, and current version labels; CI and release checks run it too. It does not
establish external-link availability or correctness of prose.
Run changed executable examples as appropriate. Documentation-only edits do not
require new classifier evaluations or live API calls.

## Package checks

```bash
uv lock --check
uv build
python scripts/check_distributions.py dist
uvx twine check --strict dist/*
```

CI also installs the wheel into a clean environment and runs the CLI outside the checkout. A release tag must match the version in `pyproject.toml`. PyPI publishing requires the repository's `pypi` environment and trusted-publisher configuration; tagging and publishing are separate release actions.

For a release, update the package/runtime version and current documentation labels,
move the relevant Unreleased changelog entries into the release record, then push
the source and require green CI on that exact commit before tagging. Verify the
publication workflow and installed PyPI artifact afterward. Documentation on GitHub
can precede a package release; the PyPI README changes only when a package is published.
