# Contributing to DLTransClass

Thank you for considering a contribution to DLTransClass. We welcome bug reports, feature requests, documentation improvements, and reproducibility-oriented patches.

## Development setup

```bash
git clone git@github.com:vectorlab5/dltransclass.git
cd dltransclass
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
```

## Style

- Follow the existing module layout under `src/dltransclass/`.
- Lint with `ruff check src/ tests/` and format with `black src/ tests/`.
- Add type hints; the project targets Python 3.10+.
- Public APIs should carry concise docstrings; reference the corresponding equation number from the paper when relevant.

## Tests

```bash
pytest -m "not slow and not gpu"        # fast unit tests
pytest -m "gpu"                         # GPU-only tests
```

New code should be covered by unit tests where feasible. Reproducibility-affecting changes (loss formulation, masking, optimizer defaults) require regression tests.

## Submitting a pull request

1. Fork the repository and create a topic branch.
2. Open a draft pull request early to discuss design choices.
3. Update `CHANGELOG.md` under the `Unreleased` section.
4. Ensure CI is green.

## Scientific reproducibility

If a change can affect reported numbers in the paper, please attach the relevant before/after metric file from `results/`. We will not merge changes that silently alter benchmark numbers.
