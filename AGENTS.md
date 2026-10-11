# Agent guide for TMT

## Scope and project

These instructions apply to the repository. Follow more specific `AGENTS.md` files within their directories.

TMT is an experimental byte-level language model built with MLX. It uses recurrent state, recurrent trace units, and latent-space prediction.
Treat model quality claims as experimental results. Small fixtures verify behavior, not model quality.

## Start here

- Read the relevant source, tests, and documentation before each change.
- Check `git status --short`. Preserve changes that belong to the user.
- Trace inputs, model state, outputs, and failure paths before a fix.
- Keep changes small. Follow local code style without unrelated changes.
- Keep related JSON keys adjacent, and preserve their order when you edit examples.
- Read directly relevant installed skills before use. Use any skill that the user names.
- Use primary sources for current or niche technical claims. Use Context7 for library documentation when available.
- State when a required tool or skill is unavailable. Do not claim checks that you did not perform.
- Delegate independent work when it materially helps. Give each agent a bounded task and avoid concurrent edits to the same files.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/tmt/main.py` | Model arithmetic, RTU traces, loss, optimizer updates, and sampling |
| `src/tmt/config.py` | Model configuration defaults |
| `src/tmt/cli.py` | CLI, configuration, checkpoint I/O, train, sweep, and generation |
| `src/tmt/validation.py` | Shared input, metadata, and array validation |
| `src/tmt/blocks.py` | Block contracts, runtime state, and RTU layers |
| `src/tmt/benchmark.py` | Frozen byte evaluation, fixed benchmark fixtures, and optional CoLA classification |
| `src/tmt/__main__.py` | Entry point for `python -m tmt` |
| `src/tmt/commands.md` | Packaged command manual, also returned by `tmt help` |
| `test/test_main.py` | Model arithmetic, gradient, recurrence, and state tests |
| `test/test_blocks.py` | Block contracts, gradients, state, and integration tests |
| `test/test_evaluation.py`, `test/test_cola_validation.py` | Byte evaluation and CoLA benchmark tests |
| `test/test_validation.py` | Shared validator imports and error order |
| `test/cli/` | CLI commands, input checks, config, checkpoint I/O, train, telemetry, generation, and sweep tests |
| `test/conftest.py`, `test/fixtures/` | Shared model fixtures and the fixed RTU baseline |
| `test/cli/conftest.py` | CLI subprocess, workspace, and manifest fixtures |
| `test/README.md` | Test layout and focused pytest commands |
| `examples/` | Default model, tiny model, and sweep grid JSON files |
| `pyproject.toml` | Package metadata, console entry point, and packaged resources |

`README.md` and `VISION.md` contain the author's project description and personal views. Preserve their voice.
Do not rewrite these documents unless the task requires it.

## Environment and checks

Use Python 3.12, 3.13, or 3.14. Model commands and most tests require a functional MLX runtime.
Use an existing suitable virtual environment when available. Otherwise, create one at the repository root:

```sh
# Use python3.12, python3.13, or python3.14.
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

The source uses the `src/` layout. Install the package before checks.
An editable installation makes source edits available to the CLI and its subprocess tests.
After a regular installation, install again before you check source changes.

Run these checks from the repository root for code changes:

```sh
python -m pytest
python -m tmt --help
python -m tmt --version
```

- Use `pytest` for tests. Keep test dependencies in the `test` extra.
- For Python compatibility changes, run the checks in Python 3.12, 3.13, and 3.14 environments.
- Run `python -m pip check` in each environment to check dependency conflicts.
- Add regression tests for meaningful behavior changes. Cover relevant failure paths.
- For documentation-only edits, check paths, commands, and claims against the source. Model runs are unnecessary.
- For package changes, verify installed entry points and access to `commands.md` outside the checkout.
- If MLX cannot run, report the limitation. Do not replace MLX or weaken tests to obtain a pass.
- Report the checks that ran and any checks that remain incomplete.

## Model and evaluation contracts

- Preserve raw-byte input and output. The encoder and decoder use 256 byte values.
- Keep `MODEL_DEFAULTS` in `src/tmt/config.py` as the source for configuration defaults.
- Direct `Model` calls require every setting. `load_config()` resolves partial CLI configuration.
- If defaults or constructor fields change, check metadata validation, example JSON files, tests, and command documentation.
- Keep imports free of file writes, prompts, argument parsing, or model runs.
- Keep CLI help and version available without an MLX import. Preserve lazy imports in `cli.py`.
- Keep `train_step()` free of sampling. It must not consume sampling RNG state.
- Preserve RTU gradient contributions and state resets at document boundaries.
- A frozen step advances recurrent state. It must not change weights, optimizer state, or RTU traces.
- Byte evaluation must restore recurrent state and RTU traces after success or failure.
- Check MLX lazy evaluation, `stop_gradient`, and compiled optimizer state when model arithmetic changes.
- Preserve the fixed byte benchmark documents and context windows for comparable scores.
- If a task changes benchmark fixtures, change their version identifier and explain the loss of comparability.
- Keep objective loss, frozen bits per byte, and CoLA MCC separate in reports.
- Record seeds, configuration, data selection, and budgets for experiments. Distinguish observations from estimates.

## Checkpoints and workspaces

- Preserve strict checkpoint validation for metadata, tensor names, shapes, and dtypes.
- Checkpoints contain model and optimizer tensors with version-1 model metadata.
- Treat format changes as compatibility changes. Provide explicit migration behavior when the task requires a new format.
- Resume restores checkpoint settings and optimizer tensors. It starts a new data pass without the previous cursor or RNG state.
- Do not describe resume as exact continuation. The trainer resets recurrent state and traces at each document.
- Relative CLI paths use the current directory. Do not add parent-directory searches without an explicit behavior change.
- Use a temporary workspace for command checks. Keep output paths separate from input paths and previous runs.
- Use tiny fixtures, such as `dim=4`, `layers=1`, `spread=4`, and a few updates, for smoke checks.
- Do not start large training runs or download datasets unless the task requires them.
- Keep datasets, checkpoints, generated samples, and build artifacts out of commits.
- The trainer saves every 500 updates and at normal completion. Ctrl-C does not save a checkpoint.
- Direct checkpoint writes can leave partial files after a process failure. Preserve the last successful checkpoint.

## Documentation and handoff

- Update `src/tmt/commands.md` when CLI behavior changes. Check examples for affected instructions.
- Keep `commands.md` in package data. Help must work outside the checkout.
- Use plain, concise technical English. Apply the installed `simplified-technical-english` skill to technical prose when available.
- Lead reviews with actionable defects, ordered by severity. Give file and line evidence.
- Report what changed, why it changed, verification, and material limits.
- Do not send external messages, publish results, or perform destructive actions without user authorization.
