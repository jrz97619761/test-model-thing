# Agent guide for TMT

## Scope and work

TMT is an experimental MLX byte-level language model with recurrent state, recurrent trace units, and latent-space prediction. Treat quality claims as experiment results. Small fixtures check behavior, not model quality.

Before a change, inspect the relevant source, tests, docs, and `git status --short`. Trace inputs, model state, outputs, and failure paths. Preserve user changes. Keep edits small and follow local style. Preserve JSON key order in examples.

Read directly relevant skills before use. Use primary sources for current technical claims and Context7 for current library docs when available. Report unavailable tools and checks you did not run.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/tmt/model.py` | Model arithmetic, RTU traces, loss, optimizer updates, and sample generation |
| `src/tmt/main.py` | Compatibility imports for the model classes |
| `src/tmt/config.py` | Model defaults and partial JSON configuration |
| `src/tmt/cli.py` | CLI parser, help, validation, and command dispatch |
| `src/tmt/checkpoint.py` | Model creation and strict checkpoint I/O |
| `src/tmt/runtime.py` | Train, evaluation, generation, and sweep operations |
| `src/tmt/data.py` | Raw-byte file selection and document reads |
| `src/tmt/records.py` | Manifest fields and JSON output |
| `src/tmt/validation.py` | Shared input, metadata, and array validation |
| `src/tmt/blocks.py` | Block contracts, runtime state, and RTU layers |
| `src/tmt/benchmark.py` | Frozen byte evaluation, fixed benchmark fixtures, and optional CoLA classification |
| `src/tmt/__main__.py` | Entry point for `python -m tmt` |
| `src/tmt/commands.md` | Packaged command manual, also returned by `tmt help` |
| `test/test_model.py` | Model arithmetic, loss, RTU recurrence, gradients, and the RTU baseline (with recorder) |
| `test/test_blocks.py` | Block contracts, state, restore, checkpoints, and schema checks |
| `test/test_benchmark.py` | Byte evaluation, CoLA row checks, benchmark run, and `python -m tmt.benchmark` |
| `test/test_config.py` | Config, init templates, examples, validation order, and checkpoint file checks |
| `test/test_cli.py` | Help, version, argument and value errors, generate, evaluate, and the installed-command smoke test |
| `test/test_train.py` | Train loop, manifests, loss rows, samples, interval evaluation, and sweep |
| `test/conftest.py`, `test/fixtures/` | Shared fixtures and the fixed RTU baseline |
| `examples/` | Default model, tiny model, and sweep grid JSON files |
| `pyproject.toml` | Package metadata, console entry point, and packaged resources |

`README.md` and `VISION.md` contain the author's project description and personal views. Preserve their voice.
Do not rewrite these documents unless the task requires it.

## Environment and checks

Use Python 3.12, 3.13, or 3.14. Model commands and most tests require a functional MLX runtime.
Use a suitable virtual environment if one is available. Otherwise, create one at the repository root:

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
.venv/bin/python -m pytest
.venv/bin/python -m tmt --help
.venv/bin/python -m tmt --version
.venv/bin/python -m pip check
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
- Direct `Model` calls require all model fields. `load_config()` resolves partial CLI configuration.
- If defaults or constructor fields change, check metadata validation, example JSON files, tests, and command documentation.
- Keep imports free of file writes, prompts, command argument checks, or model runs.
- Keep CLI help and version available without an MLX import. Preserve lazy imports in `cli.py`.
- Do not sample in `train_step()`. Keep the sample RNG state unchanged.
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
- Checkpoint loads preserve the caller's Python and MLX RNG state after success or failure. File order uses a separate seeded RNG.
- Treat format changes as compatibility changes. Provide explicit migration behavior when the task requires a new format.
- Resume restores checkpoint settings and optimizer tensors. It starts a new data pass without the previous cursor or RNG state.
- Do not describe resume as exact continuation. The trainer resets recurrent state and traces at each document.
- Relative CLI paths use the current directory. Do not add parent-directory searches without an explicit behavior change.
- Use a temporary workspace for command checks. Keep output paths separate from input paths and previous runs.
- Use tiny fixtures, such as `dim=4`, `layers=1`, `spread=4`, and a few updates, for smoke checks.
- Do not start large train runs or download datasets unless the task requires them.
- Keep datasets, checkpoints, generated samples, and build artifacts out of commits.
- The trainer saves every 500 updates and at normal completion. Ctrl-C requests a stop and a save after a complete update. Do not checkpoint partial updates.
- Checkpoint saves write a temporary file in the destination directory, then atomically replace the destination. Preserve the last successful checkpoint after a failed save.

## Documentation and handoff

- Update `src/tmt/commands.md` when CLI behavior changes. Check examples for affected instructions.
- Keep `commands.md` in package data. Help must work outside the checkout.
- Use plain, concise technical English. Apply the installed `simplified-technical-english` skill to technical prose when available.
- Lead reviews with actionable defects, ordered by severity. Give file and line evidence.
- Report what changed, why it changed, verification, and material limits.
- Do not send external messages, publish results, or perform destructive actions without user authorization.
