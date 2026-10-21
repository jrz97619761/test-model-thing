# Agent guide for TMT

TMT is an experimental byte-level language model that uses MLX.
The repository contains the model, a CLI, experiment records, and tests.
Small fixtures check numerical behavior and command workflows. They do not establish model quality.

## Source map

All package paths below are relative to `src/tmt/`.

| File | Responsibility |
| --- | --- |
| `model.py` | `Encoder`, `Decoder`, `Model`, objective loss, optimizer updates, and byte samples |
| `blocks.py` | `Block`, the `TMTv0Block`, block registry, recurrent state, and gradient traces |
| `config.py` | `MODEL_DEFAULTS` and partial JSON configuration through `load_config()` |
| `validation.py` | Input checks, checkpoint metadata and tensor checks, and block contracts |
| `checkpoint.py` | Seeded model creation and Safetensors save/load |
| `data.py` | `ByteFiles`, file selection, readability checks, and raw-byte document reads |
| `runtime.py` | `TrainingRun`, train loops, samples, checkpoint evaluation and inspection, generation, and sweeps |
| `records.py` | Typed manifest records, JSON output, and JSONL output |
| `benchmark.py` | Frozen byte evaluation, fixed byte fixtures, and optional CoLA classification |
| `cli.py`, `__main__.py` | Argument parsing, command dispatch, and `python -m tmt` |
| `commands.md` | Packaged command manual, also returned by `tmt help` |

`pyproject.toml` defines the package, dependencies, Python versions, console script, and package data.
`examples/` contains model configurations, a sweep grid, and a run manifest example.
The model import is `from tmt.model import Model`. There is no `tmt.main` compatibility module.

## Work on this repository

- Inspect `git status --short` before edits. Preserve uncommitted changes.
- Read the affected source and tests before a change. Trace inputs, recurrent state, outputs, and failure paths.
- Keep changes within the requested scope. Use the local code style.
- Keep data helpers in `data.py`, as its module documentation requests.
- Resolve discrepancies against the implementation and tests. Do not use this guide as proof that behavior is correct.
- Update this guide when a change affects its paths or contracts.

`README.md` and `VISION.md` contain the author's prose and views.
Preserve their voice. Edit them only when the task requires it.

## Setup and checks

The package declares Python `>=3.12,<3.15` and an MLX dependency.
Model operations need a functional MLX runtime.
Reuse a suitable virtual environment when one is available.
Otherwise, use Python 3.12, 3.13, or 3.14 to create one:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

The editable installation supports imports from the `src/` layout and CLI subprocess tests.
The `test` extra supplies pytest. Use the selected environment for all checks.
Type checking and linting are optional local checks. Contributors choose their own tools.

For code changes, run these commands from the repository root:

```sh
python -m pytest
python -m tmt --help
python -m tmt --version
python -m pip check
```

- Add regression tests for meaningful behavior changes and relevant failure paths.
- For Python compatibility changes, run these checks under Python 3.12, 3.13, and 3.14 when available.
- For package changes, verify installed commands and the packaged manual outside the checkout.
- For documentation-only changes, verify paths, commands, and claims against the source. Model runs are unnecessary.
- Report checks that fail or cannot run. Do not replace MLX or weaken tests to obtain a pass.

The tests are in `test/`:

| File | Coverage |
| --- | --- |
| `test_model.py` | Loss, gradients, recurrence, import behavior, and the fixed TMT-v0 baseline |
| `test_blocks.py` | Block contracts, state, gradient hooks, features, and checkpoint round trips |
| `test_benchmark.py` | Byte scores, CoLA input and output, and benchmark entry points |
| `test_config.py` | Configuration, templates, checkpoint validation, atomic saves, and RNG preservation |
| `test_cli.py` | Help, errors, byte files, generation, evaluation, interrupts, and installed commands |
| `test_train.py` | Train updates, resume, manifests, samples, evaluation intervals, and sweeps |

`test/conftest.py` supplies shared fixtures.
Use `run_subprocess` for child Python scripts, module commands, and installed-command checks.
Tests import helpers from the module that defines them. `tmt.cli` exports CLI entry points only.
`test/fixtures/tmt-v0-baseline.json` records seeded numerical results.
Do not regenerate the baseline only to hide a regression. Review any intentional baseline change.

## Model and evaluation contracts

- Preserve raw-byte documents and output. The encoder has 256 byte embeddings, and the decoder returns 256 byte logits.
- Keep defaults in `MODEL_DEFAULTS`. Direct `Model` construction requires all constructor arguments.
- Preserve config validation, examples, and checkpoint metadata when model fields change.
- Keep imports free of prompts, model runs, and file writes.
- Preserve help and version commands without MLX imports. Inspect the imports in CLI dependencies as well as `cli.py`.
- Keep `train_step()` free of samples and sample RNG changes. `Model.__call__()` also samples, so these methods are not interchangeable.
- Preserve TMT-v0 gradient contributions and the distinction between recurrent state and gradient traces.
- A frozen step advances recurrent state without optimizer updates or trace updates.
- Preserve state and trace restoration in `benchmark.evaluate()` and `runtime._training_sample()` after success or failure.
- Review MLX evaluation boundaries, `stop_gradient`, and compiled optimizer state when arithmetic changes.

`benchmark.py` defines `BYTE_FIXTURE`, `BYTE_DOCUMENTS`, and `CONTEXT_WINDOWS` for the fixed byte suite.
CoLA uses a separate classifier head while the model weights stay frozen.
The Python result uses `byte_scores` and `epoch_scores`, plus `feature_source` when CoLA runs.
The `--output` file adds `checkpoint`, `optimizer_step`, and `seed` after the result fields. The `evaluate` result ends with `checkpoint` and `optimizer_step`.
The CoLA MCC values use a scale of 100.

Keep objective loss, frozen bits per byte, and CoLA MCC distinct in reports.
Record configuration, seeds, data selection, and budgets for experiments.
Identify any fixture change that prevents direct score comparisons.

## Checkpoints and train runs

`checkpoint.py` saves model parameters, block runtime arrays, and optimizer tensors.
The Safetensors metadata contains a `tmt` JSON string with `version: 1` and the resolved model configuration.
The loader requires the current metadata fields and exact tensor names, shapes, and dtypes.

- Preserve strict checkpoint validation. Provide explicit migration behavior when a requested change alters the format.
- Preserve the caller's Python and MLX RNG state across checkpoint loads after success or failure.
- Preserve the temporary-file save and atomic replacement in the destination directory.
- Preserve the previous checkpoint when a save fails.

Resume restores checkpoint settings and tensors, then starts a new data pass.
It does not restore the previous file cursor or RNG state.
Resume appends to `loss.jsonl`, `samples.jsonl`, and `evaluation.jsonl`, and renames the earlier `run.json` to `run.replaced-<UTC timestamp>.json`. A fresh run starts empty files.
The CLI refuses a fresh run at an existing explicit checkpoint path.
The trainer resets recurrent state and traces for each nonempty document.
File order uses a separate seeded Python RNG.

`TrainingRun.after_update()` controls loss reports, terminal-byte consumption, samples, evaluation, and periodic saves at each `save_every` interval.
The trainer also saves at normal completion. It rewrites `run.json` after each save, and records `error` when an exception fails the run.
On the main thread, its SIGINT and SIGTERM handler records an interrupt request.
The trainer checks the request at complete update boundaries, after the update actions.
It saves before it raises `KeyboardInterrupt` for that request.
Worker threads do not install that handler.
`cli.console_main()` is the entry point for the console script and `python -m tmt`.
It prints `tmt: interrupted` and lets the `KeyboardInterrupt` end the process through SIGINT.
`cli.main()` raises `KeyboardInterrupt` to its caller.

- Preserve update boundaries and action order when the train loop changes.
- Resolve relative CLI paths from the current directory.
- Use temporary workspaces and tiny configurations for command checks.
- Keep check outputs separate from datasets and previous runs.
- Do not start large train runs or download datasets unless the task requires them.
- Keep datasets, checkpoints, samples, and build artifacts out of commits.

## Documentation and handoff

- Update `src/tmt/commands.md` when command behavior changes.
- A new command needs one `COMMANDS` entry in `cli.py` and one `## name` section in `src/tmt/commands.md`. `test_every_command_has_handler_and_manual_section` checks both.
- Keep the manual in package data. Verify affected examples against the implementation.
- Preserve intentional JSON field order in examples and manifest builders.
- Use concise technical English. Apply the installed `simplified-technical-english` skill when available.
- Report what changed, which checks ran, and any material limits.
