# TMT command manual
Use Python 3.12, 3.13, or 3.14. Install TMT with pip. Most commands need MLX. Help and version requests do not.

File errors and invalid values print a short error to stderr and exit with code `1`. Argument errors exit with code `2`.

`--seed`, `--sample-seed`, and sweep grid seed values must be integers from `0` to `18446744073709551615`, inclusive. Boolean grid seed values are invalid. The CLI checks these limits before model work or run folder creation, even when a seed has no effect.

Direct Python seed values use the same limits. `tmt.cli.train()`, `tmt.cli.sweep()`, `tmt.cli.generate()`, `tmt.benchmark.run()`, `tmt.checkpoint.new_model()`, and `tmt.checkpoint.load_checkpoint()` require a primary seed. Pass `seed=...` to `tmt.benchmark.run()`. The separate `sample_seed` argument remains optional and defaults to `11` for `train()` and `sweep()`.

Train, sweep, generate, and benchmark require `--seed N`. The seed sets Python's `random` seed and MLX's random seed when TMT creates or loads a model. Train and sweep use it to set file order.

A sweep requires its base seed even when the grid has a `seed` field. Each grid seed sets one candidate's seed. `--sample-seed` sets the sample seed. Evaluate loads with seed `0` and does not sample.

`--sample-prompt` must be nonempty. `--sample-bytes` and `--eval-max-bytes` must be positive. These checks apply when sample or interval evaluation options are off.

Relative paths use the current directory. The CLI does not search parent folders for files or runs. Data globs select raw files. The CLI sorts paths that match. Each file is one document, and the model resets at each document boundary. The CLI reads file contents as bytes, adds no separators, and does not decode them as text.

The model config accepts `dim`, `layers`, `spread`, `temp`, `rate`, `bound`, and `block`. Defaults are `512`, `16`, `32`, `0.75`, `0.0005`, `[40000, 120000]`, and `"rtu"`. The package registers only the `rtu` block. The optimizer uses `rate` as the weight update rate. `temp` is a unitless sampler value, and `bound` uses optimizer updates. The full TMT objective combines variance, latent-space prediction, next-byte cross entropy, and stop loss.

`tmt.config.MODEL_DEFAULTS` supplies these defaults. `load_config()` resolves partial JSON configurations.
Direct Python calls to `Model` require all seven settings. For example:

```python
from tmt.cli import load_config
from tmt.main import Model

settings = load_config()
settings.update(dim=4, layers=1, spread=4)
model = Model(**settings)
```

`dim`, `layers`, and `spread` must be positive integers. `temp` must be a finite nonnegative number. `rate` must be a finite positive number. `bound` must be a JSON array of two integers with `0 <= start < end`.

`block` must name a registered block. Boolean values are invalid for numeric fields. The CLI checks config values and sweep grid values before model creation or output creation.

## Wheel distribution

The project uses Setuptools and supports Python 3.12, 3.13, and 3.14. The `build` tool is for maintainers. It is not a runtime dependency.

From the repository root, use an active virtual environment with a supported Python version. Install the build tool:

```bash
python -m pip install build
```

Build a wheel:

```bash
python -m build --wheel
```

The build creates `dist/test_model_thing-1.0.0-py3-none-any.whl` for the current project version. It uses an isolated build environment and does not install TMT or publish files.

Install that wheel in the target virtual environment:

```bash
python -m pip install dist/test_model_thing-1.0.0-py3-none-any.whl
tmt --version
tmt help
```

Use the wheel path that matches the project version. From another directory, supply its full path. Pip installs the declared dependencies and creates the `tmt` command. The wheel contains the Python package and `commands.md`. MLX needs a compatible runtime.

For local development, use `python -m pip install -e '.[test]'`. This installs pytest and uses the source files directly. You can also install the package with `python -m pip install .`.

## Blocks

The `block` field selects one factory for every layer. The current package registers only `rtu`.
Use `"block": "rtu"` for the default block, or replace `rtu` with a factory name from `src/tmt/blocks.py`.
A custom block needs a local factory entry in `src/tmt/blocks.py`.

The Python method `Model.step()` returns `((x, state_results, auxiliary_results), (logits, stop))`.
`state_results` contains one dictionary per block, with keys from `state_fields`.
`auxiliary_results` contains one block-specific value per block.
This internal return structure does not change CLI output.

Each factory creates one fresh block per layer. The block call uses `enc`, `x`, and `probe` inputs.
It returns `(x_next, next_state, aux)`. `x_next` has shape `[dim]` and keeps the model dtype.
`next_state` is a dictionary with exactly the field names in `state_fields`.

`Block` exposes `state_fields`, `trace_fields`, `temporal_gradient`, and `feature_source` as properties without setters.
The allowed `temporal_gradient` values are `one-step-autodiff` and `rtu-v1`.
The allowed `feature_source` values are `final-output` and `recurrent-state`.
Override these getters in a custom block. For example:

```python
@property
def state_fields(self):
    return ('cache',)
```

These properties describe the block contract. Instance assignment and deletion raise `AttributeError`.

`state_fields` lists public arrays that advance on each byte. `trace_fields` lists public arrays for temporal gradient history.
Blocks create these fields before optimizer setup and freeze them as runtime state.
`Block.__init__()` does not inspect subclass arrays. `validate()` checks the completed block after its constructor returns.

All block parameters stay in the normal MLX parameter tree.
Block state shapes stay fixed for the life of the model.

`reset()` clears state and traces at each document boundary. `probe()` returns an auxiliary value for differentiation.
The default probe is an unused scalar. RTU returns a zero vector with shape `[dim]`.
The `adjust_gradients()` hook receives the byte, candidate state, auxiliary data, probe cotangent, and local gradients.
The hook returns `(grads, encoder_addition, traces)`.

The default hook returns autodiff gradients unchanged, no encoder gradient addition, and no trace updates.

An empty trace dictionary leaves traces unchanged. A trace update must contain every name in `trace_fields`.

The RTU hook uses label `rtu-v1`. It adds the prior temporal trace contributions for block decay and the shared encoder.
The default label is `one-step-autodiff`. It covers gradients within one byte step.
It does not propagate gradients through detached byte history.

The forward call returns candidate state.
It does not change persistent arrays.
The model commits candidate state after differentiation, or after a frozen step.
Frozen steps advance only `state_fields`. They do not change traces, weights, or optimizer state.

The helper names are `snapshot_state()`, `restore_state()`, `commit_state()`, and `eval_state()`.
Blocks use these helpers to copy, restore, commit, and evaluate declared fields. `Model` calls `validate()` after it creates each block.
Direct calls to `restore_state()`, `commit_state()`, or `validate_result()` build the runtime schema on first use.
The model applies snapshot, restore, and evaluation across all blocks.
Evaluation and samples restore state after success or error.

`features(x)` returns one `[dim]` vector for CoLA. The `feature_source` property reports `final-output` when the block uses `x`.
RTU uses its recurrent state and reports `recurrent-state`.
The CoLA result keeps the JSON key `feature_convention` for compatibility.

Resolved model config and checkpoint metadata both contain `block`.
The checkpoint loader requires every current config field and the `block` selector.
An earlier checkpoint without this selector fails validation, even if its version marker is `1`.
The loader does not add a default for missing metadata. Create fresh artifacts with this release.

Use `tmt.checkpoint.load_checkpoint(path, seed)` to load a checkpoint in Python. It returns `(model, settings)`. Both `tmt.checkpoint` and `tmt.cli` export `load_checkpoint`. Both modules keep `load_model(path, seed)` as a compatibility name.

## init

Create model and sweep templates in the current folder, or write one model config at a chosen path.

### Syntax
```text
tmt init [--output PATH]
```

### Requirements and options

- `--output PATH` is optional. It writes only model JSON to `PATH`. Without this option, `init` prepares the current folder.
- The destination folder must allow file and directory creation. A JSON destination that exists raises `FileExistsError` and keeps its contents.
- `init` reads `MODEL_DEFAULTS` from `tmt.config` and needs MLX. It does not create a model, checkpoint, or run record.

### Workflow

1. Go to the folder for the workspace.
2. Run `tmt init`.
3. Edit `model.json` to set small model values for a first run.
4. Put one or more raw-byte files in `data/train/`.

```bash
tmt init --output configs/model.json
```

### Behavior and files
Bare `tmt init` creates `model.json`, `grid.json`, and `run.example.json`. It also creates `data/train/`, `data/dev/`, and `runs/` if needed. `model.json` contains the configuration defaults, with `"block": "rtu"`. Edit it before a real experiment. The run template sets `sample` and `evaluation` to `null`.

`grid.json` has `dim=[4,8]`, `layers=[1]`, `spread=[4]`, `block=["rtu"]`, and `seed=[11,22]`. `run.example.json` shows the manifest fields with an empty file list. It has null values for `started_at`, `trainable_parameters`, and `temporal_gradient`. It is a template, not a run record. The command opens each JSON destination in exclusive-create mode. If a destination exists, Python raises `FileExistsError`, and the old file stays intact.

The `--output` form creates parent folders as needed. It writes only the model config and does not create workspace folders.

## train

Train a model on raw-byte documents and save a strict model and optimizer checkpoint.

### Syntax
```text
tmt train [CHECKPOINT] --data GLOB --seed N [--config PATH | --resume] [--run NAME] [--updates N] [--log-every N] [--ce-only] [--sample-every N] [--sample-prompt TEXT] [--sample-seed N] [--sample-bytes N] [--eval-data GLOB --eval-every N] [--eval-max-bytes N]
```

### Requirements and options

- `CHECKPOINT` is an optional checkpoint path for a fresh run. With `--resume`, it is required and names the checkpoint to load.
- `--data GLOB` is required. Quote the glob so the CLI, not the shell, expands it.
- `--config PATH` supplies model JSON values for a fresh run. Missing values use model defaults. The default path is `model.json` in the current folder. This option cannot be used with `--resume`.
- `--resume` loads model and optimizer tensors from `CHECKPOINT`. It cannot be used with `--run` or `--config`.
- `--run NAME` creates `runs/NAME/`. Use one relative folder name without path separators. Empty names, `.` and `..` are invalid. Do not combine it with `CHECKPOINT` or `--resume`. Use `CHECKPOINT` for an arbitrary path.
- `--updates N` sets target-byte optimizer updates. The default is `1000`. Use a positive value. On resume, this count is added to the saved optimizer step.
- `--log-every N` sets the progress interval in updates during a train run. The default is `100`. Use a positive value.
- `--seed N` sets the Python and MLX seeds. This option is required.
- `--ce-only` uses next-byte cross entropy. Without it, the command uses the full TMT objective.
- `--sample-every N` writes a sample after each N completed updates. The command writes no samples by default. Use a positive value.
- `--sample-prompt TEXT` sets the nonempty sample prompt. The default is `The `.
- `--sample-seed N` sets the sample seed. The default is `11`.
- `--sample-bytes N` sets the number of output bytes. The default is `256`. Use a positive value.
- `--eval-data GLOB` selects held-out raw-byte files for interval evaluation. Quote the glob.
- `--eval-every N` sets the evaluation interval in updates. Use a positive value. Set this option with `--eval-data`.
- `--eval-max-bytes N` sets the global input-byte budget for each evaluation. The default is `8192` bytes. Use a positive value.
- Get MLX, a valid model config, and at least one file that matches `GLOB` for a fresh run.

### Workflow

1. Run `tmt init` in the workspace.
2. Edit `model.json`, then add raw-byte files under `data/train/`.
3. Start a named run with the example command.
4. To add updates, pass that path and `--resume`.

```bash
tmt train --data 'data/train/*' --run first --updates 3 --seed 11
tmt train runs/first/model.safetensors --data 'data/train/*' --resume --updates 2 --seed 11
```

### Behavior and files
The CLI sorts matched paths and shuffles them once from `--seed`. It repeats that order until it reaches `--updates`. Each adjacent byte pair gives one target-byte update. The model resets block state and temporal traces at each file boundary. A file with fewer than two bytes gives no update. It consumes the terminal byte without an update.

A named run writes `runs/NAME/model.safetensors`. Without `CHECKPOINT` or `--run`, the CLI creates a UTC timestamp folder with microseconds under `runs/`. A named or timestamp run needs a new folder. The command checks train and evaluation selections before folder creation. Unmatched globs and unreadable files cause an input error. Correct the input and retry with the same run name.

A named run stores its manifest at `runs/NAME/manifest/run.json`. A timestamp run stores it at `runs/<TIMESTAMP>/manifest/run.json`. An explicit checkpoint stores it at `manifest/run.json` under its parent folder.

| Field | Meaning |
| --- | --- |
| `config` | Complete model settings for this invocation, with `block`. Resume uses checkpoint settings. |
| `trainable_parameters` | Count of trainable tensor elements before a full-model freeze. This count excludes runtime state and traces. |
| `temporal_gradient` | Ordered list of temporal-gradient labels, one for each block in the layer stack. |
| `seed` | Python and MLX seed for this invocation. |
| `data_glob` | The train glob as supplied. Relative globs use the invocation directory. |
| `data_files` | All matched absolute paths, in the shuffled order for each data pass. |
| `updates` | Requested target-byte optimizer updates for this invocation. Resume adds this budget to the saved optimizer step. |
| `log_every` | Update interval for progress output and loss rows. |
| `objective` | `tmt` for the full TMT loss, or `ce_only` for next-byte cross entropy. |
| `completed_updates` | Zero at start. Completed updates in this invocation at exit, after success, failure, or interrupt. |
| `resume` | `true` for resume, or `false` for a fresh run. |
| `initial_optimizer_step` | The saved optimizer step at resume start, or zero for a fresh run. |
| `sample` | Sample settings, or `null` when sampling is disabled. |
| `evaluation` | Evaluation glob, selected files, interval, and byte budget, or `null` when interval evaluation is disabled. |
| `started_at` | Invocation start time in UTC. |
| `status` | `running`, `complete`, `interrupted`, or `failed`. |
| `last_checkpoint` | Absolute path of the last successful checkpoint, or null before a fresh run saves. |

The file list records selection and pass order. The update budget can stop a pass before the trainer reads every file. Paths do not prove that file contents stay the same. The completed update count includes work after the last checkpoint. It does not imply that the trainer saved all completed updates.

The command writes the manifest at start and exit. It writes progress loss rows to `manifest/loss.jsonl` beside the checkpoint.
Each line is a JSON object with `completed_updates`, `optimizer_step`, and `objective`.
`loss` is the scalar from the latest update. `mean_loss` is the mean loss over the interval.

`mean_loss_components` contains interval means for `cross_entropy`, `latent_prediction`, `variance`, and `stop`.
The row also contains `interval_updates`, `elapsed_seconds`, and `updates_per_second`.
`elapsed_seconds` counts time from the start of the train loop.
`updates_per_second` uses time since the last report, with time for reports, samples, and evaluation from that interval.

The command writes a row at each `--log-every` interval. After at least one update, it writes a final row at exit.
It writes the row only if the last update was outside the interval.
This also applies after failure or interrupt. The command does not duplicate an interval row when the run ends on that interval.

Each invocation replaces `loss.jsonl`. This also applies to a resume.

The checkpoint contains model and optimizer tensors. The loader checks metadata, tensor names, shapes, and types before restore.

At fresh-run start, `last_checkpoint` is null. Resume starts with the input checkpoint path and replaces the invocation manifest. Each successful save sets `last_checkpoint` to the absolute path. Resume adds the requested optimizer updates and starts a new data pass. It does not restore a data cursor or random-number state.

The command prints the run folder, seed, config, and successful checkpoint saves.
At each progress interval, it prints completed updates, total optimizer step, objective, mean loss, latest loss, and update speed.
It always prints the final loss.
It saves every 500 updates and at the end. It does not save on interrupt. The last successful checkpoint remains on disk.

When `--sample-every` is set, the command samples from frozen weights at each interval.
It preserves block state, temporal traces, and global random state for the train run.
It uses `--sample-prompt`, `--sample-seed`, and `--sample-bytes` for each sample.
The command prints an escaped preview and writes raw bytes to `samples/<UTC timestamp>/step-NNNNNNNNNNNN.bin`.
The command uses the same sample seed for each sample.

It writes one JSON object per sample to `manifest/samples.jsonl`.
Each row has `completed_updates`, `optimizer_step`, `prompt`, `seed`, `bytes`, and `path`.
Each invocation resets this index, even when sample output is off.
The command keeps old sample files.

When `--eval-data` and `--eval-every` are set, the command scores held-out files at each update interval.
It also scores the model after a successful run if the final update is outside an interval.
The scorer uses frozen weights and restores block state and temporal traces.
It writes one JSON object per score to `manifest/evaluation.jsonl`.
Each row has `completed_updates`, `optimizer_step`, and the normal evaluation fields: `input_bytes`, `targets`, `bpb`, `common_targets`, `common_bpb`, and `windows`.

Each invocation resets `evaluation.jsonl`, even when interval evaluation is off.

## sweep

Run each combination in a JSON grid in sequence, then score each trained model on development bytes.

### Syntax
```text
tmt sweep --grid PATH --data GLOB --development GLOB --output DIR --seed N [--config PATH] [--updates N] [--log-every N] [--ce-only] [--max-bytes N] [--sample-every N] [--sample-prompt TEXT] [--sample-seed N] [--sample-bytes N] [--eval-data GLOB --eval-every N] [--eval-max-bytes N]
```

### Requirements and options

- `--grid PATH` is required. The JSON root must be an object. Each model field or `seed` must map to a nonempty array of values. The CLI checks the grid before it creates the output folder.
- `--data GLOB` is required and selects raw-byte train files. Quote the glob.
- `--development GLOB` is required and selects raw-byte score files. Quote the glob.
- `--output DIR` is required. The folder must be new or empty. The command rejects a nonempty folder without changes.
- `--config PATH` selects the base model JSON. The default is `model.json` in the current folder.
- `--updates N` sets target-byte updates per candidate. The default is `1000`. Use a positive value.
- `--log-every N` sets the progress interval in updates for each candidate. The default is `100`. Use a positive value.
- `--seed N` sets the base seed when the grid has no `seed` field. This option is required even when the grid has a `seed` field.
- `--ce-only` selects next-byte cross entropy. The default objective is full TMT.
- `--max-bytes N` limits development input to a global prefix across sorted files. The default is `8192` bytes. Use a positive value. Zero and negative values cause an input error.
- `--sample-every N` writes a sample after each N completed updates for each candidate. The command writes no samples by default. Use a positive value.
- `--sample-prompt TEXT` sets the nonempty sample prompt. The default is `The `.
- `--sample-seed N` sets the sample seed. The default is `11`.
- `--sample-bytes N` sets the number of output bytes. The default is `256`. Use a positive value.
- `--eval-data GLOB` selects held-out files for interval evaluation. Quote the glob.
- `--eval-every N` sets the evaluation interval in updates. Use a positive value. Set this option with `--eval-data`.
- `--eval-max-bytes N` sets the global input-byte budget for each evaluation. The default is `8192` bytes. Use a positive value.
- A sweep needs MLX, the grid and data files, and the base config file.

### Workflow

1. Run `tmt init` and edit `model.json`.
2. Edit `grid.json` to list values for the fields to compare.
3. Put train and development files in separate folders.
4. Read `results.jsonl` and compare each candidate's development BPB.

```bash
tmt sweep --grid grid.json --data 'data/train/*' --development 'data/dev/*' --output runs/sweep --updates 1000 --seed 11
```

### Behavior and files
The CLI visits value combinations in the key and value order from the JSON file. It trains one candidate at a time. A grid `seed` value selects that candidate's seed. Other grid values replace fields in the base model config. Each candidate uses a `run-NNNN/` folder under `DIR`. The folder contains `model.safetensors` and a `manifest/` folder.

Each candidate config is `run-NNNN/manifest/model.json`. Its run record is `run-NNNN/manifest/run.json`, with the same manifest fields as a standalone train invocation. Its loss rows are in `run-NNNN/manifest/loss.jsonl`. Sample and interval-evaluation files also stay in each candidate folder. The command appends one JSON result per candidate to `results.jsonl`. Each row records `run`, `settings`, `trainable_parameters`, `temporal_gradient`, `seed`, `updates`, `objective`, `data_glob`, `development_glob`, `development_max_bytes`, `development_targets`, `development_bpb`, and `checkpoint`.

The package registers only `rtu`. To compare it with a custom block, add a factory to `src/tmt/blocks.py` first.
Then add the block names and paired seeds to the grid. Keep the base config, data, objective, optimizer settings, and update budget fixed.

```json
{
  "block": ["rtu", "candidate-v1"],
  "seed": [11, 22, 33]
}
```

```bash
tmt sweep --config research.json --grid ab-grid.json \
  --data 'data/train/*' --development 'data/dev/*' \
  --output runs/ab-001 --updates 1000 --seed 11
```

Pair result rows by seed and compare every score. A shared seed repeats the RNG start.
It does not guarantee equal initial values for differently shaped blocks.
Equal updates do not mean equal parameter count, compute, or state memory.
Record the source revision and fixed factory choices with the results.

Interval evaluation uses `--eval-data` and `--eval-every`. It is separate from the development score at the end of each candidate. The development score still selects the best candidate.

At the end, the command prints the row with the lowest development BPB. The first candidate wins a tie. The command does not write a separate best-model file.

Use a new or empty output folder for each sweep. The command rejects a nonempty folder to preserve checkpoints and result rows.

The command checks train, evaluation, and development selections before output creation. An input error leaves no new output folder. Correct the input and retry with the same output path.

After execution starts, `sweep.json` records the settings, grid, data selections, status, current candidate, and completed result count. It also records the error type and message after a failure or interrupt. Completed rows remain in `results.jsonl`. To retry after an execution failure, use a new output path. The command does not resume a partial sweep.

## evaluate

Measure next-byte prediction on raw-byte documents with frozen model weights.

### Syntax
```text
tmt evaluate CHECKPOINT --data GLOB [--max-bytes N] [--windows N [N ...]] [--output PATH]
```

### Requirements and options

- `CHECKPOINT` is required. It must contain valid model and optimizer tensors.
- `--data GLOB` is required. Quote the glob so the CLI expands it.
- `--max-bytes N` sets the global input-byte limit across sorted files. The default is `8192` bytes. Use a positive value. Zero and negative values cause an input error.
- `--windows N [N ...]` selects context lengths in input bytes. The default is no context-window scores. Every value must be positive.
- `--output PATH` writes the score object as JSON. Without this option, scores go to the terminal.
- Evaluation needs MLX and at least one file that matches `GLOB`. At least one document in the selected input prefix must contain two bytes.
- Empty and one-byte documents provide no evaluation targets.

### Workflow

1. Select a checkpoint from a completed train or sweep run.
2. Put held-out raw-byte files in a separate folder.
3. Run evaluation with a quoted file glob.
4. Add `--windows` to compare limited context lengths.

```bash
tmt evaluate runs/first/model.safetensors --data 'data/dev/*' --windows 8 32 128 --output runs/first/evaluation.json
```

### Behavior and files
The command reads a global prefix from sorted files and keeps each file as a separate document. The first byte provides context for the next-byte score. The scorer consumes the terminal byte without a target score. The full score uses all earlier bytes to predict the next byte. A window score uses only its selected context length. The report includes input bytes, target count, full BPB, common-target count, common BPB, and each requested window score.

Common scores use target offsets at or above the largest requested window. If no common targets remain, common BPB and window deltas are null. The scorer snapshots and restores block state and temporal traces after success or error. It does not update weights or draw random samples. The optional JSON file contains the printed score values. The command does not alter the checkpoint.

## benchmark

Run the fixed byte benchmark suite. Optionally train a small CoLA classification head and report its score.

### Syntax
```text
tmt benchmark CHECKPOINT --seed N [--cola-data PATH] [--epochs N] [--split FRACTION]
python -m tmt.benchmark CHECKPOINT --seed N [--cola-data PATH] [--epochs N] [--split FRACTION]
```

The module command uses the same CLI as `tmt benchmark`. Both commands return this section for `--help` without MLX.

### Requirements and options

- `CHECKPOINT` is required and must pass strict model and optimizer tensor checks.
- `--cola-data PATH` selects a local four-column CoLA TSV. Labels must be `0` or `1`. Each nonblank row must have exactly four columns and a sentence with at least one non-whitespace character. The command rejects an invalid row with its TSV row number before model load or byte evaluation. Without this option, the command runs only the byte suite.
- `--epochs N` sets CoLA head epochs. The default is `1`. When you set `--cola-data`, the value must be a positive integer. Epoch count does not change the fixed byte suite.
- `--split FRACTION` sets the contiguous CoLA train fraction. The default is `0.5`.
- `--seed N` sets the checkpoint load and CoLA head seed. This option is required.
- The benchmark needs MLX. CoLA needs a readable TSV with at least two valid rows. The command does not download data.

### Workflow

1. Select a checkpoint from a train or sweep run.
2. Run the fixed byte suite.
3. To include CoLA, supply a local four-column TSV.
4. Set `--seed` for every benchmark. Set `--epochs` or `--split` when needed.

```bash
tmt benchmark runs/first/model.safetensors --seed 11
tmt benchmark runs/first/model.safetensors --cola-data data/cola.tsv --epochs 1 --split 0.5 --seed 11
```

### Behavior and files
The fixed suite uses 393 input bytes in three documents and context windows of 1, 8, 32, and 128 bytes. It reports 390 full-history targets and 104 common targets. The common count uses the suffix shared by all windows. For CoLA, the loader reads the sentence from column four and the label from column two. It keeps the input row order. The fixed suite does not establish general model quality.

The loader ignores blank lines and preserves sentence whitespace. TSV row numbers include blank lines.

The split is contiguous. Values below 0 or above 1 are clipped to that range. The model weights stay frozen, and only a separate linear classification head receives updates. The command reports train and held-out confusion counts with MCC multiplied by 100. The score range is -100 to 100.

RTU uses recurrent-state features. Other blocks use final-output features by default.
The benchmark reports the JSON key `feature_convention` as `recurrent-state` or `final-output` with the CoLA scores.

The benchmark prints results to the terminal. It does not save a report or checkpoint. A missing supplied TSV raises an ordinary file error.

## generate

Use a UTF-8 text prompt to produce a fixed number of raw output bytes from a frozen checkpoint.

### Syntax
```text
tmt generate CHECKPOINT --prompt TEXT --output PATH --seed N [--bytes N] [--log-every N] [--temperature VALUE]
```

### Requirements and options

- `CHECKPOINT` is required and must pass strict model and optimizer tensor checks.
- `--prompt TEXT` is required and must not be empty. The command encodes it as UTF-8.
- `--output PATH` is required. Its parent folder must exist and allow writes.
- `--seed N` sets the model load and sample seed. This option is required.
- `--bytes N` sets the output byte count. The default is `512`. Zero writes an empty file. Negative values are invalid.
- `--log-every N` sets the output progress interval in bytes. The default is `1000`. Use a positive value.
- `--temperature VALUE` overrides the checkpoint value. The override must be finite and nonnegative. The default uses the checkpoint value.
- Generation needs MLX. The output byte count can be zero or positive.

### Workflow

1. Select a checkpoint.
2. Choose a nonempty text prompt. Set `--bytes 0` to create an empty output file.
3. Choose an output path outside files that you need to keep.
4. Read the output as bytes. It does not include the prompt.

```bash
tmt generate runs/first/model.safetensors --prompt "The key is" --output runs/first/generated.bin --bytes 64 --seed 11
```

### Behavior and files
The command replays the prompt bytes through the model, then samples the requested number of bytes. The model weights stay frozen. The command prints the generated byte count at each progress interval and at completion. It writes only generated bytes to `PATH`. A zero byte count creates an empty file. The output file replaces a file that already exists at that path. The command does not create its parent folder. Output bytes can be invalid UTF-8.

A temperature of zero still samples. The sampler uses a minimum temperature of `0.1`.

## help

Read the bundled manual or one command section. These requests run before command argument checks and experiment work.

### Syntax
```text
tmt help [COMMAND]
tmt help --help
tmt --help COMMAND
tmt -h COMMAND
tmt COMMAND --help
tmt COMMAND -h
tmt --help
tmt -h
tmt --version
```

### Requirements and options

- `COMMAND` is optional after `tmt help`. It can be `init`, `train`, `sweep`, `evaluate`, `benchmark`, `generate`, or `help`.
- `-h` and `--help` select the same manual section in the listed positions.
- Bare `tmt --help` and `tmt -h` print a short command index.
- `tmt --version` prints the installed package version and commit information.
- Manual and version requests need no MLX, config, data, or checkpoint. They create no files.

### Workflow

1. Run `tmt --help` to view the command index.
2. Name a command to read its full section.
3. Use the syntax and workflow in that section to run an experiment.

```bash
tmt help init
tmt help help
tmt --help train
tmt generate --help
tmt --version
```

### Behavior and limits
`tmt help` prints the complete bundled manual. `tmt help COMMAND` prints one section. `tmt help --help` prints the help section. The `--help` and `-h` forms print the same section. A help request after other command arguments prints the manual before the CLI checks those arguments or starts an experiment.

An unknown help topic gives a short parser error and exit code `2`. A manual request does not need a valid file path. The version command reads installed distribution metadata from `test-model-thing`.

It prints `vcs_info.commit_id` from `direct_url.json` when present. Otherwise it prints `commit unknown`. The version command does not inspect the current folder or run Git. Local and wheel installs normally report `commit unknown`. There is no `version` subcommand.
