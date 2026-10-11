# CLI workflow

Read the [command manual](src/tmt/commands.md) or use `tmt help` for full command details.
Command sections: [init](src/tmt/commands.md#init), [train](src/tmt/commands.md#train), [sweep](src/tmt/commands.md#sweep),
[evaluate](src/tmt/commands.md#evaluate), [benchmark](src/tmt/commands.md#benchmark), [generate](src/tmt/commands.md#generate), and [help](src/tmt/commands.md#help).

## Installation and PATH

Use Python 3.12, 3.13, or 3.14. Model commands and `init` require a functional MLX runtime.
From the repository root, install the package:

```bash
# Use python3.12, python3.13, or python3.14.
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
tmt --help
tmt --version
python -m tmt --help
```

This regular installation copies the package into the virtual environment.
For development and tests, use `python -m pip install -e '.[test]'` instead.
After source edits, repeat a regular installation before you verify those edits.
Run the tests with `python -m pytest`.

Activation adds `.venv/bin` to PATH for one terminal.
For new terminals, you can add this optional line to your shell configuration:

```bash
export PATH="/absolute/path/to/main-tmt/.venv/bin:$PATH"
```

Replace the example path with your environment's absolute path.
The whole venv on PATH also selects that environment's Python and other executables.
TMT does not modify shell files or require a global installation.

`--version` reads the installed `test-model-thing` distribution version.
It includes `vcs_info.commit_id` from `direct_url.json` when present.
Otherwise it prints `commit unknown`. It does not query Git in the current folder.
Help and version require no MLX import.

## Package and workspace

The package files are `src/tmt/{__init__.py,__main__.py,blocks.py,cli.py,main.py,benchmark.py}`.
`main.py` owns model arithmetic and constructor defaults. `blocks.py` owns byte-step blocks and their factory table.
`cli.py` owns commands and the strict checkpoint loader.
`benchmark.py` owns frozen byte scores and the CoLA classifier head. Tests stay in `test/`.

The tracked examples are [model.json](examples/model.json), [tiny-model.json](examples/tiny-model.json), [grid.json](examples/grid.json), and [run.example.json](examples/run.example.json).
Installed commands do not need checkout examples.


All relative paths use the current folder. TMT does not search parent folders.
Bare `tmt init` creates `model.json`, `grid.json`, `run.example.json`, `data/train/`, `data/dev/`, and `runs/`.
`model.json` contains every constructor default, with `"block": "rtu"`.
A user JSON can override a subset of these values.

The workspace grid uses dimensions 4 and 8, one layer, spread 4, block `rtu`, and seeds 11 and 22.

`run.example.json` is a template, not a run record. Its null timestamp is a placeholder.

`init --output PATH` writes only a model JSON. Init preserves current files and can fail on a repeated invocation.

## Complete small workflow

Use a new workspace folder. These commands select a tiny model and raw-byte input:

```bash
tmt init
printf '{"dim":4,"layers":1,"spread":4,"block":"rtu"}\n' > model.json
printf abcd > data/train/tiny.bin
printf wxyz > data/dev/tiny.bin
tmt train --data 'data/train/*' --run first --updates 3
tmt train runs/first/model.safetensors --data 'data/train/*' --resume --updates 2
tmt evaluate runs/first/model.safetensors --data 'data/dev/*' --windows 1 2 --output score.json
tmt generate runs/first/model.safetensors --prompt 'The ' --output sample.bin --bytes 4
tmt benchmark runs/first/model.safetensors
tmt sweep --grid grid.json --data 'data/train/*' --development 'data/dev/*' --output runs/grid --updates 2
```

Train and sweep default to `model.json`. An explicit `--config PATH` overrides it for a fresh run.
The CLI checks config types and limits before model creation. The [command manual](src/tmt/commands.md) lists these limits.

All CLI seeds and sweep grid seeds must be integers from `0` to `18446744073709551615`, inclusive.
Evaluation and sweep require a positive `--max-bytes` value. Zero and negative values cause an input error.

Evaluation also needs at least one document with two bytes in the selected input prefix. Empty and one-byte documents provide no targets.
`--run NAME` creates `runs/NAME/model.safetensors` in a new folder.
Use one relative folder name without path separators. Empty names, `.` and `..` are invalid.
With neither a name nor a checkpoint, train uses a UTC timestamp with microseconds as the run folder name.

For an explicit path, use `tmt train other/model.safetensors --data 'data/train/*' --updates 3`.
A positional checkpoint excludes `--run`. Resume requires that positional checkpoint and forbids `--config` and `--run`.

Resume restores model and optimizer tensors, then adds the requested updates with a new data pass.
It resets block state and temporal traces at the first document. It does not restore the file cursor or RNG state.
The checkpoint config must include `block`. Checkpoints from before this field are unsupported.

The trainer writes `manifest/run.json` beside the checkpoint at start and exit.
The record contains model `config`, invocation `seed`, UTC `started_at`, `status`, and `last_checkpoint`.
It stores `null` in the sample and evaluation fields when those features are off.
Status is `running`, `complete`, `failed`, or `interrupted`. A resume replaces the record for the invocation.

The trainer writes one JSON object per line to `manifest/loss.jsonl`.
Use `--log-every N` to set the update interval. Its default is `100` updates.
Each row records `completed_updates`, `optimizer_step`, and `objective`.
It also records the latest `loss`, interval `mean_loss`, and mean `mean_loss_components` for `cross_entropy`, `latent_prediction`, `variance`, and `stop`.
The row includes `interval_updates`, `elapsed_seconds`, and `updates_per_second`.

After one or more updates, the trainer writes a final row if the last update was outside the interval.
This rule also applies after a failure or interrupt. The trainer does not duplicate an interval row when the run ends on that interval.
Each invocation replaces this file. This also applies to a resume.

The record also contains `data_glob`, absolute `data_files` in shuffled pass order, requested `updates`, `log_every`, and `objective`.
The objective is `tmt` for the full loss or `ce_only` for next-byte cross entropy.
The file list includes all matched paths. The update budget can stop a pass before the trainer reads every file.
Paths identify the file selection but do not prove that file contents stay the same.

The record also contains `trainable_parameters` and `temporal_gradient`.
`trainable_parameters` counts trainable tensor elements before a full-model freeze.
`temporal_gradient` lists the gradient rule label for each block in layer order.

`resume` identifies a resume invocation. `initial_optimizer_step` is the saved optimizer step at resume start, or zero for a fresh run.
`updates` is the budget for this invocation, with additional updates on resume.

`completed_updates` is zero at start and counts completed updates at exit, after success, failure, or interrupt.
This count includes work after the last checkpoint. It does not imply that the trainer saved all completed updates.
See the [manifest fields](src/tmt/commands.md#train) for details.

The trainer prints completed updates, total optimizer step, objective, mean loss, latest loss, and update speed at each interval.
It always prints the final loss.
`elapsed_seconds` is cumulative from the start of the train loop.
`updates_per_second` uses time since the previous report, with time for reports, samples, and evaluation in that interval.

Use `--sample-every N` to write a sample after each N completed updates.
The command writes no samples by default.
Set `--sample-prompt`, `--sample-seed`, and `--sample-bytes` to change the prompt, seed, and output length.
The defaults are `The `, `11`, and `256` bytes. The byte count must be positive.

Sample prompts must be nonempty. The CLI checks sample values and evaluation byte budgets even when samples or interval evaluation are off.

The command prints an escaped preview and writes raw sample bytes under `samples/<UTC timestamp>/`.
It records each sample in `manifest/samples.jsonl` with its update counts, prompt, seed, byte count, and path.
Each invocation replaces the sample index.
The command keeps old sample files.
The command uses the same sample seed for each sample.
The command clears the index at each invocation, even when sample output is off.

Use `--eval-data GLOB` with `--eval-every N` to score held-out files while the trainer runs.
You must set both options.
The default evaluation byte budget is `8192` bytes. Set `--eval-max-bytes N` to change it.

The command evaluates at each interval and after successful completion if the last update was outside the interval.
It writes score rows to `manifest/evaluation.jsonl`. Each row includes `completed_updates`, `optimizer_step`, and the normal evaluation score fields.
Evaluation uses frozen model weights and restores block state and temporal traces.
Each invocation replaces `evaluation.jsonl`.
The command clears this file at each invocation, even when interval evaluation is off.

It saves every 500 updates and at normal completion. It prints save feedback only after success.
Ctrl-C does not save. Keep the last successful checkpoint. Direct writes can leave partial files after a process failure.

## Scores and limits

Each file in a quoted glob is one raw-byte document. TMT adds no separators and does not decode input.
The trainer shuffles sorted file paths once per invocation. Each adjacent byte pair gives one optimizer update.
The trainer resets block state and temporal traces at each document. Empty files give no updates.

The default objective combines variance, latent-space prediction, next-byte cross entropy, and stop loss.
`--ce-only` selects next-byte cross entropy alone.

Evaluation uses frozen weights, no samples, and the first byte as context. It restores block state and temporal traces after success or failure.
The default input budget is 8192 bytes across sorted files. Each context window counts input bytes before the target.

Sweep requires a new or empty output folder. It rejects a nonempty folder without changes.
The grid must be a JSON object. Each field must contain a nonempty array.
The CLI checks the grid before it creates the output folder.

Sweep appends one row per completed candidate to `results.jsonl`. It selects the lowest development BPB, with the first candidate as tie winner.
Candidates use `run-NNNN/` folders with `model.safetensors`, `manifest/model.json`, `manifest/run.json`, and `manifest/loss.jsonl`.
Train sample and evaluation options apply to each candidate.

The development score at the end of a sweep remains separate.

## Blocks and temporal gradients

The `block` config field selects one factory for every model layer.
The package registers only `rtu`. Its temporal-gradient label is `rtu-v1`.
The model records one label per layer in `temporal_gradient`.

The Python method `Model.step()` returns `((x, state_results, auxiliary_results), (logits, stop))`.
`state_results` contains one dictionary per block. Each dictionary uses that block's `state_fields` names.
`auxiliary_results` contains one block-specific value per block.
The command-line interface does not expose this internal result structure.

Each block takes an encoded byte, the prior layer output, and a probe.
It returns the next output, candidate state, and auxiliary data for its gradient rule.
The next output has shape `[dim]`. State fields can use other fixed shapes.

Blocks list byte state in `state_fields` and temporal history in `trace_fields`.
They create all fields before optimizer setup. They freeze runtime fields and keep trainable weights as MLX parameters.
The forward call reads prior state and returns candidate state.
The call does not change persistent fields.
The model commits state after differentiation, or after a frozen step.

The base block provides `reset()`, `probe()`, `adjust_gradients()`, `features()`, and state helpers.
The helper names are `snapshot_state()`, `restore_state()`, `commit_state()`, and `eval_state()`.
`Model` applies `snapshot_state()`, `restore_state()`, and `eval_state()` across its blocks.
It calls each block's `commit_state()` for candidate state and trace values.

Evaluation and samples restore snapshots after success or failure.
Frozen steps advance state only. They leave traces, weights, and optimizer state unchanged.

The default temporal rule uses autodiff within one byte step.
It does not pass gradients through detached byte history.
The RTU hook adds the current recurrent trace contributions to the encoder and block gradients.
The hook label is `rtu-v1`.

The CoLA probe uses the final block feature vector.
RTU uses its recurrent state, with feature convention `recurrent-state`.
Other blocks use the final output, with feature convention `final-output`.
The benchmark result records `feature_convention` because different features can change CoLA MCC.

`block` is part of resolved model config and checkpoint metadata.
The loader requires complete metadata with this selector.
Earlier checkpoints without `block` fail validation, even when their version marker is `1`.
Create new checkpoint artifacts with this release.

## Local block comparison

The package ships only the RTU factory. Add a research block to `src/tmt/blocks.py` and register it in `BLOCKS`.
Keep the block name fixed for the whole experiment. Save the code revision with each result.
Use a local block name such as `candidate-v1` only after you add that factory.

Use one base config for both blocks. Keep the data, objective, optimizer settings, update budget, and seed list fixed.
Add each block and seed to a sweep grid to create paired runs.

```json
{
  "block": ["rtu", "candidate-v1"],
  "seed": [11, 22, 33]
}
```

```bash
tmt sweep --config research.json --grid ab-grid.json \
  --data 'data/train/*' --development 'data/dev/*' \
  --output runs/ab-001 --updates 1000
```

Group result rows by seed and compare every paired score.
A shared seed repeats the random-number start. It does not guarantee equal initial values for differently shaped blocks.
Equal updates also do not mean equal parameter count, compute, or state memory.
Compare `trainable_parameters` and `temporal_gradient` with BPB.
Use untouched test documents for a final comparison.

The byte-only benchmark always uses 393 input bytes, 390 targets, and 104 common targets with windows 1, 8, 32, and 128.
For optional CoLA, use `tmt benchmark runs/first/model.safetensors --cola-data cola.tsv --epochs 1 --split 0.5`.
When you set `--cola-data`, `--epochs` must be a positive integer.
Each nonblank CoLA row must have exactly four columns, a binary label, and a nonblank sentence. Invalid rows produce a row-numbered error before model load or byte evaluation. The loader ignores blank lines and preserves sentence whitespace.

`python -m tmt.benchmark` accepts the same benchmark options.
CoLA uses four-column TSV rows, a contiguous split, and head-only updates. It reports MCC times 100.
Each label must be `0` or `1`. An invalid label causes an error with its TSV row number before model load or byte evaluation.
The benchmark reports `feature_convention` with CoLA scores.

Generation encodes its nonempty prompt as UTF-8 and writes raw bytes without the prompt.
It prints byte progress at each `--log-every` interval. The default interval is `1000` bytes.
The `--temperature` override must be finite and nonnegative. The sampler uses a minimum temperature of `0.1`.

Checkpoints require complete model and optimizer tensors plus version-1 model metadata. The loader checks names, shapes, and dtypes.
Keep output paths apart from input paths. The CLI does not check path collisions or train/development overlap.

Use a positive value for `--updates`, each value in `--windows`, and `--log-every`.
Generation accepts `--bytes 0` and writes an empty file. It rejects negative byte counts.
Whole-file reads can require memory equal to the largest file.

Keep objective loss, frozen BPB, and CoLA MCC separate. Tiny fixtures do not establish general model quality.

File errors and invalid values print a short error to stderr and exit with code `1`. Argument errors exit with code `2`.
Train checks train and evaluation selections before it creates a named or timestamp run folder.
Sweep checks train, evaluation, and development selections before output creation.
Unmatched globs and unreadable files cause an input error. Correct the input and retry with the same output path.

After sweep execution starts, `sweep.json` records status, progress, and any failure or interrupt error.
For a retry after an execution failure, use a new sweep output path. The command preserves partial results and checkpoints.
