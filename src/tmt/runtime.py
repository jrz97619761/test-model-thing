"""Training, sampling, generation, and sweep runtime operations."""

from __future__ import annotations

import glob
import itertools
import json
import math
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from . import validation
from .checkpoint import (
    load_checkpoint,
    new_model,
    save_checkpoint as write_checkpoint,
)
from .config import load_config
from .data import ByteFiles
from .records import (
    RunManifest,
    RunStatus,
    SweepManifest,
    append_jsonl,
    build_run_manifest,
    build_sweep_manifest,
    write_json,
)

if TYPE_CHECKING:
    import mlx.core as mx

    from .main import Model


def _input_files(pattern: str | PathLike[str]) -> list[str]:
    """Resolve a selection and check that every selected file is readable."""
    return ByteFiles(pattern).paths


def documents(
    pattern: str | PathLike[str], max_bytes: int | None = None
) -> Iterator[bytes]:
    """Yield sorted raw files with an optional prefix across all file bytes."""
    yield from ByteFiles(pattern).documents(max_bytes)


def _training_sample(model: Model, prompt: str, count: int, seed: int) -> bytes:
    """Sample with a local RNG key and restore recurrent state after any exit."""
    validation.seed(seed, 'sample-seed')
    import mlx.core as mx

    snapshot = model.snapshot_state()
    try:
        model.reset()
        for byte in prompt.encode('utf-8'):
            _, (logits, _) = model.step(mx.array(byte), frozen=True)
        key = mx.random.key(seed)
        generated = bytearray()
        for _ in range(count):
            key, sample_key = mx.random.split(key)
            byte = int(model.sample(logits, key=sample_key).item())
            generated.append(byte)
            _, (logits, _) = model.step(mx.array(byte), frozen=True)
        model.eval_state()
        return bytes(generated)
    finally:
        model.restore_state(snapshot)
        model.eval_state()


@dataclass(frozen=True)
class MonitorOptions:
    """Sample and evaluation settings shared by a training run and a sweep."""

    sample_every: int | None = None
    sample_prompt: str = 'The '
    sample_seed: int = 11
    sample_bytes: int = 256
    eval_data: str | None = None
    eval_every: int | None = None
    eval_max_bytes: int = 8192


@dataclass
class TrainingRun:
    """Mutable state and actions for one training run."""

    settings: dict[str, Any] | None
    path: str | PathLike[str]
    pattern: str | PathLike[str]
    updates: int
    seed: int
    ce_only: bool
    resume: bool
    log_every: int
    monitor: MonitorOptions
    new_run: bool = False
    files: list[str] = field(default_factory=list)
    evaluation_data: list[bytes] | None = None
    model: Model = field(init=False)
    checkpoint: Path = field(init=False)
    manifest_dir: Path = field(init=False)
    manifest: Path = field(init=False)
    metrics: Path = field(init=False)
    record: RunManifest | None = None
    completed: int = 0
    last_loss: mx.array | None = None
    logged: int = 0
    evaluated: int = 0
    loss_sum: float = 0.0
    component_sums: dict[str, float] = field(default_factory=dict)
    started: float = 0.0
    last_report: float = 0.0
    sample_dir: Path = field(init=False)
    status: RunStatus = 'failed'
    mx: Any = None

    def setup(self) -> None:
        """Load inputs and model, then create the run manifest paths."""
        import mlx.core as mx

        self.mx = mx
        self.files = _input_files(self.pattern)
        self.evaluation_data = (
            None
            if self.monitor.eval_data is None
            else list(documents(self.monitor.eval_data, self.monitor.eval_max_bytes))
        )
        if self.resume:
            self.model, self.settings = load_checkpoint(self.path, self.seed)
        else:
            self.model = new_model(self.settings, self.seed)
        random.shuffle(self.files)

        self.checkpoint = Path(self.path)
        self.checkpoint.parent.mkdir(parents=True, exist_ok=not self.new_run)
        self.manifest_dir = self.checkpoint.parent / 'manifest'
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = self.manifest_dir / 'run.json'
        self.metrics = self.manifest_dir / 'loss.jsonl'
        self.record = build_run_manifest(
            config=self.settings,
            trainable_parameters=self.model.count(),
            temporal_gradient=self.model.temporal_gradient,
            seed=self.seed,
            data_glob=self.pattern,
            data_files=[str(Path(filename).resolve()) for filename in self.files],
            updates=self.updates,
            log_every=self.log_every,
            objective='ce_only' if self.ce_only else 'tmt',
            resume=self.resume,
            initial_optimizer_step=(
                int(self.model.optimizer.state['step'].item()) if self.resume else 0
            ),
            sample=None
            if self.monitor.sample_every is None
            else {
                'every': self.monitor.sample_every,
                'prompt': self.monitor.sample_prompt,
                'seed': self.monitor.sample_seed,
                'bytes': self.monitor.sample_bytes,
            },
            evaluation=None
            if self.monitor.eval_every is None
            else {
                'every': self.monitor.eval_every,
                'data_glob': self.monitor.eval_data,
                'max_bytes': self.monitor.eval_max_bytes,
                'data_files': [
                    str(Path(name).resolve())
                    for name in sorted(
                        glob.glob(self.monitor.eval_data, recursive=True)
                    )
                ],
            },
            started_at=datetime.now(timezone.utc).isoformat(),
            last_checkpoint=(
                str(self.checkpoint.resolve()) if self.resume else None
            ),
        )
        self.started = self.last_report = time.perf_counter()
        self.sample_dir = (
            self.checkpoint.parent
            / 'samples'
            / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        )

    def report_loss(self) -> None:
        """Write and print the current objective loss row."""
        if self.record is None or self.last_loss is None:
            raise RuntimeError('training run is not ready to report loss')
        now = time.perf_counter()
        interval_updates = self.completed - self.logged
        loss = float(self.last_loss.item())
        row = {
            'completed_updates': self.completed,
            'optimizer_step': int(self.model.optimizer.state['step'].item()),
            'objective': self.record['objective'],
            'loss': loss,
            'mean_loss': self.loss_sum / interval_updates,
            'mean_loss_components': {
                name: value / interval_updates
                for name, value in self.component_sums.items()
            },
            'interval_updates': interval_updates,
            'elapsed_seconds': now - self.started,
            'updates_per_second': interval_updates / max(now - self.last_report, 1e-9),
        }
        append_jsonl(self.metrics, row)
        print(
            f'updates: {self.completed}, objective: {self.record["objective"]}, '
            f'loss: {loss:.6f}, '
            f'mean loss: {row["mean_loss"]:.6f}, step: {row["optimizer_step"]}, '
            f'updates/s: {row["updates_per_second"]:.2f}',
            flush=True,
        )
        self.logged = self.completed
        self.last_report = now
        self.loss_sum = 0.0
        self.component_sums.clear()

    def report_sample(self) -> None:
        """Write a seeded training sample and its manifest row."""
        sample = _training_sample(
            self.model,
            self.monitor.sample_prompt,
            self.monitor.sample_bytes,
            self.monitor.sample_seed,
        )
        step = int(self.model.optimizer.state['step'].item())
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        destination = self.sample_dir / f'step-{step:012d}.bin'
        destination.write_bytes(sample)
        append_jsonl(
            self.manifest_dir / 'samples.jsonl',
            {
                'completed_updates': self.completed,
                'optimizer_step': step,
                'prompt': self.monitor.sample_prompt,
                'seed': self.monitor.sample_seed,
                'bytes': len(sample),
                'path': str(destination.resolve()),
            },
        )
        print(
            f'sample at update {self.completed}, step {step}: {sample!r}', flush=True
        )

    def report_evaluation(self) -> None:
        """Score held-out documents and write an evaluation row."""
        if self.evaluation_data is None:
            raise RuntimeError('evaluation data is not configured')
        from tmt.benchmark import evaluate

        result = evaluate(self.model, self.evaluation_data)
        append_jsonl(
            self.manifest_dir / 'evaluation.jsonl',
            {
                'completed_updates': self.completed,
                'optimizer_step': int(self.model.optimizer.state['step'].item()),
                **result,
            },
        )
        print(
            f'evaluation at update {self.completed}: BPB {result["bpb"]:.6f}, '
            f'targets: {result["targets"]}',
            flush=True,
        )
        self.evaluated = self.completed

    def save_checkpoint(self) -> None:
        """Save model and optimizer state, then update the manifest record."""
        if self.record is None:
            raise RuntimeError('training run is not ready to save')
        write_checkpoint(self.checkpoint, self.model, self.settings)
        self.record['last_checkpoint'] = str(self.checkpoint.resolve())
        print(f'saved checkpoint: {self.checkpoint}, updates: {self.completed}')

    def consume_terminal_byte(self, byte: int) -> None:
        """Advance frozen recurrent state over a document's final byte."""
        self.model.step(self.mx.array(byte), frozen=True)
        self.model.eval_state()

    def after_update(self, end: bool, terminal_byte: int) -> None:
        """Run every action triggered by an optimizer update in stable order."""
        if self.completed % self.log_every == 0:
            self.report_loss()
        if end:
            self.consume_terminal_byte(terminal_byte)
        if (
            self.monitor.sample_every is not None
            and self.completed % self.monitor.sample_every == 0
        ):
            self.report_sample()
        if (
            self.monitor.eval_every is not None
            and self.completed % self.monitor.eval_every == 0
        ):
            self.report_evaluation()
        if self.completed % 500 == 0:
            self.save_checkpoint()

    def one_data_pass(self) -> None:
        """Visit the selected files once, in the run's fixed shuffled order."""
        before_pass = self.completed
        for filename in self.files:
            data = Path(filename).read_bytes()
            if not data:
                continue
            self.model.reset()
            if len(data) == 1:
                self.consume_terminal_byte(data[0])
                continue
            for index in range(len(data) - 1):
                end = index == len(data) - 2
                components: dict[str, float] = {}
                _, _, self.last_loss = self.model.train_step(
                    data[index],
                    data[index + 1],
                    end,
                    self.ce_only,
                    metrics=components,
                )
                self.completed += 1
                self.loss_sum += float(self.last_loss.item())
                for name, value in components.items():
                    self.component_sums[name] = self.component_sums.get(name, 0.0) + value
                self.after_update(end, data[-1])
                if self.completed >= self.updates:
                    break
            if self.completed >= self.updates:
                break
        if self.completed == before_pass:
            raise ValueError('no training targets found in data pass')

    def finalize(self) -> None:
        """Write final run status and any pending partial loss row."""
        if self.record is None:
            raise RuntimeError('training run is not ready to finalize')
        self.record['status'] = self.status
        self.record['completed_updates'] = self.completed
        write_json(self.manifest, self.record)
        if self.last_loss is not None and self.completed != self.logged:
            self.report_loss()

    def run(self) -> Model:
        """Run data passes and preserve the run's failure and interrupt status."""
        if self.record is None:
            raise RuntimeError('training run must be set up before it can run')
        try:
            self.metrics.write_text('')
            (self.manifest_dir / 'samples.jsonl').write_text('')
            (self.manifest_dir / 'evaluation.jsonl').write_text('')
            write_json(self.manifest, self.record)
            print(
                f'run folder: {self.checkpoint.parent}, seed: {self.seed}, '
                f'config: {json.dumps(self.settings)}'
            )
            while self.completed < self.updates:
                self.one_data_pass()
            if (
                self.monitor.eval_every is not None
                and self.completed != self.evaluated
            ):
                self.report_evaluation()
            self.save_checkpoint()
            self.status = 'complete'
        except KeyboardInterrupt:
            self.status = 'interrupted'
            raise
        finally:
            self.finalize()

        print(f'updates: {self.completed}, checkpoint: {self.checkpoint}')
        return self.model


def train(
    settings: dict[str, Any] | None,
    path: str | PathLike[str],
    pattern: str | PathLike[str],
    updates: int = 1000,
    seed: int | None = None,
    ce_only: bool = False,
    resume: bool = False,
    log_every: int = 100,
    sample_every: int | None = None,
    sample_prompt: str = 'The ',
    sample_seed: int = 11,
    sample_bytes: int = 256,
    eval_data: str | None = None,
    eval_every: int | None = None,
    eval_max_bytes: int = 8192,
    *,
    new_run: bool = False,
) -> Model:
    """Train on adjacent raw-byte pairs and record the run manifest."""
    validation.training_options(seed, sample_seed, updates, log_every)
    validation.monitoring(
        sample_every, sample_prompt, sample_bytes, eval_data, eval_every, eval_max_bytes
    )
    run = TrainingRun(
        settings=settings,
        path=path,
        pattern=pattern,
        updates=updates,
        seed=seed,
        ce_only=ce_only,
        resume=resume,
        log_every=log_every,
        monitor=MonitorOptions(
            sample_every=sample_every,
            sample_prompt=sample_prompt,
            sample_seed=sample_seed,
            sample_bytes=sample_bytes,
            eval_data=eval_data,
            eval_every=eval_every,
            eval_max_bytes=eval_max_bytes,
        ),
        new_run=new_run,
    )
    run.setup()
    return run.run()


def evaluate_checkpoint(
    checkpoint: str | PathLike[str],
    pattern: str | PathLike[str],
    max_bytes: int,
    windows: list[int],
) -> dict[str, Any]:
    """Load a checkpoint and evaluate it on selected byte documents."""
    from tmt.benchmark import evaluate

    model, _ = load_checkpoint(checkpoint, 0)
    data = list(documents(pattern, max_bytes))
    return evaluate(model, data, windows)


def generate(
    checkpoint: str | PathLike[str],
    prompt: str,
    output: str | PathLike[str],
    seed: int | None = None,
    count: int = 512,
    temperature: float | None = None,
    log_every: int = 1000,
) -> bytes:
    """Replay a UTF-8 prompt and write a fixed count of sampled bytes."""
    validation.seed(seed)
    validation.require(prompt, 'generation needs a nonempty prompt')
    validation.require(count >= 0, 'bytes must be nonnegative')
    validation.positive(log_every, 'log-every')
    validation.require(
        temperature is None or (math.isfinite(temperature) and temperature >= 0),
        'temperature must be a finite nonnegative number',
    )

    import mlx.core as mx

    model, _ = load_checkpoint(checkpoint, seed)
    model.reset()
    if temperature is not None:
        model.temp = temperature
    mx.random.seed(seed)

    prompt = prompt.encode('utf-8')
    for byte in prompt:
        _, (logits, _) = model.step(mx.array(byte), frozen=True)
    generated = bytearray()
    for _ in range(count):
        byte = int(model.sample(logits).item())
        generated.append(byte)
        _, (logits, _) = model.step(mx.array(byte), frozen=True)
        if len(generated) % log_every == 0:
            print(f'generated bytes: {len(generated)}/{count}', flush=True)

    with open(output, 'wb') as file:
        file.write(generated)
    print(f'Generated {len(generated)} bytes to {output}')
    return bytes(generated)


@dataclass(frozen=True)
class SweepCandidate:
    """Resolved model settings and seed for one grid row."""

    index: int
    settings: dict[str, Any]
    seed: int


class SweepExecution(NamedTuple):
    """Model, checkpoint, and development score for one completed run."""

    model: Model
    checkpoint: Path
    score: dict[str, Any]


def _resolve_sweep_candidate(
    index: int,
    values: tuple[Any, ...],
    keys: list[str],
    settings: dict[str, Any],
    seed: int,
) -> SweepCandidate:
    """Resolve one grid point while preserving key and product order."""
    overrides = dict(zip(keys, values))
    run_seed = overrides.pop('seed', seed)
    run_settings = dict(settings)
    run_settings.update(overrides)
    validation.model_settings(run_settings, 'model config')
    return SweepCandidate(index=index, settings=run_settings, seed=run_seed)


def _run_sweep_candidate(
    candidate: SweepCandidate,
    output: Path,
    pattern: str | PathLike[str],
    updates: int,
    ce_only: bool,
    log_every: int,
    monitor: MonitorOptions,
    development_data: list[bytes],
    evaluate: Callable[[Model, list[bytes]], dict[str, Any]],
) -> SweepExecution:
    """Train and score one already-resolved sweep candidate."""
    run_path = output / f'run-{candidate.index:04d}'
    run_path.mkdir(parents=True, exist_ok=True)
    manifest_dir = run_path / 'manifest'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_dir / 'model.json', candidate.settings)
    checkpoint = run_path / 'model.safetensors'
    model = train(
        candidate.settings,
        checkpoint,
        pattern,
        updates,
        candidate.seed,
        ce_only,
        resume=False,
        log_every=log_every,
        sample_every=monitor.sample_every,
        sample_prompt=monitor.sample_prompt,
        sample_seed=monitor.sample_seed,
        sample_bytes=monitor.sample_bytes,
        eval_data=monitor.eval_data,
        eval_every=monitor.eval_every,
        eval_max_bytes=monitor.eval_max_bytes,
    )
    score = evaluate(model, development_data)
    return SweepExecution(model=model, checkpoint=checkpoint, score=score)


def _sweep_result(
    candidate: SweepCandidate,
    run: SweepExecution,
    pattern: str | PathLike[str],
    development: str | PathLike[str],
    max_bytes: int,
    updates: int,
    ce_only: bool,
) -> dict[str, Any]:
    """Build the stable results row for one completed sweep candidate."""
    model = run.model
    score = run.score
    checkpoint = run.checkpoint
    return {
        'run': candidate.index,
        'settings': candidate.settings,
        'trainable_parameters': model.count(),
        'temporal_gradient': model.temporal_gradient,
        'seed': candidate.seed,
        'updates': updates,
        'objective': 'ce_only' if ce_only else 'tmt',
        'data_glob': pattern,
        'development_glob': development,
        'development_max_bytes': max_bytes,
        'development_targets': score['targets'],
        'development_bpb': score['bpb'],
        'checkpoint': str(checkpoint.resolve()),
    }


def sweep(
    grid_path: str | PathLike[str],
    pattern: str | PathLike[str],
    development: str | PathLike[str],
    output: str | PathLike[str],
    config: str | PathLike[str] = 'model.json',
    updates: int = 1000,
    seed: int | None = None,
    ce_only: bool = False,
    max_bytes: int = 8192,
    log_every: int = 100,
    sample_every: int | None = None,
    sample_prompt: str = 'The ',
    sample_seed: int = 11,
    sample_bytes: int = 256,
    eval_data: str | None = None,
    eval_every: int | None = None,
    eval_max_bytes: int = 8192,
) -> dict[str, Any]:
    """Run each model and seed pair in JSON product order."""
    validation.training_options(seed, sample_seed, updates, log_every)
    validation.positive(max_bytes, 'max-bytes')
    validation.monitoring(
        sample_every, sample_prompt, sample_bytes, eval_data, eval_every, eval_max_bytes
    )
    from tmt.benchmark import evaluate

    settings = load_config(config)
    with open(grid_path) as file:
        grid = json.load(file)
    validation.sweep_grid(grid, settings)
    keys = list(grid)
    _input_files(pattern)
    if eval_data is not None:
        list(documents(eval_data, eval_max_bytes))
    development_data = list(documents(development, max_bytes))
    output_path = Path(output)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f'sweep output folder {output_path} is not empty; choose a new output folder'
        )
    output_path.mkdir(parents=True, exist_ok=True)
    results_path = output_path / 'results.jsonl'
    rows: list[dict[str, Any]] = []

    sweep_manifest = output_path / 'sweep.json'
    record: SweepManifest = build_sweep_manifest(
        config=settings,
        grid=grid,
        seed=seed,
        updates=updates,
        objective='ce_only' if ce_only else 'tmt',
        data_glob=pattern,
        development_glob=development,
        development_max_bytes=max_bytes,
        log_every=log_every,
        sample_every=sample_every,
        sample_prompt=sample_prompt,
        sample_seed=sample_seed,
        sample_bytes=sample_bytes,
        eval_data=eval_data,
        eval_every=eval_every,
        eval_max_bytes=eval_max_bytes,
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    write_json(sweep_manifest, record)
    monitor = MonitorOptions(
        sample_every=sample_every,
        sample_prompt=sample_prompt,
        sample_seed=sample_seed,
        sample_bytes=sample_bytes,
        eval_data=eval_data,
        eval_every=eval_every,
        eval_max_bytes=eval_max_bytes,
    )
    try:
        for index, values in enumerate(
            itertools.product(*(grid[key] for key in keys)), 1
        ):
            record['current_run'] = index
            candidate = _resolve_sweep_candidate(index, values, keys, settings, seed)
            run = _run_sweep_candidate(
                candidate,
                output_path,
                pattern,
                updates,
                ce_only,
                log_every,
                monitor,
                development_data,
                evaluate,
            )
            row = _sweep_result(
                candidate,
                run,
                pattern,
                development,
                max_bytes,
                updates,
                ce_only,
            )
            append_jsonl(results_path, row)
            rows.append(row)
            record['completed_runs'] = len(rows)
            write_json(sweep_manifest, record)
        record['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as error:
        record['status'] = (
            'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        )
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        write_json(sweep_manifest, record)

    best = min(rows, key=lambda row: row['development_bpb'])
    print(f'best row: {json.dumps(best)}')
    return best
