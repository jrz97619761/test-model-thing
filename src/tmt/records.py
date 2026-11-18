"""Typed manifest records and JSON output helpers."""

from __future__ import annotations

import json
from os import PathLike
from typing import Any, Literal, TypedDict

type RunStatus = Literal['running', 'complete', 'failed', 'interrupted']


class FileSelection(TypedDict):
    glob: str
    files: list[str]


class BudgetedFileSelection(FileSelection):
    max_bytes: int


class RunData(TypedDict):
    train: FileSelection
    evaluation: BudgetedFileSelection | None


class SweepData(TypedDict):
    train: FileSelection
    development: BudgetedFileSelection
    evaluation: BudgetedFileSelection | None


class TrainingSettings(TypedDict):
    seed: int
    updates: int
    objective: str
    resume: bool


class SampleOptions(TypedDict):
    every: int
    prompt: str
    seed: int
    bytes: int


class EvaluationInterval(TypedDict):
    every: int


class MonitoringOptions(TypedDict):
    log_every: int
    save_every: int
    sample: SampleOptions | None
    evaluation: EvaluationInterval | None


class RunProgress(TypedDict):
    completed_updates: int
    initial_optimizer_step: int
    optimizer_step: int


class SweepProgress(TypedDict):
    completed_runs: int
    current_run: int | None


class RunError(TypedDict):
    type: str
    message: str


class RunDetails(TypedDict):
    started_at: str | None
    status: RunStatus
    last_checkpoint: str | None
    error: RunError | None


class SweepDetails(TypedDict):
    started_at: str
    status: RunStatus
    error: RunError | None


class RunEvaluationOptions(TypedDict):
    every: int
    data_glob: str
    data_files: list[str]
    max_bytes: int


class RunManifest(TypedDict):
    model: dict[str, Any]
    hparams: dict[str, Any]
    data: RunData
    training: TrainingSettings
    monitoring: MonitoringOptions
    progress: RunProgress
    run: RunDetails


class SweepManifest(TypedDict):
    model: dict[str, Any]
    hparams: dict[str, Any]
    grid: dict[str, list[Any]]
    data: SweepData
    training: TrainingSettings
    monitoring: MonitoringOptions
    progress: SweepProgress
    run: SweepDetails


def _grouped_settings(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split flat settings into manifest model and hyperparameter groups."""
    from .config import group_model_config

    grouped = group_model_config(config)
    model = grouped['model']
    hparams = grouped['hparams']
    return model, hparams


def build_run_manifest(
    *,
    config: dict[str, Any],
    trainable_parameters: int | None,
    temporal_gradient: list[str] | None,
    seed: int,
    data_glob: str,
    data_files: list[str],
    updates: int,
    log_every: int,
    objective: str,
    save_every: int = 500,
    completed_updates: int = 0,
    resume: bool = False,
    initial_optimizer_step: int = 0,
    optimizer_step: int = 0,
    sample: SampleOptions | None = None,
    evaluation: RunEvaluationOptions | None = None,
    started_at: str | None = None,
    status: RunStatus = 'running',
    last_checkpoint: str | None = None,
    error: RunError | None = None,
) -> RunManifest:
    """Build a run manifest in the stable JSON field order."""
    model, hparams = _grouped_settings(config)
    model.update(
        trainable_parameters=trainable_parameters,
        temporal_gradient=temporal_gradient,
    )

    data_evaluation: BudgetedFileSelection | None = (
        None
        if evaluation is None
        else {
            'glob': evaluation['data_glob'],
            'files': evaluation['data_files'],
            'max_bytes': evaluation['max_bytes'],
        }
    )
    monitor_evaluation: EvaluationInterval | None = (
        None if evaluation is None else {'every': evaluation['every']}
    )

    return {
        'model': model,
        'hparams': hparams,
        'data': {
            'train': {'glob': data_glob, 'files': data_files},
            'evaluation': data_evaluation,
        },
        'training': {
            'seed': seed,
            'updates': updates,
            'objective': objective,
            'resume': resume,
        },
        'monitoring': {
            'log_every': log_every,
            'save_every': save_every,
            'sample': sample,
            'evaluation': monitor_evaluation,
        },
        'progress': {
            'completed_updates': completed_updates,
            'initial_optimizer_step': initial_optimizer_step,
            'optimizer_step': optimizer_step,
        },
        'run': {
            'started_at': started_at,
            'status': status,
            'last_checkpoint': last_checkpoint,
            'error': error,
        },
    }


def build_sweep_manifest(
    *,
    config: dict[str, Any],
    grid: dict[str, list[Any]],
    seed: int,
    updates: int,
    objective: str,
    data_glob: str,
    development_glob: str,
    development_max_bytes: int,
    log_every: int,
    save_every: int,
    sample_every: int | None,
    sample_prompt: str,
    sample_seed: int,
    sample_bytes: int,
    eval_data: str | None,
    eval_every: int | None,
    eval_max_bytes: int,
    started_at: str,
    status: RunStatus = 'running',
    completed_runs: int = 0,
    current_run: int | None = None,
    error: RunError | None = None,
    data_files: list[str] | None = None,
    evaluation_files: list[str] | None = None,
    development_files: list[str] | None = None,
) -> SweepManifest:
    """Build a sweep manifest in the stable JSON field order."""
    model, hparams = _grouped_settings(config)
    model.update(trainable_parameters=None, temporal_gradient=None)

    return {
        'model': model,
        'hparams': hparams,
        'grid': grid,
        'data': {
            'train': {
                'glob': data_glob,
                'files': [] if data_files is None else data_files,
            },
            'development': {
                'glob': development_glob,
                'files': [] if development_files is None else development_files,
                'max_bytes': development_max_bytes,
            },
            'evaluation': None
            if eval_data is None
            else {
                'glob': eval_data,
                'files': [] if evaluation_files is None else evaluation_files,
                'max_bytes': eval_max_bytes,
            },
        },
        'training': {
            'seed': seed,
            'updates': updates,
            'objective': objective,
            'resume': False,
        },
        'monitoring': {
            'log_every': log_every,
            'save_every': save_every,
            'sample': None
            if sample_every is None
            else {
                'every': sample_every,
                'prompt': sample_prompt,
                'seed': sample_seed,
                'bytes': sample_bytes,
            },
            'evaluation': None
            if eval_every is None
            else {'every': eval_every},
        },
        'progress': {
            'completed_runs': completed_runs,
            'current_run': current_run,
        },
        'run': {'started_at': started_at, 'status': status, 'error': error},
    }


def read_json(path: str | PathLike[str]) -> Any:
    """Read one JSON value and name the file in a syntax error."""
    with open(path) as file:
        try:
            return json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f'invalid JSON in {path}: {error}') from error


def write_json(
    path: str | PathLike[str], record: object, *, exclusive: bool = False
) -> None:
    """Write indented JSON followed by a newline."""
    with open(path, 'x' if exclusive else 'w') as file:
        json.dump(record, file, indent=2)
        file.write('\n')


def append_jsonl(path: str | PathLike[str], row: object) -> None:
    """Append one compact JSON value followed by a newline."""
    with open(path, 'a') as file:
        file.write(json.dumps(row) + '\n')
