"""Typed manifest records and JSON output helpers."""

from __future__ import annotations

import json
from os import PathLike
from typing import Any, Literal, TypedDict


RunStatus = Literal['running', 'complete', 'failed', 'interrupted']


class RunManifest(TypedDict):
    config: dict[str, Any]
    trainable_parameters: int | None
    temporal_gradient: list[str] | None
    seed: int
    data_glob: str
    data_files: list[str]
    updates: int
    log_every: int
    objective: str
    completed_updates: int
    resume: bool
    initial_optimizer_step: int
    sample: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    started_at: str | None
    status: RunStatus
    last_checkpoint: str | None


class SweepManifest(TypedDict):
    config: dict[str, Any]
    grid: dict[str, list[Any]]
    seed: int
    updates: int
    objective: str
    data_glob: str
    development_glob: str
    development_max_bytes: int
    log_every: int
    sample_every: int | None
    sample_prompt: str
    sample_seed: int
    sample_bytes: int
    eval_data: str | None
    eval_every: int | None
    eval_max_bytes: int
    started_at: str
    status: RunStatus
    completed_runs: int
    current_run: int | None
    error: dict[str, str] | None


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
    completed_updates: int = 0,
    resume: bool = False,
    initial_optimizer_step: int = 0,
    sample: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    started_at: str | None = None,
    status: RunStatus = 'running',
    last_checkpoint: str | None = None,
) -> RunManifest:
    """Build a run manifest in the stable JSON field order."""
    return {
        'config': config,
        'trainable_parameters': trainable_parameters,
        'temporal_gradient': temporal_gradient,
        'seed': seed,
        'data_glob': data_glob,
        'data_files': data_files,
        'updates': updates,
        'log_every': log_every,
        'objective': objective,
        'completed_updates': completed_updates,
        'resume': resume,
        'initial_optimizer_step': initial_optimizer_step,
        'sample': sample,
        'evaluation': evaluation,
        'started_at': started_at,
        'status': status,
        'last_checkpoint': last_checkpoint,
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
    error: dict[str, str] | None = None,
) -> SweepManifest:
    """Build a sweep manifest in the stable JSON field order."""
    return {
        'config': config,
        'grid': grid,
        'seed': seed,
        'updates': updates,
        'objective': objective,
        'data_glob': data_glob,
        'development_glob': development_glob,
        'development_max_bytes': development_max_bytes,
        'log_every': log_every,
        'sample_every': sample_every,
        'sample_prompt': sample_prompt,
        'sample_seed': sample_seed,
        'sample_bytes': sample_bytes,
        'eval_data': eval_data,
        'eval_every': eval_every,
        'eval_max_bytes': eval_max_bytes,
        'started_at': started_at,
        'status': status,
        'completed_runs': completed_runs,
        'current_run': current_run,
        'error': error,
    }


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
