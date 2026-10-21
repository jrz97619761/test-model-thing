"""Command-line parsing, help routing, and dispatch to runtime operations."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Protocol, Sequence

from . import validation
from .config import load_config
from .records import build_run_manifest, write_json
from .runtime import evaluate_checkpoint, generate, inspect_checkpoint, sweep, train


def version_text() -> str:
    from importlib.metadata import distribution

    installed = distribution('test-model-thing')
    direct_url = installed.read_text('direct_url.json')
    commit = None
    if direct_url:
        commit = json.loads(direct_url).get('vcs_info', {}).get('commit_id')
    return f'tmt {installed.version}, commit {commit or "unknown"}'


def command_manual(topic: str | None = None) -> str:
    from importlib.resources import files

    manual = files('tmt').joinpath('commands.md').read_text(encoding='utf-8')
    if topic is None:
        return manual
    if topic not in HELP_TOPICS:
        raise ValueError(f'unknown help topic: {topic}')
    heading = f'## {topic}\n'
    section = manual.split(heading, 1)[1].split('\n## ', 1)[0]
    return heading + section.rstrip() + '\n'


type ArgumentSpec = tuple[tuple[str, ...], dict[str, Any]]

INIT_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    (
        ('--output',),
        {'help': 'write only model JSON to PATH; default: initialize current folder'},
    ),
)

SHARED_TRAINING_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    (
        ('--data',),
        {
            'required': True,
            'help': 'quoted raw-file train glob, one document per file',
        },
    ),
    (
        ('--updates',),
        {
            'type': int,
            'default': 1000,
            'help': 'target-byte optimizer updates per run; default: 1000',
        },
    ),
    (
        ('--log-every',),
        {
            'type': int,
            'default': 100,
            'help': 'positive loss report interval in updates; default: 100',
        },
    ),
    (
        ('--save-every',),
        {
            'type': int,
            'default': 500,
            'help': 'positive checkpoint save interval in updates; default: 500',
        },
    ),
    (
        ('--seed',),
        {
            'type': int,
            'required': True,
            'help': 'model initialization and file-order seed',
        },
    ),
    (
        ('--ce-only',),
        {
            'action': 'store_true',
            'help': 'next-byte cross entropy only; default: full TMT objective',
        },
    ),
    (
        ('--sample-every',),
        {'type': int, 'help': 'positive sample interval in updates; default: disabled'},
    ),
    (
        ('--sample-prompt',),
        {
            'default': 'The ',
            'help': 'fixed nonempty sample prompt; default: "The "',
        },
    ),
    (
        ('--sample-seed',),
        {'type': int, 'default': 11, 'help': 'fixed sample seed; default: 11'},
    ),
    (
        ('--sample-bytes',),
        {
            'type': int,
            'default': 256,
            'help': 'positive sample byte count; default: 256',
        },
    ),
    (
        ('--eval-data',),
        {'help': 'quoted held-out byte glob; requires --eval-every'},
    ),
    (
        ('--eval-every',),
        {
            'type': int,
            'help': 'positive held-out BPB interval in updates; requires --eval-data',
        },
    ),
    (
        ('--eval-max-bytes',),
        {
            'type': int,
            'default': 8192,
            'help': 'positive held-out input byte budget; default: 8192',
        },
    ),
)

TRAIN_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    (
        ('checkpoint',),
        {
            'nargs': '?',
            'help': 'explicit checkpoint path; required with --resume',
        },
    ),
    (
        ('--run',),
        {
            'help': 'one relative folder name under runs/; '
            'default: UTC timestamp with microseconds; '
            'excludes checkpoint and --resume',
        },
    ),
)

TRAIN_SETTINGS_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    (
        ('--config',),
        {
            'help': 'fresh model JSON; default: model.json in current folder; '
            'excludes --resume',
        },
    ),
    (
        ('--resume',),
        {
            'action': 'store_true',
            'help': 'load checkpoint settings and add --updates with a new data pass; '
            'no exact cursor/RNG resume',
        },
    ),
)

EVALUATE_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    (('checkpoint',), {'help': 'strict model and optimizer checkpoint'}),
    (
        ('--data',),
        {'required': True, 'help': 'quoted raw-file glob, one document per file'},
    ),
    (
        ('--max-bytes',),
        {
            'type': int,
            'default': 8192,
            'help': 'positive global input-byte prefix across sorted files; '
            'default: 8192',
        },
    ),
    (
        ('--windows',),
        {
            'type': int,
            'nargs': '+',
            'default': (),
            'help': 'context lengths in input bytes; default: no windows',
        },
    ),
    (
        ('--output',),
        {'help': 'optional JSON score file; default: terminal only'},
    ),
)

BENCHMARK_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    (('checkpoint',), {'help': 'strict checkpoint; fixed byte suite always runs'}),
    (
        ('--epochs',),
        {
            'type': int,
            'default': 1,
            'help': 'positive CoLA classifier epochs; default: 1',
        },
    ),
    (
        ('--split',),
        {
            'type': float,
            'default': 0.5,
            'help': 'contiguous CoLA train fraction; default: 0.5',
        },
    ),
    (
        ('--cola-data',),
        {'help': 'optional four-column CoLA TSV; default: byte suite only'},
    ),
    (('--seed',), {'type': int, 'required': True, 'help': 'CoLA head seed'}),
    (
        ('--output',),
        {'help': 'optional JSON result file; default: terminal only'},
    ),
)

GENERATE_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    (('checkpoint',), {'help': 'strict checkpoint for frozen generation'}),
    (
        ('--prompt',),
        {'required': True, 'help': 'nonempty text encoded as UTF-8'},
    ),
    (
        ('--output',),
        {'required': True, 'help': 'raw-byte output file without the prompt'},
    ),
    (('--seed',), {'type': int, 'required': True, 'help': 'sample seed'}),
    (
        ('--bytes',),
        {
            'type': int,
            'default': 512,
            'help': 'number of output bytes; default: 512',
        },
    ),
    (
        ('--log-every',),
        {
            'type': int,
            'default': 1000,
            'help': 'positive progress interval in output bytes; default: 1000',
        },
    ),
    (
        ('--temperature',),
        {
            'type': float,
            'help': 'finite nonnegative sampler temperature; effective minimum: 0.1; '
            'default: checkpoint value',
        },
    ),
)

INSPECT_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    (('checkpoint',), {'help': 'strict checkpoint to summarize'}),
)

SWEEP_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    (
        ('--grid',),
        {
            'required': True,
            'help': 'JSON object with nonempty arrays of model values or seed',
        },
    ),
    (
        ('--development',),
        {'required': True, 'help': 'quoted raw-file development glob'},
    ),
    (
        ('--output',),
        {'required': True, 'help': 'sweep folder for run-NNNN and results.jsonl'},
    ),
    (
        ('--config',),
        {
            'default': 'model.json',
            'help': 'base model JSON; default: model.json in current folder',
        },
    ),
    (
        ('--max-bytes',),
        {
            'type': int,
            'default': 8192,
            'help': 'positive global development input-byte prefix; default: 8192',
        },
    ),
)

MONITORING_OPTIONS = (
    'sample_every',
    'sample_prompt',
    'sample_seed',
    'sample_bytes',
    'eval_data',
    'eval_every',
    'eval_max_bytes',
)


class _ArgumentTarget(Protocol):
    def add_argument(self, *names: str, **options: Any) -> argparse.Action: ...


def _register_arguments(
    parser: _ArgumentTarget, specifications: Sequence[ArgumentSpec]
) -> None:
    for names, options in specifications:
        parser.add_argument(*names, **options)


HELP_FLAGS = ('-h', '--help')


def _manual_topic(argv: Sequence[str]) -> tuple[bool, str | None]:
    if not argv:
        return False, None
    first, rest = argv[0], argv[1:]
    if first == 'help':
        if any(flag in rest for flag in HELP_FLAGS):
            return True, 'help'
        return True, rest[0] if rest else None
    if first in HELP_FLAGS:
        if rest and not rest[0].startswith('-'):
            return True, rest[0]
        return False, None
    if not first.startswith('-') and any(flag in rest for flag in HELP_FLAGS):
        return True, first
    return False, None


def _route_manual_request(argv: Sequence[str], parser: argparse.ArgumentParser) -> bool:
    if argv[:1] == ['help'] and len(argv) > 2 and not any(
        flag in argv for flag in HELP_FLAGS
    ):
        parser.error('help accepts at most one topic')
    requested, topic = _manual_topic(argv)
    if not requested:
        return False
    try:
        print(command_manual(topic), end='')
    except ValueError as error:
        parser.error(str(error))
    return True


def _monitoring_options(args: argparse.Namespace) -> dict[str, Any]:
    return {name: getattr(args, name) for name in MONITORING_OPTIONS}


def _option_default(specifications: Sequence[ArgumentSpec], option: str) -> Any:
    return next(
        options['default'] for names, options in specifications if option in names
    )


def _bpb(value: float | None) -> str:
    return 'none' if value is None else f'{value:.6f}'


def _validate_training_args(args: argparse.Namespace) -> None:
    validation.training_options(args.seed, args.updates, args.log_every)
    validation.positive(args.save_every, 'save-every')
    validation.monitoring(**_monitoring_options(args))


def _resolve_train_path(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[dict[str, Any] | None, str | PathLike[str], bool]:
    if args.checkpoint is not None and args.run is not None:
        parser.error('checkpoint and --run cannot be used together')
    if args.resume and (args.checkpoint is None or args.run is not None):
        parser.error('--resume requires an explicit checkpoint and forbids --run')
    if args.run is not None:
        validation.run_name(args.run)
    if not glob.glob(args.data, recursive=True):
        raise FileNotFoundError(f'no files matched {args.data!r}')
    settings = None if args.resume else load_config(args.config or 'model.json')
    new_run = args.checkpoint is None
    checkpoint: str | PathLike[str]
    if new_run:
        name = args.run or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        folder = Path('runs') / name
        if folder.exists():
            raise FileExistsError(
                f'run folder {folder} already exists; choose a new run name'
            )
        checkpoint = folder / 'model.safetensors'
    else:
        checkpoint = args.checkpoint
        if not args.resume and Path(checkpoint).exists():
            raise FileExistsError(
                f'checkpoint {checkpoint} exists; use --resume, or remove it'
            )
    return settings, checkpoint, new_run


def _run_init(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    settings = load_config()
    example_run = build_run_manifest(
        config=settings,
        trainable_parameters=None,
        temporal_gradient=None,
        seed=11,
        data_glob='data/train/*',
        data_files=[],
        updates=_option_default(SHARED_TRAINING_ARGUMENTS, '--updates'),
        log_every=_option_default(SHARED_TRAINING_ARGUMENTS, '--log-every'),
        save_every=_option_default(SHARED_TRAINING_ARGUMENTS, '--save-every'),
        objective='tmt',
    )
    templates = (
        {args.output: settings}
        if args.output is not None
        else {
            'model.json': settings,
            'grid.json': {
                'dim': [4, 8],
                'layers': [1],
                'spread': [4],
                'block': ['tmt-v0'],
                'seed': [11, 22],
            },
            'run.example.json': example_run,
        }
    )
    for destination, value in templates.items():
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, value, exclusive=True)
        print(f'wrote template: {destination}')
    if args.output is None:
        for directory in ('data/train', 'data/dev', 'runs'):
            Path(directory).mkdir(parents=True, exist_ok=True)


def _run_train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _validate_training_args(args)
    settings, checkpoint, new_run = _resolve_train_path(args, parser)
    train(
        settings,
        checkpoint,
        args.data,
        args.updates,
        args.seed,
        ce_only=args.ce_only,
        resume=args.resume,
        log_every=args.log_every,
        new_run=new_run,
        save_every=args.save_every,
        **_monitoring_options(args),
    )


def _run_evaluate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    result = evaluate_checkpoint(
        args.checkpoint, args.data, args.max_bytes, args.windows
    )
    print(
        f'input bytes: {result["input_bytes"]}, targets: {result["targets"]}, '
        f'BPB: {_bpb(result["bpb"])}'
    )
    print(
        f'common targets: {result["common_targets"]}, '
        f'common BPB: {_bpb(result["common_bpb"])}'
    )
    for window in result['windows']:
        delta = window['delta_bpb']
        print(
            f'window {window["window"]}: BPB {_bpb(window["bpb"])}, '
            f'common BPB {_bpb(window["common_bpb"])}, '
            f'delta {"none" if delta is None else f"{delta:+.6f}"}'
        )
    if args.output is not None:
        write_json(args.output, result)


def _run_benchmark(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validation.seed(args.seed)
    from .benchmark import run

    run(
        args.checkpoint,
        args.epochs,
        args.split,
        args.cola_data,
        seed=args.seed,
        output=args.output,
    )


def _run_generate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validation.seed(args.seed)
    validation.positive(args.log_every, 'log-every')
    generate(
        args.checkpoint,
        args.prompt,
        args.output,
        args.seed,
        count=args.bytes,
        temperature=args.temperature,
        log_every=args.log_every,
    )


def _run_sweep(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    _validate_training_args(args)
    sweep(
        args.grid,
        args.data,
        args.development,
        args.output,
        args.config,
        args.updates,
        args.seed,
        ce_only=args.ce_only,
        max_bytes=args.max_bytes,
        log_every=args.log_every,
        save_every=args.save_every,
        **_monitoring_options(args),
    )


def _run_inspect(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    summary = inspect_checkpoint(args.checkpoint)
    print(
        f'checkpoint: {args.checkpoint}, step: {summary["optimizer_step"]}, '
        f'trainable parameters: {summary["trainable_parameters"]}'
    )
    print(f'config: {json.dumps(summary["config"])}')
    run = summary['run']
    if run is None:
        print('run manifest: none')
    else:
        print(
            f'run status: {run["status"]}, objective: {run["objective"]}, '
            f'seed: {run["seed"]}, updates: {run["completed_updates"]}'
        )


def _run_help(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    print(command_manual(), end='')


type CommandHandler = Callable[[argparse.Namespace, argparse.ArgumentParser], None]


@dataclass(frozen=True)
class CommandSpec:
    name: str
    help: str
    handler: CommandHandler
    arguments: tuple[ArgumentSpec, ...] = ()
    shared_training: bool = False
    exclusive: tuple[ArgumentSpec, ...] = ()


COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec('init', 'write default model settings', _run_init, INIT_ARGUMENTS),
    CommandSpec(
        'train',
        'train a model on byte files',
        _run_train,
        TRAIN_ARGUMENTS,
        shared_training=True,
        exclusive=TRAIN_SETTINGS_ARGUMENTS,
    ),
    CommandSpec(
        'evaluate', 'score next-byte predictions', _run_evaluate, EVALUATE_ARGUMENTS
    ),
    CommandSpec(
        'benchmark',
        'run byte and CoLA benchmarks',
        _run_benchmark,
        BENCHMARK_ARGUMENTS,
    ),
    CommandSpec(
        'generate',
        'generate a fixed number of bytes',
        _run_generate,
        GENERATE_ARGUMENTS,
    ),
    CommandSpec(
        'sweep',
        'run a sequential model grid',
        _run_sweep,
        SWEEP_ARGUMENTS,
        shared_training=True,
    ),
    CommandSpec(
        'inspect',
        'show checkpoint state and run status',
        _run_inspect,
        INSPECT_ARGUMENTS,
    ),
    CommandSpec('help', 'read the bundled command manual or one topic', _run_help),
)
HELP_TOPICS = tuple(command.name for command in COMMANDS)


def _command_options(command: CommandSpec) -> tuple[str, ...]:
    shared = SHARED_TRAINING_ARGUMENTS if command.shared_training else ()
    return tuple(
        name
        for names, _ in (*shared, *command.arguments, *command.exclusive)
        for name in names
        if name.startswith('--')
    )


def _misplaced_option_error(argv: Sequence[str]) -> str | None:
    """Explain a command option that comes before the command name."""
    if not argv or not argv[0].startswith('--'):
        return None
    name = argv[0].split('=', 1)[0]
    commands = [
        command.name for command in COMMANDS if name in _command_options(command)
    ]
    if not commands:
        return None
    return (
        f'{name} must come after the command name; '
        f'commands that accept it: {", ".join(commands)}'
    )


def build_parser() -> argparse.ArgumentParser:
    training_options = argparse.ArgumentParser(add_help=False)
    _register_arguments(training_options, SHARED_TRAINING_ARGUMENTS)

    parser = argparse.ArgumentParser(
        prog='tmt',
        description='TMT experiment commands',
        epilog='Use tmt help for the complete command manual.',
    )
    parser.add_argument('--version', action='version', version=version_text())
    commands = parser.add_subparsers(dest='command', required=True)
    for command in COMMANDS:
        subparser = commands.add_parser(
            command.name,
            parents=[training_options] if command.shared_training else [],
            help=command.help,
        )
        _register_arguments(subparser, command.arguments)
        if command.exclusive:
            _register_arguments(
                subparser.add_mutually_exclusive_group(), command.exclusive
            )
        subparser.set_defaults(handler=command.handler)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        parser = build_parser()
        if _route_manual_request(argv, parser):
            return
        misplaced = _misplaced_option_error(argv)
        if misplaced is not None:
            parser.error(misplaced)
        args = parser.parse_args(argv)
        args.handler(args, parser)
    except (OSError, ValueError) as error:
        print(f'tmt: error: {error}', file=sys.stderr)
        raise SystemExit(1) from None


def console_main(argv: Sequence[str] | None = None) -> None:
    """Run main() as the process entry point.

    Ctrl-C prints one line instead of a traceback. The KeyboardInterrupt still
    propagates, so Python finalizes and then exits through SIGINT, as shells expect.
    """
    try:
        main(argv)
    except KeyboardInterrupt:
        previous_hook = sys.excepthook

        def report_interrupt(kind, value, traceback) -> None:
            if issubclass(kind, KeyboardInterrupt):
                print('tmt: interrupted', file=sys.stderr)
            else:
                previous_hook(kind, value, traceback)

        sys.excepthook = report_interrupt
        raise


if __name__ == '__main__':
    console_main()
