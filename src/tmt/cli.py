from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Sequence

from . import validation
from .checkpoint import (
    load_checkpoint,
    load_model,
    new_model,
    save_checkpoint,
)
from .config import load_config
from .records import build_run_manifest, write_json
from .runtime import (
    _training_sample,
    documents,
    evaluate_checkpoint,
    generate,
    sweep,
    train,
)


def evaluate_command(args: argparse.Namespace) -> None:
    validation.positive(args.max_bytes, 'max-bytes')
    for window in args.windows:
        validation.positive(window, 'evaluation windows')
    result = evaluate_checkpoint(
        args.checkpoint, args.data, args.max_bytes, args.windows
    )
    print(
        f'input bytes: {result["input_bytes"]}, targets: {result["targets"]}, '
        f'BPB: {result["bpb"]}'
    )
    print(
        f'common targets: {result["common_targets"]}, '
        f'common BPB: {result["common_bpb"]}'
    )
    for window in result['windows']:
        print(
            f'window {window["window"]}: BPB {window["bpb"]}, '
            f'common BPB {window["common_bpb"]}, delta {window["delta_bpb"]}'
        )
    if args.output is not None:
        write_json(args.output, result)


def init_config(path: str | PathLike[str] | None = None) -> None:
    settings = load_config()
    example_run = build_run_manifest(
        config=settings,
        trainable_parameters=None,
        temporal_gradient=None,
        seed=11,
        data_glob='data/train/*',
        data_files=[],
        updates=1000,
        log_every=100,
        objective='tmt',
    )
    templates = (
        {path: settings}
        if path is not None
        else {
            'model.json': settings,
            'grid.json': {
                'dim': [4, 8],
                'layers': [1],
                'spread': [4],
                'block': ['rtu'],
                'seed': [11, 22],
            },
            'run.example.json': example_run,
        }
    )
    for destination, value in templates.items():
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, value, exclusive=True)
        print(f'Wrote template to {destination}')
    if path is None:
        for directory in ('data/train', 'data/dev', 'runs'):
            Path(directory).mkdir(parents=True, exist_ok=True)


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
    if topic not in (
        'init',
        'train',
        'sweep',
        'evaluate',
        'benchmark',
        'generate',
        'help',
    ):
        raise ValueError(f'unknown help topic: {topic}')
    heading = f'## {topic}\n'
    section = manual.split(heading, 1)[1].split('\n## ', 1)[0]
    return heading + section.rstrip() + '\n'


type ArgumentSpec = tuple[tuple[str, ...], dict[str, Any]]

TRAINING_ARGUMENTS: tuple[ArgumentSpec, ...] = (
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
            'help': 'target-byte optimizer updates per run, default: 1000',
        },
    ),
    (
        ('--log-every',),
        {
            'type': int,
            'default': 100,
            'help': 'positive loss report interval in updates, default: 100',
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
            'help': 'fixed nonempty sample prompt, default: "The "',
        },
    ),
    (
        ('--sample-seed',),
        {'type': int, 'default': 11, 'help': 'fixed sample seed, default: 11'},
    ),
    (
        ('--sample-bytes',),
        {
            'type': int,
            'default': 256,
            'help': 'positive sample byte count, default: 256',
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
            'help': 'positive held-out input byte budget, default: 8192',
        },
    ),
)

EVALUATION_ARGUMENTS: tuple[ArgumentSpec, ...] = (
    (
        ('--data',),
        {'required': True, 'help': 'quoted raw-file glob, one document per file'},
    ),
    (
        ('--max-bytes',),
        {
            'type': int,
            'default': 8192,
            'help': 'positive global input-byte prefix across sorted files, default: 8192',
        },
    ),
    (
        ('--windows',),
        {
            'type': int,
            'nargs': '+',
            'default': [],
            'help': 'context lengths in input bytes; default: no windows',
        },
    ),
    (
        ('--output',),
        {'help': 'optional JSON score file; default: terminal only'},
    ),
)


def _register_arguments(
    parser: argparse.ArgumentParser, specifications: Sequence[ArgumentSpec]
) -> None:
    for names, options in specifications:
        parser.add_argument(*names, **options)


def build_parser() -> argparse.ArgumentParser:
    training_options = argparse.ArgumentParser(add_help=False)
    _register_arguments(training_options, TRAINING_ARGUMENTS)

    parser = argparse.ArgumentParser(
        prog='tmt',
        description='TMT experiment commands',
        epilog='Use tmt help for the complete command manual.',
    )
    parser.add_argument('--version', action='version', version=version_text())
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='write default model settings')
    init.add_argument(
        '--output',
        help='write only model JSON to PATH; default: initialize current folder',
    )
    training = commands.add_parser(
        'train', parents=[training_options], help='train a model on byte files'
    )
    training.add_argument(
        'checkpoint', nargs='?', help='explicit checkpoint path; required with --resume'
    )
    training.add_argument(
        '--run',
        help='one relative folder name under runs/; '
        'default: UTC timestamp with microseconds; excludes checkpoint and --resume',
    )
    settings = training.add_mutually_exclusive_group()
    settings.add_argument(
        '--config',
        help='fresh model JSON, default: model.json in current folder; '
        'excludes --resume',
    )
    settings.add_argument(
        '--resume',
        action='store_true',
        help='load checkpoint settings and add --updates with a new data pass; '
        'no exact cursor/RNG resume',
    )
    evaluation = commands.add_parser('evaluate', help='score next-byte predictions')
    evaluation.add_argument('checkpoint', help='strict model and optimizer checkpoint')
    _register_arguments(evaluation, EVALUATION_ARGUMENTS)
    benchmark = commands.add_parser('benchmark', help='run byte and CoLA benchmarks')
    benchmark.add_argument(
        'checkpoint', help='strict checkpoint; fixed byte suite always runs'
    )
    benchmark.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='positive CoLA classifier epochs, default: 1',
    )
    benchmark.add_argument(
        '--split',
        type=float,
        default=0.5,
        help='contiguous CoLA train fraction, default: 0.5',
    )
    benchmark.add_argument(
        '--cola-data', help='optional four-column CoLA TSV; default: byte suite only'
    )
    benchmark.add_argument(
        '--seed', type=int, required=True, help='CoLA head seed'
    )
    generation = commands.add_parser(
        'generate', help='generate a fixed number of bytes'
    )
    generation.add_argument(
        'checkpoint', help='strict checkpoint for frozen generation'
    )
    generation.add_argument(
        '--prompt', required=True, help='nonempty text encoded as UTF-8'
    )
    generation.add_argument(
        '--output', required=True, help='raw-byte output file without the prompt'
    )
    generation.add_argument(
        '--seed', type=int, required=True, help='sample seed'
    )
    generation.add_argument(
        '--bytes', type=int, default=512, help='number of output bytes, default: 512'
    )
    generation.add_argument(
        '--log-every',
        type=int,
        default=1000,
        help='positive progress interval in output bytes, default: 1000',
    )
    generation.add_argument(
        '--temperature',
        type=float,
        help='finite nonnegative sampler temperature; effective minimum: 0.1; '
        'default: checkpoint value',
    )
    sweeping = commands.add_parser(
        'sweep', parents=[training_options], help='run a sequential model grid'
    )
    sweeping.add_argument(
        '--grid',
        required=True,
        help='JSON object with nonempty arrays of model values or seed',
    )
    sweeping.add_argument(
        '--development', required=True, help='quoted raw-file development glob'
    )
    sweeping.add_argument(
        '--output', required=True, help='sweep folder for run-NNNN and results.jsonl'
    )
    sweeping.add_argument(
        '--config',
        default='model.json',
        help='base model JSON, default: model.json in current folder',
    )
    sweeping.add_argument(
        '--max-bytes',
        type=int,
        default=8192,
        help='positive global development input-byte prefix, default: 8192',
    )
    commands.add_parser('help', help='read the bundled command manual or one topic')
    return parser


def _route_manual_request(argv: Sequence[str], parser: argparse.ArgumentParser) -> bool:
    help_flags = ('-h', '--help')
    if argv and (
        argv[0] == 'help'
        or (len(argv) > 1 and any(flag in argv for flag in help_flags))
    ):
        if argv[0] == 'help':
            if any(flag in argv for flag in help_flags):
                topic = 'help'
            else:
                if len(argv) > 2:
                    parser.error('help accepts at most one topic')
                topic = argv[1] if len(argv) == 2 else None
        else:
            topic = argv[1] if argv[0] in help_flags else argv[0]
        try:
            print(command_manual(topic), end='')
        except ValueError as error:
            parser.error(str(error))
        return True
    return False


def _validate_namespace(args: argparse.Namespace) -> dict[str, Any]:
    if hasattr(args, 'seed'):
        validation.seed(args.seed)
    if args.command in ('train', 'sweep'):
        validation.seed(args.sample_seed, 'sample-seed')
        validation.positive(args.updates, 'updates')
        monitoring = {
            name: getattr(args, name)
            for name in (
                'sample_every',
                'sample_prompt',
                'sample_seed',
                'sample_bytes',
                'eval_data',
                'eval_every',
                'eval_max_bytes',
            )
        }
        validation.monitoring(
            **{
                name: value
                for name, value in monitoring.items()
                if name != 'sample_seed'
            }
        )
    if args.command in ('train', 'sweep', 'generate'):
        validation.positive(args.log_every, 'log-every')
    if args.command in ('train', 'sweep'):
        return monitoring
    return {}


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
    return settings, checkpoint, new_run


def _dispatch_init(args: argparse.Namespace) -> None:
    init_config(args.output)


def _dispatch_train(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    monitoring: dict[str, Any],
) -> None:
    settings, checkpoint, new_run = _resolve_train_path(args, parser)
    train(
        settings,
        checkpoint,
        args.data,
        args.updates,
        args.seed,
        args.ce_only,
        args.resume,
        args.log_every,
        new_run=new_run,
        **monitoring,
    )


def _dispatch_evaluate(args: argparse.Namespace) -> None:
    evaluate_command(args)


def _dispatch_benchmark(args: argparse.Namespace) -> None:
    from tmt.benchmark import run

    run(args.checkpoint, args.epochs, args.split, args.cola_data, seed=args.seed)


def _dispatch_generate(args: argparse.Namespace) -> None:
    generate(
        args.checkpoint,
        args.prompt,
        args.output,
        args.seed,
        args.bytes,
        args.temperature,
        args.log_every,
    )


def _dispatch_sweep(args: argparse.Namespace, monitoring: dict[str, Any]) -> None:
    sweep(
        args.grid,
        args.data,
        args.development,
        args.output,
        args.config,
        args.updates,
        args.seed,
        args.ce_only,
        args.max_bytes,
        args.log_every,
        **monitoring,
    )


def _dispatch_command(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    monitoring: dict[str, Any],
) -> None:
    if args.command == 'init':
        _dispatch_init(args)
    elif args.command == 'train':
        _dispatch_train(args, parser, monitoring)
    elif args.command == 'evaluate':
        _dispatch_evaluate(args)
    elif args.command == 'benchmark':
        _dispatch_benchmark(args)
    elif args.command == 'generate':
        _dispatch_generate(args)
    elif args.command == 'sweep':
        _dispatch_sweep(args, monitoring)


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        parser = build_parser()
        if _route_manual_request(argv, parser):
            return
        args = parser.parse_args(argv)
        monitoring = _validate_namespace(args)
        _dispatch_command(args, parser, monitoring)
    except (OSError, ValueError) as error:
        print(f'tmt: error: {error}', file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == '__main__':
    main()
