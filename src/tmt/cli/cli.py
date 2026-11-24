from __future__ import annotations

import tmt.cli.commands as commands
import tmt.cli.handlers as handlers
import tmt.util.validation as validation

import argparse
import sys
import json

from typing import Sequence

def version() -> str:
    from importlib.metadata import distribution

    dist = distribution('test-model-thing')
    direct_url = dist.read_text('direct_url.json')

    commit = None
    if direct_url: commit = json.loads(direct_url).get('vcs_info', {}).get('commit_id')

    return f'tmt {dist.version}, commit {commit or "unknown"}'

# cmd manual

def manual(topic: str | None = None) -> str:
    from importlib.resources import files
    manual = files('tmt').joinpath('COMMANDS.md').read_text(encoding = 'utf-8')

    if topic is None: return manual
    if topic not in tuple(cmd.name for cmd in commands.COMMANDS): raise ValueError(f'unknown help topic: {topic}')

    heading = f'## {topic}\n'
    section = manual.split(heading, 1)[1].split('\n## ', 1)[0]
    return heading + section.rstrip() + '\n'

HELP_FLAGS = ('-h', '--help')
has_help_flag = lambda args: any(flag in args for flag in HELP_FLAGS)

def get_manual(argv: Sequence[str]) -> tuple[bool, str | None]:
    if not argv: return False, None
    first, rest = argv[0], argv[1:]

    if first == 'help':
        if has_help_flag(rest): return True, 'help'
        return True, rest[0] if rest else None

    if first in HELP_FLAGS:
        if rest and not rest[0].startswith('-'): return True, rest[0]
        return False, None

    if not first.startswith('-') and has_help_flag(rest): return True, first
    return False, None

def route_manual(argv: Sequence[str], parser: argparse.ArgumentParser) -> bool:
    if argv[:1] == ['help'] and len(argv) > 2 and not has_help_flag(argv):
        parser.error('help accepts at most one topic')

    requested, topic = get_manual(argv)
    if not requested: return False

    try: print(manual(topic), end = '')
    except ValueError as e: parser.error(str(e))

    return True

# errors

SHARED = ('train', 'sweep')

def check_misplaced(argv: Sequence[str], parser: argparse.ArgumentParser) -> None:
    if not argv or not argv[0].startswith('--'): return
    name = argv[0].split('=', 1)[0]

    cmds = [
        cmd.name for cmd in commands.COMMANDS
        if name in cmd.options((commands.TRAIN_BASE, ) if cmd.name in SHARED else ())
    ]
    if not cmds: return

    parser.error(f'{name} must come after the command name, commands that accept it are: {', '.join(cmds)}')

# final setup

def setup() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(**commands.METADATA)
    parser.add_argument('--version', action = 'version', version = version())

    subparser = parser.add_subparsers(dest = 'command', required = True)

    train_base = argparse.ArgumentParser(add_help = False)
    commands.argument(train_base, commands.TRAIN_BASE)

    handlers.HANDLERS['help'] = lambda *_: print(manual(), end = '')

    for cmd in commands.COMMANDS:
        commands.register(
            subparser, cmd,
            [train_base] if cmd.name in SHARED else [],
            handlers.HANDLERS[cmd.name]
        )

    return parser

SILENT = ('help', 'inspect', '--version')

def log_notice(argv: Sequence[str], type: str) -> None:
    if len(argv) > 0 and argv[0] not in SILENT: print(f'tmt: {type} command \'{argv[0]}\'.')

@commands.wrap_interrupt
def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    try:
        parser = setup()
        if route_manual(argv, parser): return
        check_misplaced(argv, parser)

        log_notice(argv, 'running')
        
        args = parser.parse_args(argv)
        args.handler(args, parser)

        log_notice(argv, 'finished')

    except (validation.ValidationError, OSError) as e:
        print(f'tmt: error ({validation.format_type(type(e))}): {e}', file = sys.stderr)
        raise SystemExit(1) from None