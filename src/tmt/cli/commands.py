from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

# setup

type Arguments = dict[str, dict[str, Any]]
type Handler = Callable[[argparse.Namespace, argparse.ArgumentParser], None]

@dataclass(frozen = True)
class Command:
    name: str
    help: str
    args: tuple[Arguments, ...] = ()
    exclusive_args: tuple[Arguments, ...] = ()

    def options(self, shared: tuple[Arguments, ...]) -> tuple[str, ...]:
        return tuple(
            name for name in (
                *{key for arg in shared for key in arg},
                *{key for arg in self.args for key in arg},
                *{key for arg in self.exclusive_args for key in arg}
            )
            if name.startswith('--')
        )

def argument(parser: argparse.ArgumentParser | argparse._MutuallyExclusiveGroup, args: Arguments) -> None:
    for name, params in args.items():
        parser.add_argument(name, **params)

def nothing(args: argparse.Namespace, _: argparse.ArgumentParser) -> None: print(args.command)

def register(subparser: argparse._SubParsersAction[argparse.ArgumentParser], command: Command, parents: list[argparse.ArgumentParser], handler: Handler = nothing) -> None:
    parser = subparser.add_parser(command.name, help = command.help, parents = parents)
    for args in command.args: argument(parser, args)
    for args in command.exclusive_args: argument(parser.add_mutually_exclusive_group(), args)
    parser.set_defaults(handler = handler)

def fetch_defaults(args: Arguments, names: tuple[str, ...]) -> tuple[Any]:
    defaults = []
    for name in names: defaults.append(next(options['default'] for key, options in args.items() if key == name))

    return tuple(defaults)

# decorator

def wrap_interrupt(fn):
    def wrapper(*args, **kwargs):
        try: fn(*args, **kwargs)

        except KeyboardInterrupt:
            import sys
            previous_hook = sys.excepthook

            def interrupted(kind, value, traceback) -> None:
                if issubclass(kind, KeyboardInterrupt): print('tmt: interrupted', file = sys.stderr)
                else: previous_hook(kind, value, traceback)

            sys.excepthook = interrupted
            raise

    return wrapper

# arguments

METADATA: dict[str, Any] = {
    'prog': 'tmt',
    'description': 'TMT experiment commands',
    'epilog': 'Use tmt help for the complete command manual.'
}

INIT: Arguments = {
    '--output': {
        'help': 'write only model JSON to PATH; default: initialize current folder'
    }
}

TOKENIZER_CONVERT: Arguments = {
    'format': {
        'choices': ('huggingface', 'tiktoken'),
        'help': 'source tokenizer format',
    },

    'input': {
        'help': 'source tokenizer file'
    },
    'output': {
        'help': 'new TMT tokenizer JSON file'
    },
}

TRAIN_BASE: Arguments = {
    '--seed': {
        'type': int,
        'required': True,
        'help': 'model initialization and file-order seed'
    },

    '--data': {
        'default': 'data/train/**',
        'help': 'quoted raw-file train glob, one document per file'
    },
    '--eval-data': {
        'default': 'data/eval/**',
        'help': 'quoted held-out train glob, one document per file'
    },

    '--updates': {
        'type': int,
        'default': 10000,
        'help': 'target-token optimizer updates per run; default: 10000'
    },
    '--max-bytes': {
        'type': int,
        'default': 8192,
        'help': 'positive input byte budget (held-out and sweep development); default: 8192'
    },

    '--log-interval': {
        'type': int,
        'default': 100,
        'help': 'positive loss report interval in updates; default: 100'
    },
    '--save-interval': {
        'type': int,
        'default': 500,
        'help': 'positive checkpoint save interval in updates; default: 500'
    },
    '--eval-interval': {
        'type': int,
        'default': 1000,
        'help': 'positive held-out BPB interval in updates; default: 1000'
    },
    '--sample-interval': {
        'type': int,
        'default': 500,
        'help': 'maximum sample interval in updates; default: 500'
    },

    '--sample-prompt': {
        'default': 'The ',
        'help': 'fixed nonempty sample prompt; default: \'The \''
    },
    '--sample-seed': {
        'type': int,
        'default': 11,
        'help': 'fixed sample seed; default: 11'
    },
    '--sample-length': {
        'type': int,
        'default': 256,
        'help': 'positive sample token count; default: 256'
    }
}

TRAIN: Arguments = {
    'checkpoint': {
        'nargs': '?',
        'help': 'explicit checkpoint path; required with --resume'
    },

    '--run': {
        'help': 'one relative folder name under runs/; default: UTC timestamp with microseconds; excludes checkpoint and --resume'
    },
    '--bpe': {
        'help': 'BPE tokenizer JSON for a fresh run; default: byte mode'
    },
}

TRAIN_SETTINGS: Arguments = {
    '--config': {
        'help': 'fresh model JSON; default: model.json in current folder; excludes --resume'
    },

    '--resume': {
        'action': 'store_true',
        'help': 'load checkpoint settings and add --updates with a new data pass; no exact cursor/RNG resume'
    }
}

EVAL: Arguments = {
    'checkpoint': {
        'help': 'strict model and optimizer checkpoint'
    },
    'output': {
        'help': 'JSON score file'
    },

    '--data': {
        'default': 'data/eval/**',
        'help': 'quoted raw-file glob, one document per file'
    },

    '--max-bytes': {
        'type': int,
        'default': 8192,
        'help': 'positive global input-byte prefix across sorted files; default: 8192'
    },
    '--windows': {
        'type': int,
        'nargs': '+',
        'default': (),
        'help': 'context lengths in input bytes; default: (1, 8, 32, 128)'
    }
}

BENCHMARK: Arguments = {
    'checkpoint': {
        'help': 'strict checkpoint; fixed byte suite always runs'
    },
    'output': {
        'help': 'JSON result file'
    },

    'cola_data': {
        'help': 'four-column CoLA TSV path'
    },
    
    '--seed': {
        'type': int,
        'required': True,
        'help': 'CoLA head seed'
    },

    '--epochs': {
        'type': int,
        'default': 1,
        'help': 'positive CoLA classifier epochs; default: 1'
    },
    '--split': {
        'type': float,
        'default': 0.5,
        'help': 'contiguous CoLA train fraction; default: 0.5'
    },
}

GENERATE: Arguments = {
    'checkpoint': {
        'help': 'strict checkpoint for frozen generation'
    },
    'output': {
        'help': 'raw-byte output file without the prompt'
    },

    'prompt': {
        'help': 'nonempty text encoded as UTF-8'
    },
    '--seed': {
        'type': int,
        'required': True,
        'help': 'sample seed'
    },

    '--steps': {
        'type': int,
        'default': 512,
        'help': 'maximum output steps; default: 512'
    },
    '--temperature': {
        'type': float,
        'default': -1.0,
        'help': 'finite nonnegative sampler temperature; effective minimum: 0.1; default: checkpoint value'
    }
}

INSPECT: Arguments = {
    'checkpoint': {
        'help': 'strict checkpoint to summarize'
    }
}

SWEEP: Arguments = {
    'grid': {
        'help': 'JSON object with nonempty arrays of model values or seed'
    },
    'output': {
        'help': 'sweep folder for run-NNNN and results.jsonl'
    },

    '--dev-data': {
        'default': 'data/dev/**',
        'help': 'quoted raw-file development glob'
    },
    '--config': {
        'default': 'model.json',
        'help': 'base model JSON; default: model.json in current folder'
    }
}

# commands

COMMANDS: list[Command] = [
    Command('init', 'write default model settings', (INIT, )),
    Command('tokenizer-convert', 'convert external byte BPE merges', (TOKENIZER_CONVERT, )),
    Command('train', 'train a model on byte files', (TRAIN, ), (TRAIN_SETTINGS, )),
    Command('evaluate', 'score next-token predictions', (EVAL, )),
    Command('benchmark', 'run byte and CoLA benchmarks', (BENCHMARK, )),
    Command('generate', 'generate bytes until stop or byte limit', (GENERATE, )),
    Command('sweep', 'run a sequential model grid', (SWEEP, )),
    Command('inspect', 'show checkpoint state and run status', (INSPECT, )),
    Command('help', 'read the bundled command manual or one topic')
]