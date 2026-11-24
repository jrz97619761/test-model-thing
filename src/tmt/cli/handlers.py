from __future__ import annotations

import tmt.runtime.runtime as runtime
import tmt.runtime.checkpoint as checkpoint
import tmt.runtime.benchmark as benchmark
import tmt.model.tokenizers as tokenizers
import tmt.cli.commands as commands
import tmt.util.validation as validation

import argparse
import glob
import json
from pathlib import Path

# handlers

def handle_init(args: argparse.Namespace, _: argparse.ArgumentParser) -> None:
    updates, max_bytes, log_interval, save_interval, eval_interval, sample_interval, sample_prompt, sample_seed, sample_length = commands.fetch_defaults(
        commands.TRAIN_BASE,
        (
            '--updates', '--max-bytes',
            '--log-interval', '--save-interval', '--eval-interval', '--sample-interval',
            '--sample-prompt', '--sample-seed', '--sample-length'
        )
    )

    example_run = runtime.run_manifest(
        checkpoint.DEFAULT_CONFIG,
        checkpoint.DummyByteFiles('data/train/**'), checkpoint.DummyByteFiles('data/eval/**'),
        11,

        updates, False, (log_interval, save_interval, eval_interval, sample_interval),
        (sample_prompt, sample_seed, sample_length), max_bytes,
        0, 0, 0, None, None, 'example', None, None
    )

    config = checkpoint.group_config(checkpoint.DEFAULT_CONFIG)

    templates = (
        {args.output: config}
        if args.output is not None
        else {
            'model.json': config,
            'grid.json': checkpoint.DEFAULT_GRID,
            'run.example.json': example_run
        }
    )

    for dest, value in templates.items():
        dest = Path(dest)
        dest.parent.mkdir(parents = True, exist_ok = True)

        checkpoint.write_json(dest, value, exclusive = True)

    if args.output is None:
        for directory in ('data/train', 'data/eval', 'data/dev', 'runs'):
            Path(directory).mkdir(parents = True, exist_ok = True)

def handle_tokenizer_convert(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    match args.format:
        case 'huggingface': merges = tokenizers.BPEConverter.huggingface(checkpoint.read_json(args.input))
        case 'tiktoken': merges = tokenizers.BPEConverter.tiktoken(Path(args.input).read_text(encoding = 'utf-8'))

    if not merges: parser.error(f'{args.input} gave no merges, is the tokenizer file empty or invalid?')

    _ = tokenizers.BPE(merges)
    checkpoint.write_json(args.output, {'merges': merges}, exclusive = True)

def handle_train(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    # arg conflicts

    if args.checkpoint is not None and args.run is not None:
        parser.error('checkpoint and --run cannot be used together')

    if args.resume and args.checkpoint is None:
        parser.error('--resume requires an explicit checkpoint and forbids --run')

    if args.resume and args.bpe is not None:
        parser.error('--bpe cannot be used with --resume')

    # validation

    if args.run is not None: validation.file_name(args.run)
    validation.require(glob.glob(args.data, recursive = True), f'no files matched {args.data!r}')

    validation.positive(args.max_bytes, 'max bytes')

    # config loading

    config = tokenizer = None
    if not args.resume:
        config = checkpoint.load_config(args.config or 'model.json')

        if args.bpe is not None:
            tokenizer = checkpoint.read_json(args.bpe)
            tokenizers.BPE(tokenizers.BPEConverter.tmt(tokenizer))

            if not tokenizer['merges']:
                parser.error(f'{args.bpe} gave no merges, is the tokenizer file empty or invalid?')

    # folder handling

    new = args.checkpoint is None

    if new:
        name = args.run or checkpoint.stamp('file')
        folder = Path('runs') / name
        validation.require(not folder.exists(), f'run folder {folder} already exists; choose a new run name')
        checkpoint_path = folder / 'model.safetensors'
    else:
        checkpoint_path = args.checkpoint
        validation.require(args.resume or not Path(checkpoint_path).exists(), f'checkpoint {checkpoint_path} exists; use --resume, or remove it')

    # run

    runtime.TrainingRun(
        checkpoint_path, args.data, args.eval_data, config, tokenizer,
        args.seed, args.updates,
        
        (args.log_interval, args.save_interval, args.eval_interval, args.sample_interval),
        (args.sample_prompt, args.sample_seed, args.sample_length),

        args.resume, args.max_bytes, new
    ).run()

def handle_evaluate(args: argparse.Namespace, _: argparse.ArgumentParser) -> None:
    validation.positive(args.max_bytes, 'max bytes')

    checkpoint.write_json(args.output, benchmark.bpb_score(
        checkpoint.load(args.checkpoint)[0],
        list(checkpoint.ByteFiles.load(args.data, args.max_bytes)),
        args.windows
    ))

def handle_benchmark(args: argparse.Namespace, _: argparse.ArgumentParser) -> None:
    validation.seed(args.seed)
    checkpoint.write_json(args.output, benchmark.benchmark(
        checkpoint.load(args.checkpoint)[0],
        args.cola_data, args.seed,
        args.epochs, args.split
    ))

def handle_generate(args: argparse.Namespace, _: argparse.ArgumentParser) -> None:
    validation.seed(args.seed)
    validation.file_parent(args.output)

    generated = runtime.generate(
        checkpoint.load(args.checkpoint)[0],
        args.prompt, args.steps, args.seed, args.temperature
    )

    with open(args.output, 'wb') as file:
        file.write(generated)

def handle_sweep(args: argparse.Namespace, _: argparse.ArgumentParser) -> None:
    validation.positive(args.max_bytes, 'max bytes')
    
    runtime.sweep(
        args.data, args.eval_data, args.dev_data, args.grid, args.config, args.output,
        args.seed, args.updates,
        
        (args.log_interval, args.save_interval, args.eval_interval, args.sample_interval),
        (args.sample_prompt, args.sample_seed, args.sample_length), args.max_bytes
    )

def handle_inspect(args: argparse.Namespace, _: argparse.ArgumentParser) -> None:
    summary = runtime.inspect(args.checkpoint)
    print(json.dumps(summary))