import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    'command',
    [
        ['-m', 'tmt', 'benchmark'],
        ['-m', 'tmt.benchmark'],
    ],
    ids=['main-module', 'benchmark-module'],
)
def test_benchmark_rejects_malformed_cola_row_without_scores(tmp_path, command):
    data = tmp_path / 'cola.tsv'
    data.write_text('src\t1\tx\talpha\nsrc\t0\tx\tbeta\nsrc\t1\tx\n', encoding='utf-8')

    result = subprocess.run(
        [
            sys.executable,
            *command,
            'missing.safetensors',
            '--cola-data',
            str(data),
            '--epochs',
            '1',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1, result.stderr
    assert 'CoLA TSV row 3 has 3 columns; expected 4' in result.stderr
    assert 'Traceback' not in result.stderr
    assert result.stdout == ''


def test_benchmark_entry_points_report_same_byte_scores(
    saved_checkpoint, cli_runner, tmp_path
):
    checkpoint = saved_checkpoint.path
    expected = cli_runner('benchmark', str(checkpoint), success=True)
    result = subprocess.run(
        [sys.executable, '-m', 'tmt.benchmark', str(checkpoint)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == expected.stdout, (
        'module entry point changed benchmark scores'
    )
    assert result.stderr == expected.stderr, (
        'module entry point changed benchmark diagnostics'
    )
    assert '390 targets' in result.stdout, 'fixed byte benchmark result is missing'
