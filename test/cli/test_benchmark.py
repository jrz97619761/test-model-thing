import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    'command',
    [
        ['-m', 'tmt', 'benchmark'],
        ['-m', 'tmt.benchmark'],
    ],
)
def test_cli_rejects_malformed_row_without_scores(tmp_path, command):
    data = tmp_path / 'cola.tsv'
    data.write_text('src\t1\tx\talpha\nsrc\t0\tx\tbeta\nsrc\t1\tx\n', encoding='utf-8')

    result = subprocess.run(
        [sys.executable, *command, 'missing.safetensors', '--cola-data', str(data), '--epochs', '1'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 1, result.stderr
    assert 'CoLA TSV row 3 has 3 columns; expected 4' in result.stderr
    assert 'Traceback' not in result.stderr
    assert result.stdout == ''
