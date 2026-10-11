import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tmt.benchmark import cola, run


def test_cola_rejects_malformed_rows_before_work(subtests):
    cases = (
        ('src\t1\tx', 'has 3 columns; expected 4'),
        ('src\t1\tx\talpha\textra', 'has 5 columns; expected 4'),
        ('src\t1\tx\talpha\t', 'has 5 columns; expected 4'),
        ('\tsrc\t1\tx\talpha', 'has 5 columns; expected 4'),
        ('src\t1\tx\t', 'has an empty sentence'),
        ('src\t1\tx\t   ', 'has an empty sentence'),
    )
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / 'cola.tsv'
        for row, error in cases:
            with subtests.test(row=row):
                data.write_text('src\t1\tx\talpha\n\nsrc\t0\tx\tbeta\n' + row + '\n', encoding='utf-8')
                model = Mock()
                with patch('tmt.cli.load_model') as load_model, patch('tmt.benchmark.evaluate') as evaluate:
                    for supplied in (None, model):
                        with pytest.raises(ValueError, match='CoLA TSV row 4 ' + error):
                            run('missing.safetensors', data=str(data), model=supplied)
                    load_model.assert_not_called()
                    evaluate.assert_not_called()
                    model.freeze.assert_not_called()


def test_cola_preserves_fields_and_sentence_bytes():
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / 'cola.tsv'
        data.write_bytes('\r\n  \r\n\t1\t\t  café  \r\nsrc\t0\tx\tbeta'.encode('utf-8'))
        assert cola(str(data)) == [('  café  '.encode('utf-8'), 1), (b'beta', 0)]


def test_commands_reject_malformed_row_without_scores(subtests):
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / 'cola.tsv'
        data.write_text('src\t1\tx\talpha\nsrc\t0\tx\tbeta\nsrc\t1\tx\n', encoding='utf-8')
        for command in (['-m', 'tmt', 'benchmark'], ['-m', 'tmt.benchmark']):
            with subtests.test(command=command):
                result = subprocess.run(
                    [sys.executable, *command, 'missing.safetensors', '--cola-data', str(data), '--epochs', '1'],
                    cwd=directory, capture_output=True, text=True, timeout=60,
                )
                assert result.returncode == 1, result.stderr
                assert 'CoLA TSV row 3 has 3 columns; expected 4' in result.stderr
                assert 'Traceback' not in result.stderr
                assert result.stdout == ''


def test_cola_rejects_nonbinary_label_with_tsv_row():
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / 'cola.tsv'
        data.write_text(
            'src\t1\tx\talpha\n'
            'src\t0\tx\tbeta\n'
            'src\t2\tx\tgamma\n'
            'src\t2\tx\tdelta\n',
            encoding='utf-8',
        )

        with pytest.raises(ValueError, match=r'row 3 has invalid label .2.; expected 0 or 1'):
            cola(str(data))


def test_cola_rejects_noninteger_label_with_tsv_row():
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / 'cola.tsv'
        data.write_text('src\t1\tx\talpha\nsrc\tno\tx\tbeta\n', encoding='utf-8')

        with pytest.raises(ValueError, match=r'row 2 has invalid label .no.; expected 0 or 1'):
            cola(str(data))


def test_run_validates_labels_before_model_or_benchmark_work():
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / 'cola.tsv'
        data.write_text(
            'src\t1\tx\talpha\n'
            'src\t0\tx\tbeta\n'
            'src\t2\tx\tgamma\n'
            'src\t2\tx\tdelta\n',
            encoding='utf-8',
        )

        with patch('tmt.cli.load_model') as load_model, patch('tmt.benchmark.evaluate') as evaluate:
            with pytest.raises(ValueError, match=r'row 3 has invalid label .2.; expected 0 or 1'):
                run('missing.safetensors', data=str(data))

            load_model.assert_not_called()
            evaluate.assert_not_called()
