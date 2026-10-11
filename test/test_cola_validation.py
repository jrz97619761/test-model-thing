import pytest

from tmt import benchmark, cli
from tmt.benchmark import cola, run


@pytest.mark.parametrize(
    ('row', 'error'),
    [
        ('src\t1\tx', 'has 3 columns; expected 4'),
        ('src\t1\tx\talpha\textra', 'has 5 columns; expected 4'),
        ('src\t1\tx\talpha\t', 'has 5 columns; expected 4'),
        ('\tsrc\t1\tx\talpha', 'has 5 columns; expected 4'),
        ('src\t1\tx\t', 'has an empty sentence'),
        ('src\t1\tx\t   ', 'has an empty sentence'),
    ],
)
@pytest.mark.parametrize('supply_model', [False, True])
def test_rejects_malformed_rows_before_work(
    tmp_path, monkeypatch, forbid_call, row, error, supply_model
):
    data = tmp_path / 'cola.tsv'
    data.write_text(
        'src\t1\tx\talpha\n\nsrc\t0\tx\tbeta\n' + row + '\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(cli, 'load_model', forbid_call)
    monkeypatch.setattr(benchmark, 'evaluate', forbid_call)
    model = object() if supply_model else None

    with pytest.raises(ValueError, match=f'CoLA TSV row 4 {error}'):
        run('missing.safetensors', data=str(data), model=model)


def test_preserves_sentence_fields_and_bytes(tmp_path):
    data = tmp_path / 'cola.tsv'
    data.write_bytes('\r\n  \r\n\t1\t\t  café  \r\nsrc\t0\tx\tbeta'.encode('utf-8'))

    assert cola(str(data)) == [('  café  '.encode('utf-8'), 1), (b'beta', 0)]


@pytest.mark.parametrize(
    ('label', 'row_number', 'error'),
    [
        ('2', 3, r'row 3 has invalid label .2.; expected 0 or 1'),
        ('no', 2, r'row 2 has invalid label .no.; expected 0 or 1'),
    ],
)
def test_rejects_invalid_labels(tmp_path, label, row_number, error):
    data = tmp_path / 'cola.tsv'
    data.write_text(
        'src\t1\tx\talpha\n'
        + ('src\t0\tx\tbeta\n' if row_number == 3 else '')
        + f'src\t{label}\tx\tgamma\n'
        + ('src\t2\tx\tdelta\n' if row_number == 3 else ''),
        encoding='utf-8',
    )

    with pytest.raises(ValueError, match=error):
        cola(str(data))


def test_run_rejects_invalid_labels_before_work(tmp_path, monkeypatch, forbid_call):
    data = tmp_path / 'cola.tsv'
    data.write_text(
        'src\t1\tx\talpha\nsrc\t0\tx\tbeta\nsrc\t2\tx\tgamma\nsrc\t2\tx\tdelta\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(cli, 'load_model', forbid_call)
    monkeypatch.setattr(benchmark, 'evaluate', forbid_call)

    with pytest.raises(
        ValueError, match=r'row 3 has invalid label .2.; expected 0 or 1'
    ):
        run('missing.safetensors', data=str(data))
