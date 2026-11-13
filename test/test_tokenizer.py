import base64

import pytest

from tmt.data import BPETokenizer, huggingface_merges, tiktoken_merges


def test_ranked_merges_round_trip_bytes_and_reserve_stop():
    tokenizer = BPETokenizer([[97, 98], [257, 97], [258, 98]])
    raw = bytes(range(256)) + b'abababab\xff\x00'
    tokens = tokenizer.encode(raw)
    assert b''.join(tokenizer.vocabulary[token] for token in tokens) == raw
    assert 256 not in tokens
    assert tokenizer.encode(b'abab') == [257, 257]
    assert tokenizer.vocab_size == 260


def test_overlapping_merges_follow_rank_and_leftmost_ties():
    tokenizer = BPETokenizer([[97, 97], [257, 97], [97, 257]])
    assert tokenizer.encode(b'aaaaa') == [257, 258]
    assert tokenizer.encode(b'aaa') == [258]


@pytest.mark.parametrize(
    'value',
    [
        [],
        {},
        {'version': 1, 'merges': []},
        {'merges': {}},
        {'merges': [[0, 256]]},
        {'merges': [[0, 257]]},
        {'merges': [[True, 2]]},
        {'merges': [[0]]},
        {'merges': [3]},
        {'merges': [[0, 1], [0, 1]]},
    ],
)
def test_invalid_tokenizer_files(value):
    with pytest.raises(ValueError, match='BPE'):
        BPETokenizer.from_dict(value)


def test_huggingface_merges_use_the_byte_alphabet():
    source = {'model': {'type': 'BPE', 'merges': ['a b', ['Ġ', 'ab']]}}
    assert huggingface_merges(source) == [(97, 98), (32, 257)]
    source['pre_tokenizer'] = {'type': 'ByteLevel'}
    with pytest.raises(ValueError, match='pre-tokenizer'):
        huggingface_merges(source)


def test_tiktoken_merges_and_invalid_rank_files():
    rows = [(bytes([i]), i) for i in range(256)] + [(b'ab', 256), (b'aba', 257)]
    text = '\n'.join(
        f'{base64.b64encode(token).decode()} {rank}'
        for token, rank in rows
    )
    assert tiktoken_merges(text) == [(97, 98), (257, 97)]
    with pytest.raises(ValueError, match='rank file'):
        tiktoken_merges('not-base64! 0')
    with pytest.raises(ValueError, match='every byte'):
        tiktoken_merges('YQ== 0')
