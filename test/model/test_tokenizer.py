from __future__ import annotations
import pytest

import base64

from tmt.model.tokenizers import BPE, BPEConverter
from tmt.util.validation import ValidationError

def test_token_ranked_merge():
    tokenizer = BPE([[97, 98], [257, 97], [258, 98]])

    raw = bytes(range(256)) + b'abababab\xff\x00'
    tokens = tokenizer.encode(raw)
    
    assert b''.join(tokenizer.vocab[token] for token in tokens) == raw
    assert 256 not in tokens
    assert tokenizer.encode(b'abab') == [257, 257]
    assert tokenizer.size == 260

def test_token_overlapping_merge():
    tokenizer = BPE([[97, 97], [257, 97], [97, 257]])

    assert tokenizer.encode(b'aaaaa') == [257, 258]
    assert tokenizer.encode(b'aaa') == [258]

@pytest.mark.parametrize('value', [
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
])
def test_token_invalid_merge(value):
    with pytest.raises(ValidationError, match = 'bpe'): BPE(BPEConverter.tmt(value))

def test_token_hf_converter():
    source = {'model': {'type': 'BPE', 'merges': ['a b', ['Ġ', 'ab']]}}
    assert BPEConverter.huggingface(source) == [(97, 98), (32, 257)]

    source['pre_tokenizer'] = {'type': 'ByteLevel'}
    with pytest.raises(ValidationError, match = 'extra keys'): BPEConverter.huggingface(source)

def test_token_tiktoken_converter():
    rows = [(bytes([i]), i) for i in range(256)] + [(b'ab', 256), (b'aba', 257)]
    text = '\n'.join(
        f'{base64.b64encode(token).decode()} {rank}'
        for token, rank in rows
    )

    assert BPEConverter.tiktoken(text) == [(97, 98), (257, 97)]
    with pytest.raises(ValidationError, match = 'rank file'): BPEConverter.tiktoken('not-base64! 0')
    with pytest.raises(ValidationError, match = 'every byte'): BPEConverter.tiktoken('YQ== 0')
