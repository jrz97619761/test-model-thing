from __future__ import annotations

import tmt.util.validation as validation

from typing import Any, Iterable, Sequence

import base64
import math

class Tokenizer:
    def __init__(self) -> None:
        self.vocab: list[bytes] = [bytes([byte]) for byte in range(256)] + [b'']

    @property
    def size(self) -> int: return len(self.vocab)

    def encode(self, data: bytes) -> list[int]: return list(data)
    def decode(self, token: int) -> bytes: return self.vocab[token]

type BytePair = Sequence[int, int]

class BPE(Tokenizer):
    def __init__(self, merges: Iterable[BytePair]) -> None:
        super().__init__()
        self.ranks: dict[BytePair, int] = {}

        for pair in merges:
            validation.instance(pair, list | tuple, 'bpe pair')
            validation.require(len(pair) == 2, f'bpe pair length must be 2, found length {len(pair)}')

            for token in pair:
                validation.instance(token, int, 'bpe token id')
                validation.require(not isinstance(token, bool), 'bpe token id cannot be boolean')

                validation.require(token != 256, f'bpe token id {token} must not be equal to stop token (256)')
                validation.require(0 <= token < self.size, f'bpe merge must be pair of two existing non-stop tokens, not {token}')
                validation.require(tuple(pair) not in self.ranks, f'bpe pair {tuple(pair)} is duplicated')

            self.ranks[tuple(pair)] = self.size
            self.vocab.append(self.vocab[pair[0]] + self.vocab[pair[1]])

    def encode(self, data: bytes) -> list[int]: return BPE.merge(data, self.ranks)

    @staticmethod
    def merge(data: bytes, ranks: dict[BytePair, int]) -> list[int]:
        tokens = list(data)
        while len(tokens) > 1:
            pair = min(zip(tokens, tokens[1:]), key = lambda item: ranks.get(item, math.inf))
            if pair not in ranks: break

            merged, index = [], 0

            while index < len(tokens):
                if tuple(tokens[index : index + 2]) == pair:
                    merged.append(ranks[pair])
                    index += 2

                else:
                    merged.append(tokens[index])
                    index += 1

            tokens = merged

        return tokens

class BPEConverter:
    @staticmethod
    def tmt(config: Any) -> list[BytePair]:
        validation.instance(config, dict, 'tmt tokenizer bpe')
        validation.key_difference(config, ['merges'], 'tmt tokenizer bpe')
        validation.instance(config['merges'], list, 'tmt tokenizer bpe merges')

        return config['merges']

    @staticmethod
    def huggingface(config: Any) -> list[BytePair]:
        validation.instance(config, dict, 'hf tokenizer')
        validation.key_difference(config, ['model'], 'hf tokenizer')

        validation.instance(config.get('model'), dict, 'hf tokenizer model')
        validation.instance(config['model'].get('merges'), list, 'hf tokenizer merges list')
        validation.require(
            config['model'].get('type') == 'BPE', f'hf tokenizer must be of type BPE, currently type \'{config['model'].get(type)}\''
        )

        visible = [*range(33, 127), *range(161, 173), *range(174, 256)]
        hidden = [byte for byte in range(256) if byte not in visible]

        ids = {chr(byte): byte for byte in visible}
        ids.update({chr(256 + index): byte for index, byte in enumerate(hidden)})

        merges = []

        for item in config['model']['merges']:
            pair = item.split(' ') if isinstance(item, str) else item

            validation.instance(pair, list, 'byte pair')
            validation.require(len(pair) == 2, f'byte pair is of length {len(pair)}, should be 2')

            for symbol in pair:
                validation.instance(symbol, str, 'hf bpe symbol')
                validation.require(symbol in ids, 'hf bpe symbol is not of valid id')

            validation.require(''.join(pair) not in ids, f'byte pair {tuple(pair)} is duplicated')

            ids[''.join(pair)] = 257 + len(merges)
            merges.append((ids[pair[0]], ids[pair[1]]))

        return merges

    @staticmethod
    def tiktoken(config: Any) -> list[BytePair]:
        validation.instance(config, str, 'tiktoken rank file')

        with validation.trycatch(ValueError, 'invalid tiktoken rank file'):
            rows = [line.split() for line in config.splitlines() if line]
            ranked = sorted(
                (int(rank), base64.b64decode(token, validate = True))
                for token, rank in rows
            )

        validation.require(
            len({token for _, token in ranked if len(token) == 1}) == 256,
            'tiktoken ranks must contain every byte'
        )

        ranks: dict[BytePair] = {}
        for _, token in ranked:
            if len(token) > 1:
                pair = tuple(BPE.merge(token, ranks))
                validation.require(len(pair) == 2, f'tiktoken token {token} is not one BPE merge')

                ranks[pair] = 257 + len(ranks)

        return list(ranks)