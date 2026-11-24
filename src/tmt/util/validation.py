from __future__ import annotations

import math
import operator
from functools import reduce
from typing import Any, Generator, Annotated, Union, TypeAliasType, NewType, TypeVar, get_origin, get_args, is_typeddict
from types import UnionType, NoneType
from collections.abc import Mapping, Iterable
from os import PathLike
from pathlib import Path
from contextlib import contextmanager

class ValidationError(Exception): pass

# basic stuff

def require(cond: Any, msg: str) -> None:
    if not cond: raise ValidationError(msg)

def convert(tp) -> type | UnionType | tuple:
    while isinstance(tp, (TypeAliasType, NewType)):
        tp = tp.__value__ if isinstance(tp, TypeAliasType) else tp.__supertype__

    if tp is Any: return object
    if tp is None: return NoneType

    if is_typeddict(tp): return dict

    origin = get_origin(tp)

    if origin is Annotated: return convert(get_args(tp)[0])

    if origin is UnionType or origin is Union:
        args = tuple(convert(arg) for arg in get_args(tp))
        return reduce(operator.or_, args)

    if origin is not None: return origin
    return tp

def format_type(tp) -> str:
    return str(tp).replace('|', 'or').replace('<class \'', '').replace('\'>', '')

def instance(obj: Any, tp: Any, name: str) -> None:
    tp = convert(tp)
    require(isinstance(obj, tp), f'{name} should be of type {format_type(tp)}')

@contextmanager
def trycatch(errors: type | tuple[type, ...], msg: str, trace: bool = False) -> Generator[None, None, None]:
    try: yield

    except errors as e:
        raise ValidationError(f'({format_type(type(e))}) {msg}') from (e if trace else None)

# numbers

def number(value: Any) -> None:
    instance(value, int | float, 'number')

def positive(value: Any, name: str, strict = True) -> None:
    number(value)
    require(value > 0 if strict else value >= 0, f'number \'{name}\' should be positive')

def finite(value: Any, name: str) -> None:
    number(value)
    require(math.isfinite(value), f'number \'{name}\' should be finite')

def seed(value: Any, name: str = 'seed') -> None:
    instance(value, int, name)
    require(0 <= value < 2 ** 64, f'number \'{name}\' should be between 0 and {2 ** 64 - 1} (aka 2 ** 64 - 1)')

# lists and dicts

def tuplify(obj: list[Any] | dict[str, Any] | Any) -> tuple[Any, ...] | dict[str, Any]:
    if isinstance(obj, list):
        convert = []
        for item in obj: convert.append(tuplify(item))
        return tuple(convert)

    if isinstance(obj, dict):
        convert = {}
        for key, value in obj.items(): convert[key] = tuplify(value)
        return convert

    return obj

def format_iterable(iterable: Iterable[str]) -> str: return f'\'{'\', \''.join(iterable)}\''

def mapping(value: Any) -> None:
    instance(value, Mapping, 'mapping')

def iterable(value: Any) -> None:
    instance(value, Iterable, 'iterable')

def iterable_length(actual: Any, expected: Any, name: str) -> None:
    iterable(actual)
    iterable(expected)

    require(len(actual) == len(expected), f'iterable \'{name}\' has non-matching length')

def key_exists(key: Any, map: Any) -> None:
    mapping(map)
    instance(key, str, 'key')

    require(key in map, f'unknown key \'{key}\', valid keys: {format_iterable(sorted(map))}')

def key_difference(actual: Any, expected: Any, name: str, strict = True) -> None:
    mapping(actual)
    iterable(expected)

    missing = sorted(set(expected) - set(actual))
    require(not missing, f'mapping \'{name}\' has missing keys: {format_iterable(missing)}')

    if strict:
        extra = sorted(set(actual) - set(expected))
        require(not extra, f'mapping \'{name}\' has extra keys: {format_iterable(extra)}')

# mlx arrays

def array(arr: Any) -> None:
    import mlx.core as mx
    instance(arr, mx.array, 'array')

def array_shape(arr: Any, shape: tuple[int, ...], dtype: Any, name: str) -> None:
    array(arr)
    require(arr.shape == shape and arr.dtype == dtype, f'array \'{name}\' has non-matching shape and/or dtype')

def array_match(arr: Any, expected: Any, name: str) -> None:
    array(expected)
    array_shape(arr, expected.shape, expected.dtype, name)

def arrays_match(actual: Any, expected: Any, name: str) -> None:
    mapping(actual)
    mapping(expected)

    key_difference(actual, expected, name)
    for key in expected: array_match(actual[key], expected[key], f'{name}: {key}')

# files

type FilePath = str | PathLike[str]

def file_name(file: Any) -> None:
    instance(file, str, 'file name')
    require(
        file and file not in ('.', '..')
        and not any(char in file for char in ('/', '\\', '\0')),
        f'\'{file}\' is not a valid file or folder name'
    )

def file_parent(path: Any) -> None:
    instance(path, FilePath, 'file path')

    parent = Path(path).parent
    require(parent.is_dir(), f'parent folder \'{parent}\' of file does not exist')