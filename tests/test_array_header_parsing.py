"""Canonical array headers take a fast path that returns exactly what literal_eval does."""

import ast
import json
import random
from pathlib import Path

import numpy as np
import pytest

from amber_lmdb import LMDBReader, LMDBWriter, codec
from amber_lmdb.arrays import _TENSOR_ITEM_SIZES

_SIMPLE_DTYPES = [
    np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
    np.uint64, np.float16, np.float32, np.float64, np.longdouble, np.complex64,
    np.complex128, ">f4", ">i8", "<U10", "S100", "V16", "M8[ns]", "m8[s]",
]  # fmt: skip
_SHAPES = [(), (0,), (7,), (3, 4), (2, 0, 5), (1, 2, 3, 4)]


def _header_text(value: bytes) -> str:
    _, _, length = codec._HEAD.unpack_from(value)
    return value[codec._HEAD.size : codec._HEAD.size + length].decode("latin1")


def _same(left, right) -> bool:
    """Equal values of identical types, recursively; 1 == True alone would not do."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return list(left) == list(right) and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, tuple):
        return len(left) == len(right) and all(map(_same, left, right))
    return left == right


def test_numpy_headers_for_simple_dtypes_take_the_fast_path():
    for dtype in _SIMPLE_DTYPES:
        for shape in _SHAPES:
            array = np.zeros(shape, dtype)
            value = codec.ndarray_to_bytes(array)
            text = _header_text(value)
            meta = codec._canonical_meta(text)
            assert meta is not None, text
            assert _same(meta, ast.literal_eval(text)), text
            restored = codec.bytes_to_ndarray(value)
            assert restored.dtype == array.dtype and restored.shape == array.shape


def test_tensor_headers_take_the_fast_path_without_torch():
    for name in [*_TENSOR_ITEM_SIZES, "future_dtype"]:
        for shape in _SHAPES:
            text = _header_text(codec._pack({"kind": "torch", "dtype": name, "shape": shape}, b""))
            meta = codec._canonical_meta(text)
            assert meta is not None, text
            assert _same(meta, ast.literal_eval(text)), text


def test_chunked_headers_take_the_fast_path():
    for shape in [(1,), (7,), (3, 4), (2, 0, 5)]:
        meta, _ = codec._ndarray_meta(np.zeros(shape, np.float32))
        text = _header_text(codec._pack({**meta, "chunk_rows": 3}, b""))
        fast = codec._canonical_meta(text)
        assert fast is not None and fast["chunk_rows"] == 3, text
        assert _same(fast, ast.literal_eval(text)), text


def test_historical_array_header_takes_the_fast_path():
    fixture = json.loads((Path(__file__).parent / "fixtures/mmfabric_0_2_2.json").read_text())
    (row,) = [row for row in fixture["rows"] if row["tag"] == "write_ndarray"]
    text = _header_text(bytes.fromhex(row["value"]))
    assert _same(codec._canonical_meta(text), ast.literal_eval(text))


_NUMPY = "{'kind': 'numpy', 'descr': '<f4', 'shape': %s}"


@pytest.mark.parametrize(
    "text",
    [
        _NUMPY % "(5)",  # an int, not a tuple
        _NUMPY % "(05,)",  # SyntaxError in literal_eval
        _NUMPY % "(1_0,)",
        _NUMPY % "(1,2)",
        _NUMPY % "(1, 2,)",
        _NUMPY % "( 1,)",
        _NUMPY % "[1, 2]",
        _NUMPY % "(-1,)",
        _NUMPY % "(True,)",
        _NUMPY % "(0x10,)",
        _NUMPY % ("(" + "9" * 19 + ",)"),
        _NUMPY % f"({chr(0x661)},)",  # a non-ASCII decimal digit
        _NUMPY % "(1,)" + "\n",
        " " + _NUMPY % "(1,)",
        '{"kind": "numpy", "descr": "<f4", "shape": (1,)}',
        "{'descr': '<f4', 'kind': 'numpy', 'shape': (1,)}",
        "{'kind': 'numpy', 'descr': '<f4', 'shape': (1,), 'dtype_name': 'bfloat16', 'itemsize': 2}",
        "{'kind': 'numpy', 'descr': [('a', '<f4')], 'shape': (1,)}",
        "{'kind': 'numpy', 'descr': '<f\\x34', 'shape': (1,)}",
        "{'kind': 'numpy', 'descr': '', 'shape': (1,)}",
        "{'kind': 'torch', 'dtype': 'Float32', 'shape': (1,)}",
        "{'kind': 'torch', 'shape': (2, 3), 'dtype': 'float32'}",
        "{'kind': 'tensor', 'dtype': 'float32', 'shape': (1,)}",
        "{'kind': 'numpy', 'descr': '<f4', 'shape': (1,), 'chunk_rows': 0}",
        "{'kind': 'numpy', 'descr': '<f4', 'shape': (1,), 'chunk_rows': 01}",
        "{'kind': 'numpy', 'descr': '<f4', 'shape': (1,), 'chunk_rows': -1}",
        "{'kind': 'numpy', 'descr': '<f4', 'chunk_rows': 2, 'shape': (1,)}",
        "{'kind': 'torch', 'dtype': 'float32', 'shape': (1,), 'chunk_rows': 2}",
    ],
)
def test_non_canonical_metadata_still_goes_through_literal_eval(text):
    assert codec._canonical_meta(text) is None


def test_fast_path_agrees_with_literal_eval_on_mutated_headers():
    rng = random.Random(0)
    seeds = [
        _header_text(codec.ndarray_to_bytes(np.zeros((3, 12, 0), ">u8"))),
        _header_text(codec.ndarray_to_bytes(np.zeros((), "M8[ns]"))),
        _header_text(codec._pack({"kind": "torch", "dtype": "float8_e4m3fn", "shape": (1,)}, b"")),
        _NUMPY % "(10, 20, 30)",
        "{'kind': 'numpy', 'descr': '<f4', 'shape': (40, 2), 'chunk_rows': 16}",
    ]
    # Weighted toward shape characters so that many mutations stay canonical.
    alphabet = "0123456789" * 3 + "(), " * 3 + "'{}:<>|=[]_-+.ejxL\"\\\nnumpytorchkindshapedescr"
    matched = 0
    for _ in range(20_000):
        text = list(rng.choice(seeds))
        for _ in range(rng.randint(1, 3)):
            # Most edits land near the end, where the shape and dtype are.
            start = max(0, len(text) - 24) if rng.random() < 0.7 else 0
            position = rng.randrange(start, len(text) + 1)
            edit = rng.random()
            if edit < 0.4:
                text.insert(position, rng.choice(alphabet))
            elif edit < 0.7 and position < len(text):
                text[position] = rng.choice(alphabet)
            elif position < len(text):
                del text[position]
        candidate = "".join(text)
        meta = codec._canonical_meta(candidate)
        if meta is not None:
            matched += 1
            assert _same(meta, ast.literal_eval(candidate)), candidate
    assert matched > 1_000


def test_reads_are_identical_with_and_without_the_fast_path(tmp_path, monkeypatch):
    path = tmp_path / "arrays.lmdb"
    arrays = [np.arange(n, dtype=np.float32) for n in (0, 1, 5, 17)]
    arrays += [np.ones((n, 3), ">i8") for n in (1, 4)]
    arrays += [np.zeros((), np.complex64), np.zeros(2, [("a", "<f4"), ("b", "<i2")])]
    with LMDBWriter(path) as writer:
        for index, array in enumerate(arrays):
            writer.write_ndarray(array, key="values", index=index)
    keys, indices = ["values"] * len(arrays), list(range(len(arrays)))

    def read():
        with LMDBReader(path) as reader:
            values = reader.read_many(keys, indices=indices)
            headers = reader.read_array_headers_many(keys, indices=indices)
            shapes = [reader.read_as_shape("values", index=index) for index in indices]
        return values, headers, shapes

    fast = read()
    monkeypatch.setattr(codec, "_canonical_meta", lambda text: None)
    slow = read()
    for got, expected, array in zip(fast[0], slow[0], arrays, strict=True):
        assert got.dtype == expected.dtype == array.dtype
        assert np.array_equal(got, expected) and np.array_equal(got, array)
    assert fast[1] == slow[1] and fast[2] == slow[2]
