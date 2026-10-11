"""Typed-value decoding: same results and errors as before, lowered limits, node budgets."""

import pickle
import random

import msgpack
import numpy as np
import pytest
from frozendict import frozendict

from amber_lmdb import (
    LMDBOverlayReader,
    LMDBReader,
    LMDBWriter,
    ReadLimitError,
    RecordTable,
    ValueLimits,
    codec,
    verify_store,
)
from amber_lmdb.codec import any_to_bytes

C = codec


def _reference_decode(node, *, allow_custom, depth, items):
    """The decoder as it was before the inline-leaf rewrite, kept as the oracle."""
    if depth > C._MAX_VALUE_DEPTH:
        raise ValueError("value exceeds the maximum nesting depth")
    items[0] += 1
    if items[0] > C._MAX_VALUE_ITEMS:
        raise ValueError("value exceeds the maximum item count")
    if type(node) is not list or not node or type(node[0]) is not int:
        raise ValueError("value contains an invalid schema node")
    tag = node[0]
    if tag == C._NONE:
        if len(node) != 1:
            raise ValueError("invalid none value")
        return None
    if tag == C._BOOL:
        if len(node) != 2 or type(node[1]) is not bool:
            raise ValueError("invalid bool value")
        return node[1]
    if tag == C._INT:
        if len(node) != 2 or type(node[1]) is not int:
            raise ValueError("invalid int value")
        return node[1]
    if tag == C._BIG_INT:
        if (
            len(node) != 3
            or type(node[1]) is not bool
            or type(node[2]) is not bytes
            or not node[2]
            or node[2][0] == 0
        ):
            raise ValueError("invalid large int value")
        value = int.from_bytes(node[2], "big")
        value = -value if node[1] else value
        if C._MSGPACK_MIN_INT <= value <= C._MSGPACK_MAX_INT:
            raise ValueError("large int value is not canonical")
        return value
    if tag == C._FLOAT:
        if len(node) != 2 or type(node[1]) is not float:
            raise ValueError("invalid float value")
        return node[1]
    if tag == C._COMPLEX:
        if len(node) != 3 or type(node[1]) is not float or type(node[2]) is not float:
            raise ValueError("invalid complex value")
        return complex(node[1], node[2])
    if tag == C._STR:
        if len(node) != 2 or type(node[1]) is not str:
            raise ValueError("invalid str value")
        return node[1]
    if tag == C._BYTES:
        if len(node) != 2 or type(node[1]) is not bytes:
            raise ValueError("invalid bytes value")
        return node[1]
    if tag == C._TYPE_REFERENCE:
        if len(node) != 2 or type(node[1]) is not str or node[1] not in C._VALUE_TYPES:
            raise ValueError("unsupported type reference")
        return C._VALUE_TYPES[node[1]]
    if tag in C._ARRAY_NODE_TAGS:
        if len(node) != 2 or type(node[1]) is not bytes:
            raise ValueError("invalid array value")
        return C.bytes_to_ndarray(node[1])
    if tag in C._CONTAINER_NODE_TAGS:
        if len(node) != 2 or type(node[1]) is not list:
            raise ValueError("invalid container value")
        values = [
            _reference_decode(value, allow_custom=allow_custom, depth=depth + 1, items=items)
            for value in node[1]
        ]
        if tag == C._LIST:
            return values
        if tag == C._TUPLE:
            return tuple(values)
        result = set()
        for value in values:
            try:
                duplicate = value in result
            except TypeError as exc:
                raise ValueError("set value is not hashable") from exc
            if duplicate:
                raise ValueError("set contains duplicate values")
            result.add(value)
        return result if tag == C._SET else frozenset(result)
    if tag in C._MAPPING_NODE_TAGS:
        if len(node) != 2 or type(node[1]) is not list:
            raise ValueError("invalid mapping value")
        mapping = {}
        for pair in node[1]:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("mapping entries must be key/value pairs")
            key = _reference_decode(
                pair[0], allow_custom=allow_custom, depth=depth + 1, items=items
            )
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise ValueError("mapping key is not hashable") from exc
            if duplicate:
                raise ValueError("mapping contains colliding keys")
            mapping[key] = _reference_decode(
                pair[1], allow_custom=allow_custom, depth=depth + 1, items=items
            )
        return mapping if tag == C._DICT else frozendict(mapping)
    if tag == C._CUSTOM:
        if len(node) != 3 or type(node[1]) is not str:
            raise ValueError("invalid custom value")
        raise ValueError("custom value decoding requires allow_custom=True")
    raise ValueError(f"unsupported value tag: {tag!r}")


def _outcome(function, node):
    try:
        return ("value", function(node))
    except (ValueError, TypeError) as error:
        return ("error", type(error).__name__, str(error))


def _same(left, right):
    """Structural equality of outcomes, comparing NumPy arrays by value and dtype."""
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return left.dtype == right.dtype and np.array_equal(left, right)
    if isinstance(left, (dict, frozendict)):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(map(_same, left, right))
    return left == right


def _new(node):
    return C._decode_value_node(node, False, 0, C._new_nodes(C._DEFAULT_LIMITS, None))


def _old(node):
    return _reference_decode(node, allow_custom=False, depth=0, items=[0])


SEEDS = [
    {"id": 3, "name": "x", "labels": ["a", "b"], "bounds": (0.5, 1.5), "flags": {True, False}},
    [1, 2.5, "s", b"b", None, True, 2**70, -(2**70), 1 + 2j],
    frozendict({"a": frozenset({1, 2}), "b": ((1, 2), [3])}),
    {"nested": [[[[1]]]], "empty": [], "map": {}, (1, "t"): "tuple key"},
    {"array": np.arange(3, dtype=np.int16), "type": dict},
]


def _mutate(rng, node):
    """Replace one random schema node, usually deep inside containers, with another.

    Replacements are valid or invalid nodes, so both decoded values and every error
    path are compared.
    """
    nested = (
        type(node) is list
        and len(node) == 2
        and node[0] in (*C._CONTAINER_NODE_TAGS, *C._MAPPING_NODE_TAGS)
        and type(node[1]) is list
        and node[1]
    )
    if nested and rng.random() < 0.8:
        children = list(node[1])
        position = rng.randrange(len(children))
        child = children[position]
        if node[0] in C._MAPPING_NODE_TAGS and type(child) is list and len(child) == 2:
            side = rng.randrange(2)
            pair = list(child)
            pair[side] = _mutate(rng, child[side])
            children[position] = pair
        else:
            children[position] = _mutate(rng, child)
        return [node[0], children]
    return rng.choice(
        [
            [C._INT, 5],
            [C._STR, "v"],
            [C._NONE],
            [C._FLOAT, 0.5],
            [C._TUPLE, [[C._INT, 1]]],
            None,
            [],
            [C._INT],
            [C._INT, "1"],
            [C._STR, 1],
            [C._BOOL, 1],
            [C._FLOAT, 1],
            [C._BYTES, "x"],
            [C._LIST, [[C._INT, 1], [C._LIST, "x"]]],
            [C._SET, [[C._INT, 1], [C._INT, 1]]],
            [C._SET, [[C._LIST, []]]],
            [C._DICT, [[[C._STR, "k"], [C._INT, 1]], [[C._STR, "k"], [C._INT, 2]]]],
            [C._DICT, [[[C._LIST, []], [C._INT, 1]]]],
            [C._DICT, [[[C._STR, "k"]]]],
            [C._CUSTOM, "x", [C._NONE]],
            [99, 1],
            [C._BIG_INT, False, b"\x01"],
            "leaf",
            7,
        ]
    )


def test_decoder_matches_the_previous_implementation_on_valid_and_mutated_nodes():
    rng = random.Random(1)
    nodes = [
        msgpack.unpackb(any_to_bytes(seed)[5:], raw=False, strict_map_key=True) for seed in SEEDS
    ]
    checked = errors = 0
    for node in nodes:
        assert _same(_outcome(_new, node), _outcome(_old, node))
    for _ in range(20_000):
        node = _mutate(rng, rng.choice(nodes))
        new, old = _outcome(_new, node), _outcome(_old, node)
        assert _same(new, old), node
        checked += 1
        errors += new[0] == "error"
    assert checked == 20_000 and 2_000 < errors < 19_000


def test_depth_and_item_limits_match_the_previous_implementation():
    deep = [C._INT, 1]
    for _ in range(C._MAX_VALUE_DEPTH):
        deep = [C._LIST, [deep]]
    assert _outcome(_new, deep) == _outcome(_old, deep)
    deeper = [C._LIST, [deep]]
    assert _outcome(_new, deeper) == _outcome(_old, deeper)
    assert _outcome(_new, deeper)[2] == "value exceeds the maximum nesting depth"
    empty_at_limit = [C._LIST, []]
    for _ in range(C._MAX_VALUE_DEPTH):
        empty_at_limit = [C._LIST, [empty_at_limit]]
    assert _outcome(_new, empty_at_limit) == _outcome(_old, empty_at_limit)
    wide = [C._LIST, [[C._INT, index] for index in range(C._MAX_VALUE_ITEMS)]]
    assert _outcome(_new, wide) == _outcome(_old, wide)
    assert _outcome(_new, wide)[2] == "value exceeds the maximum item count"


def test_value_limits_validation():
    assert ValueLimits() == ValueLimits(64 * 1024**2, 1_000_000, 100)
    for options in ({"max_nodes": 0}, {"max_depth": -1}, {"max_value_bytes": 1.5}):
        with pytest.raises(ValueError, match="positive integer"):
            ValueLimits(**options)
    for options in ({"max_nodes": 1_000_001}, {"max_depth": 101}, {"max_value_bytes": 2**27}):
        with pytest.raises(ValueError, match="may only lower"):
            ValueLimits(**options)
    with pytest.raises(TypeError, match="ValueLimits"):
        LMDBReader("x", value_limits={"max_nodes": 10})


def _store(path):
    with LMDBWriter(path, map_size=64 * 1024**2) as writer:
        for index in range(8):
            writer.write_dict({"values": list(range(100)), "id": index}, key="rows", index=index)
        writer.write_sequence([[[[1]]]], key="deep")
        writer.write_any("x" * 5000, key="long")
        writer.write_ndarray(np.zeros(100_000, np.uint8), key="array")
    return path


def test_reader_value_limits_apply_to_every_messagepack_read(tmp_path):
    path = _store(tmp_path / "store.lmdb")
    # 105 nodes per row with lists of 100 items: msgpack's length limit still passes.
    limits = ValueLimits(max_nodes=104, max_depth=3, max_value_bytes=1024)
    with LMDBReader(path, value_limits=limits) as reader:
        assert reader.value_limits == limits
        for read in (
            lambda: reader.read_as_dict("rows", 0),
            lambda: reader.read("rows", 0),
            lambda: reader.read_many(["rows"], indices=[0]),
            lambda: reader.read_as_dict_many(["rows"], indices=[0]),
            lambda: list(reader.iter_dict_pages(["rows"], indices=[0])),
            lambda: list(reader.iter_read_pages(["rows"], indices=[0])),
            lambda: RecordTable(reader, "rows", lambda row: row, count=1).get(0),
        ):
            with pytest.raises(ValueError, match="maximum item count"):
                read()
        with pytest.raises(ValueError, match="nesting depth"):
            reader.read_as_list("deep")
        with pytest.raises(ValueError, match="maximum byte count"):
            reader.read_as_any("long")
        assert reader.read_as_ndarray("array").shape == (100_000,)  # Arrays are unaffected.
        restored = pickle.loads(pickle.dumps(reader))
        assert restored.value_limits == limits
        restored.close()
    with (
        LMDBOverlayReader([path], value_limits=limits) as overlay,
        pytest.raises(ValueError, match="maximum item count"),
    ):
        overlay.read_as_dict("rows", 0)
    with LMDBReader(path) as reader:
        assert reader.read_as_dict("rows", 0)["id"] == 0


def test_batch_node_budget(tmp_path):
    path = _store(tmp_path / "store.lmdb")
    nodes_per_row = 1 + (1 + 1 + 100) + (1 + 1)  # dict; key, list, items; key, id
    with LMDBReader(path) as reader:
        rows = reader.read_many(
            ["rows"] * 3, indices=[0, 1, 2], max_decoded_nodes=3 * nodes_per_row
        )
        assert [row["id"] for row in rows] == [0, 1, 2]
        with pytest.raises(ReadLimitError) as error:
            reader.read_many(
                ["rows"] * 3, indices=[0, 1, 2], max_decoded_nodes=3 * nodes_per_row - 1
            )
        assert error.value.kind == "nodes" and error.value.limit == 3 * nodes_per_row - 1
        with pytest.raises(ReadLimitError):
            reader.read_as_dict_many(["rows"] * 2, indices=[0, 1], max_decoded_nodes=nodes_per_row)
        pages = list(
            reader.iter_read_pages(
                ["rows"] * 4, indices=range(4), max_items=2, max_decoded_nodes=2 * nodes_per_row
            )
        )
        assert [len(page) for page in pages] == [2, 2]  # The budget applies per page.
        with pytest.raises(ReadLimitError):
            list(reader.iter_dict_pages(["rows"] * 2, indices=[0, 1], max_decoded_nodes=10))
        for invalid in (0, -1, 1.5):
            with pytest.raises(ValueError, match="max_decoded_nodes"):
                reader.read_many(["rows"], indices=[0], max_decoded_nodes=invalid)


def test_verify_store_with_lowered_limits(tmp_path):
    path = _store(tmp_path / "store.lmdb")
    assert verify_store(path).ok
    report = verify_store(path, value_limits=ValueLimits(max_nodes=104))
    assert not report.ok and report.error_count == 8
    with pytest.raises(TypeError, match="ValueLimits"):
        verify_store(path, value_limits=50)
