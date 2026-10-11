"""Encode and validate typed values and arrays; import PyTorch only when needed."""

from __future__ import annotations

import ast
import ctypes
import lzma
import math
import re
import struct
import sys
import threading
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import msgpack
import numpy as np
from frozendict import frozendict
from numpy import ndarray
from numpy.typing import NDArray

if TYPE_CHECKING:
    import torch
    from torch import Tensor


def _is_tensor_type(value_type: type[Any]) -> bool:
    """Check for the exact Tensor type without importing PyTorch."""
    module = sys.modules.get("torch")
    return module is not None and value_type is getattr(module, "Tensor", None)


_VALUE_PREFIX = b"MMFV\x01"
_MAX_VALUE_BYTES = 64 * 1024**2
_MAX_VALUE_ITEMS = 1_000_000
_MAX_VALUE_DEPTH = 100
_MSGPACK_MIN_INT = -(1 << 63)
_MSGPACK_MAX_INT = (1 << 64) - 1

_NONE = 0
_BOOL = 1
_INT = 2
_BIG_INT = 3
_FLOAT = 4
_COMPLEX = 5
_STR = 6
_BYTES = 7
_LIST = 8
_TUPLE = 9
_SET = 10
_FROZENSET = 11
_DICT = 12
_FROZENDICT = 13
_CUSTOM = 14
_TYPE_REFERENCE = 15
_NDARRAY_VALUE = 16
_TENSOR_VALUE = 17

_ARRAY_NODE_TAGS = frozenset({_NDARRAY_VALUE, _TENSOR_VALUE})
_CONTAINER_NODE_TAGS = frozenset({_LIST, _TUPLE, _SET, _FROZENSET})
_MAPPING_NODE_TAGS = frozenset({_DICT, _FROZENDICT})

# Stored type names must not trigger arbitrary imports; only allowlisted types resolve.
_VALUE_TYPES = {
    f"{cls.__module__}.{cls.__qualname__}": cls
    for cls in (
        bool,
        int,
        float,
        complex,
        str,
        bytes,
        list,
        tuple,
        set,
        frozenset,
        dict,
        frozendict,
        ndarray,
        type(None),
        type,
    )
}

_CUSTOM_CODECS_BY_NAME: dict[str, tuple[type[Any], Callable[[Any], Any], Callable[[Any], Any]]] = {}
_CUSTOM_CODECS_BY_TYPE: dict[type[Any], tuple[str, Callable[[Any], Any], Callable[[Any], Any]]] = {}
_CUSTOM_CODEC_LOCK = threading.RLock()

type StorageKind = Literal[
    "tensor",
    "ndarray",
    "frozendict",
    "dict",
    "frozenset",
    "set",
    "list",
    "tuple",
    "str",
    "primitive",
    "any",
]
"""LMDB storage category selected from a value's exact type."""

"""Compatibility alias for the general StorageKind category."""

_STORAGE_KINDS: dict[type[Any], StorageKind] = {
    ndarray: "ndarray",
    frozendict: "frozendict",
    dict: "dict",
    frozenset: "frozenset",
    set: "set",
    list: "list",
    tuple: "tuple",
    str: "str",
    bool: "primitive",
    int: "primitive",
    float: "primitive",
    complex: "primitive",
    bytes: "any",
    type(None): "any",
    type: "any",
}


def storage_kind_for_type(value_type: type[Any]) -> StorageKind:
    """Select a storage category for an exact Python type.

    Args:
        value_type: Exact type of the value to store.

    Returns:
        Storage category. The fallback ``"any"`` does not mean an unsupported
        type can be serialized without registering a codec.
    """
    if _is_tensor_type(value_type):
        return "tensor"
    return _STORAGE_KINDS.get(value_type, "any")


def is_registered_bytes_type(value_type: type[Any]) -> bool:
    """Check whether a codec is explicitly registered for a class.

    Args:
        value_type: Exact Python type to look up.

    Returns:
        True if a codec is registered for that exact type.
    """
    with _CUSTOM_CODEC_LOCK:
        return value_type in _CUSTOM_CODECS_BY_TYPE


def register_bytes_codec[T](
    name: str,
    value_type: type[T],
    encoder: Callable[[T], Any],
    decoder: Callable[[Any], T],
) -> None:
    """Register a codec for storing instances of a custom class.

    Args:
        name: Stable ASCII identifier recorded in stored values.
        value_type: Exact class to register; subclasses do not inherit the codec.
        encoder: Convert an instance to a supported value.
        decoder: Restore an instance from a supported value.

    Raises:
        TypeError: The class or callbacks have invalid types.
        ValueError: The name is invalid or the name or class is already registered.

    Note:
        Decoders run only when reading with ``allow_custom=True``. They receive
        external input and must validate its types and ranges.
    """
    if (
        type(name) is not str
        or not name
        or len(name) > 128
        or not name.isascii()
        or any(not (char.isalnum() or char in "._-") for char in name)
    ):
        raise ValueError("codec name must be 1-128 ASCII letters, digits, '.', '_', or '-'")
    if not isinstance(value_type, type):
        raise TypeError("value_type must be a type")
    if not callable(encoder) or not callable(decoder):
        raise TypeError("encoder and decoder must be callable")
    with _CUSTOM_CODEC_LOCK:
        if name in _CUSTOM_CODECS_BY_NAME:
            raise ValueError(f"codec name is already registered: {name!r}")
        if value_type in _CUSTOM_CODECS_BY_TYPE:
            raise ValueError(f"class already has a registered codec: {value_type!r}")
        _CUSTOM_CODECS_BY_NAME[name] = (value_type, encoder, decoder)
        _CUSTOM_CODECS_BY_TYPE[value_type] = (name, encoder, decoder)


def unregister_bytes_codec(name: str) -> None:
    """Remove a codec registered with ``register_bytes_codec``.

    Values stored with it can no longer be decoded until it is registered again. The
    registry is process-wide, so tests and plugins should unregister what they add.

    Raises:
        KeyError: No codec is registered under ``name``.
    """
    with _CUSTOM_CODEC_LOCK:
        if name not in _CUSTOM_CODECS_BY_NAME:
            raise KeyError(f"no codec is registered under {name!r}")
        value_type, _, _ = _CUSTOM_CODECS_BY_NAME.pop(name)
        _CUSTOM_CODECS_BY_TYPE.pop(value_type, None)


def _encode_value_node(
    obj: Any,
    *,
    active: set[int],
    depth: int,
    items: list[int],
) -> list[Any]:
    """Recursively encode a value as a type-tagged MessagePack schema node.

    Args:
        obj: Supported scalar, container, array, type reference or custom value.
        active: Container and custom-value identities on the current recursion path.
        depth: Current nesting depth.
        items: Shared one-element list tracking the total number of nodes.

    Returns:
        A schema node containing an integer type tag and encoded payload.

    Raises:
        TypeError: The value type is unsupported or needs an unregistered codec.
        ValueError: Nesting or item limits are exceeded, or a cycle is detected.
    """
    if depth > _MAX_VALUE_DEPTH:
        raise ValueError("value exceeds the maximum nesting depth")
    items[0] += 1
    if items[0] > _MAX_VALUE_ITEMS:
        raise ValueError("value exceeds the maximum item count")

    if obj is None:
        return [_NONE]
    if type(obj) is bool:
        return [_BOOL, obj]
    if type(obj) is int:
        if _MSGPACK_MIN_INT <= obj <= _MSGPACK_MAX_INT:
            return [_INT, obj]
        magnitude = abs(obj).to_bytes((abs(obj).bit_length() + 7) // 8, "big")
        return [_BIG_INT, obj < 0, magnitude]
    if type(obj) is float:
        return [_FLOAT, obj]
    if type(obj) is complex:
        return [_COMPLEX, obj.real, obj.imag]
    if type(obj) is str:
        return [_STR, obj]
    if type(obj) is bytes:
        return [_BYTES, obj]
    if isinstance(obj, type):
        if obj in _VALUE_TYPES.values() or _is_tensor_type(obj):
            return [_TYPE_REFERENCE, f"{obj.__module__}.{obj.__qualname__}"]
        with _CUSTOM_CODEC_LOCK:
            reference = _CUSTOM_CODECS_BY_TYPE.get(obj)
        if reference is None:
            raise TypeError("only built-in or registered type references can be serialized")
        return [_TYPE_REFERENCE, f"codec:{reference[0]}"]
    if type(obj) is ndarray:
        return [_NDARRAY_VALUE, ndarray_to_bytes(obj)]
    if _is_tensor_type(type(obj)):
        return [_TENSOR_VALUE, tensor_to_bytes(obj)]

    if type(obj) in {list, tuple, set, frozenset, dict, frozendict}:
        identity = id(obj)
        if identity in active:
            raise ValueError("cyclic containers cannot be serialized")
        active.add(identity)
        try:
            if type(obj) is list:
                return [
                    _LIST,
                    [
                        _encode_value_node(value, active=active, depth=depth + 1, items=items)
                        for value in obj
                    ],
                ]
            if type(obj) is tuple:
                return [
                    _TUPLE,
                    [
                        _encode_value_node(value, active=active, depth=depth + 1, items=items)
                        for value in obj
                    ],
                ]
            if type(obj) in {set, frozenset}:
                return [
                    _SET if type(obj) is set else _FROZENSET,
                    [
                        _encode_value_node(value, active=active, depth=depth + 1, items=items)
                        for value in obj
                    ],
                ]
            return [
                _DICT if type(obj) is dict else _FROZENDICT,
                [
                    [
                        _encode_value_node(key, active=active, depth=depth + 1, items=items),
                        _encode_value_node(value, active=active, depth=depth + 1, items=items),
                    ]
                    for key, value in obj.items()
                ],
            ]
        finally:
            active.remove(identity)

    with _CUSTOM_CODEC_LOCK:
        codec = _CUSTOM_CODECS_BY_TYPE.get(type(obj))
    if codec is None:
        if isinstance(obj, np.generic):
            raise TypeError(_numpy_scalar_message(obj))
        raise TypeError(
            f"{type(obj).__module__}.{type(obj).__qualname__} is not a supported "
            "value type; register it with register_bytes_codec()"
        )
    name, encoder, _ = codec
    identity = id(obj)
    if identity in active:
        raise ValueError("cyclic custom values cannot be serialized")
    active.add(identity)
    try:
        payload = encoder(obj)
        return [
            _CUSTOM,
            name,
            _encode_value_node(payload, active=active, depth=depth + 1, items=items),
        ]
    finally:
        active.remove(identity)


def _numpy_scalar_message(obj: np.generic) -> str:
    return (
        f"NumPy scalar numpy.{type(obj).__name__} is not stored as a value; convert it with "
        ".item() to a Python number, or store np.asarray(value) with write_ndarray"
    )


def _pack_value(obj: Any) -> bytes:
    """Pack a supported value into versioned, type-tagged MessagePack bytes.

    Args:
        obj: Value supported by ``_encode_value_node``.

    Returns:
        MessagePack payload with a format version prefix.

    Raises:
        TypeError: The value is unsupported or the encoder does not return bytes.
        ValueError: Nesting, item or byte limits are exceeded, or a cycle is detected.
    """
    node = _encode_value_node(obj, active=set(), depth=0, items=[0])
    payload = msgpack.packb(node, use_bin_type=True, strict_types=True)
    if not isinstance(payload, bytes):
        raise TypeError("MessagePack encoder did not return bytes")
    if len(payload) + len(_VALUE_PREFIX) > _MAX_VALUE_BYTES:
        raise ValueError("encoded value exceeds the maximum byte count")
    return _VALUE_PREFIX + payload


@dataclass(frozen=True, slots=True)
class ValueLimits:
    """Limits for decoding one MessagePack value; they may only lower the built-in ones.

    Attributes:
        max_value_bytes: Encoded bytes per value (built-in limit 64 MiB).
        max_nodes: Schema nodes per value: every scalar, container and mapping key
            counts once (built-in limit 1,000,000).
        max_depth: Container nesting depth (built-in limit 100).

    Decoding a value peaks at roughly 17-25 times its encoded size in Python objects,
    about 120-170 bytes per node, so lowering ``max_nodes`` bounds memory per value.
    Arrays are unaffected: their allocation equals their payload. Custom decoders are
    caller code and are not bounded.
    """

    max_value_bytes: int = _MAX_VALUE_BYTES
    max_nodes: int = _MAX_VALUE_ITEMS
    max_depth: int = _MAX_VALUE_DEPTH

    def __post_init__(self) -> None:
        for name, value, ceiling in (
            ("max_value_bytes", self.max_value_bytes, _MAX_VALUE_BYTES),
            ("max_nodes", self.max_nodes, _MAX_VALUE_ITEMS),
            ("max_depth", self.max_depth, _MAX_VALUE_DEPTH),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value > ceiling:
                raise ValueError(f"{name} may only lower the built-in limit of {ceiling}")


_DEFAULT_LIMITS = ValueLimits()


class _NodeBudget:
    """Schema nodes a whole batch or page may decode (``max_decoded_nodes``)."""

    __slots__ = ("limit", "used")

    def __init__(self, limit: int) -> None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("max_decoded_nodes must be a positive integer or None")
        self.limit = limit
        self.used = 0


# Node counter of one value, as a list for speed: [count, limit, max_depth, budget].
# limit is the value's max_nodes, or a batch budget's remainder when that is smaller.
_COUNT, _LIMIT, _DEPTH, _BUDGET = 0, 1, 2, 3


def _new_nodes(limits: ValueLimits, budget: _NodeBudget | None) -> list[Any]:
    """Return a fresh node counter for one value."""
    if budget is None:
        return [0, limits.max_nodes, limits.max_depth, None]
    remaining = budget.limit - budget.used
    if remaining < limits.max_nodes:
        return [0, remaining, limits.max_depth, budget]
    return [0, limits.max_nodes, limits.max_depth, None]


def _overflow(nodes: list[Any]) -> ValueError:
    """Return the error for one node too many: the batch's or the value's limit."""
    budget = nodes[_BUDGET]
    if budget is not None:
        from .budgets import ReadLimitError

        return ReadLimitError(
            kind="nodes", limit=budget.limit, required=budget.used + nodes[_COUNT]
        )
    return ValueError("value exceeds the maximum item count")


# Leaf tags decoded without a recursive call: exact Python type and its error message.
_LEAVES: dict[int, tuple[type[Any], str]] = {
    _BOOL: (bool, "invalid bool value"),
    _INT: (int, "invalid int value"),
    _FLOAT: (float, "invalid float value"),
    _STR: (str, "invalid str value"),
    _BYTES: (bytes, "invalid bytes value"),
}


def _decode_value_node(node: Any, allow_custom: bool, depth: int, nodes: list[Any]) -> Any:
    """Recursively decode and validate a type-tagged MessagePack schema node.

    Args:
        node: Schema node returned by ``msgpack.unpackb``.
        allow_custom: Whether registered custom decoders may run.
        depth: Current nesting depth.
        nodes: Counter from ``_new_nodes`` enforcing this value's node and depth limits.

    Returns:
        Python value represented by the node.

    Raises:
        TypeError: A custom decoder returns a value of the wrong exact type.
        ValueError: The schema is invalid, a limit is exceeded, container values
            collide or are unhashable, or a custom codec is missing or disallowed.
        ReadLimitError: The batch's ``max_decoded_nodes`` budget is exhausted.

    Note:
        Container and mapping loops decode leaf children inline, with the same checks
        in the same order as a recursive call, because most nodes are leaves.
    """
    if depth > nodes[2]:
        raise ValueError("value exceeds the maximum nesting depth")
    nodes[0] += 1
    if nodes[0] > nodes[1]:
        raise _overflow(nodes)
    if type(node) is not list or not node or type(node[0]) is not int:
        raise ValueError("value contains an invalid schema node")
    tag = node[0]
    leaf = _LEAVES.get(tag)
    if leaf is not None:
        if len(node) != 2 or type(node[1]) is not leaf[0]:
            raise ValueError(leaf[1])
        return node[1]

    if tag == _NONE:
        if len(node) != 1:
            raise ValueError("invalid none value")
        return None
    if tag == _BIG_INT:
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
        if _MSGPACK_MIN_INT <= value <= _MSGPACK_MAX_INT:
            raise ValueError("large int value is not canonical")
        return value
    if tag == _COMPLEX:
        if len(node) != 3 or type(node[1]) is not float or type(node[2]) is not float:
            raise ValueError("invalid complex value")
        return complex(node[1], node[2])
    if tag == _TYPE_REFERENCE:
        if len(node) != 2 or type(node[1]) is not str:
            raise ValueError("unsupported type reference")
        if node[1].startswith("codec:"):
            with _CUSTOM_CODEC_LOCK:
                registered = _CUSTOM_CODECS_BY_NAME.get(node[1][6:])
            if registered is None:
                raise ValueError("type reference codec is not registered")
            return registered[0]
        if len(node) == 2 and node[1] == "torch.Tensor":
            from torch import Tensor

            return Tensor
        if len(node) != 2 or type(node[1]) is not str or node[1] not in _VALUE_TYPES:
            raise ValueError("unsupported type reference")
        return _VALUE_TYPES[node[1]]
    if tag in _ARRAY_NODE_TAGS:
        if len(node) != 2 or type(node[1]) is not bytes:
            raise ValueError("invalid array value")
        if tag == _NDARRAY_VALUE:
            return bytes_to_ndarray(node[1])
        return bytes_to_tensor(node[1])

    child_depth = depth + 1
    if tag in _CONTAINER_NODE_TAGS:
        if len(node) != 2 or type(node[1]) is not list:
            raise ValueError("invalid container value")
        values: list[Any] = []
        append = values.append
        inline = child_depth <= nodes[2]
        leaves = _LEAVES
        for child in node[1]:
            if (
                inline
                and type(child) is list
                and child
                and type(child[0]) is int
                and (kind := leaves.get(child[0])) is not None
            ):
                nodes[0] += 1
                if nodes[0] > nodes[1]:
                    raise _overflow(nodes)
                if len(child) != 2 or type(child[1]) is not kind[0]:
                    raise ValueError(kind[1])
                append(child[1])
            else:
                append(_decode_value_node(child, allow_custom, child_depth, nodes))
        if tag == _LIST:
            return values
        if tag == _TUPLE:
            return tuple(values)
        set_result: set[Any] = set()
        for value in values:
            try:
                duplicate = value in set_result
            except TypeError as exc:
                raise ValueError("set value is not hashable") from exc
            if duplicate:
                raise ValueError("set contains duplicate values")
            set_result.add(value)
        return set_result if tag == _SET else frozenset(set_result)

    if tag in _MAPPING_NODE_TAGS:
        if len(node) != 2 or type(node[1]) is not list:
            raise ValueError("invalid mapping value")
        mapping_result: dict[Any, Any] = {}
        inline = child_depth <= nodes[2]
        leaves = _LEAVES
        for pair in node[1]:
            if type(pair) is not list or len(pair) != 2:
                raise ValueError("mapping entries must be key/value pairs")
            key_node, value_node = pair
            if (
                inline
                and type(key_node) is list
                and key_node
                and type(key_node[0]) is int
                and (kind := leaves.get(key_node[0])) is not None
            ):
                nodes[0] += 1
                if nodes[0] > nodes[1]:
                    raise _overflow(nodes)
                if len(key_node) != 2 or type(key_node[1]) is not kind[0]:
                    raise ValueError(kind[1])
                key = key_node[1]
            else:
                key = _decode_value_node(key_node, allow_custom, child_depth, nodes)
            # The key is checked before its value is decoded, as before.
            try:
                duplicate = key in mapping_result
            except TypeError as exc:
                raise ValueError("mapping key is not hashable") from exc
            if duplicate:
                raise ValueError("mapping contains colliding keys")
            if (
                inline
                and type(value_node) is list
                and value_node
                and type(value_node[0]) is int
                and (kind := leaves.get(value_node[0])) is not None
            ):
                nodes[0] += 1
                if nodes[0] > nodes[1]:
                    raise _overflow(nodes)
                if len(value_node) != 2 or type(value_node[1]) is not kind[0]:
                    raise ValueError(kind[1])
                mapping_result[key] = value_node[1]
            else:
                mapping_result[key] = _decode_value_node(
                    value_node, allow_custom, child_depth, nodes
                )
        return mapping_result if tag == _DICT else frozendict(mapping_result)

    if tag == _CUSTOM:
        if len(node) != 3 or type(node[1]) is not str:
            raise ValueError("invalid custom value")
        if not allow_custom:
            raise ValueError("custom value decoding requires allow_custom=True")
        with _CUSTOM_CODEC_LOCK:
            codec = _CUSTOM_CODECS_BY_NAME.get(node[1])
        if codec is None:
            raise ValueError(f"custom value codec is not registered: {node[1]!r}")
        cls, _, decoder = codec
        payload = _decode_value_node(node[2], allow_custom, child_depth, nodes)
        custom_result = decoder(payload)
        if type(custom_result) is not cls:
            raise TypeError(
                f"custom decoder {node[1]!r} returned {type(custom_result)!r}, expected {cls!r}"
            )
        return custom_result

    raise ValueError(f"unsupported value tag: {tag!r}")


def _unpack_value(
    buffer: bytes | bytearray | memoryview,
    *,
    allow_custom: bool = False,
    limits: ValueLimits | None = None,
    budget: _NodeBudget | None = None,
) -> Any:
    """Validate versioned MessagePack bytes and restore a Python value.

    Args:
        buffer: Bytes-like buffer produced by ``_pack_value``.
        allow_custom: Whether registered custom decoders may run.
        limits: Lower limits than the built-in ones, or None.
        budget: Batch node budget shared by several values, or None.

    Returns:
        Python value represented by the buffer.

    Raises:
        TypeError: The buffer is not bytes-like or ``allow_custom`` is not bool.
        ValueError: Size, prefix, MessagePack, schema or value constraints are violated.
        ReadLimitError: ``budget`` is exhausted.
    """
    if not isinstance(buffer, (bytes, bytearray, memoryview)):
        raise TypeError("buffer must be bytes-like")
    if type(allow_custom) is not bool:
        raise TypeError("allow_custom must be bool")
    if limits is None:  # The common case: built-in limits, no attribute lookups.
        max_bytes, max_nodes = _MAX_VALUE_BYTES, _MAX_VALUE_ITEMS
    else:
        max_bytes, max_nodes = limits.max_value_bytes, limits.max_nodes
    packed = bytes(buffer)
    if len(packed) > max_bytes:
        raise ValueError("encoded value exceeds the maximum byte count")
    if not packed.startswith(_VALUE_PREFIX):
        raise ValueError("unsupported value format or version")
    try:
        node = msgpack.unpackb(
            packed[len(_VALUE_PREFIX) :],
            raw=False,
            strict_map_key=True,
            max_str_len=max_bytes,
            max_bin_len=max_bytes,
            max_array_len=max_nodes,
            max_map_len=max_nodes,
            max_ext_len=max_bytes,
        )
    except (msgpack.exceptions.UnpackException, ValueError) as exc:
        raise ValueError("invalid MessagePack value") from exc
    if budget is None:
        nodes = (
            [0, _MAX_VALUE_ITEMS, _MAX_VALUE_DEPTH, None]
            if limits is None
            else [0, max_nodes, limits.max_depth, None]
        )
        return _decode_value_node(node, allow_custom, 0, nodes)
    nodes = _new_nodes(_DEFAULT_LIMITS if limits is None else limits, budget)
    try:
        return _decode_value_node(node, allow_custom, 0, nodes)
    finally:
        budget.used += nodes[0]


def _unpack_expected(
    buffer: bytes | bytearray | memoryview,
    expected: type[Any],
    *,
    allow_custom: bool = False,
    limits: ValueLimits | None = None,
    budget: _NodeBudget | None = None,
) -> Any:
    """Restore a value and require its exact type to match ``expected``.

    Args:
        buffer: Bytes-like buffer produced by ``_pack_value``.
        expected: Exact type required for the decoded value.
        allow_custom: Whether registered custom decoders may run.
        limits: Lower limits than the built-in ones, or None.
        budget: Batch node budget shared by several values, or None.

    Returns:
        Decoded value with exactly the required type; no container conversion occurs.

    Raises:
        TypeError: An option has the wrong type or a custom decoder returns the wrong type.
        ValueError: The decoded type differs from ``expected`` or the payload is invalid.
    """
    result = _unpack_value(buffer, allow_custom=allow_custom, limits=limits, budget=budget)
    if type(result) is not expected:
        raise ValueError(
            f"encoded value contains {type(result).__name__}, expected {expected.__name__}"
        )
    return result


def any_to_bytes(obj: Any) -> bytes:
    """Encode a supported value as versioned, type-tagged MessagePack bytes.

    Args:
        obj: Supported scalar, container, array, type reference or custom value.

    Returns:
        Typed bytes that ``bytes_to_any`` can restore.
    """
    return _pack_value(obj)


def bytes_to_any(buffer: bytes | bytearray | memoryview, *, allow_custom: bool = False) -> Any:
    """Restore a value from versioned, type-tagged MessagePack bytes.

    Args:
        buffer: Buffer produced by ``any_to_bytes``.
        allow_custom: Whether registered custom decoders may run. Disabled by default.

    Returns:
        The restored value, retaining its encoded Python type.
    """
    return _unpack_value(buffer, allow_custom=allow_custom)


def primitive_to_bytes(obj: bool | int | float | complex) -> bytes:
    """Encode a boolean or number as type-tagged MessagePack bytes.

    Args:
        obj: Boolean, integer, floating-point number or complex number.

    Returns:
        Bytes that ``bytes_to_primitive`` can restore.

    Raises:
        TypeError: The exact type is not bool, int, float or complex.
    """
    if type(obj) not in {bool, int, float, complex}:
        if isinstance(obj, np.generic):
            raise TypeError(_numpy_scalar_message(obj))
        raise TypeError(f"obj must be bool, int, float, or complex, got {type(obj).__name__}")
    return _pack_value(obj)


def bytes_to_primitive(
    buffer: bytes | bytearray | memoryview,
) -> bool | int | float | complex:
    """Restore a boolean or number from type-tagged MessagePack bytes.

    Args:
        buffer: Buffer produced by ``primitive_to_bytes``.

    Returns:
        Boolean, integer, floating-point number or complex number of the stored type.

    Raises:
        ValueError: The payload is invalid or contains a different value type.
    """
    result = _unpack_value(buffer)
    if type(result) not in {bool, int, float, complex}:
        raise ValueError("encoded value is not a primitive")
    return result


def str_to_bytes(obj: str) -> bytes:
    r"""Encode a string as UTF-8 bytes.

    Args:
        obj: String to encode.

    Returns:
        UTF-8 bytes, without a MessagePack wrapper.

    Note:
        Use ``bytes_to_str`` to decode the result.
    """
    return obj.encode("utf-8")


def bytes_to_str(buffer: bytes | bytearray | memoryview) -> str:
    r"""Decode UTF-8 bytes into a string.

    Args:
        buffer: UTF-8 bytes or a memoryview of them.

    Returns:
        Decoded string.

    Raises:
        UnicodeDecodeError: The input is not valid UTF-8.
    """
    return bytes(buffer).decode("utf-8")


def sequence_to_bytes(obj: tuple[Any, ...] | list[Any]) -> bytes:
    """Encode a tuple or list as type-tagged MessagePack bytes.

    Args:
        obj: Tuple or list to encode.

    Returns:
        Bytes retaining the exact container type.

    Raises:
        TypeError: The exact type is neither tuple nor list.
    """
    if type(obj) not in {tuple, list}:
        raise TypeError("obj must be a tuple or list")
    return _pack_value(obj)


def bytes_to_tuple(
    buffer: bytes | bytearray | memoryview, *, allow_custom: bool = False
) -> tuple[Any, ...]:
    """Restore a tuple from type-tagged MessagePack bytes.

    Args:
        buffer: Buffer produced by ``sequence_to_bytes`` from a tuple.
        allow_custom: Whether registered custom decoders may run.

    Returns:
        Restored tuple. A stored list is rejected rather than converted.
    """
    return _unpack_expected(buffer, tuple, allow_custom=allow_custom)


def bytes_to_list(
    buffer: bytes | bytearray | memoryview, *, allow_custom: bool = False
) -> list[Any]:
    """Restore a list from type-tagged MessagePack bytes.

    Args:
        buffer: Buffer produced by ``sequence_to_bytes`` from a list.
        allow_custom: Whether registered custom decoders may run.

    Returns:
        Restored list. A stored tuple is rejected rather than converted.
    """
    return _unpack_expected(buffer, list, allow_custom=allow_custom)


def set_to_bytes(obj: set[Any] | frozenset[Any]) -> bytes:
    """Encode a set or frozenset as type-tagged MessagePack bytes.

    Args:
        obj: Set or frozenset to encode.

    Returns:
        Bytes retaining the exact container type.

    Raises:
        TypeError: The exact type is neither set nor frozenset.
    """
    if type(obj) not in {set, frozenset}:
        raise TypeError("obj must be a set or frozenset")
    return _pack_value(obj)


def bytes_to_set(buffer: bytes | bytearray | memoryview, *, allow_custom: bool = False) -> set[Any]:
    """Restore a set from type-tagged MessagePack bytes.

    Args:
        buffer: Buffer produced by ``set_to_bytes`` from a set.
        allow_custom: Whether registered custom decoders may run.

    Returns:
        Restored set. A stored frozenset is rejected rather than converted.
    """
    return _unpack_expected(buffer, set, allow_custom=allow_custom)


def bytes_to_frozenset(
    buffer: bytes | bytearray | memoryview, *, allow_custom: bool = False
) -> frozenset[Any]:
    """Restore a frozenset from type-tagged MessagePack bytes.

    Args:
        buffer: Buffer produced by ``set_to_bytes`` from a frozenset.
        allow_custom: Whether registered custom decoders may run.

    Returns:
        Restored frozenset. A stored set is rejected rather than converted.
    """
    return _unpack_expected(buffer, frozenset, allow_custom=allow_custom)


def dict_to_bytes(obj: dict[Any, Any] | frozendict[Any, Any]) -> bytes:
    """Encode a dict or frozendict as type-tagged MessagePack bytes.

    Args:
        obj: Dict or frozendict to encode.

    Returns:
        Bytes retaining the exact mapping type and key/value order.

    Raises:
        TypeError: The exact type is neither dict nor frozendict.
    """
    if type(obj) not in {dict, frozendict}:
        raise TypeError("obj must be a dict or frozendict")
    return _pack_value(obj)


def bytes_to_dict(
    buffer: bytes | bytearray | memoryview, *, allow_custom: bool = False
) -> dict[Any, Any]:
    """Restore a dict from type-tagged MessagePack bytes.

    Args:
        buffer: Buffer produced by ``dict_to_bytes`` from a dict.
        allow_custom: Whether registered custom decoders may run.

    Returns:
        Restored dict. A stored frozendict is rejected rather than converted.
    """
    return _unpack_expected(buffer, dict, allow_custom=allow_custom)


def bytes_to_frozendict(
    buffer: bytes | bytearray | memoryview, *, allow_custom: bool = False
) -> frozendict[Any, Any]:
    """Restore a frozendict from type-tagged MessagePack bytes.

    Args:
        buffer: Buffer produced by ``dict_to_bytes`` from a frozendict.
        allow_custom: Whether registered custom decoders may run.

    Returns:
        Restored frozendict. A stored dict is rejected rather than converted.
    """
    return _unpack_expected(buffer, frozendict, allow_custom=allow_custom)


# Layout: [magic(4)][version(1)][header_len(4, LE)][header(repr dict)][raw data]
_MAGIC = b"NTB\x00"  # numpy/torch bytes
_VERSION = 2
# Compressed NumPy arrays use version 3, so readers without compression refuse them.
_COMPRESSED_VERSION = 3
_HEAD = struct.Struct("<4sBI")  # magic, version, header length

#: Array compressions ``write_ndarray`` accepts; zstd needs Python 3.14.
COMPRESSIONS = ("zlib", "lzma", "zstd")
_DEFAULT_LEVELS = {"zlib": 6, "lzma": 6, "zstd": 3}


def _zstd() -> Any:
    try:
        from compression import zstd  # pyright: ignore[reportMissingImports]  # Python 3.14+
    except ImportError:
        raise ValueError("zstd array compression requires Python 3.14 or later") from None
    return zstd


def _check_compression(name: object, level: object) -> None:
    """Validate a compression name and optional level before anything is written."""
    if name not in COMPRESSIONS:
        raise ValueError(f"compression must be one of {COMPRESSIONS} or None, got {name!r}")
    if name == "zstd":
        _zstd()
    if level is not None and (isinstance(level, bool) or not isinstance(level, int)):
        raise TypeError("compression_level must be an integer or None")


def _compress(name: str, data: bytes | memoryview, level: int | None) -> bytes:
    """Compress one payload; invalid levels raise ValueError."""
    chosen = _DEFAULT_LEVELS[name] if level is None else level
    try:
        if name == "zlib":
            return zlib.compress(data, chosen)
        if name == "lzma":
            return lzma.compress(data, preset=chosen)
        return _zstd().compress(data, level=chosen)
    except (zlib.error, lzma.LZMAError) as exc:
        raise ValueError(f"invalid {name} compression level {chosen}: {exc}") from None


def _decompress(name: str, data: bytes | memoryview, expected: int) -> bytes:
    """Decompress exactly ``expected`` bytes; any other outcome is a damaged payload.

    The output is bounded by ``expected``, so a hostile payload cannot expand further.
    """
    try:
        if name == "zlib":
            inflater = zlib.decompressobj()
            output = inflater.decompress(data, expected + 1)
            complete = inflater.eof and not inflater.unconsumed_tail and not inflater.unused_data
        elif name == "lzma":
            decoder = lzma.LZMADecompressor()
            output = decoder.decompress(data, max_length=expected + 1)
            complete = decoder.eof and not decoder.unused_data
        elif name == "zstd":
            decoder = _zstd().ZstdDecompressor()
            output = decoder.decompress(data, max_length=expected + 1)
            complete = decoder.eof and not decoder.unused_data
        else:
            raise ValueError(f"unknown array compression: {name!r}")
    except (zlib.error, lzma.LZMAError) as exc:
        raise ValueError(f"compressed array payload is damaged: {exc}") from None
    except Exception as exc:  # zstd raises compression.zstd.ZstdError
        if type(exc).__name__ != "ZstdError":
            raise
        raise ValueError(f"compressed array payload is damaged: {exc}") from None
    if not complete or len(output) != expected:
        raise ValueError("compressed array payload does not decompress to its shape")
    return output


# Exact repr of the metadata that ndarray_to_bytes and tensor_to_bytes write for simple
# dtypes. Matching it is much faster than ast.literal_eval; any other text,
# including extension or structured dtypes, still goes through literal_eval. Dimensions
# are limited to 18 digits so int() never meets the integer string-conversion limit.
_SHAPE_DIM = r"(?:0|[1-9][0-9]{0,17})"
_SHAPE_TEXT = rf"\((|{_SHAPE_DIM},|(?:{_SHAPE_DIM}, )+{_SHAPE_DIM})\)"
_NUMPY_META = re.compile(
    rf"\{{'kind': 'numpy', 'descr': '([<>|=][A-Za-z][0-9A-Za-z\[\]]*)', 'shape': {_SHAPE_TEXT}"
    r"(?:, 'chunk_rows': ([1-9][0-9]{0,17}))?\}"
)
_TORCH_META = re.compile(
    rf"\{{'kind': 'torch', 'dtype': '([a-z][0-9a-z_]*)', 'shape': {_SHAPE_TEXT}\}}"
)


def _pack(meta: dict, data: bytes, version: int = _VERSION) -> bytes:
    """Pack an array header, its byte length and the payload."""
    header = repr(meta).encode("latin1")
    return _HEAD.pack(_MAGIC, version, len(header)) + header + data


def _canonical_meta(text: str) -> dict[str, Any] | None:
    """Return what ast.literal_eval gives for canonical metadata text, else None."""
    chunk_rows = None
    if match := _NUMPY_META.fullmatch(text):
        meta: dict[str, Any] = {"kind": "numpy", "descr": match[1]}
        chunk_rows = match[3]
    elif match := _TORCH_META.fullmatch(text):
        meta = {"kind": "torch", "dtype": match[1]}
    else:
        return None
    dims = match[2]
    meta["shape"] = tuple(int(dim) for dim in dims.rstrip(",").split(", ")) if dims else ()
    if chunk_rows is not None:  # Chunked array headers written by write_ndarray(chunk_rows=).
        meta["chunk_rows"] = int(chunk_rows)
    return meta


def _unpack(buffer: bytes | bytearray | memoryview) -> tuple[dict, memoryview]:
    """Validate an array header and return metadata and a raw-payload view.

    Raises:
        ValueError: The header is truncated, the format or version is unsupported,
            or the metadata is not a dictionary.
        SyntaxError: The metadata is not a valid Python literal.
    """
    mv = memoryview(buffer)
    if len(mv) < _HEAD.size:
        raise ValueError("Truncated array header")
    magic, version, hlen = _HEAD.unpack_from(mv, 0)
    if magic != _MAGIC:
        raise ValueError(f"Unsupported array format (magic={magic!r})")
    if version not in (1, _VERSION, _COMPRESSED_VERSION):
        raise ValueError(
            f"Unsupported array version: {version}; supported versions are 1 to "
            f"{_COMPRESSED_VERSION}"
        )
    start = _HEAD.size
    if hlen > len(mv) - start:
        raise ValueError("Truncated array metadata")
    text = bytes(mv[start : start + hlen]).decode("latin1")
    meta = _canonical_meta(text)
    if meta is None:
        meta = ast.literal_eval(text)
    if not isinstance(meta, dict):
        raise ValueError("Array metadata must be a dict")
    compression = meta.get("compression")
    if (version == _COMPRESSED_VERSION) != (compression is not None):
        raise ValueError(f"array version {version} does not match its compression metadata")
    if compression is not None and (compression not in COMPRESSIONS or meta.get("kind") != "numpy"):
        raise ValueError(f"unsupported array compression: {compression!r}")
    return meta, mv[start + hlen :]  # Keep the payload as a zero-copy view.


def _checked_shape(meta: dict, data: memoryview, itemsize: int) -> tuple[int, ...]:
    """Validate dimensions and payload size before allocating the restored array."""
    shape = meta.get("shape")
    if not isinstance(shape, (tuple, list)) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise ValueError("Array shape must contain nonnegative integers")
    if math.prod(shape) * itemsize != len(data):
        raise ValueError("Array shape and payload size do not match")
    return tuple(shape)


def _resolve_named_numpy_dtype(name: str) -> np.dtype | None:
    """Resolve a dtype name through NumPy or the optional ml-dtypes package.

    Args:
        name: Dtype name to resolve.

    Returns:
        Available dtype, or None if neither provider recognizes the name.
    """
    try:
        return np.dtype(name)
    except TypeError:
        pass
    try:
        import ml_dtypes
    except ImportError:
        return None
    obj = getattr(ml_dtypes, name, None)
    if obj is None:
        return None
    try:
        return np.dtype(obj)
    except TypeError:
        return None


def _numpy_dtype_descr(dtype: np.dtype) -> tuple[list | str, bool]:
    r"""Return a dtype descriptor and whether its name must also be stored.

    Args:
        dtype: NumPy dtype to describe.

    Returns:
        ``(descr, needs_name)``. The flag is True when the descriptor alone cannot
        round-trip the exact dtype.

    Note:
        Extension dtypes such as bfloat16 and float8 may produce void or unusable
        descriptors. Their callers must also store the dtype name and item size.
    """
    try:
        descr = np.lib.format.dtype_to_descr(dtype)
    except Exception:
        descr = ""
    try:
        roundtrips = bool(descr) and np.lib.format.descr_to_dtype(descr) == dtype
    except Exception:
        roundtrips = False
    return descr, (not roundtrips)


def ndarray_to_bytes(
    array: NDArray[Any], *, compression: str | None = None, compression_level: int | None = None
) -> bytes:
    """Encode a NumPy array with its shape, dtype and format version.

    Supports scalar and empty arrays, structured dtypes and non-native byte order.
    Values are stored in C order; original strides and Fortran layout are not retained.

    Args:
        array: NumPy array to encode.
        compression: ``"zlib"``, ``"lzma"`` or ``"zstd"`` (Python 3.14+) compresses
            the values; None stores them raw.
        compression_level: Level for ``compression``; None uses 6 (zlib, lzma) or 3.

    Returns:
        Self-describing bytes that ``bytes_to_ndarray`` can restore.

    Raises:
        TypeError: The dtype contains Python objects.

    Examples:
        >>> array = np.arange(6, dtype=np.float32).reshape(2, 3)
        >>> bytes_to_ndarray(ndarray_to_bytes(array)).shape
        (2, 3)
    """
    meta, contiguous = _ndarray_meta(array)
    if compression is None:
        return _pack(meta, contiguous.tobytes())
    _check_compression(compression, compression_level)
    meta["compression"] = compression
    payload = _compress(compression, _raw_view(contiguous), compression_level)
    return _pack(meta, payload, _COMPRESSED_VERSION)


def _raw_view(contiguous: NDArray[Any]) -> memoryview:
    """Return the bytes of a C-contiguous array without copying them."""
    return memoryview(contiguous.reshape(-1).view(np.uint8))


def _ndarray_meta(array: NDArray[Any]) -> tuple[dict[str, Any], NDArray[Any]]:
    """Return an array's header metadata and its C-contiguous values.

    Raises:
        TypeError: The dtype contains Python objects.
    """
    # Contiguous conversion turns scalars into 1-D arrays; retain the original shape.
    shape = tuple(array.shape)
    dtype = array.dtype
    if dtype.hasobject:
        raise TypeError("Object dtype cannot be stored as raw bytes")
    contiguous = np.ascontiguousarray(array)
    descr, is_ext = _numpy_dtype_descr(dtype)
    meta: dict[str, Any] = {
        "kind": "numpy",
        "descr": descr,
        "shape": shape,
    }
    if is_ext:
        # Extension dtypes that cannot round-trip via descr need a name and item size.
        meta["dtype_name"] = dtype.name
        meta["itemsize"] = int(dtype.itemsize)
    return meta, contiguous


def _meta_to_numpy_dtype(meta: dict) -> np.dtype:
    """Restore a NumPy dtype from array metadata.

    Args:
        meta: Metadata with a descriptor or an extension dtype name and item size.

    Returns:
        Restored dtype. Unavailable extension dtypes fall back to a void dtype
        with the same item size.
    """
    name = meta.get("dtype_name")
    if name is not None:
        # Resolve extension dtypes, including ml-dtypes, by name.
        dtype = _resolve_named_numpy_dtype(name)
        if dtype is None:
            # An unavailable dtype falls back to void, retaining bytes and item size.
            dtype = np.dtype(f"V{meta['itemsize']}")
        return dtype
    return np.lib.format.descr_to_dtype(meta["descr"])


def bytes_to_ndarray(buffer: bytes | bytearray | memoryview) -> NDArray[Any]:
    """Restore a NumPy array from self-describing bytes.

    Args:
        buffer: Bytes produced by ``ndarray_to_bytes``.

    Returns:
        Array with the stored shape, dtype and values.

    Raises:
        TypeError: The dtype contains Python objects.
        ValueError: The header or payload size is invalid.

    Note:
        Unavailable extension dtypes fall back to void, retaining item size and
        raw bytes. The array references the input buffer: read-only over bytes,
        writable over a bytearray.
    """
    meta, data = _unpack(buffer)
    if meta.get("kind") != "numpy":
        raise ValueError("Expected NumPy array metadata")
    dtype = _meta_to_numpy_dtype(meta)
    if dtype.hasobject:
        raise TypeError("Object dtype cannot be restored from raw bytes")
    compression = meta.get("compression")
    if compression is not None:
        shape = meta.get("shape")
        if not isinstance(shape, (tuple, list)) or any(
            type(size) is not int or size < 0 for size in shape
        ):
            raise ValueError("Array shape must contain nonnegative integers")
        data = memoryview(_decompress(compression, data, math.prod(shape) * dtype.itemsize))
    shape = _checked_shape(meta, data, dtype.itemsize)
    try:
        flat = np.frombuffer(data, dtype=dtype)
    except (ValueError, TypeError):
        flat = np.frombuffer(data, dtype=np.uint8).view(dtype)
    return flat.reshape(shape)


def tensor_to_bytes(tensor: Tensor) -> bytes:
    """Encode a PyTorch tensor with its shape, dtype and format version.

    Detaches the tensor and makes it contiguous on the CPU before storing its bytes.

    Args:
        tensor: Tensor to encode. Requires the optional PyTorch dependency.

    Returns:
        Self-describing bytes that ``bytes_to_tensor`` can restore.

    Note:
        The original device, strides and ``requires_grad`` are not retained.

    Examples:
        >>> import torch
        >>> tensor = torch.arange(6).reshape(2, 3)
        >>> bytes_to_tensor(tensor_to_bytes(tensor)).shape
        torch.Size([2, 3])
    """
    tensor = tensor.detach().cpu().contiguous()
    nbytes = tensor.nelement() * tensor.element_size()
    ptr = tensor.data_ptr()
    data = ctypes.string_at(ptr, nbytes)
    meta = {
        "kind": "torch",
        "dtype": str(tensor.dtype).removeprefix(
            "torch."
        ),  # 'float32', 'bfloat16', 'float8_e4m3fn', ...
        "shape": tuple(tensor.shape),
    }
    return _pack(meta, data)


def bytes_to_tensor(buffer: bytes) -> Tensor:
    """Restore a CPU tensor from self-describing bytes.

    Args:
        buffer: Bytes produced by ``tensor_to_bytes``. Requires PyTorch.

    Returns:
        CPU tensor with the stored shape, dtype and values, including empty tensors.

    Raises:
        TypeError: The dtype name does not identify a ``torch.dtype``.
        ValueError: The header or payload size is invalid.

    Note:
        The result has no gradient history and ``requires_grad=False``.
    """
    return _tensor_from_buffer(buffer, owned=False)


def _tensor_from_owned(buffer: bytearray) -> Tensor:
    """Restore a tensor that takes over a private writable body without copying it.

    Readers copy tensor bodies from LMDB straight into a bytearray that nothing
    else references, so the tensor can use it in place. Its payload starts after
    the header and may be unaligned; PyTorch CPU operations accept that.
    """
    return _tensor_from_buffer(buffer, owned=True)


def _tensor_from_buffer(buffer: bytes | bytearray, *, owned: bool) -> Tensor:
    import torch

    meta, data = _unpack(buffer)
    if meta.get("kind") != "torch":
        raise ValueError("Expected Tensor metadata")
    if "compression" in meta:
        raise ValueError("compressed tensors are not supported")
    dtype = getattr(torch, meta["dtype"])
    if not isinstance(dtype, torch.dtype):
        raise TypeError(
            f"Expected torch.dtype, got {type(dtype)!r} (dtype string: {meta['dtype']!r})"
        )
    shape = _checked_shape(meta, data, torch.empty(0, dtype=dtype).element_size())
    if len(data) == 0:
        # torch.frombuffer rejects empty buffers; construct empty tensors directly.
        return torch.empty(shape, dtype=dtype)
    # A caller's buffer is copied; an owned bytearray is used in place.
    buf = data if owned else bytearray(data)
    try:
        flat = torch.frombuffer(buf, dtype=dtype)
    except (ValueError, RuntimeError, TypeError):
        # For unsupported frombuffer dtypes, read uint8 bytes and reinterpret them.
        flat = torch.frombuffer(buf, dtype=torch.uint8).view(dtype)
    return flat.reshape(shape)


def bytes_to_shape(buffer: bytes | bytearray | memoryview) -> tuple[int, ...]:
    """Read the shape from an array or tensor header without decoding its payload.

    Args:
        buffer: Buffer produced by ``ndarray_to_bytes`` or ``tensor_to_bytes``.

    Returns:
        Stored dimensions. No array or tensor is constructed.
    """
    meta, _ = _unpack(buffer)
    return tuple(meta["shape"])


def bytes_to_dtype(buffer: bytes | bytearray | memoryview) -> np.dtype | torch.dtype:
    """Read a dtype from an array or tensor header without decoding its payload.

    Args:
        buffer: Buffer produced by ``ndarray_to_bytes`` or ``tensor_to_bytes``.

    Returns:
        ``numpy.dtype`` for arrays or ``torch.dtype`` for tensors. Resolving a
        tensor dtype requires the optional PyTorch dependency.

    Raises:
        TypeError: The tensor dtype name does not identify a ``torch.dtype``.
        ValueError: The header kind is neither ``"numpy"`` nor ``"torch"``.
    """
    meta, _ = _unpack(buffer)
    kind = meta.get("kind")
    if kind == "torch":
        import torch

        dtype = getattr(torch, meta["dtype"])
        if not isinstance(dtype, torch.dtype):
            raise TypeError(
                f"Expected torch.dtype, got {type(dtype)!r} (dtype string: {meta['dtype']!r})"
            )
        return dtype
    if kind == "numpy":
        return _meta_to_numpy_dtype(meta)
    raise ValueError(f"Unknown array kind: {kind!r}")


__all__ = [
    "StorageKind",
    "any_to_bytes",
    "bytes_to_any",
    "bytes_to_dict",
    "bytes_to_dtype",
    "bytes_to_frozendict",
    "bytes_to_frozenset",
    "bytes_to_list",
    "bytes_to_ndarray",
    "bytes_to_primitive",
    "bytes_to_set",
    "bytes_to_shape",
    "bytes_to_str",
    "bytes_to_tensor",
    "bytes_to_tuple",
    "dict_to_bytes",
    "is_registered_bytes_type",
    "ndarray_to_bytes",
    "primitive_to_bytes",
    "register_bytes_codec",
    "sequence_to_bytes",
    "set_to_bytes",
    "storage_kind_for_type",
    "str_to_bytes",
    "tensor_to_bytes",
    "unregister_bytes_codec",
]
