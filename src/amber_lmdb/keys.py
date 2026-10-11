"""Logical keys: unambiguous, length-prefixed LMDB key encoding and its inverse."""

from __future__ import annotations

import operator
import struct
from pathlib import PurePath

_KEY_MAGIC = b"MMFK\x01"
_MAX_INDEX = 2**64 - 1


def _index_value(index: object) -> int | None:
    """Return a validated index: None, or an integer (NumPy integers too) in [0, 2**64)."""
    if index is None:
        return None
    if isinstance(index, bool):
        raise TypeError("index must be an integer or None, not bool")
    try:
        value = operator.index(index)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(f"index must be an integer or None, got {type(index).__name__}") from None
    if value < 0 or value > _MAX_INDEX:
        raise ValueError("index must be in the range [0, 2**64)")
    return value


def get_loc(key: str | PurePath | tuple[str, ...], index: int | None = None) -> bytes:
    """Encode a logical key into unambiguous, length-prefixed LMDB bytes.

    Args:
        key: String, path or tuple of string components. Paths are encoded as their
            POSIX form (``a/b`` on every operating system), like a string key.
        index: Optional integer index in ``[0, 2**64)`` (Python or NumPy integers);
            None omits it.

    Returns:
        Bytes retaining the key kind, component boundaries and index presence.
        A string key and a one-component tuple key remain distinct.

    Raises:
        TypeError: The key, its components or the index have unsupported types.
        ValueError: The index is outside its valid range.

    Examples:
        >>> get_loc("features") != get_loc(("features",))
        True
    """
    if isinstance(key, str):
        components: tuple[str, ...] = (key,)
        key_kind = 0
    elif type(key) is tuple:
        if not all(isinstance(component, str) for component in key):
            raise TypeError("tuple key components must all be strings")
        components = key
        key_kind = 1
    elif isinstance(key, PurePath):
        components = (key.as_posix(),)
        key_kind = 0
    else:
        raise TypeError(f"key must be str, Path, or tuple[str, ...], got {type(key)!r}")
    # Plain ints in range, the common case, skip the general conversion.
    value = (
        index
        if index is None or (type(index) is int and 0 <= index <= _MAX_INDEX)
        else (_index_value(index))
    )
    encoded = bytearray(_KEY_MAGIC)
    encoded.extend((key_kind,))
    encoded.extend(struct.pack(">I", len(components)))
    for component in components:
        raw = component.encode("utf-8")
        encoded.extend(struct.pack(">Q", len(raw)))
        encoded.extend(raw)
    encoded.extend(b"\x00" if value is None else b"\x01" + struct.pack(">Q", value))
    return bytes(encoded)


def decode_key(location: bytes) -> tuple[str | tuple[str, ...], int | None]:
    """Reverse ``get_loc`` while validating lengths, UTF-8 and the optional index.

    Raises:
        TypeError: The input is not bytes.
        ValueError: The key is truncated, noncanonical or malformed.
    """
    if not isinstance(location, bytes):
        raise TypeError("location must be bytes")
    if not location.startswith(_KEY_MAGIC) or len(location) < len(_KEY_MAGIC) + 6:
        raise ValueError("invalid typed key header")
    offset = len(_KEY_MAGIC)
    kind = location[offset]
    count = struct.unpack_from(">I", location, offset + 1)[0]
    offset += 5
    if kind not in (0, 1) or (kind == 0 and count != 1):
        raise ValueError("invalid key kind or component count")
    components: list[str] = []
    # Each component needs at least an eight-byte length; reserve the index flag.
    if count > (len(location) - offset - 1) // 8:
        raise ValueError("truncated key components")
    for _ in range(count):
        if offset + 8 > len(location):
            raise ValueError("truncated key component length")
        size = struct.unpack_from(">Q", location, offset)[0]
        offset += 8
        if size > len(location) - offset - 1:
            raise ValueError("truncated key component")
        try:
            components.append(location[offset : offset + size].decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("invalid key UTF-8") from exc
        offset += size
    tail = location[offset:]
    if tail == b"\x00":
        index = None
    elif len(tail) == 9 and tail[0] == 1:
        index = struct.unpack_from(">Q", tail, 1)[0]
    else:
        raise ValueError("invalid key index or trailing bytes")
    key = components[0] if kind == 0 else tuple(components)
    return key, index


def describe_location(location: bytes | None) -> str:
    """Return a readable description of an encoded key for messages.

    Examples:
        >>> describe_location(get_loc(("s01", "pitch"), 3))
        "key ('s01', 'pitch') index 3"
    """
    if location is None:
        return "an unknown key"
    try:
        key, index = decode_key(location)
    except (TypeError, ValueError):
        return f"location {location.hex()}"
    return f"key {key!r}" if index is None else f"key {key!r} index {index}"


__all__ = ["decode_key", "describe_location", "get_loc"]
