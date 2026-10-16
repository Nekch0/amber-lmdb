"""File locations of value bodies, recorded at write time to prefetch bodies on reads.

A cold read through LMDB's memory map faults a body in one page at a time; on
network storage every page is a round trip. A writer therefore records, for each
indexed body of at least :data:`MIN_BYTES`, where it lies in ``data.mdb``. The table
of one logical key holds ``(index, offset, length)`` rows sorted by index, as
little-endian ``uint64`` bytes in a dictionary ``{"version": 1, "rows": ...}``
stored under the reserved key ``("amber-lmdb", "locations", <key prefix hex>)``. A
dictionary rather than an array keeps the tables out of array statistics, also
those of earlier versions. A reader looks a batch up in these
tables and asks the kernel for the bodies with ``posix_fadvise(WILLNEED)`` before
copying them, in pieces no larger than :data:`PIECE` because the kernel trims one
request to about the readahead size.

The tables are hints only: values are always read and checked through LMDB, so a
missing, stale or damaged table can cost speed but never change a result.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from .keys import decode_key, get_loc

RESERVED_COMPONENT = "amber-lmdb"  # The same reserved namespace as amber_lmdb.lmdb.
MIN_BYTES = 8 * 1024
"""Bodies smaller than this are not recorded; they span only a few pages."""

PIECE = 1024 * 1024
"""Largest single prefetch request, in bytes."""

GAP = 64 * 1024
"""Neighboring bodies closer than this are requested as one range."""

_TABLE_PREFIX = get_loc((RESERVED_COMPONENT, "locations", ""))[:-9]
"""Encoded prefix shared by every location table key (third component excluded)."""

FADVISE: Callable[[int, int, int, int], None] | None = getattr(os, "posix_fadvise", None)
WILLNEED: int = getattr(os, "POSIX_FADV_WILLNEED", 3)


def group_of(location: bytes) -> tuple[bytes, int] | None:
    """Split an indexed location into its key prefix and index; None otherwise.

    An indexed location is ``prefix + b"\\x01" + <8-byte big-endian index>``. An
    unindexed one ends with ``b"\\x00"``; decoding tells the two apart even when
    the bytes of an unindexed key happen to look like an index marker.
    """
    if len(location) < 9 or location[-9] != 1:
        return None
    prefix = location[:-9]
    try:
        _, index = decode_key(location)
    except ValueError:
        return None
    if index is None:
        return None
    return prefix, index


def table_location(prefix: bytes) -> bytes:
    """Return the reserved location of the table for the logical key ``prefix``."""
    return get_loc((RESERVED_COMPONENT, "locations", prefix.hex()))


def is_table_location(location: bytes) -> bool:
    """Whether ``location`` is a location table key."""
    return location.startswith(_TABLE_PREFIX)


def body_offset(view: memoryview, base: int) -> int:
    """Return the file offset of a body read through a memory map starting at ``base``."""
    return int(np.frombuffer(view, dtype=np.uint8).ctypes.data) - base


def merge_tables(parts: list[NDArray[np.uint64]]) -> NDArray[np.uint64]:
    """Merge recorded rows in write order; the last row of an index wins."""
    rows = np.concatenate(parts)[::-1]
    _, first = np.unique(rows[:, 0], return_index=True)
    return np.ascontiguousarray(rows[first])


def encode_table(table: NDArray[np.uint64]) -> dict[str, object]:
    """Return the stored form of a table."""
    return {"version": 1, "rows": table.astype("<u8", copy=False).tobytes()}


def decode_table(value: object) -> NDArray[np.uint64] | None:
    """Restore a table from its stored form; None when it is not one."""
    if not isinstance(value, dict) or value.get("version") != 1:
        return None
    rows = value.get("rows")
    if not isinstance(rows, bytes) or len(rows) % 24:
        return None
    return np.frombuffer(rows, dtype="<u8").reshape(-1, 3).astype(np.uint64)


def lookup(table: NDArray[np.uint64], index: int) -> tuple[int, int] | None:
    """Return ``(offset, length)`` of ``index`` in a table, or None."""
    position = int(np.searchsorted(table[:, 0], index))
    if position < table.shape[0] and int(table[position, 0]) == index:
        return int(table[position, 1]), int(table[position, 2])
    return None


def advise(descriptor: int, spans: Iterable[tuple[int, int]]) -> None:
    """Ask the kernel to read ``(offset, length)`` spans, merged and in pieces."""
    if FADVISE is None:
        return
    ordered = sorted(spans)
    if not ordered:
        return
    start, end = ordered[0][0], ordered[0][0] + ordered[0][1]
    for offset, length in ordered[1:]:
        if offset - end <= GAP:
            end = max(end, offset + length)
            continue
        _advise_range(descriptor, start, end)
        start, end = offset, offset + length
    _advise_range(descriptor, start, end)


def _advise_range(descriptor: int, start: int, end: int) -> None:
    assert FADVISE is not None
    for piece in range(start, end, PIECE):
        try:
            FADVISE(descriptor, piece, min(PIECE, end - piece), WILLNEED)
        except OSError:
            return
