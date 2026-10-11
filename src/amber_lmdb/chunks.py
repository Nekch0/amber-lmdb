"""Chunked NumPy arrays: a header entry plus raw row chunks, each with its own envelope.

The header entry sits at the array's own key with writer tag ``write_ndarray_chunked``.
Its body is an array-codec header (kind, dtype, shape and ``chunk_rows``) followed by
a fixed trailer (payload bytes, rows per chunk, bytes per full chunk, chunk count) and
the 32-byte envelope digest of every chunk. Chunk ``n`` holds rows
``[n * chunk_rows, (n + 1) * chunk_rows)`` in C order under the reserved key
``("amber-lmdb", "chunk", <parent key hex>)`` with index ``n`` and writer tag
``write_array_chunk``. Because the header lists the chunk digests, the header's own
envelope covers the whole array, and a chunk must carry exactly the digest its header
lists. The trailer lets a reader plan chunk copies inside a transaction without
parsing the metadata text.

A compressed chunked array (array format version 3, ``compression`` in the metadata)
compresses every chunk on its own and lists the stored size of every chunk (8 bytes
each) between the trailer and the digests; the trailer keeps the uncompressed sizes.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import codec as _codec
from .integrity import _DIGEST_BYTES, _LENGTHS, _PREFIX_BYTES, EntryMetadata

CHUNKED_TAG = "write_ndarray_chunked"
CHUNK_TAG = "write_array_chunk"
# Payload bytes, rows per chunk, bytes per full chunk, chunk count.
_TRAILER = struct.Struct(">QQQQ")
# Integrity envelope size of every chunk, counted against read budgets.
_CHUNK_ENVELOPE_BYTES = _PREFIX_BYTES + _LENGTHS.size + len(CHUNK_TAG) + _DIGEST_BYTES


@dataclass(frozen=True, slots=True)
class _Layout:
    """Sizes and chunk digests from a header body's binary trailer."""

    payload_bytes: int
    chunk_rows: int
    chunk_bytes: int
    count: int
    digests: bytes
    stored: tuple[int, ...] | None = None  # stored chunk sizes of a compressed array

    @property
    def row_bytes(self) -> int:
        """Bytes per row; zero when the array has no chunks."""
        return self.chunk_bytes // self.chunk_rows if self.count else 0

    @property
    def rows(self) -> int | None:
        """Row count, or None when there are no chunks to derive it from."""
        return self.payload_bytes // self.row_bytes if self.count else None

    def size(self, number: int) -> int:
        """Return the byte length of chunk ``number``; only the last may be shorter."""
        if number == self.count - 1:
            return self.payload_bytes - number * self.chunk_bytes
        return self.chunk_bytes

    def stored_size(self, number: int) -> int:
        """Return the stored byte length of chunk ``number`` (compressed or raw)."""
        return self.size(number) if self.stored is None else self.stored[number]

    def digest(self, number: int) -> bytes:
        """Return the envelope digest that chunk ``number`` must carry."""
        return self.digests[number * _DIGEST_BYTES : (number + 1) * _DIGEST_BYTES]


def _encode_header(
    meta: dict[str, Any],
    payload_bytes: int,
    chunk_bytes: int,
    digests: list[bytes],
    stored: list[int] | None = None,
) -> bytes:
    """Return a header body: codec header, trailer, stored sizes and chunk digests."""
    trailer = _TRAILER.pack(payload_bytes, meta["chunk_rows"], chunk_bytes, len(digests))
    if stored is None:
        return _codec._pack(meta, trailer + b"".join(digests))
    sizes = struct.pack(f">{len(stored)}Q", *stored)
    return _codec._pack(meta, trailer + sizes + b"".join(digests), _codec._COMPRESSED_VERSION)


def _layout(view: bytes | memoryview) -> _Layout:
    """Read the trailer without parsing the metadata text, so it can run in a transaction.

    Raises:
        ValueError: The header body is truncated or its sizes are inconsistent.
    """
    if len(view) < _codec._HEAD.size:
        raise ValueError("truncated chunked array header")
    magic, version, length = _codec._HEAD.unpack_from(view)
    if magic != _codec._MAGIC:
        raise ValueError("chunked array header has an unsupported array format")
    start = _codec._HEAD.size + length
    if len(view) < start + _TRAILER.size:
        raise ValueError("truncated chunked array trailer")
    payload_bytes, chunk_rows, chunk_bytes, count = _TRAILER.unpack_from(view, start)
    compressed = version == _codec._COMPRESSED_VERSION
    sizes_bytes = 8 * count if compressed else 0
    if len(view) != start + _TRAILER.size + sizes_bytes + count * _DIGEST_BYTES:
        raise ValueError("chunked array header does not list one digest per chunk")
    if chunk_rows == 0:
        raise ValueError("chunked array chunk_rows must be positive")
    if payload_bytes == 0:
        consistent = count == 0
    else:
        consistent = (
            chunk_bytes > 0
            and chunk_bytes % chunk_rows == 0
            and payload_bytes % (chunk_bytes // chunk_rows) == 0
            and count == -(-payload_bytes // chunk_bytes)
        )
    if not consistent:
        raise ValueError("chunked array chunk count does not match its sizes")
    digests_start = start + _TRAILER.size + sizes_bytes
    stored = struct.unpack_from(f">{count}Q", view, start + _TRAILER.size) if compressed else None
    return _Layout(
        payload_bytes, chunk_rows, chunk_bytes, count, bytes(view[digests_start:]), stored
    )


@dataclass(frozen=True, slots=True)
class _ArrayInfo:
    """Validated dtype, shape and row size of a stored NumPy array."""

    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    row_bytes: int
    compression: str | None = None


def _numpy_info(meta: dict[str, Any]) -> _ArrayInfo:
    """Validate NumPy metadata of an array with at least one dimension."""
    if meta.get("kind") != "numpy":
        raise ValueError("row reads need a NumPy array")
    dtype = _codec._meta_to_numpy_dtype(meta)
    if dtype.hasobject:
        raise TypeError("object dtype cannot be restored from raw bytes")
    shape = meta.get("shape")
    if not isinstance(shape, (tuple, list)) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise ValueError("array shape must contain nonnegative integers")
    if not shape:
        raise ValueError("row reads need an array with at least one dimension")
    return _ArrayInfo(
        dtype, tuple(shape), math.prod(shape[1:]) * dtype.itemsize, meta.get("compression")
    )


def _plain_info(header: bytes | memoryview) -> _ArrayInfo:
    """Return the dtype, shape and row size of an unchunked array header."""
    meta, _ = _codec._unpack(header)
    return _numpy_info(meta)


def _chunked_info(header: bytes) -> _ArrayInfo:
    """Parse a chunked header's metadata and check it against its trailer."""
    layout = _layout(header)
    meta, _ = _codec._unpack(header)
    info = _numpy_info(meta)
    if meta.get("chunk_rows") != layout.chunk_rows:
        raise ValueError("chunked array chunk_rows does not match its trailer")
    if layout.payload_bytes != info.shape[0] * info.row_bytes or (
        layout.count and layout.row_bytes != info.row_bytes
    ):
        raise ValueError("chunked array shape and chunk sizes do not match")
    return info


type _Check = tuple[bytes, EntryMetadata, int, int]


class _ArrayRows:
    """Owned rows of a stored NumPy array, copied in a transaction and checked after it.

    ``data`` holds rows starting at row ``first_row``; ``start`` and ``stop`` are the
    requested rows, clamped to the array when its shape is known. ``checks`` lists the
    chunk byte ranges of ``data`` whose digests the reader verifies, and ``whole``
    marks ``data`` as the entire unchunked body, verified against the entry itself.
    ``len()`` is the number of bytes counted against read budgets.
    """

    __slots__ = (
        "checks",
        "chunked",
        "copied",
        "data",
        "first_row",
        "header",
        "start",
        "stop",
        "whole",
    )

    def __init__(
        self,
        *,
        header: bytes,
        chunked: bool,
        data: bytes | NDArray[np.uint8],
        first_row: int,
        start: int,
        stop: int,
        copied: int,
        checks: list[_Check],
        whole: bool,
    ) -> None:
        self.header = header
        self.chunked = chunked
        self.data = data
        self.first_row = first_row
        self.start = start
        self.stop = stop
        self.copied = copied
        self.checks = checks
        self.whole = whole

    def __len__(self) -> int:
        return self.copied

    def _raw(self, info: _ArrayInfo) -> tuple[bytes | NDArray[np.uint8], int]:
        """Return uncompressed values from row ``first_row`` on, and their byte offset."""
        if info.compression is None:
            offset = 0
            if self.whole:
                offset = _codec._HEAD.size + _codec._HEAD.unpack_from(self.header)[2]
            return self.data, offset
        view = memoryview(self.data)
        if self.whole:
            head = _codec._HEAD.size + _codec._HEAD.unpack_from(self.header)[2]
            expected = info.shape[0] * info.row_bytes
            return _codec._decompress(info.compression, view[head:], expected), 0
        layout = _layout(self.header)
        first = self.first_row // layout.chunk_rows
        parts = [
            _codec._decompress(info.compression, view[begin:end], layout.size(first + position))
            for position, (_, _, begin, end) in enumerate(self.checks)
        ]
        return b"".join(parts), 0

    def array(self) -> NDArray[Any]:
        """Return the requested rows as a read-only array that owns its memory."""
        info = _chunked_info(self.header) if self.chunked else _plain_info(self.header)
        rows = info.shape[0]
        start = min(self.start, rows)
        stop = min(max(self.stop, start), rows)
        size = (stop - start) * info.row_bytes
        if size == 0:
            empty = np.empty((stop - start, *info.shape[1:]), dtype=info.dtype)
            empty.flags.writeable = False
            return empty
        raw, offset = self._raw(info)
        offset += (start - self.first_row) * info.row_bytes
        if offset < 0 or offset + size > len(raw):
            raise ValueError("stored rows do not match the array header")
        if self.whole and size < len(raw):
            # Keep only the requested rows, not the whole verified body.
            data: bytes | NDArray[np.uint8] = bytes(memoryview(raw)[offset : offset + size])
            offset = 0
        else:
            data = raw
        flat = np.frombuffer(data, dtype=np.uint8, count=size, offset=offset)
        try:
            values = flat.view(info.dtype)
        except (ValueError, TypeError):
            values = np.frombuffer(flat.tobytes(), dtype=info.dtype)
        values.flags.writeable = False
        return values.reshape((stop - start, *info.shape[1:]))


__all__: list[str] = []
