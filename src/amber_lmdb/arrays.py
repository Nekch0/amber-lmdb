"""Owned array headers and shape bounds, without importing a tensor framework."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from . import codec

# Tensor wire headers contain names, not item sizes. Unknown future names remain
# inspectable, but payload-size validation needs the corresponding tensor codec.
_TENSOR_ITEM_SIZES = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "uint16": 2,
    "int16": 2,
    "float16": 2,
    "bfloat16": 2,
    "uint32": 4,
    "int32": 4,
    "float32": 4,
    "complex32": 4,
    "uint64": 8,
    "int64": 8,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
    "float8_e4m3fn": 1,
    "float8_e4m3fnuz": 1,
    "float8_e5m2": 1,
    "float8_e5m2fnuz": 1,
    "float8_e8m0fnu": 1,
    "float4_e2m1fn_x2": 1,
}


@dataclass(frozen=True, slots=True)
class ArrayHeader:
    """An array's kind, shape, dtype and byte counts, without its payload.

    Tensor dtype names are strings, so header inspection needs no PyTorch import.
    Unknown tensor names have itemsize=None; they are not fully size-validated.
    Payload byte counts include no envelope/header bytes. Scalars have shape=().
    ``chunk_rows`` is set for arrays written with ``write_ndarray(..., chunk_rows=)``;
    their payload lives in separate chunk entries and ``payload_bytes`` counts it.
    ``compression`` names the codec of a compressed array; ``payload_bytes`` is then
    the uncompressed size, which the stored bytes do not reveal without decoding.
    """

    kind: Literal["numpy", "torch"]
    shape: tuple[int, ...]
    dtype: str
    itemsize: int | None
    header_bytes: int
    payload_bytes: int
    chunk_rows: int | None = None
    compression: str | None = None

    @property
    def numel(self) -> int:
        """Return the element count; scalars contain one element, empty axes zero."""
        return math.prod(self.shape)


def _header_view(value: bytes | memoryview, maximum: int) -> memoryview:
    view = memoryview(value)
    if len(view) >= codec._HEAD.size:
        _, _, length = codec._HEAD.unpack_from(view)
        if length <= maximum:
            return view[: codec._HEAD.size + length]
    return view[: codec._HEAD.size]


def _copy_header(value: bytes | memoryview, maximum: int) -> bytes:
    return bytes(_header_view(value, maximum))


def parse_array_header(
    header: bytes,
    *,
    stored_bytes: int,
    max_header_bytes: int = 64 * 1024,
) -> ArrayHeader:
    """Validate owned header bytes against a stored value's total byte length.

    No array body is needed or decoded. Check magic/version, kind, shape, dtype,
    and payload length for NumPy and recognized tensor names. Object dtypes fail.
    The caller supplies stored_bytes, including the envelope, and copies only the
    prefix/header. Invalid/truncated metadata or byte counts raise ValueError or
    TypeError. Header size limits apply before parsing the Python literal.
    """
    if not isinstance(header, bytes):
        raise TypeError("header must be owned bytes")
    if type(stored_bytes) is not int or stored_bytes < 0:
        raise ValueError("stored_bytes must be a nonnegative integer")
    if type(max_header_bytes) is not int or max_header_bytes <= 0:
        raise ValueError("max_header_bytes must be a positive integer")
    if len(header) < codec._HEAD.size:
        raise ValueError("truncated array header")
    _, _, length = codec._HEAD.unpack_from(header)
    if length > max_header_bytes:
        raise ValueError("array header exceeds inspection limit")
    header_size = codec._HEAD.size + length
    if stored_bytes < header_size:
        raise ValueError("truncated stored array header")
    metadata, _ = codec._unpack(header)
    shape = metadata.get("shape")
    if not isinstance(shape, (tuple, list)) or any(
        type(size) is not int or size < 0 for size in shape
    ):
        raise ValueError("invalid array shape")
    kind = metadata.get("kind")
    itemsize: int | None
    if kind == "numpy":
        dtype = codec._meta_to_numpy_dtype(metadata)
        if dtype.hasobject:
            raise TypeError("object dtype cannot be restored from raw bytes")
        dtype_name = str(dtype)
        itemsize = int(dtype.itemsize)
    elif kind == "torch":
        name = metadata.get("dtype")
        if not isinstance(name, str) or not name or not name.isascii():
            raise ValueError("invalid tensor dtype")
        dtype_name = f"torch.{name}"
        itemsize = _TENSOR_ITEM_SIZES.get(name)
    else:
        raise ValueError("unknown array kind")
    payload_size = stored_bytes - header_size
    compression = metadata.get("compression")
    chunk_rows = metadata.get("chunk_rows")
    if chunk_rows is not None:
        # A chunked header stores a trailer, chunk sizes (when compressed) and chunk
        # digests instead of the payload.
        if kind != "numpy" or type(chunk_rows) is not int or chunk_rows <= 0 or not shape:
            raise ValueError("invalid chunked array header")
        assert itemsize is not None
        chunks = math.ceil(shape[0] / chunk_rows) if math.prod(shape) else 0
        per_chunk = 40 if compression is not None else 32
        if payload_size != 32 + per_chunk * chunks:
            raise ValueError("chunked array header does not list one digest per chunk")
        payload_size = math.prod(shape) * itemsize
        return ArrayHeader(
            kind,
            tuple(shape),
            dtype_name,
            itemsize,
            header_size,
            payload_size,
            chunk_rows,
            compression,
        )
    if compression is not None:
        assert itemsize is not None
        payload_size = math.prod(shape) * itemsize
    elif itemsize is not None and math.prod(shape) * itemsize != payload_size:
        raise ValueError("array shape and payload size do not match")
    return ArrayHeader(
        kind, tuple(shape), dtype_name, itemsize, header_size, payload_size, None, compression
    )


__all__ = ["ArrayHeader", "parse_array_header"]
