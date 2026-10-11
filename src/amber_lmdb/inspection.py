"""Inspect a published generation without decoding array bodies or custom objects."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb

from . import codec
from .arrays import _header_view, parse_array_header
from .budgets import DEFAULT_MAX_BYTES, ReadLimitError
from .integrity import (
    _MAX_METADATA_BYTES,
    ChecksumAlgorithm,
    EntryMetadata,
    _parse_metadata,
    _verify_checksum,
)
from .lmdb import (
    _ARRAY_KINDS,
    _METADATA_DB_NAME,
    _VALID_READERS,
    LMDBSnapshot,
    _acquire_env,
    _check_array_tag,
    _environment_key,
    _is_metadata_descriptor,
    _release_env,
    decode_key,
    get_loc,
)

_MAX_ARRAY_HEADER = 64 * 1024


@dataclass(frozen=True, slots=True)
class EntryInfo:
    """A key, its writer tag and optional array header, never its array values.

    ``key`` is None when the binary key is malformed. ``error`` describes malformed
    keys, tags or array headers. ``size_bytes`` is the stored value's byte length.
    ``checksum`` names the envelope digest; it is not verified for header samples.
    """

    location: bytes
    key: str | tuple[str, ...] | None
    index: int | None
    writer_tag: str | None
    size_bytes: int
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    error: str | None = None
    checksum: ChecksumAlgorithm | None = None


@dataclass(frozen=True, slots=True)
class StoreInfo:
    """Statistics, optional metadata and a bounded page from one read transaction.

    ``entries`` is LMDB's main database count, including named database descriptors.
    ``data_entries`` excludes the amber-lmdb tag database descriptor if present.
    ``used_bytes`` approximates allocated pages, not live data or all generations.
    ``meta`` is decoded after the transaction ends, with custom decoding disabled.
    ``next_after`` is an opaque pagination token. Subsequent calls must compare
    snapshots because they open fresh transactions and may see a new generation.
    ``lib_version`` is the opened native engine version, not the binding version.
    """

    snapshot: LMDBSnapshot
    entries: int
    data_entries: int
    used_bytes: int
    page_size: int
    meta: dict[Any, Any] | None = None
    meta_error: str | None = None
    samples: tuple[EntryInfo, ...] = ()
    next_after: bytes | None = None
    lib_version: tuple[int, int, int] | None = None


def _entry_info(
    location: bytes,
    header_bytes: bytes,
    size_bytes: int,
    raw_tag: bytes | None,
    *,
    check_array: bool = True,
) -> EntryInfo:
    key: str | tuple[str, ...] | None = None
    index = None
    tag = None
    shape = None
    dtype = None
    error = None
    checksum = None
    try:
        key, index = decode_key(location)
        if raw_tag is None:
            raise ValueError("missing writer tag")
        metadata = _parse_metadata(raw_tag)
        tag, checksum = metadata.writer_tag, metadata.checksum
        if metadata.size_bytes != size_bytes:
            raise ValueError("encoded value length does not match integrity metadata")
        if tag not in _VALID_READERS:
            raise ValueError(f"unknown writer tag: {tag!r}")
        if check_array and tag in _ARRAY_KINDS:
            header = parse_array_header(header_bytes, stored_bytes=size_bytes)
            _check_array_tag(tag, header)
            shape, dtype = header.shape, header.dtype
    except (ValueError, TypeError, KeyError, SyntaxError, UnicodeError, OverflowError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return EntryInfo(location, key, index, tag, size_bytes, shape, dtype, error, checksum)


def inspect_store(
    path: str | Path,
    *,
    with_meta: bool = False,
    entry_limit: int = 0,
    after: bytes | None = None,
    max_meta_bytes: int = 64 * 1024**2,
    max_batch_bytes: int = DEFAULT_MAX_BYTES,
    metadata_key: str | Path | tuple[str, ...] = "meta",
    generation: str | None = None,
    cooperative: bool = True,
) -> StoreInfo:
    """Inspect the actually opened generation using normal LMDB locks.

    Default inspection reads statistics only. ``entry_limit`` is 0 through 1024;
    array inspection reads headers via transaction-local views, without copying
    array payloads. Optional ``meta`` must have a dictionary payload and a matching
    writer tag. Missing or invalid metadata is reported in ``meta_error``.
    ``metadata_key`` selects a caller-defined unindexed dictionary key; its
    default preserves the legacy ``"meta"`` convention.

    max_batch_bytes bounds combined owned bodies/headers/integrity metadata copies;
    max_meta_bytes additionally bounds the optional metadata body/envelope. Neither
    bounds decoder allocations or runtime. Header samples do not verify body checksums.

    The read transaction is short and closes before metadata decoding. This is an
    inspector for the published amber-lmdb layout, not arbitrary LMDB
    databases. It does not establish NFS safety or validate every stored value.
    ``generation`` inspects a named generation instead of ``CURRENT``;
    ``cooperative=False`` reads immutable media without coordination locks.

    Raises:
        TypeError: An option has the wrong type.
        ValueError: A limit, layout or generation pointer is invalid.
        OSError: Files cannot be accessed.
        lmdb.Error: The generation cannot be opened or read.
    """
    if type(with_meta) is not bool or type(entry_limit) is not int:
        raise TypeError("with_meta must be bool and entry_limit must be int")
    if after is not None and not isinstance(after, bytes):
        raise TypeError("after must be bytes or None")
    if type(max_meta_bytes) is not int or type(max_batch_bytes) is not int:
        raise TypeError("inspection byte limits must be integers")
    if not 0 <= entry_limit <= 1024 or max_meta_bytes <= 0 or max_batch_bytes <= 0:
        raise ValueError("invalid inspection limit")
    meta_location = get_loc(metadata_key)
    key = _environment_key(
        path,
        readonly=True,
        readahead=False,
        max_readers=512,
        map_size=512 * 1024**3,
        generation=generation,
        cooperative=cooperative,
    )
    environment = _acquire_env(key, validated=True)
    raw_meta: bytes | None = None
    meta_integrity: EntryMetadata | None = None
    meta_error = None
    raw_entries: list[tuple[bytes, bytes, int, bytes | None]] = []
    next_after = None
    copied = 0
    try:
        metadata_db = None
        with suppress(lmdb.NotFoundError):
            metadata_db = environment.open_db(_METADATA_DB_NAME, create=False)
        with environment.begin(write=False, buffers=True) as transaction:
            stats = transaction.stat()
            info = environment.info()
            snapshot = LMDBSnapshot(key[0], key[1], transaction.id())
            lib_version = environment.lib_version()
            if with_meta:
                raw = transaction.get(meta_location)
                tag = (
                    transaction.get(meta_location, db=metadata_db)
                    if metadata_db is not None
                    else None
                )
                if raw is None or tag is None:
                    meta_error = "missing meta entry or invalid writer tag"
                elif len(raw) + len(tag) > max_meta_bytes:
                    meta_error = "metadata exceeds inspection limit"
                else:
                    if len(raw) + len(tag) > max_batch_bytes:
                        raise ReadLimitError(
                            kind="bytes",
                            limit=max_batch_bytes,
                            required=len(raw) + len(tag),
                            location=meta_location,
                        )
                    try:
                        meta_integrity = _parse_metadata(tag)
                        if meta_integrity.writer_tag not in ("write_dict", "write_any"):
                            raise ValueError("invalid metadata writer tag")
                        raw_meta = bytes(raw)
                        copied += len(raw) + len(tag)
                    except ValueError as error:
                        meta_error = str(error)
            if entry_limit:
                with transaction.cursor() as cursor:
                    found = cursor.first() if after is None else cursor.set_range(after)
                    if found and after is not None and bytes(cursor.key()) == after:
                        found = cursor.next()
                    while found:
                        location = bytes(cursor.key())
                        if not _is_metadata_descriptor(location):
                            if len(raw_entries) == entry_limit:
                                next_after = raw_entries[-1][0]
                                break
                            tag = (
                                transaction.get(location, db=metadata_db)
                                if metadata_db is not None
                                else None
                            )
                            value = cursor.value()
                            tag_size = 0 if tag is None else min(len(tag), _MAX_METADATA_BYTES + 1)
                            if copied + tag_size > max_batch_bytes:
                                raise ReadLimitError(
                                    kind="bytes",
                                    limit=max_batch_bytes,
                                    required=copied + tag_size,
                                    location=location,
                                )
                            tag_bytes = (
                                None if tag is None else bytes(tag[: _MAX_METADATA_BYTES + 1])
                            )
                            header_view = memoryview(b"")
                            if tag_bytes is not None:
                                try:
                                    writer_tag = _parse_metadata(tag_bytes).writer_tag
                                except ValueError:
                                    writer_tag = None
                                if writer_tag in _ARRAY_KINDS:
                                    header_view = _header_view(value, _MAX_ARRAY_HEADER)
                            required = copied + tag_size + len(header_view)
                            if required > max_batch_bytes:
                                raise ReadLimitError(
                                    kind="bytes",
                                    limit=max_batch_bytes,
                                    required=required,
                                    location=location,
                                )
                            copied = required
                            raw_entries.append(
                                (location, bytes(header_view), len(value), tag_bytes)
                            )
                        found = cursor.next()
    finally:
        # Release the acquired generation, not whichever CURRENT now selects.
        _release_env(key)
    meta = None
    if raw_meta is not None:
        try:
            assert meta_integrity is not None
            _verify_checksum(meta_location, meta_integrity, raw_meta)
            meta = codec.bytes_to_dict(raw_meta, allow_custom=False)
        except (ValueError, TypeError, KeyError, ImportError, SyntaxError) as exc:
            meta_error = f"{type(exc).__name__}: {exc}"
    entries = int(stats["entries"])
    return StoreInfo(
        snapshot=snapshot,
        entries=entries,
        data_entries=entries - int(metadata_db is not None),
        used_bytes=(int(info["last_pgno"]) + 1) * int(stats["psize"]),
        page_size=int(stats["psize"]),
        meta=meta,
        meta_error=meta_error,
        samples=tuple(_entry_info(*entry) for entry in raw_entries),
        next_after=next_after,
        lib_version=lib_version,
    )


__all__ = ["EntryInfo", "StoreInfo", "decode_key", "inspect_store"]
