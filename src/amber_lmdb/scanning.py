"""Streaming array statistics and whole-store validation with short transactions."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import lmdb

from . import _locations
from .arrays import _header_view, parse_array_header
from .budgets import DEFAULT_MAX_BYTES, ReadLimitError
from .chunks import CHUNK_TAG, CHUNKED_TAG, _chunked_info, _layout
from .codec import ValueLimits
from .collections import SourceChangedError
from .inspection import _entry_info, decode_key
from .integrity import _MAX_METADATA_BYTES, EntryMetadata, _parse_metadata, _verify_checksum
from .lmdb import (
    _ARRAY_KINDS,
    _METADATA_DB_NAME,
    LMDBReader,
    LMDBSnapshot,
    _chunk_location,
    _chunk_parent,
    _is_metadata_descriptor,
    _is_reserved,
    _opened_store,
)
from .progress import CancelEvent, ProgressCallback, _Tracker


@dataclass(frozen=True, slots=True)
class ArrayGroup:
    """Shape statistics for one writer tag, dtype and rank.

    Axis minima/maxima are independent bounds, not necessarily stored shapes.
    Smallest/largest shapes belong to real samples selected by element count;
    equal-sized ties keep the first binary-key-ordered sample. Scalars and empty
    arrays count as one sample each. Byte counts exclude array envelopes.
    """

    writer_tag: str
    dtype: str
    rank: int
    count: int
    min_shape: tuple[int, ...]
    max_shape: tuple[int, ...]
    smallest_shape: tuple[int, ...]
    largest_shape: tuple[int, ...]
    smallest_location: bytes
    largest_location: bytes
    payload_bytes: int


@dataclass(frozen=True, slots=True)
class ArrayStatistics:
    """Array groups from one pinned, transaction-ID-checked generation.

    All main-database values are visited, but only array headers are copied.
    Non-array bodies and custom decoders are never read. Memory is bounded by
    page/header limits and max_groups, not by the number of stored samples.
    """

    snapshot: LMDBSnapshot
    scanned_entries: int
    array_entries: int
    groups: tuple[ArrayGroup, ...]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A malformed entry or a value that could not be fully verified."""

    location: bytes | None
    category: Literal["error", "unverified"]
    message: str

    def _decoded(self) -> tuple[str | tuple[str, ...] | None, int | None]:
        if self.location is None:
            return None, None
        try:
            return decode_key(self.location)
        except ValueError:
            return None, None

    @property
    def key(self) -> str | tuple[str, ...] | None:
        """Decoded key of ``location``; None for store-level issues or malformed keys."""
        return self._decoded()[0]

    @property
    def index(self) -> int | None:
        """Decoded index of ``location``; None when absent or not decodable."""
        return self._decoded()[1]


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Whole-store coverage, with bounded issue details and uncapped issue counts.

    complete is false if any value was skipped due to byte limits, an unavailable
    optional dependency or blocked custom decoding, or if the scan was cancelled
    (a partial report carried by OperationCancelled). ok additionally requires no
    detected errors. Bounded bodies also require their key/tag/length-bound checksum
    to match. This detects corruption, including numeric body changes, but does not
    authenticate the source: an attacker can replace both a value and its checksum.

    ``unverified_by_reason`` counts every unverified value by reason (for example
    ``"optional dependency unavailable: torch"``) and ``entries_by_tag`` counts checked
    entries by writer tag; unlike ``issues`` neither is capped.
    """

    snapshot: LMDBSnapshot
    checked_entries: int
    tag_entries: int
    decoded_values: int
    skipped_values: int
    error_count: int
    issues: tuple[ValidationIssue, ...]
    cancelled: bool = False
    unverified_by_reason: dict[str, int] = field(default_factory=lambda: dict[str, int]())
    entries_by_tag: dict[str, int] = field(default_factory=lambda: dict[str, int]())

    @property
    def complete(self) -> bool:
        """Whether every value could be checked, independent of detected errors."""
        return self.skipped_values == 0 and not self.cancelled

    @property
    def ok(self) -> bool:
        """Whether coverage is complete and no invalid data was found."""
        return self.complete and self.error_count == 0


@dataclass(frozen=True, slots=True)
class _RawEntry:
    location: bytes
    header: bytes
    size: int
    tag: bytes | None
    payload: bytes | None
    skipped: bool


def _limits(batch_size: int, max_batch_bytes: int) -> None:
    if type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise ValueError("batch_size must be between 1 and 1024")
    if type(max_batch_bytes) is not int or max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be a positive integer")


@contextmanager
def _opened(
    path: str | Path, *, generation: str | None = None, cooperative: bool = True
) -> Iterator[tuple[Any, Any, LMDBSnapshot]]:
    """Open ``CURRENT`` (or ``generation``) read-only with its integrity database."""
    with _opened_store(path, generation=generation, cooperative=cooperative) as (env, key):
        tags = None
        with suppress(lmdb.NotFoundError):
            tags = env.open_db(_METADATA_DB_NAME, create=False)
        yield env, tags, LMDBSnapshot(key[0], key[1], env.info()["last_txnid"])


def _entry_total(env: Any, tags: Any) -> int:
    """Main-database entries, without the integrity database's descriptor."""
    return env.stat()["entries"] - int(tags is not None)


def _pages(
    env: Any,
    tags: Any,
    snapshot: LMDBSnapshot,
    *,
    batch_size: int,
    max_batch_bytes: int,
    max_value_bytes: int,
    with_payload: bool,
    prefetcher: _locations.ScanPrefetcher | None = None,
) -> Iterator[list[_RawEntry]]:
    after: bytes | None = None
    while True:
        page: list[_RawEntry] = []
        copied = 0
        with env.begin(write=False, buffers=True) as transaction:
            if transaction.id() != snapshot.last_txnid:
                raise SourceChangedError("store changed between scan pages")
            with transaction.cursor() as cursor:
                found = cursor.first() if after is None else cursor.set_range(after)
                if found and after is not None and bytes(cursor.key()) == after:
                    found = cursor.next()
                while found and len(page) < batch_size:
                    location = bytes(cursor.key())
                    if not _is_metadata_descriptor(location):
                        if prefetcher is not None:
                            prefetcher.before(transaction, location)
                        value = cursor.value()
                        raw_tag = None if tags is None else transaction.get(location, db=tags)
                        tag_size = (
                            0 if raw_tag is None else min(len(raw_tag), _MAX_METADATA_BYTES + 1)
                        )
                        skipped = with_payload and (
                            len(value) > max_value_bytes or len(value) + tag_size > max_batch_bytes
                        )
                        base_cost = tag_size + (len(value) if with_payload and not skipped else 0)
                        if copied + base_cost > max_batch_bytes:
                            if page:
                                break
                            raise ReadLimitError(
                                kind="bytes",
                                limit=max_batch_bytes,
                                required=base_cost,
                                location=location,
                            )
                        tag = None if raw_tag is None else bytes(raw_tag[: _MAX_METADATA_BYTES + 1])
                        header_view = memoryview(b"")
                        if not with_payload and tag is not None:
                            try:
                                writer_tag = _parse_metadata(tag).writer_tag
                            except ValueError:
                                writer_tag = None
                            if writer_tag in _ARRAY_KINDS:
                                header_view = _header_view(value, 64 * 1024)
                        cost = base_cost + len(header_view)
                        if copied + cost > max_batch_bytes:
                            if page:
                                break
                            raise ReadLimitError(
                                kind="bytes",
                                limit=max_batch_bytes,
                                required=cost,
                                location=location,
                            )
                        payload = bytes(value) if with_payload and not skipped else None
                        # Full scans reuse the owned payload for header parsing, not a second copy.
                        header = payload if payload is not None else bytes(header_view)
                        copied += cost
                        page.append(_RawEntry(location, header, len(value), tag, payload, skipped))
                    found = cursor.next()
        if not page:
            return
        after = page[-1].location
        yield page


def array_statistics(
    path: str | Path,
    *,
    batch_size: int = 512,
    max_groups: int = 1024,
    generation: str | None = None,
    cooperative: bool = True,
    progress: ProgressCallback | None = None,
    cancel: CancelEvent | None = None,
) -> ArrayStatistics:
    """Scan all array headers, grouping by dtype, writer tag and rank.

    Transactions close between pages. Abort with SourceChangedError if the pinned
    generation changes in place; CURRENT replacement alone does not change it.
    Raise ValueError for malformed entries or more than max_groups groups. Never
    copy array bodies or import PyTorch; varying ranks get separate groups.
    ``progress`` receives a report after pages, outside transactions; a set
    ``cancel`` event raises OperationCancelled between pages, without a partial result.
    ``generation`` scans a named generation instead of ``CURRENT``;
    ``cooperative=False`` reads immutable media without coordination locks (see
    ``LMDBReader``).
    """
    _limits(batch_size, 1)
    if type(max_groups) is not int or max_groups <= 0:
        raise ValueError("max_groups must be a positive integer")
    groups: dict[tuple[str, str, int], ArrayGroup] = {}
    scanned = arrays = 0
    with _opened(path, generation=generation, cooperative=cooperative) as (env, tags, snapshot):
        tracker = _Tracker("array_statistics", progress, cancel, total=_entry_total(env, tags))
        tracker.check()
        for page in _pages(
            env,
            tags,
            snapshot,
            batch_size=batch_size,
            max_batch_bytes=DEFAULT_MAX_BYTES,
            max_value_bytes=1,
            with_payload=False,
        ):
            for raw in page:
                scanned += 1
                entry = _entry_info(raw.location, raw.header, raw.size, raw.tag)
                if entry.error is not None:
                    raise ValueError(f"invalid entry {raw.location.hex()}: {entry.error}")
                if entry.writer_tag not in _ARRAY_KINDS or _is_reserved(raw.location):
                    continue
                header = parse_array_header(raw.header, stored_bytes=raw.size)
                arrays += 1
                tag = str(entry.writer_tag)
                group_key = tag, header.dtype, len(header.shape)
                previous = groups.get(group_key)
                shape, numel = header.shape, header.numel
                if previous is None:
                    if len(groups) == max_groups:
                        raise ValueError("array statistics exceed max_groups")
                    groups[group_key] = ArrayGroup(
                        tag,
                        header.dtype,
                        len(shape),
                        1,
                        shape,
                        shape,
                        shape,
                        shape,
                        raw.location,
                        raw.location,
                        header.payload_bytes,
                    )
                else:
                    smallest = numel < math.prod(previous.smallest_shape)
                    largest = numel > math.prod(previous.largest_shape)
                    groups[group_key] = ArrayGroup(
                        tag,
                        header.dtype,
                        len(shape),
                        previous.count + 1,
                        tuple(min(a, b) for a, b in zip(previous.min_shape, shape, strict=True)),
                        tuple(max(a, b) for a, b in zip(previous.max_shape, shape, strict=True)),
                        shape if smallest else previous.smallest_shape,
                        shape if largest else previous.largest_shape,
                        raw.location if smallest else previous.smallest_location,
                        raw.location if largest else previous.largest_location,
                        previous.payload_bytes + header.payload_bytes,
                    )
            tracker.page(len(page), sum(len(raw.header) for raw in page))
        if env.info()["last_txnid"] != snapshot.last_txnid:
            raise SourceChangedError("store changed while statistics were assembled")
    tracker.finish()
    return ArrayStatistics(snapshot, scanned, arrays, tuple(groups[key] for key in sorted(groups)))


def _chunk_links(
    env: Any,
    tags: Any,
    snapshot: LMDBSnapshot,
    headers: list[tuple[bytes, bytes]],
    chunks: list[tuple[bytes, EntryMetadata]],
) -> list[tuple[bytes, str]]:
    """Check that chunked headers and chunks list each other, in one short transaction.

    Every chunk a verified header lists must exist with the listed digest and size,
    and every chunk must be listed by its parent header; chunk bodies are verified
    as entries of their own.
    """
    problems: list[tuple[bytes, str]] = []
    with env.begin(write=False, buffers=True) as transaction:
        if transaction.id() != snapshot.last_txnid:
            raise SourceChangedError("store changed during chunk validation")
        for location, payload in headers:
            layout = _layout(payload)
            for number in range(layout.count):
                chunk = _chunk_location(location, number)
                raw, body = transaction.get(chunk, db=tags), transaction.get(chunk)
                try:
                    metadata = None if raw is None else _parse_metadata(raw)
                except ValueError:
                    metadata = None
                if (
                    metadata is None
                    or body is None
                    or metadata.writer_tag != CHUNK_TAG
                    or metadata.size_bytes != layout.stored_size(number)
                    or len(body) != metadata.size_bytes
                    or metadata.digest != layout.digest(number)
                ):
                    problems.append((location, f"chunk {number} is missing or does not match"))
                    break
        for location, metadata in chunks:
            link = _chunk_parent(location)
            if link is None:
                problems.append((location, "malformed chunk key"))
                continue
            parent, number = link
            raw, header = transaction.get(parent, db=tags), transaction.get(parent)
            try:
                listed = (
                    raw is not None
                    and header is not None
                    and _parse_metadata(raw).writer_tag == CHUNKED_TAG
                    and number < (layout := _layout(header)).count
                    and layout.digest(number) == metadata.digest
                )
            except ValueError:
                listed = False
            if not listed:
                problems.append((location, "chunk is not listed by an array header"))
    return problems


def verify_store(
    path: str | Path,
    *,
    batch_size: int = 512,
    max_value_bytes: int = 64 * 1024**2,
    max_batch_bytes: int | None = None,
    max_issues: int = 1000,
    allow_custom: bool = False,
    generation: str | None = None,
    cooperative: bool = True,
    progress: ProgressCallback | None = None,
    cancel: CancelEvent | None = None,
    value_limits: ValueLimits | None = None,
) -> ValidationReport:
    """Validate every key/tag and decode bounded values after short transactions.

    Also scan the tag database for malformed keys and orphan tags. Memory-backed
    views never cross transactions. Payload copies are bounded by both byte limits;
    oversized values, missing optional packages and blocked custom codecs are
    unverified, never silently certified. Issue storage is capped by max_issues.
    All source pages must retain the same opened generation and transaction ID.

    Custom decoders are disabled by default. Explicit allow_custom=True runs caller
    code and is unsuitable for untrusted stores. This checks the typed format,
    not application invariants, numeric authenticity or physical LMDB page repair.

    ``progress`` receives a report after pages, outside transactions. A set
    ``cancel`` event raises OperationCancelled between pages; its ``partial`` report
    is marked cancelled, so it is never complete or ok. Reader options such as
    ``verify_checksums`` do not apply here: every bounded body is verified.
    ``value_limits`` lowers MessagePack decoding limits; values beyond them are errors.

    ``max_batch_bytes=None`` sizes pages to hold any value up to ``max_value_bytes``
    (64 MiB or ``max_value_bytes`` plus 1 MiB, whichever is larger), so raising
    ``max_value_bytes`` alone verifies larger values. ``generation`` and
    ``cooperative`` select the generation and locking as in ``array_statistics``.
    """
    if type(max_value_bytes) is not int or max_value_bytes <= 0:
        raise ValueError("max_value_bytes must be a positive integer")
    if max_batch_bytes is None:
        max_batch_bytes = max(64 * 1024**2, max_value_bytes + 1024**2)
    _limits(batch_size, max_batch_bytes)
    if type(max_issues) is not int or max_issues < 0:
        raise ValueError("max_issues must be a nonnegative integer")
    if type(allow_custom) is not bool:
        raise TypeError("allow_custom must be bool")
    if value_limits is not None and not isinstance(value_limits, ValueLimits):
        raise TypeError("value_limits must be a ValueLimits instance or None")
    issues: list[ValidationIssue] = []
    checked = decoded = skipped = errors = tag_count = 0
    reasons: dict[str, int] = {}
    tags_seen: dict[str, int] = {}

    def record(
        location: bytes | None,
        category: Literal["error", "unverified"],
        message: str,
        reason: str | None = None,
    ) -> None:
        nonlocal errors, skipped
        if category == "error":
            errors += 1
        else:
            skipped += 1
            key = reason or message
            reasons[key] = reasons.get(key, 0) + 1
        if len(issues) < max_issues:
            issues.append(ValidationIssue(location, category, message))

    def report(cancelled: bool = False) -> ValidationReport:
        return ValidationReport(
            snapshot,
            checked,
            tag_count,
            decoded,
            skipped,
            errors,
            tuple(issues),
            cancelled,
            dict(reasons),
            dict(sorted(tags_seen.items())),
        )

    def partial() -> ValidationReport:
        return report(cancelled=True)

    with (
        _opened(path, generation=generation, cooperative=cooperative) as (env, tags, snapshot),
        closing(_locations.ScanPrefetcher(snapshot.generation)) as prefetcher,
    ):
        tracker = _Tracker("verify_store", progress, cancel, total=_entry_total(env, tags))
        tracker.check(partial)
        if tags is None:
            record(None, "error", "missing writer-tag database")
        for page in _pages(
            env,
            tags,
            snapshot,
            batch_size=batch_size,
            max_batch_bytes=max_batch_bytes,
            max_value_bytes=max_value_bytes,
            with_payload=True,
            prefetcher=prefetcher,
        ):
            headers: list[tuple[bytes, bytes]] = []
            chunks: list[tuple[bytes, EntryMetadata]] = []
            for raw in page:
                checked += 1
                entry = _entry_info(
                    raw.location, raw.header, raw.size, raw.tag, check_array=not raw.skipped
                )
                if entry.error is not None:
                    record(raw.location, "error", entry.error)
                    continue
                assert entry.writer_tag is not None
                tags_seen[entry.writer_tag] = tags_seen.get(entry.writer_tag, 0) + 1
                if raw.skipped:
                    record(
                        raw.location,
                        "unverified",
                        "value exceeds the configured copy byte limit",
                        "value exceeds the copy limit",
                    )
                    continue
                assert raw.payload is not None and entry.writer_tag is not None
                try:
                    assert raw.tag is not None
                    metadata = _parse_metadata(raw.tag)
                    _verify_checksum(raw.location, metadata, raw.payload)
                    if entry.writer_tag == CHUNKED_TAG:
                        _chunked_info(raw.payload)
                        headers.append((raw.location, raw.payload))
                    elif entry.writer_tag == CHUNK_TAG:
                        chunks.append((raw.location, metadata))
                    else:
                        LMDBReader._decode_entry(
                            entry.writer_tag,
                            raw.payload,
                            allow_custom=allow_custom,
                            limits=value_limits,
                        )
                    decoded += 1
                except ImportError as error:
                    record(
                        raw.location,
                        "unverified",
                        f"optional dependency unavailable: {error}",
                        f"optional dependency unavailable: {error.name or error}",
                    )
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    SyntaxError,
                    UnicodeError,
                    OverflowError,
                ) as error:
                    # The codec uses this exact diagnostic for explicit consent.
                    blocked = (
                        not allow_custom
                        and str(error) == "custom value decoding requires allow_custom=True"
                    )
                    record(
                        raw.location,
                        "unverified" if blocked else "error",
                        f"{type(error).__name__}: {error}",
                        "custom decoding not allowed" if blocked else None,
                    )
            if headers or chunks:
                for location, message in _chunk_links(env, tags, snapshot, headers, chunks):
                    record(location, "error", message)
            tracker.page(len(page), sum(len(raw.payload or b"") for raw in page), partial)
        if tags is not None:
            after: bytes | None = None
            while True:
                problems: list[tuple[bytes, str]] = []
                locations: list[bytes] = []
                with env.begin(write=False, buffers=True) as transaction:
                    if transaction.id() != snapshot.last_txnid:
                        raise SourceChangedError("store changed during tag validation")
                    with transaction.cursor(db=tags) as cursor:
                        found = cursor.first() if after is None else cursor.set_range(after)
                        if found and after is not None and bytes(cursor.key()) == after:
                            found = cursor.next()
                        while found and len(locations) < batch_size:
                            location = bytes(cursor.key())
                            locations.append(location)
                            if (
                                _is_metadata_descriptor(location)
                                or transaction.get(location) is None
                            ):
                                problems.append((location, "orphan writer tag"))
                            found = cursor.next()
                if not locations:
                    break
                tag_count += len(locations)
                after = locations[-1]
                for location in locations:
                    try:
                        decode_key(location)
                    except (TypeError, ValueError) as error:
                        record(location, "error", f"invalid tag key: {error}")
                for location, message in problems:
                    record(location, "error", message)
                tracker.check(partial)
        if env.info()["last_txnid"] != snapshot.last_txnid:
            raise SourceChangedError("store changed while validation report was assembled")
    tracker.finish()
    return report()


__all__ = [
    "ArrayGroup",
    "ArrayStatistics",
    "ValidationIssue",
    "ValidationReport",
    "array_statistics",
    "verify_store",
]
