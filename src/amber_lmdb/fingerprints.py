"""Content fingerprints and differences of generations, from integrity metadata only."""

from __future__ import annotations

import hashlib
import heapq
import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chunks import CHUNKED_TAG, _layout
from .collections import SourceChangedError
from .integrity import ChecksumAlgorithm, EntryMetadata, _parse_metadata
from .lmdb import (
    _RESERVED_MARK,
    LMDBSnapshot,
    _chunk_location,
    _is_reserved,
    decode_key,
)
from .progress import CancelEvent, ProgressCallback, _Tracker
from .scanning import _entry_total, _opened

_DOMAIN = b"amber-lmdb fingerprint v1\x00"

type KeyRef = tuple[str | tuple[str, ...], int | None]


@dataclass(frozen=True, slots=True)
class StoreFingerprint:
    """A digest of every entry's key and integrity envelope, in key order.

    Each envelope binds the key, writer tag, body length and body digest, so equal
    fingerprints mean equal logical content written with the same digest
    algorithms. The value does not depend on LMDB page layout, engine, compaction
    or write order. The reserved namespace (manifest, array chunks) is excluded; a
    chunked array's header envelope covers its chunks, whose digests it lists.

    Attributes:
        snapshot: Generation that was read.
        entries: Number of entries included.
        algorithms: Envelope digest algorithms present, sorted.
        digest: BLAKE2b-256 hex digest.
    """

    snapshot: LMDBSnapshot
    entries: int
    algorithms: tuple[ChecksumAlgorithm, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class StoreDiff:
    """Entries added, removed, changed or undetermined between two generations.

    ``changed`` covers a different tag, length or (same algorithm) digest.
    ``undetermined`` entries have equal tag and length but envelopes of different
    digest algorithms; ``compare_bodies=True`` resolves them by reading bodies.
    Key lists hold at most ``max_listed`` decoded ``(key, index)`` pairs each; the
    counts are complete. The reserved namespace (manifest, array chunks) is excluded,
    so a changed chunk shows as its changed array.
    """

    before: LMDBSnapshot
    after: LMDBSnapshot
    unchanged: int
    added_count: int
    removed_count: int
    changed_count: int
    undetermined_count: int
    added: tuple[KeyRef, ...] = field(default=())
    removed: tuple[KeyRef, ...] = field(default=())
    changed: tuple[KeyRef, ...] = field(default=())
    undetermined: tuple[KeyRef, ...] = field(default=())

    @property
    def identical(self) -> bool:
        """Whether both generations hold the same entries with determined equality."""
        return not (
            self.added_count or self.removed_count or self.changed_count or self.undetermined_count
        )


def _new_digest() -> Any:
    """Return the fingerprint hasher, already fed with its domain string."""
    return hashlib.blake2b(_DOMAIN, digest_size=32)


def _add_entry(digest: Any, location: bytes, envelope: bytes) -> None:
    """Feed one entry, length-prefixed so that entry boundaries are unambiguous."""
    digest.update(struct.pack(">I", len(location)) + location)
    digest.update(struct.pack(">I", len(envelope)) + envelope)


def _metadata_pages(
    env: Any, tags: Any, snapshot: LMDBSnapshot, page_size: int
) -> Iterator[list[tuple[bytes, bytes]]]:
    """Yield owned (location, envelope) pairs in key order, one short transaction per page."""
    after: bytes | None = None
    while True:
        page: list[tuple[bytes, bytes]] = []
        with env.begin(write=False, buffers=True) as transaction:
            if transaction.id() != snapshot.last_txnid:
                raise SourceChangedError("store changed between metadata pages")
            with transaction.cursor(db=tags) as cursor:
                found = cursor.first() if after is None else cursor.set_range(after + b"\x00")
                while found and len(page) < page_size:
                    location = bytes(cursor.key())
                    if _is_reserved(location):
                        # The manifest and array chunks are not content entries: chunk
                        # digests are listed in their array's header envelope.
                        found = cursor.set_range(location[:10] + _RESERVED_MARK + b"\x01")
                        continue
                    page.append((location, bytes(cursor.value())))
                    after = location
                    found = cursor.next()
        if page:
            yield page
        if not found:
            return


def _validated(path: str | Path, tags: Any) -> None:
    if tags is None:
        raise ValueError(f"store has no integrity metadata database: {path}")


def _page_size(page_size: int) -> None:
    if type(page_size) is not int or not 1 <= page_size <= 65536:
        raise ValueError("page_size must be between 1 and 65536")


def fingerprint_store(
    path: str | Path,
    *,
    generation: str | None = None,
    cooperative: bool = True,
    page_size: int = 4096,
    progress: ProgressCallback | None = None,
    cancel: CancelEvent | None = None,
) -> StoreFingerprint:
    """Fingerprint a dataset root's ``CURRENT`` generation, or its named ``generation``.

    Reads only integrity metadata, in bounded pages with short transactions; the
    pinned generation must not change in place (``SourceChangedError``). Bodies are
    not read, so this identifies content as recorded by the envelopes; use
    ``verify_store`` to check bodies against them. ``progress`` and ``cancel`` work
    as in ``verify_store``; cancellation gives no partial result.
    ``cooperative=False`` reads immutable media without coordination locks.
    """
    _page_size(page_size)
    digest = _new_digest()
    algorithms: set[ChecksumAlgorithm] = set()
    entries = 0
    with _opened(path, generation=generation, cooperative=cooperative) as (env, tags, snapshot):
        _validated(path, tags)
        tracker = _Tracker("fingerprint_store", progress, cancel, total=_entry_total(env, tags))
        tracker.check()
        for page in _metadata_pages(env, tags, snapshot, page_size):
            for location, envelope in page:
                algorithms.add(_parse_metadata(envelope).checksum)
                _add_entry(digest, location, envelope)
            entries += len(page)
            tracker.page(len(page), sum(len(loc) + len(env_) for loc, env_ in page))
    tracker.finish()
    return StoreFingerprint(snapshot, entries, tuple(sorted(algorithms)), digest.hexdigest())


def _compare(
    first: EntryMetadata, second: EntryMetadata, raw_first: bytes, raw_second: bytes
) -> str:
    if raw_first == raw_second:
        return "unchanged"
    if first.writer_tag != second.writer_tag or first.size_bytes != second.size_bytes:
        return "changed"
    if first.checksum == second.checksum:
        return "changed"
    return "undetermined"


def diff_stores(
    before: str | Path,
    after: str | Path,
    *,
    before_generation: str | None = None,
    after_generation: str | None = None,
    cooperative: bool = True,
    max_listed: int = 1000,
    compare_bodies: bool = False,
    max_value_bytes: int = 64 * 1024**2,
    page_size: int = 4096,
    progress: ProgressCallback | None = None,
    cancel: CancelEvent | None = None,
) -> StoreDiff:
    """List entries added, removed, changed or undetermined from ``before`` to ``after``.

    Each side is a dataset root's ``CURRENT`` generation, or the generation named by
    ``before_generation``/``after_generation``, so two generations of one root can
    be compared.

    A merge of both generations' integrity metadata in key order; bodies are read
    only with ``compare_bodies=True`` and only for entries whose envelopes use
    different digest algorithms with equal tag and length (bodies larger than
    ``max_value_bytes`` stay undetermined). Both generations must stay unchanged in
    place. ``progress`` and ``cancel`` work as in ``verify_store``, without a
    partial result. ``max_listed`` caps each key list (counts stay complete);
    ``cooperative=False`` reads immutable media without coordination locks.
    """
    _page_size(page_size)
    if type(max_listed) is not int or max_listed < 0:
        raise ValueError("max_listed must be a nonnegative integer")
    if type(compare_bodies) is not bool:
        raise TypeError("compare_bodies must be bool")
    if type(max_value_bytes) is not int or max_value_bytes <= 0:
        raise ValueError("max_value_bytes must be a positive integer")
    counts = {"unchanged": 0, "added": 0, "removed": 0, "changed": 0, "undetermined": 0}
    samples: dict[str, list[KeyRef]] = {name: [] for name in counts}

    def record(kind: str, location: bytes) -> None:
        counts[kind] += 1
        if kind != "unchanged" and len(samples[kind]) < max_listed:
            samples[kind].append(decode_key(location))

    with (
        _opened(before, generation=before_generation, cooperative=cooperative) as (
            env_a,
            tags_a,
            snap_a,
        ),
        _opened(after, generation=after_generation, cooperative=cooperative) as (
            env_b,
            tags_b,
            snap_b,
        ),
    ):
        _validated(before, tags_a)
        _validated(after, tags_b)
        total = _entry_total(env_a, tags_a) + _entry_total(env_b, tags_b)
        tracker = _Tracker("diff_stores", progress, cancel, total=total)
        tracker.check()

        def stream(env: Any, tags: Any, snap: LMDBSnapshot, side: int):
            for page in _metadata_pages(env, tags, snap, page_size):
                yield from ((location, side, envelope) for location, envelope in page)
                tracker.page(len(page))

        def bodies_equal(location: bytes, tag: str) -> bool | None:
            with (
                env_a.begin(write=False, buffers=True) as first,
                env_b.begin(write=False, buffers=True) as second,
            ):
                if first.id() != snap_a.last_txnid or second.id() != snap_b.last_txnid:
                    raise SourceChangedError("store changed while comparing bodies")
                pairs = [(first.get(location), second.get(location))]
                if tag == CHUNKED_TAG:
                    # Chunk digests differ with the algorithm; compare the rest and the chunks.
                    try:
                        layout = _layout(pairs[0][0] or b"")
                    except ValueError:
                        return None
                    end = len(pairs[0][0] or b"") - 32 * layout.count
                    pairs = [(pairs[0][0][:end], (pairs[0][1] or b"")[:end])] + [
                        (first.get(chunk), second.get(chunk))
                        for chunk in (_chunk_location(location, n) for n in range(layout.count))
                    ]
                for value_a, value_b in pairs:
                    if value_a is None or value_b is None:
                        return None
                    if len(value_a) > max_value_bytes or len(value_b) > max_value_bytes:
                        return None
                    if value_a != value_b:
                        return False
                return True

        merged = heapq.merge(
            stream(env_a, tags_a, snap_a, 0),
            stream(env_b, tags_b, snap_b, 1),
            key=lambda item: (item[0], item[1]),
        )
        pending: tuple[bytes, int, bytes] | None = None
        for item in merged:
            if pending is not None and pending[0] == item[0]:
                location, raw_a, raw_b = item[0], pending[2], item[2]
                kind = _compare(_parse_metadata(raw_a), _parse_metadata(raw_b), raw_a, raw_b)
                if kind == "undetermined" and compare_bodies:
                    equal = bodies_equal(location, _parse_metadata(raw_a).writer_tag)
                    if equal is not None:
                        kind = "unchanged" if equal else "changed"
                record(kind, location)
                pending = None
                continue
            if pending is not None:
                record("removed" if pending[1] == 0 else "added", pending[0])
            pending = item
        if pending is not None:
            record("removed" if pending[1] == 0 else "added", pending[0])
    tracker.finish()
    return StoreDiff(
        snap_a,
        snap_b,
        counts["unchanged"],
        counts["added"],
        counts["removed"],
        counts["changed"],
        counts["undetermined"],
        tuple(samples["added"]),
        tuple(samples["removed"]),
        tuple(samples["changed"]),
        tuple(samples["undetermined"]),
    )


__all__ = ["KeyRef", "StoreDiff", "StoreFingerprint", "diff_stores", "fingerprint_store"]
