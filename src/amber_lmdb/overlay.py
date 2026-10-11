"""Read several published stores as one keyspace in which no entry may repeat."""

from __future__ import annotations

import heapq
import struct
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from .budgets import DEFAULT_MAX_BYTES, ReadLimitError
from .chunks import _ArrayRows
from .codec import ValueLimits
from .errors import AmberLMDBError, KeyNotFoundError
from .integrity import EntryMetadata
from .keys import describe_location
from .lmdb import (
    KeyGroup,
    LMDBReader,
    LMDBSnapshot,
    TypedReader,
    _Body,
    _is_reserved,
    _key_prefix,
    _safe_key,
    _validate_page_size,
    get_loc,
)
from .progress import CancelEvent, ProgressCallback, _Tracker

if TYPE_CHECKING:
    from .manifest import StoreManifest


def _tagged[T](items: Iterable[T], layer: int) -> Iterator[tuple[T, int]]:
    """Pair every item with its layer number, bound now rather than when iterated."""
    for item in items:
        yield item, layer


class OverlayConflictError(AmberLMDBError, ValueError):
    """An entry exists in more than one layer of an overlay.

    Attributes:
        location: Encoded key of the repeated entry.
        layers: Paths of the layers that contain it, in overlay order.
    """

    def __init__(self, location: bytes, layers: tuple[Path, ...]) -> None:
        names = ", ".join(str(layer) for layer in layers)
        super().__init__(
            f"{describe_location(location)} exists in more than one overlay layer: {names}"
        )
        self.location = location
        self.layers = layers


class LMDBOverlayReader(TypedReader):
    """Read published stores as one keyspace, for example base features plus new ones.

    Each layer is an ordinary pinned ``LMDBReader`` with the same options. A key and
    index must exist in exactly one layer: reads, ``contains``, ``count``,
    ``iter_indices`` and ``iter_keys`` raise ``OverlayConflictError`` when an entry
    they resolve or list is in several layers, and ``check_disjoint`` verifies the
    whole overlay up front. Nothing is copied or modified.

    Batches look up integrity metadata in every layer, then copy each layer's share
    in one short transaction per layer; byte and item budgets cover the whole batch
    and results keep request order. ``get_snapshot`` returns one snapshot per layer.

    Args:
        paths: Published dataset directories, at least one, all distinct.
        map_size: Memory-map size limit for newly opened layer environments.
        readahead: Whether to enable OS readahead for newly opened environments.
        verify_checksums: Passed to every layer; see ``LMDBReader``.
        value_limits: Passed to every layer; see ``LMDBReader``.
        verify_threads: Passed to every layer; see ``LMDBReader``.
        cooperative: Passed to every layer; False reads immutable media without
            coordination locks (see ``LMDBReader``).

    Note:
        An entry whose integrity metadata is missing counts as absent, so a read
        reports ``KeyError`` rather than the corruption a single reader would report.
        Use ``verify_store`` on each layer to diagnose stores.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        *,
        map_size: int = 512 * 1024**3,
        readahead: bool = False,
        verify_checksums: bool = True,
        value_limits: ValueLimits | None = None,
        verify_threads: int = 0,
        cooperative: bool = True,
    ) -> None:
        if isinstance(paths, (str, bytes, Path)):
            raise TypeError("paths must be a sequence of dataset paths, not a single path")
        if not paths:
            raise ValueError("an overlay needs at least one layer")
        resolved = [Path(path).resolve() for path in paths]
        if len(set(resolved)) != len(resolved):
            raise ValueError("overlay layers must be distinct datasets")
        self._layers = tuple(
            LMDBReader(
                path,
                map_size=map_size,
                readahead=readahead,
                verify_checksums=verify_checksums,
                value_limits=value_limits,
                verify_threads=verify_threads,
                cooperative=cooperative,
            )
            for path in paths
        )
        self._value_limits = value_limits

    @property
    def paths(self) -> tuple[Path, ...]:
        """Return the layer paths in overlay order, as supplied."""
        return tuple(layer.path for layer in self._layers)

    @property
    def verify_checksums(self) -> bool:
        """Return whether body reads verify integrity digests before decoding."""
        return self._layers[0].verify_checksums

    @property
    def value_limits(self) -> ValueLimits | None:
        """Return the lowered MessagePack decoding limits, or None for the built-in ones."""
        return self._value_limits

    @property
    def verify_threads(self) -> int:
        """Return every layer's digest verification thread count."""
        return self._layers[0].verify_threads

    @property
    def cooperative(self) -> bool:
        """Return whether the layers take cooperative locks (False: read-only media)."""
        return self._layers[0].cooperative

    @property
    def closed(self) -> bool:
        """Return whether ``close()`` was called; a closed overlay cannot read again."""
        return self._layers[0].closed

    @property
    def generations(self) -> tuple[str | None, ...]:
        """Return every layer's pinned generation (None before its first read)."""
        return tuple(layer.generation for layer in self._layers)

    def __repr__(self) -> str:
        layers = getattr(self, "_layers", ())  # also during a failed construction
        names = ", ".join(repr(str(layer.path)) for layer in layers)
        state = "closed" if layers and layers[0].closed else "open"
        return f"LMDBOverlayReader([{names}], {state})"

    def get_snapshot(self) -> tuple[LMDBSnapshot, ...]:
        """Return the opened generation identity of every layer, in overlay order."""
        return tuple(layer.get_snapshot() for layer in self._layers)

    def manifests(self) -> tuple[StoreManifest | None, ...]:
        """Return every layer's writer manifest (or None), in overlay order."""
        return tuple(layer.manifest() for layer in self._layers)

    def get_lib_versions(self) -> tuple[tuple[int, int, int], ...]:
        """Return the LMDB engine version of every layer, in overlay order."""
        return tuple(layer.get_lib_version() for layer in self._layers)

    def _conflict(self, location: bytes, owners: Sequence[int]) -> OverlayConflictError:
        return OverlayConflictError(location, tuple(self._layers[i].path for i in owners))

    def _presence(self, locations: Sequence[bytes]) -> list[tuple[int, ...]]:
        """Return the layers holding each location; raise on entries in several layers."""
        found = [layer._present(locations) for layer in self._layers]
        holders = []
        for position, location in enumerate(locations):
            owners = tuple(i for i, present in enumerate(found) if present[position])
            if len(owners) > 1:
                raise self._conflict(location, owners)
            holders.append(owners)
        return holders

    def _owners(self, locations: Sequence[bytes]) -> list[int]:
        owners = []
        for location, holders in zip(locations, self._presence(locations), strict=True):
            if not holders:
                key, index = _safe_key(location)
                raise KeyNotFoundError(
                    f"no entry for {describe_location(location)} in any overlay layer",
                    key=key,
                    index=index,
                    location=location,
                )
            owners.append(holders[0])
        return owners

    def _owned_entries(
        self,
        locations: Sequence[bytes],
        read_method: str,
        *,
        max_bytes: int | None,
        partial: bool = False,
        snapshot: Any = None,
        mutable: bool = False,
    ) -> list[tuple[bytes, EntryMetadata, _Body]]:
        owners = self._owners(locations)
        snapshots = snapshot if snapshot is not None else (None,) * len(self._layers)
        used = 0
        if partial:
            # Pages must be a prefix of the request, so read runs of consecutive owners.
            result: list[tuple[bytes, EntryMetadata, _Body]] = []
            start = 0
            while start < len(locations):
                end = start
                while end < len(locations) and owners[end] == owners[start]:
                    end += 1
                layer = owners[start]
                remaining = None if max_bytes is None else max_bytes - used
                if remaining is not None and remaining <= 0 and result:
                    break
                try:
                    page = self._layers[layer]._owned_entries(
                        locations[start:end],
                        read_method,
                        max_bytes=remaining,
                        partial=True,
                        snapshot=snapshots[layer],
                        mutable=mutable,
                    )
                except ReadLimitError:
                    # A later run that does not fit ends this page, as in one store.
                    if result:
                        break
                    raise
                result.extend(page)
                used += sum(len(value) + metadata.metadata_bytes for _, metadata, value in page)
                if len(page) < end - start:
                    break
                start = end
            return result
        ordered: list[tuple[bytes, EntryMetadata, _Body] | None] = [None] * len(locations)
        for layer, reader in enumerate(self._layers):
            positions = [i for i, owner in enumerate(owners) if owner == layer]
            if not positions:
                continue
            remaining = None if max_bytes is None else max_bytes - used
            entries = reader._owned_entries(
                [locations[i] for i in positions],
                read_method,
                max_bytes=remaining,
                snapshot=snapshots[layer],
                mutable=mutable,
            )
            used += sum(len(value) + metadata.metadata_bytes for _, metadata, value in entries)
            for position, entry in zip(positions, entries, strict=True):
                ordered[position] = entry
        return [entry for entry in ordered if entry is not None]

    def _owned_rows(
        self,
        locations: Sequence[bytes],
        ranges: Sequence[tuple[int, int]],
        *,
        max_bytes: int | None,
    ) -> list[tuple[EntryMetadata, _ArrayRows]]:
        owners = self._owners(locations)
        ordered: list[tuple[EntryMetadata, _ArrayRows] | None] = [None] * len(locations)
        used = 0
        for layer, reader in enumerate(self._layers):
            positions = [i for i, owner in enumerate(owners) if owner == layer]
            if not positions:
                continue
            rows = reader._owned_rows(
                [locations[i] for i in positions],
                [ranges[i] for i in positions],
                max_bytes=None if max_bytes is None else max_bytes - used,
            )
            used += sum(len(item) + metadata.metadata_bytes for metadata, item in rows)
            for position, entry in zip(positions, rows, strict=True):
                ordered[position] = entry
        return [entry for entry in ordered if entry is not None]

    def _array_header_entries(
        self,
        locations: Sequence[bytes],
        max_header_bytes: int,
        max_bytes: int,
    ) -> list[tuple[str, bytes, int]]:
        owners = self._owners(locations)
        ordered: list[tuple[str, bytes, int] | None] = [None] * len(locations)
        used = 0
        for layer, reader in enumerate(self._layers):
            positions = [i for i, owner in enumerate(owners) if owner == layer]
            if not positions:
                continue
            entries, spent = reader._header_entries_used(
                [locations[i] for i in positions], max_header_bytes, max_bytes - used
            )
            used += spent
            for position, entry in zip(positions, entries, strict=True):
                ordered[position] = entry
        return [entry for entry in ordered if entry is not None]

    def contains(self, key: str | Path | tuple[str, ...], index: int | None = None) -> bool:
        """Return whether one layer holds the entry; raise if several do."""
        return self.contains_many([key], indices=[index])[0]

    def contains_many(
        self,
        keys: Sequence[str | Path | tuple[str, ...]],
        *,
        indices: Sequence[int | None] | None = None,
        max_items: int = 4096,
    ) -> list[bool]:
        """Check several keys against every layer; raise on entries in several layers."""
        locations = self._batch_locations(keys, indices, max_items, DEFAULT_MAX_BYTES)
        return [bool(holders) for holders in self._presence(locations)] if locations else []

    def index_range(self, key: str | Path | tuple[str, ...]) -> tuple[int, int] | None:
        """Return the smallest and largest index over all layers, with two seeks per layer.

        Bounds alone cannot reveal repeated entries; ``count`` and ``iter_indices`` do.
        """
        ranges = [bounds for layer in self._layers if (bounds := layer.index_range(key))]
        if not ranges:
            return None
        return min(first for first, _ in ranges), max(last for _, last in ranges)

    def count(self, key: str | Path | tuple[str, ...]) -> int:
        """Count indexed entries over all layers; raise if an index is in several layers."""
        return sum(len(page) for page in self.iter_indices(key, page_size=65536))

    def iter_indices(
        self,
        key: str | Path | tuple[str, ...],
        *,
        start: int = 0,
        page_size: int = 4096,
    ) -> Iterator[list[int]]:
        """Merge the layers' indices in ascending pages; raise on an index in several layers."""
        _validate_page_size(page_size)
        prefix = _key_prefix(key)
        streams = [
            _tagged(
                (index for page in reader.iter_indices(key, start=start) for index in page), layer
            )
            for layer, reader in enumerate(self._layers)
        ]
        page: list[int] = []
        previous: tuple[int, int] | None = None
        for index, layer in heapq.merge(*streams):
            if previous is not None and previous[0] == index:
                location = prefix + b"\x01" + struct.pack(">Q", index)
                raise self._conflict(location, (previous[1], layer))
            previous = (index, layer)
            page.append(index)
            if len(page) == page_size:
                yield page
                page = []
        if page:
            yield page

    def iter_keys(
        self, *, prefix: tuple[str, ...] = (), page_size: int = 1024
    ) -> Iterator[list[KeyGroup]]:
        """Merge the layers' key listings in encoded order; raise on repeated entries.

        A key present in several layers is allowed when its entries do not repeat:
        its unindexed entry may be in one layer only, and when index ranges overlap
        the indices are compared one by one.
        """
        _validate_page_size(page_size)
        streams = [
            (
                (get_loc(group.key), owner, group)
                for group, owner in _tagged(
                    (group for page in reader.iter_keys(prefix=prefix) for group in page), layer
                )
            )
            for layer, reader in enumerate(self._layers)
        ]
        page: list[KeyGroup] = []
        pending: list[tuple[int, KeyGroup]] = []

        def merged() -> KeyGroup:
            groups = [group for _, group in pending]
            unindexed = [layer for layer, group in pending if group.unindexed]
            if len(unindexed) > 1:
                raise self._conflict(get_loc(groups[0].key), unindexed)
            ranges = sorted(
                (group.first_index, group.last_index, layer)
                for layer, group in pending
                if group.first_index is not None and group.last_index is not None
            )
            if any(ranges[i + 1][0] <= ranges[i][1] for i in range(len(ranges) - 1)):
                # Overlapping ranges may still interleave; compare indices one by one.
                for _ in self.iter_indices(groups[0].key):
                    pass
            return KeyGroup(
                groups[0].key,
                bool(unindexed),
                min((r[0] for r in ranges), default=None),
                max((r[1] for r in ranges), default=None),
            )

        current: bytes | None = None
        for location, layer, group in heapq.merge(*streams, key=lambda item: item[0]):
            if location != current and pending:
                page.append(merged())
                pending = []
                if len(page) == page_size:
                    yield page
                    page = []
            current = location
            pending.append((layer, group))
        if pending:
            page.append(merged())
        if page:
            yield page

    def check_disjoint(
        self,
        *,
        page_size: int = 4096,
        progress: ProgressCallback | None = None,
        cancel: CancelEvent | None = None,
    ) -> int:
        """Verify that no entry is in more than one layer and return the entry count.

        Merges every layer's sorted integrity-metadata keys in bounded pages without
        reading bodies, and raises ``OverlayConflictError`` at the first repeat. Layer
        manifests and array chunks are not counted; chunks follow their array, and
        manifests never conflict (read them with ``manifests``).
        """
        _validate_page_size(page_size)
        total = 0
        for reader in self._layers:
            total += reader._get_env().stat()["entries"] - 1  # minus the integrity database
        tracker = _Tracker("check_disjoint", progress, cancel, total=total)
        tracker.check()
        streams = [
            _tagged(
                (
                    location
                    for page in reader._iter_locations(page_size)
                    for location in page
                    if not _is_reserved(location)
                ),
                layer,
            )
            for layer, reader in enumerate(self._layers)
        ]
        checked = 0
        previous: tuple[bytes, int] | None = None
        for location, layer in heapq.merge(*streams):
            if previous is not None and previous[0] == location:
                raise self._conflict(location, (previous[1], layer))
            previous = (location, layer)
            checked += 1
            if checked % page_size == 0:
                tracker.page(page_size)
        tracker.page(checked % page_size)
        tracker.finish()
        return checked

    def close(self) -> None:
        """Release every layer's environment reference; repeated calls are harmless."""
        for layer in self._layers:
            layer.close()

    def refresh(self) -> None:
        """Release every layer so each resolves its ``CURRENT`` again on the next read."""
        for layer in self._layers:
            layer.refresh()

    def __enter__(self) -> LMDBOverlayReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False


__all__ = ["LMDBOverlayReader", "OverlayConflictError"]
