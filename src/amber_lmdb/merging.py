"""Merge disjoint stores into one new store by copying entries and their envelopes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from .integrity import _parse_metadata, _verify_checksum
from .keys import describe_location
from .lmdb import (
    _MANIFEST_PREFIX,
    _METADATA_DB_NAME,
    LMDBWriter,
    _is_metadata_descriptor,
    _is_reserved,
    _opened_store,
)
from .overlay import LMDBOverlayReader
from .progress import CancelEvent, ProgressCallback, _Tracker

_PAGE_ENTRIES = 1024
_COMMIT_BYTES = 256 * 1024**2


def merge_stores(
    paths: Sequence[str | Path],
    destination: str | Path,
    *,
    overwrite: bool = False,
    map_size: int = 512 * 1024**3,
    lib_version: Literal[0, 1] | None = None,
    verify_checksums: bool = True,
    manifest: bool = False,
    provenance: Mapping[str, Any] | None = None,
    progress: ProgressCallback | None = None,
    cancel: CancelEvent | None = None,
) -> int:
    """Publish one store holding every entry of several disjoint stores.

    Typical use: fold a base store and the feature stores layered over it with
    ``LMDBOverlayReader`` into one self-contained store. Entries are copied as stored,
    bodies with their integrity envelopes (no decoding, no re-encoding), so chunked
    arrays and every value type survive unchanged; layer manifests are not copied
    (``manifest=True`` writes a new one). The sources must be disjoint
    (``OverlayConflictError`` otherwise) and stay unchanged.

    Args:
        paths: Source dataset roots, read at their ``CURRENT`` generations.
        destination: Dataset root to publish; see ``LMDBWriter``.
        overwrite: Replace a published ``destination``.
        map_size: Memory-map size limit of the new generation.
        lib_version: Engine of the new generation; see ``LMDBWriter``.
        verify_checksums: Verify every body against its envelope while copying.
        manifest: Write a manifest for the merged generation.
        provenance: Manifest provenance; requires ``manifest=True``.
        progress: Called between pages with a ``Progress`` report.
        cancel: A set event stops the merge before publication; nothing is published.

    Returns:
        Number of entries copied, counted like ``fingerprint_store`` (array chunks,
        which are copied with their arrays, are not counted).
    """
    if isinstance(paths, (str, bytes, Path)) or not paths:
        raise ValueError("paths must be a nonempty sequence of dataset roots")
    if type(verify_checksums) is not bool:
        raise TypeError("verify_checksums must be bool")
    with LMDBOverlayReader(paths) as overlay:
        total = overlay.check_disjoint()
    tracker = _Tracker("merge_stores", progress, cancel, total=total)
    tracker.check()
    copied = 0
    with LMDBWriter(
        destination,
        map_size=map_size,
        overwrite=overwrite,
        lib_version=lib_version,
        manifest=manifest,
        provenance=provenance,
    ) as writer:
        pending = 0
        for source in paths:
            with _opened_store(source) as (environment, _key):
                tags = environment.open_db(_METADATA_DB_NAME, create=False)
                after: bytes | None = None
                while True:
                    page: list[tuple[bytes, bytes, bytes]] = []
                    with (
                        environment.begin(write=False, buffers=True) as transaction,
                        transaction.cursor() as cursor,
                    ):
                        found = cursor.first() if after is None else cursor.set_range(after)
                        if found and after is not None and bytes(cursor.key()) == after:
                            found = cursor.next()
                        while found and len(page) < _PAGE_ENTRIES:
                            location = bytes(cursor.key())
                            if not _is_metadata_descriptor(location) and not location.startswith(
                                _MANIFEST_PREFIX
                            ):
                                envelope = transaction.get(location, db=tags)
                                if envelope is None:
                                    raise ValueError(
                                        f"entry without integrity metadata in {source}: "
                                        f"{describe_location(location)}"
                                    )
                                page.append((location, bytes(cursor.value()), bytes(envelope)))
                            after = location
                            found = cursor.next()
                    if not page:
                        break
                    target = writer._get_txn()
                    for location, body, envelope in page:
                        if verify_checksums:
                            _verify_checksum(location, _parse_metadata(envelope), body)
                        target.put(location, body)
                        target.put(location, envelope, db=writer._metadata_db)
                        pending += len(body) + len(envelope)
                    copied += sum(not _is_reserved(location) for location, _, _ in page)
                    if pending >= _COMMIT_BYTES:
                        writer.commit()
                        pending = 0
                    tracker.page(len(page), sum(len(item[1]) for item in page))
                    if not found:
                        break
        if writer._txn is not None:
            writer.commit()
        tracker.check()
    tracker.finish()
    return copied


__all__ = ["merge_stores"]
