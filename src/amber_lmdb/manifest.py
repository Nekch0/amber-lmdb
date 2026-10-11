"""Optional writer manifests: counts, array shapes, provenance and the content fingerprint."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import lmdb

from . import codec as _codec
from .arrays import _header_view, parse_array_header
from .chunks import CHUNKED_TAG
from .fingerprints import _add_entry, _new_digest
from .integrity import _parse_metadata
from .keys import describe_location
from .lmdb import _MANIFEST_KEY, _is_reserved, decode_key, get_loc

if TYPE_CHECKING:
    from .lmdb import LMDBWriter

_FORMAT = "amber-lmdb-manifest-v1"
_ARRAY_TAGS = ("write_ndarray", "write_tensor", CHUNKED_TAG)


@dataclass(frozen=True, slots=True)
class ManifestArray:
    """Array shapes of one writer tag, dtype and rank within a key group."""

    writer_tag: str
    dtype: str
    rank: int
    count: int
    min_shape: tuple[int, ...]
    max_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ManifestGroup:
    """One logical key: writer tags, unindexed entry, index count and bounds, arrays."""

    key: str | tuple[str, ...]
    tags: tuple[tuple[str, int], ...]
    unindexed: bool
    indexed: int
    first_index: int | None
    last_index: int | None
    arrays: tuple[ManifestArray, ...]


@dataclass(frozen=True, slots=True)
class StoreManifest:
    """Summary written by ``LMDBWriter(..., manifest=True)`` just before publication.

    Attributes:
        format: Manifest format marker.
        created_utc: ISO 8601 UTC time when the manifest was built.
        amber_lmdb_version: Writing package version, or ``"unknown"``.
        py_lmdb_version: py-lmdb version used by the writer.
        engine: LMDB engine version of the generation.
        checksums: Envelope digest algorithms present.
        entries: Entries other than the reserved manifest and array chunks.
        body_bytes: Total encoded body bytes of those entries, plus chunk bodies.
        fingerprint: ``fingerprint_store`` digest of those entries.
        groups_truncated: True when more key groups existed than were itemized.
        groups: Itemized key groups in encoded key order.
        provenance: Caller-supplied mapping, or None.
        base: Generation copied by ``LMDBWriter(..., base=...)``, or None.
    """

    format: str
    created_utc: str
    amber_lmdb_version: str
    py_lmdb_version: str
    engine: tuple[int, int, int]
    checksums: tuple[str, ...]
    entries: int
    body_bytes: int
    fingerprint: str
    groups_truncated: bool
    groups: tuple[ManifestGroup, ...]
    provenance: dict[str, Any] | None
    base: str | None


def _package_version() -> str:
    try:
        return importlib.metadata.version("amber-lmdb")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _validate_provenance(provenance: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return a plain dict that the dictionary codec can store, or None."""
    if provenance is None:
        return None
    if not isinstance(provenance, Mapping) or not all(isinstance(k, str) for k in provenance):
        raise TypeError("provenance must be a mapping with string keys")
    plain = dict(provenance)
    _codec.dict_to_bytes(plain)  # Reject unsupported values before writing anything.
    return plain


class _Group:
    __slots__ = ("arrays", "first", "indexed", "key", "last", "tags", "unindexed")

    def __init__(self, key: str | tuple[str, ...]) -> None:
        self.key = key
        self.tags: dict[str, int] = {}
        self.unindexed = False
        self.indexed = 0
        self.first: int | None = None
        self.last: int | None = None
        self.arrays: dict[tuple[str, str, int], list[Any]] = {}

    def add(self, index: int | None, tag: str, shape: tuple[int, ...] | None, dtype: str) -> None:
        self.tags[tag] = self.tags.get(tag, 0) + 1
        if index is None:
            self.unindexed = True
        else:
            self.indexed += 1
            self.first = index if self.first is None else min(self.first, index)
            self.last = index if self.last is None else max(self.last, index)
        if shape is not None:
            slot = self.arrays.get((tag, dtype, len(shape)))
            if slot is None:
                self.arrays[(tag, dtype, len(shape))] = [1, shape, shape]
            else:
                slot[0] += 1
                slot[1] = tuple(min(a, b) for a, b in zip(slot[1], shape, strict=True))
                slot[2] = tuple(max(a, b) for a, b in zip(slot[2], shape, strict=True))

    def encoded(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "tags": dict(sorted(self.tags.items())),
            "unindexed": self.unindexed,
            "indexed": self.indexed,
            "first_index": self.first,
            "last_index": self.last,
            "arrays": tuple(
                {
                    "writer_tag": tag,
                    "dtype": dtype,
                    "rank": rank,
                    "count": count,
                    "min_shape": low,
                    "max_shape": high,
                }
                for (tag, dtype, rank), (count, low, high) in sorted(self.arrays.items())
            ),
        }


def _build(writer: LMDBWriter, max_groups: int, provenance: dict[str, Any] | None) -> dict:
    """Scan the writer's staging generation once and return the manifest dictionary."""
    env = writer._get_env()
    digest = _new_digest()
    checksums: set[str] = set()
    groups: dict[bytes, _Group] = {}
    entries = body_bytes = 0
    truncated = False
    with (
        env.begin(write=False, buffers=True) as transaction,
        transaction.cursor(db=writer._metadata_db) as cursor,
    ):
        for raw_location, raw_envelope in cursor:
            location, envelope = bytes(raw_location), bytes(raw_envelope)
            if _is_reserved(location):  # The manifest itself and array chunks.
                continue
            metadata = _parse_metadata(envelope)
            _add_entry(digest, location, envelope)
            checksums.add(metadata.checksum)
            entries += 1
            body_bytes += metadata.size_bytes
            key, index = decode_key(location)
            prefix = location[: len(location) - (1 if index is None else 9)]
            group = groups.get(prefix)
            if group is None:
                if len(groups) == max_groups:
                    truncated = True
                    continue
                group = groups[prefix] = _Group(key)
            shape, dtype = None, ""
            if metadata.writer_tag in _ARRAY_TAGS:
                value = transaction.get(location)
                if value is None:
                    raise ValueError(
                        f"{describe_location(location)} has integrity metadata but no body"
                    )
                header = parse_array_header(
                    bytes(_header_view(value, 64 * 1024)), stored_bytes=len(value)
                )
                shape, dtype = header.shape, header.dtype
                if header.chunk_rows is not None:
                    body_bytes += header.payload_bytes  # Chunk bodies of this array.
            group.add(index, metadata.writer_tag, shape, dtype)
    return {
        "format": _FORMAT,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "amber_lmdb_version": _package_version(),
        "py_lmdb_version": lmdb.__version__,
        "engine": tuple(env.lib_version()),
        "checksums": tuple(sorted(checksums)),
        "entries": entries,
        "body_bytes": body_bytes,
        "fingerprint": digest.hexdigest(),
        "groups_truncated": truncated,
        "groups": tuple(groups[prefix].encoded() for prefix in sorted(groups)),
        "provenance": provenance,
        "base": writer._base_name,
    }


def _write_manifest(writer: LMDBWriter, max_groups: int, provenance: dict[str, Any] | None) -> None:
    """Build and store the manifest as the last committed entry of the staging generation."""
    manifest = _build(writer, max_groups, provenance)
    writer._put_tagged(
        get_loc(_MANIFEST_KEY), _codec.dict_to_bytes(manifest), "write_dict", reserved=True
    )
    writer.commit()


def _shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)) or not all(type(v) is int for v in value):
        raise ValueError("invalid manifest shape")
    return tuple(value)


def _manifest_from_dict(raw: Mapping[Any, Any]) -> StoreManifest:
    """Validate a stored manifest dictionary and convert it to dataclasses."""
    if raw.get("format") != _FORMAT:
        raise ValueError(f"unsupported manifest format: {raw.get('format')!r}")
    try:
        groups = tuple(
            ManifestGroup(
                group["key"] if isinstance(group["key"], str) else tuple(group["key"]),
                tuple((str(tag), int(count)) for tag, count in group["tags"].items()),
                bool(group["unindexed"]),
                int(group["indexed"]),
                group["first_index"],
                group["last_index"],
                tuple(
                    ManifestArray(
                        str(item["writer_tag"]),
                        str(item["dtype"]),
                        int(item["rank"]),
                        int(item["count"]),
                        _shape(item["min_shape"]),
                        _shape(item["max_shape"]),
                    )
                    for item in group["arrays"]
                ),
            )
            for group in raw["groups"]
        )
        engine = tuple(int(part) for part in raw["engine"])
        if len(engine) != 3:
            raise ValueError("invalid manifest engine version")
        provenance = raw["provenance"]
        base = raw.get("base")
        if base is not None and type(base) is not str:
            raise ValueError("invalid manifest base generation")
        return StoreManifest(
            str(raw["format"]),
            str(raw["created_utc"]),
            str(raw["amber_lmdb_version"]),
            str(raw["py_lmdb_version"]),
            (engine[0], engine[1], engine[2]),
            tuple(str(name) for name in raw["checksums"]),
            int(raw["entries"]),
            int(raw["body_bytes"]),
            str(raw["fingerprint"]),
            bool(raw["groups_truncated"]),
            groups,
            None if provenance is None else dict(provenance),
            base,
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed manifest: {exc}") from exc


__all__ = ["ManifestArray", "ManifestGroup", "StoreManifest"]
