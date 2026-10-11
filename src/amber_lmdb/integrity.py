"""Versioned entry integrity metadata binding key, writer tag, size and body."""

from __future__ import annotations

import hashlib
import hmac
import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .errors import AmberLMDBError
from .keys import describe_location

type ChecksumAlgorithm = Literal["blake2b-256", "sha256"]

_DEFAULT_CHECKSUM: ChecksumAlgorithm = "blake2b-256"
_MAGIC = b"TLMI"
_PREFIX_BYTES = len(_MAGIC) + 1
_DIGEST_BYTES = 32
_LENGTHS = struct.Struct(">BQ")
_MAX_METADATA_BYTES = _PREFIX_BYTES + _LENGTHS.size + 63 + _DIGEST_BYTES
# Envelope version byte -> (algorithm, display name, 32-byte digest constructor).
# The version byte is hashed first, so equal bodies never share a digest across versions.
_VERSIONS: dict[int, tuple[ChecksumAlgorithm, str, Callable[[], Any]]] = {
    1: ("sha256", "SHA-256", hashlib.sha256),
    2: ("blake2b-256", "BLAKE2b-256", lambda: hashlib.blake2b(digest_size=_DIGEST_BYTES)),
}
_VERSION_OF: dict[str, int] = {algorithm: version for version, (algorithm, *_) in _VERSIONS.items()}


class IntegrityError(AmberLMDBError, ValueError):
    """An entry's bounded integrity metadata, encoded size or checksum is invalid."""


@dataclass(frozen=True, slots=True)
class EntryMetadata:
    """Parsed immutable writer tag, encoded byte length, expected digest and its algorithm.

    ``checksum`` is ``"sha256"`` for envelope version 1 and ``"blake2b-256"`` for
    version 2. Both digests are 32 bytes and are verified the same way on any CPU.
    """

    writer_tag: str
    size_bytes: int
    digest: bytes
    checksum: ChecksumAlgorithm

    @property
    def metadata_bytes(self) -> int:
        """Return the canonical encoded envelope length, excluding the value body."""
        return _PREFIX_BYTES + _LENGTHS.size + len(self.writer_tag) + _DIGEST_BYTES


def _validate_checksum(checksum: object) -> ChecksumAlgorithm:
    if type(checksum) is not str or checksum not in _VERSION_OF:
        raise ValueError("checksum must be 'blake2b-256' or 'sha256'")
    return _VERSIONS[_VERSION_OF[checksum]][0]


def _parse_metadata(raw: bytes | memoryview) -> EntryMetadata:
    if (
        len(raw) > _MAX_METADATA_BYTES
        or len(raw) < _PREFIX_BYTES + _LENGTHS.size + 1 + _DIGEST_BYTES
    ):
        raise IntegrityError("invalid entry integrity metadata length")
    value = bytes(raw)
    version = value[len(_MAGIC)]
    if not value.startswith(_MAGIC) or version not in _VERSIONS:
        raise IntegrityError("unsupported entry integrity metadata format or version")
    tag_size, size = _LENGTHS.unpack_from(value, _PREFIX_BYTES)
    start = _PREFIX_BYTES + _LENGTHS.size
    if not 1 <= tag_size <= 63 or len(value) != start + tag_size + _DIGEST_BYTES:
        raise IntegrityError("invalid entry writer tag length or trailing metadata")
    try:
        tag = value[start : start + tag_size].decode("ascii")
    except UnicodeError as error:
        raise IntegrityError("entry writer tag is not ASCII") from error
    return EntryMetadata(tag, size, value[-_DIGEST_BYTES:], _VERSIONS[version][0])


def _hasher(location: bytes, tag: str, size: int, checksum: ChecksumAlgorithm) -> Any:
    version = _VERSION_OF[checksum]
    encoded = tag.encode("ascii")
    digest = _VERSIONS[version][2]()
    # One update of the joined prefix: the same digest as separate updates, fewer calls.
    digest.update(
        b"".join(
            (
                _MAGIC,
                bytes((version,)),
                struct.pack(">I", len(location)),
                location,
                _LENGTHS.pack(len(encoded), size),
                encoded,
            )
        )
    )
    return digest


def _encode_metadata(
    location: bytes,
    tag: str,
    value: bytes | memoryview,
    checksum: ChecksumAlgorithm = _DEFAULT_CHECKSUM,
) -> bytes:
    checksum = _validate_checksum(checksum)
    encoded = tag.encode("ascii")
    if not 1 <= len(encoded) <= 63:
        raise ValueError("writer tag must have 1..63 ASCII bytes")
    digest = _hasher(location, tag, len(value), checksum)
    digest.update(value)
    prefix = _MAGIC + bytes((_VERSION_OF[checksum],))
    return prefix + _LENGTHS.pack(len(encoded), len(value)) + encoded + digest.digest()


def _verify_checksum(
    location: bytes, metadata: EntryMetadata, value: bytes | bytearray | memoryview
) -> None:
    if len(value) != metadata.size_bytes:
        raise IntegrityError("encoded value length does not match integrity metadata")
    digest = _hasher(location, metadata.writer_tag, metadata.size_bytes, metadata.checksum)
    digest.update(value)
    if not hmac.compare_digest(digest.digest(), metadata.digest):
        name = _VERSIONS[_VERSION_OF[metadata.checksum]][1]
        raise IntegrityError(
            f"entry {name} checksum mismatch for {describe_location(location)} "
            f"(location {location.hex()})"
        )


__all__ = ["ChecksumAlgorithm", "EntryMetadata", "IntegrityError"]
