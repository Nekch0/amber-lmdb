"""Manage typed LMDB writes, shared reads and environment lifetimes."""

from __future__ import annotations

import functools
import logging
import math
import os
import shutil
import stat
import struct
import sys
import tempfile
import threading as _threading
import uuid
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path, PurePath
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, cast

import lmdb
import numpy as np
from frozendict import frozendict
from lmdb import Environment, Transaction
from numpy.typing import NDArray

from . import chunks as _chunks
from . import codec as _codec
from ._ownership import _FileLock, _generation_lock, _root_gate
from .arrays import ArrayHeader, _header_view, parse_array_header
from .budgets import (
    DEFAULT_MAX_BYTES,
    ReadLimitError,
    _validate_limits,
    _validate_max_bytes,
)
from .chunks import _ArrayRows
from .codec import ValueLimits
from .errors import KeyNotFoundError, ReaderClosedError, ReaderMismatchError, StoreNotFoundError
from .history import HISTORY_NAME, _append_history, _create_history, _record_current, read_history
from .integrity import (
    ChecksumAlgorithm,
    EntryMetadata,
    _encode_metadata,
    _parse_metadata,
    _validate_checksum,
    _verify_checksum,
)
from .keys import _KEY_MAGIC, decode_key, describe_location, get_loc

_log = logging.getLogger(__name__)
_log.addHandler(logging.NullHandler())

if TYPE_CHECKING:
    import torch
    from torch import Tensor

    from .manifest import StoreManifest

if os.name == "nt":
    import msvcrt as _file_lock
else:
    import fcntl as _file_lock

# Key/body codecs remain stable; the new store envelope requires integrity metadata.
_METADATA_DB_NAME = b"amber-lmdb.entries.v1"
_FORMAT_MARKER = "amber-lmdb-v1\n"


def _is_metadata_descriptor(location: bytes) -> bool:
    # LMDB 1.0 stores named-database descriptors with a terminating NUL.
    return location in (_METADATA_DB_NAME, _METADATA_DB_NAME + b"\x00")


#: Valid typed read methods for each stored writer tag.
_VALID_READERS: dict[str, frozenset[str]] = {
    # read_as_any decodes every type-tagged MessagePack body, whichever writer stored it.
    "write_any": frozenset({"read_as_any"}),
    "write_primitive": frozenset({"read_as_primitive", "read_as_any"}),
    "write_str": frozenset({"read_as_str"}),
    "write_sequence": frozenset({"read_as_list", "read_as_tuple", "read_as_any"}),
    "write_set": frozenset({"read_as_set", "read_as_frozenset", "read_as_any"}),
    "write_dict": frozenset({"read_as_dict", "read_as_frozendict", "read_as_any"}),
    "write_ndarray": frozenset(
        {"read_as_ndarray", "read_as_tensor", "read_as_shape", "read_as_dtype", "read_array_rows"}
    ),
    "write_tensor": frozenset({"read_as_tensor", "read_as_shape", "read_as_dtype"}),
    _chunks.CHUNKED_TAG: frozenset(
        {"read_as_ndarray", "read_as_tensor", "read_as_shape", "read_as_dtype", "read_array_rows"}
    ),
    # Chunks are internal parts of a chunked array, read only through its header.
    _chunks.CHUNK_TAG: frozenset({"_read_chunk"}),
}

#: Owned body returned by reads: bytes, a tensor's private bytearray, or array rows.
type _Body = bytes | bytearray | _ArrayRows

#: Writer tags whose bodies are array headers, mapped to their array kind.
_ARRAY_KINDS = {"write_ndarray": "numpy", "write_tensor": "torch", _chunks.CHUNKED_TAG: "numpy"}

_WRITE_METHODS = {
    "tensor": "write_tensor",
    "ndarray": "write_ndarray",
    "dict": "write_dict",
    "frozendict": "write_dict",
    "set": "write_set",
    "frozenset": "write_set",
    "list": "write_sequence",
    "tuple": "write_sequence",
    "str": "write_str",
    "primitive": "write_primitive",
    "any": "write_any",
}


# Indices of one logical key are contiguous after its unindexed entry: get_loc ends
# with 0x00 (no index) or 0x01 + big-endian index, so prefix + 0x02 bounds the range.
_RESERVED_COMPONENT = "amber-lmdb"
_RESERVED_MARK = struct.pack(">Q", len(_RESERVED_COMPONENT)) + _RESERVED_COMPONENT.encode()
_MANIFEST_KEY = (_RESERVED_COMPONENT, "manifest")
_MANIFEST_PREFIX = get_loc(_MANIFEST_KEY)[:-1]
from .keys import _MAX_INDEX  # noqa: E402


def _is_reserved(location: bytes) -> bool:
    """Whether a location is a tuple key whose first component is ``"amber-lmdb"``.

    That namespace holds the manifest and array chunks; writers refuse it for user
    data, and listings, fingerprints, diffs and overlay checks skip it.
    """
    return location[5:6] == b"\x01" and location[10 : 10 + len(_RESERVED_MARK)] == _RESERVED_MARK


def _safe_key(location: bytes) -> tuple[str | tuple[str, ...] | None, int | None]:
    """Return the decoded (key, index) of a location, or (None, None) if malformed."""
    try:
        return decode_key(location)
    except (TypeError, ValueError):
        return None, None


def _key_not_found(location: bytes) -> KeyNotFoundError:
    """Return the error for a missing entry, naming its decoded key."""
    key, index = _safe_key(location)
    return KeyNotFoundError(
        f"no entry for {describe_location(location)}", key=key, index=index, location=location
    )


def _refuse_reserved(location: bytes) -> None:
    """Raise ValueError for a user write or delete in the reserved namespace."""
    if _is_reserved(location):
        raise ValueError(
            f"tuple keys starting with {_RESERVED_COMPONENT!r} are reserved for the store "
            "manifest and array chunks"
        )


@functools.lru_cache(maxsize=4096)
def _chunk_prefix(parent: bytes) -> bytes:
    """Return the chunk locations' shared prefix (key and index marker) for ``parent``."""
    return get_loc((_RESERVED_COMPONENT, "chunk", parent.hex()), 0)[:-8]


def _chunk_location(parent: bytes, number: int) -> bytes:
    """Return the reserved location of chunk ``number`` of the array at ``parent``."""
    return _chunk_prefix(parent) + struct.pack(">Q", number)


def _chunk_parent(location: bytes) -> tuple[bytes, int] | None:
    """Return the parent location and chunk number of a chunk location, else None."""
    if not _is_reserved(location):
        return None
    key, index = decode_key(location)
    if not isinstance(key, tuple) or len(key) != 3 or key[1] != "chunk" or index is None:
        return None
    try:
        return bytes.fromhex(key[2]), index
    except ValueError:
        return None


def _key_prefix(key: str | Path | tuple[str, ...]) -> bytes:
    """Return the encoded logical key without its index marker."""
    return get_loc(key)[:-1]


def _index_of(location: bytes, low: bytes) -> int | None:
    """Return the index of an indexed location under ``low`` (prefix + 0x01), else None."""
    if not location.startswith(low):
        return None
    if len(location) != len(low) + 8:
        raise ValueError(f"malformed indexed key: {describe_location(location)}")
    return struct.unpack_from(">Q", location, len(low))[0]


def _boundary_index(cursor: Any, prefix: bytes, *, last: bool) -> int | None:
    """Seek the smallest or largest index of one logical key; None without indices."""
    if last:
        found = cursor.set_range(prefix + b"\x02")
        found = cursor.prev() if found else cursor.last()
    else:
        found = cursor.set_range(prefix + b"\x01")
    return _index_of(bytes(cursor.key()), prefix + b"\x01") if found else None


def _as_bytes(value: bytes | bytearray) -> bytes:
    """Return a non-tensor body as bytes; only tensor bodies are owned bytearrays."""
    return value if isinstance(value, bytes) else bytes(value)


def _check_array_tag(tag: str | None, info: ArrayHeader) -> None:
    """Raise ValueError unless an array header matches the writer tag that stored it."""
    if _ARRAY_KINDS.get(tag or "") != info.kind or (tag == _chunks.CHUNKED_TAG) != (
        info.chunk_rows is not None
    ):
        raise ValueError("array kind does not match writer tag")


#: Bodies and chunks at least this large are verified in the thread pool, if any.
_PARALLEL_DIGEST_BYTES = 64 * 1024


def _row_checks(
    location: bytes, metadata: EntryMetadata, rows: _ArrayRows
) -> list[tuple[bytes, EntryMetadata, Any]]:
    """Digest checks of copied rows: the header (or whole body) and every chunk."""
    view = memoryview(rows.data)
    checks: list[tuple[bytes, EntryMetadata, Any]] = [
        (location, metadata, view if rows.whole else rows.header)
    ]
    checks.extend((chunk, meta, view[begin:end]) for chunk, meta, begin, end in rows.checks)
    return checks


def _node_budget(limit: int | None) -> _codec._NodeBudget | None:
    """Return a fresh batch node budget, or None without ``max_decoded_nodes``."""
    return None if limit is None else _codec._NodeBudget(limit)


def _plain(value: bytes | bytearray | _ArrayRows) -> bytes | bytearray:
    """Return an owned body that typed non-array reads accept; array rows never are."""
    if isinstance(value, _ArrayRows):
        raise TypeError("array rows are decoded only by array reads")
    return value


def _tensor(value: bytes | bytearray) -> Tensor:
    """Decode a tensor body, using an owned bytearray in place instead of copying it."""
    if isinstance(value, bytearray):
        return _codec._tensor_from_owned(value)
    return _codec.bytes_to_tensor(value)


def _writable(array: NDArray[Any]) -> NDArray[Any]:
    """Make an owned read result writable: in place over a mutable body, else a copy."""
    if not array.flags.writeable:
        try:
            array.flags.writeable = True
        except ValueError:  # the memory is immutable bytes
            array = array.copy()
    return array


_CASTINGS = frozenset({"no", "equiv", "safe", "same_kind", "unsafe"})


def _converted[T: np.generic](
    array: NDArray[Any], dtype: type[T] | np.dtype[Any] | str | None, casting: str
) -> NDArray[T]:
    """Apply an optional dtype conversion under a NumPy casting rule."""
    if casting not in _CASTINGS:
        raise ValueError(f"casting must be one of {sorted(_CASTINGS)}")
    if dtype is None:
        return array
    try:
        return array.astype(dtype, casting=casting)  # type: ignore[call-overload]
    except TypeError as exc:
        raise TypeError(
            f"cannot convert {array.dtype} to {np.dtype(dtype)} with casting={casting!r}; "
            "pass casting='unsafe' to allow lossy conversions"
        ) from exc


def _index_sequence(indices: object) -> Sequence[int | None]:
    """Validate batch indices: a sequence, or a one-dimensional integer NumPy array."""
    if isinstance(indices, np.ndarray):
        if indices.ndim != 1 or indices.dtype.kind not in "iu":
            raise TypeError("indices array must be one-dimensional with an integer dtype")
        return cast(list[int], indices.tolist())
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Sequence):
        raise TypeError(
            f"indices must be a sequence of integers or None, not {type(indices).__name__}"
        )
    return cast(Sequence[int | None], indices)


def _pairs(keys: Iterable[Any], indices: Iterable[Any]) -> Iterator[tuple[Any, Any]]:
    """Pair streamed keys with indices; unequal lengths raise ValueError when reached."""
    key_iterator, index_iterator = iter(keys), iter(indices)
    done = object()
    while True:
        key, index = next(key_iterator, done), next(index_iterator, done)
        if key is done and index is done:
            return
        if key is done or index is done:
            raise ValueError("keys and indices have different lengths")
        yield key, index


def _validate_page_size(page_size: int) -> None:
    if type(page_size) is not int or page_size < 1:
        raise ValueError("page_size must be a positive integer")


def _lstat(path: Path) -> os.stat_result | None:
    """Return ``os.lstat`` of a path, or None when it does not exist."""
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _is_plain_dir(path: Path) -> bool:
    """Whether a path is a directory and not a symlink, with one ``lstat`` call."""
    info = _lstat(path)
    return info is not None and stat.S_ISDIR(info.st_mode)


def _is_missing_or_plain_file(path: Path) -> bool:
    """Whether a path is absent or a regular file that is not a symlink (one ``lstat``)."""
    info = _lstat(path)
    return info is None or stat.S_ISREG(info.st_mode)


def _validate_dataset_root(root: Path) -> None:
    """Validate the dataset root and its exact ``FORMAT`` marker.

    Reject symlinks for the root and marker, nonregular entries and unsupported
    marker contents with ValueError.
    """
    if not _is_plain_dir(root):
        if _lstat(root) is None:
            raise StoreNotFoundError(f"LMDB dataset not found: {root}")
        raise ValueError(f"LMDB dataset root must be a regular directory: {root}")
    format_path = root / "FORMAT"
    if not _is_missing_or_plain_file(format_path):
        raise ValueError(f"invalid LMDB FORMAT entry: {format_path}")
    try:
        marker = format_path.read_text(encoding="ascii")
    except FileNotFoundError as exc:
        raise ValueError(f"LMDB dataset has no FORMAT marker: {root}") from exc
    if marker != _FORMAT_MARKER:
        raise ValueError(f"unsupported LMDB dataset format: {marker!r}")


def _is_version_name(value: str) -> bool:
    """Check for ``v-`` followed by a 32-digit lowercase hexadecimal UUID."""
    if len(value) != 34 or not value.startswith("v-"):
        return False
    try:
        return uuid.UUID(hex=value[2:]).hex == value[2:]
    except ValueError:
        return False


def _read_current(root: Path) -> str:
    """Validate ``CURRENT`` and return the generation name it selects.

    Require a regular ASCII file containing one valid generation name and a
    trailing newline. A missing pointer raises FileNotFoundError; invalid layout,
    symlinks, pointer syntax or UUIDs raise ValueError.
    """
    _validate_dataset_root(root)
    current = root / "CURRENT"
    if not _is_missing_or_plain_file(current):
        raise ValueError(f"invalid LMDB CURRENT entry: {current}")
    try:
        pointer = current.read_text(encoding="ascii")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"LMDB dataset has no CURRENT pointer: {root}") from exc
    if not pointer.endswith("\n") or pointer.count("\n") != 1:
        raise ValueError(f"invalid LMDB CURRENT pointer: {pointer!r}")
    version = pointer[:-1]
    if not _is_version_name(version):
        raise ValueError(f"invalid LMDB CURRENT pointer: {version!r}")
    return version


def _resolve_lmdb_path(path: str | Path) -> tuple[str, str]:
    """Validate a dataset and resolve its root and selected generation paths.

    Reject a symlink root and require the selected generation to resolve to a
    directory directly beneath ``versions``. Return two resolved absolute paths.
    A missing pointer raises FileNotFoundError; invalid layout raises ValueError.
    """
    requested = Path(path).absolute()
    if requested.is_symlink():
        raise ValueError(f"LMDB dataset root cannot be a symlink: {requested}")
    root = requested.resolve()
    name = _read_current(root)
    versions = root / "versions"
    candidate = versions / name
    if not _is_plain_dir(versions):
        raise ValueError("LMDB versions must be a regular directory inside the dataset root")
    resolved_versions = versions.resolve()  # Junctions are not symlinks; resolve follows them.
    if resolved_versions.parent != root:
        raise ValueError("LMDB versions must be a regular directory inside the dataset root")
    if not _is_plain_dir(candidate):
        raise ValueError("LMDB CURRENT must select a regular generation directory")
    version_path = candidate.resolve()
    if version_path.parent != resolved_versions:
        raise ValueError("LMDB CURRENT points outside the versions directory")
    return str(root), str(version_path)


def _fsync_file(path: Path) -> None:
    """Open a file read/write and fsync its contents; propagate I/O errors."""
    with path.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    """Fsync a directory where supported; otherwise do nothing.

    Propagate errors from opening or syncing the directory.
    """
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        return
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _acquire_writer_lock(lock_path: Path) -> Any:
    """Take the non-blocking OS writer lock beside a dataset root and return its handle.

    Convert lock contention to BlockingIOError; the lock file stays in place.
    """
    handle = lock_path.open("a+b")
    try:
        if os.name == "nt":
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            _file_lock.locking(handle.fileno(), _file_lock.LK_NBLCK, 1)
        else:
            _file_lock.flock(handle.fileno(), _file_lock.LOCK_EX | _file_lock.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise BlockingIOError(f"writer lock is already held: {lock_path}") from exc
    return handle


def _release_writer_lock(handle: Any) -> None:
    """Release a writer lock handle; unlock errors during cleanup are ignored."""
    try:
        if os.name == "nt":
            handle.seek(0)
            _file_lock.locking(handle.fileno(), _file_lock.LK_UNLCK, 1)
        else:
            _file_lock.flock(handle.fileno(), _file_lock.LOCK_UN)
    except OSError:
        pass
    finally:
        handle.close()


def _writer_lock_path(root: Path) -> Path:
    """Return the writer lock file beside a dataset root."""
    return root.with_name(f".{root.name}.write.lock")


def _write_current(root: Path, name: str) -> None:
    """Atomically point ``CURRENT`` at a generation through a synced temporary file."""
    descriptor, temporary_name = tempfile.mkstemp(dir=root, prefix=".CURRENT.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        try:
            content = f"{name}\n".encode("ascii")
            if os.write(descriptor, content) != len(content):
                raise OSError("incomplete CURRENT pointer write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, root / "CURRENT")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _remove_unpublished(root: Path, name: str) -> None:
    """Remove a never-published generation under its exclusive lease, through ``.trash``."""
    generation = root / "versions" / name
    trash = root / ".trash"
    with _root_gate(root):
        try:
            lease = _generation_lock(root, name, shared=False, blocking=False)
        except BlockingIOError as exc:
            raise BlockingIOError(f"unpublished generation {name} is in use") from exc
        try:
            trash.mkdir(mode=0o700, exist_ok=True)
            os.replace(generation, trash / name)
            _fsync_directory(root / "versions")
        except BaseException:
            lease.close()
            raise
    try:
        shutil.rmtree(trash / name)
        _log.warning("removed unpublished generation %s left by an interrupted write", name)
    finally:
        lease.close()


class LMDBWriter:
    """Write a unique generation and publish it by atomically switching ``CURRENT``.

    Acquire a writer lock beside the dataset, write to a unique staging directory
    and publish on successful context exit. The previous generation remains
    available until the pointer switches.

    Args:
        path: Dataset directory containing generations and the ``CURRENT`` pointer.
        map_size: LMDB memory-map size limit in bytes.
        overwrite: Allow replacement of a published dataset. If False, entering
            the context fails when a valid ``CURRENT`` already exists. Ignored with
            ``base``, which always replaces.
        lib_version: Engine for the new generation: 1 selects LMDB 1.0, 0 selects
            LMDB 0.9. None (default) selects 1, or the base's engine with ``base``;
            an explicit value must then match it. This does not convert generations.
        base: Start the new generation from a copy of a published one of this
            dataset: ``"current"`` or a generation name ``"v-<32 hex>"``. Entering
            the context copies it compactly into the staging directory while holding
            its shared lease, so cooperative deletion cannot remove it meanwhile.
            Copied entries keep their envelopes; writes add or replace entries and
            ``delete``/``delete_key`` remove them. The base stays until deleted
            explicitly. A copied manifest is dropped (``manifest=True`` rebuilds it).
        verify_on_publish: Offline-check the completed private staging data file
            before publication. False avoids the extra full-file verification cost.
        checksum: Digest bound into each entry's integrity envelope. The default
            ``"blake2b-256"`` writes envelope version 2; ``"sha256"`` writes version 1.
            Readers verify both on any CPU; only hashing speed differs.
        manifest: Store a summary (counts, index bounds, array shapes, envelope
            algorithms, content fingerprint, provenance) as the last entry before
            publication; read it with ``LMDBReader.manifest()``. Building it scans
            the staging generation once. Off by default.
        provenance: Mapping with string keys stored in the manifest; requires
            ``manifest=True``. Values must be storable by ``write_dict``.
        manifest_max_groups: Maximum key groups itemized in the manifest; beyond it
            only totals are kept and the manifest is marked truncated.

    Examples:
        Write a temporary dataset and remove it when the example ends.

        >>> from contextlib import redirect_stdout
        >>> from io import StringIO
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> import numpy as np
        >>> output = StringIO()
        >>> with TemporaryDirectory() as tmp, redirect_stdout(output):
        ...     path = Path(tmp) / "dataset.lmdb"
        ...     with LMDBWriter(path) as writer:
        ...         _ = writer.write_ndarray(np.arange(4), key="features", index=0)
        ...         writer.commit()
        ...     created = path.exists()
        >>> created
        True

    Note:
        Use this class as a context manager. A concurrent writer for the same
        path raises BlockingIOError. Process exit releases the OS lock; a later
        writer validates and removes recognized remnants of interrupted writes.
        This does not guarantee safe operation on network filesystems.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        map_size: int = 512 * 1024**3,
        overwrite: bool = False,
        lib_version: Literal[0, 1] | None = None,
        verify_on_publish: bool = False,
        checksum: ChecksumAlgorithm = "blake2b-256",
        manifest: bool = False,
        provenance: Mapping[str, Any] | None = None,
        manifest_max_groups: int = 10_000,
        base: str | None = None,
    ):
        if type(map_size) is not int or map_size <= 0:
            raise ValueError("map_size must be a positive integer number of bytes")
        if type(overwrite) is not bool:
            raise TypeError("overwrite must be bool")
        if type(manifest) is not bool:
            raise TypeError("manifest must be bool")
        if provenance is not None and not manifest:
            raise ValueError("provenance requires manifest=True")
        if type(manifest_max_groups) is not int or manifest_max_groups <= 0:
            raise ValueError("manifest_max_groups must be a positive integer")
        if lib_version is not None and (type(lib_version) is not int or lib_version not in (0, 1)):
            raise ValueError("lib_version must be 0 (LMDB 0.9), 1 (LMDB 1.0) or None")
        if type(verify_on_publish) is not bool:
            raise TypeError("verify_on_publish must be bool")
        if base is not None:
            if type(base) is not str:
                raise TypeError("base must be None, 'current' or a generation name")
            if base != "current" and not _is_version_name(base):
                raise ValueError("base must be 'current' or a generation name such as v-<32 hex>")
        self._base = base
        self._base_name: str | None = None  # Generation actually copied.
        self._lib_version = lib_version
        self._engine: Literal[0, 1] = 1 if lib_version is None else lib_version
        self._verify_on_publish = verify_on_publish
        self._checksum: ChecksumAlgorithm = _validate_checksum(checksum)
        from .manifest import _validate_provenance

        self._manifest = manifest
        self._provenance = _validate_provenance(provenance)
        self._manifest_max_groups = manifest_max_groups
        self._path = Path(path).absolute()
        self._tmp_path: Path | None = None
        self._lock_path = self._path.with_name(f".{self._path.name}.write.lock")
        self._lock_handle: Any = None
        self._version_name: str | None = None
        self._map_size = map_size
        self._overwrite = overwrite
        self._env: Environment | None = None
        self._txn: Transaction | None = None
        self._metadata_db: Any = None
        self._finalized = False  # Set after the CURRENT pointer switches.
        # Writes and deletes look for chunked arrays to replace only once a chunked
        # array was written or copied from a base, so plain writers pay nothing.
        self._may_hold_chunks = base is not None

    @property
    def path(self) -> Path:
        """Return the dataset root this writer publishes to."""
        return self._path

    @property
    def staging_path(self) -> Path | None:
        """Return the private staging directory during an active write, else None."""
        return self._tmp_path

    def __repr__(self) -> str:
        state = (
            "writing"
            if self._tmp_path is not None
            else ("published" if self._finalized else "idle")
        )
        return f"LMDBWriter({str(self._path)!r}, {state})"

    def _get_env(self) -> Environment:
        """Lazily open the staging environment and its writer-tag database.

        Raise RuntimeError before context entry. Propagate LMDB errors when
        opening the environment or named database fails.
        """
        if self._env is None:
            if self._tmp_path is None:
                raise RuntimeError("LMDBWriter must be entered before writing")
            # The staging directory already exists. py-lmdb's create=True probes and
            # makes directories through narrow Windows calls that misread non-ASCII
            # paths unless the system code page is UTF-8; LMDB itself opens UTF-16.
            env = lmdb.open(
                str(self._tmp_path),
                map_size=self._map_size,
                subdir=True,
                create=False,
                lock=True,
                readahead=False,
                meminit=False,
                max_dbs=1,
                lib_version=self._engine,
                max_spare_txns=0,
            )
            if env is None:  # LMDB normally raises on failure; guard a missing return value.
                raise RuntimeError(f"lmdb.open returned None for {self._tmp_path}")
            self._env = env
            self._metadata_db = env.open_db(_METADATA_DB_NAME, create=True)
        return self._env

    def _get_txn(self) -> Transaction:
        """Lazily start a write transaction and reuse it until commit or exit."""
        if self._txn is None:
            self._txn = self._get_env().begin(write=True)
        return self._txn

    def _put(self, key: bytes, value: bytes) -> bool:
        """Store a key/value pair in the current write transaction."""
        return self._get_txn().put(key, value)

    def _put_tagged(
        self, loc: bytes, value: bytes, writer_tag: str, *, reserved: bool = False
    ) -> bool:
        """Atomically store the body and its key/tag/length-bound checksum envelope.

        User writes refuse the reserved namespace and replace a chunked array's chunks.
        """
        if not reserved:
            _refuse_reserved(loc)
            if self._may_hold_chunks:
                self._drop_chunks(loc)
        return self._store(loc, value, writer_tag)[0]

    def _store(self, loc: bytes, value: bytes | memoryview, writer_tag: str) -> tuple[bool, bytes]:
        """Store a body and its envelope without checks; return the put result and envelope."""
        metadata = _encode_metadata(loc, writer_tag, value, self._checksum)
        transaction = self._get_txn()
        ok = transaction.put(loc, value)
        transaction.put(loc, metadata, db=self._metadata_db)
        return ok, metadata

    def _drop_chunks(self, location: bytes) -> None:
        """Remove the chunks of the chunked array stored at ``location``, if it is one."""
        transaction = self._get_txn()
        raw = transaction.get(location, db=self._metadata_db)
        if raw is None or _parse_metadata(raw).writer_tag != _chunks.CHUNKED_TAG:
            return
        low = _chunk_location(location, 0)[:-8]
        for database in (self._metadata_db, None):
            with transaction.cursor(db=database) as cursor:
                cursor.set_range(low)
                while bytes(cursor.key()).startswith(low):
                    if not cursor.delete():
                        break

    def _put_chunked(
        self,
        location: bytes,
        array: NDArray[Any],
        chunk_rows: int,
        compression: str | None = None,
        compression_level: int | None = None,
    ) -> bool:
        """Store a NumPy array as a header entry plus row chunks; see ``write_ndarray``."""
        if type(chunk_rows) is not int or chunk_rows <= 0:
            raise ValueError("chunk_rows must be a positive integer")
        if array.ndim == 0:
            raise ValueError("chunked arrays need a NumPy array with at least one dimension")
        _refuse_reserved(location)
        meta, contiguous = _codec._ndarray_meta(array)
        if compression is not None:
            meta["compression"] = compression
        # More rows per chunk than the array has would only store one chunk anyway.
        chunk_rows = min(chunk_rows, max(array.shape[0], 1))
        meta["chunk_rows"] = chunk_rows
        raw = memoryview(contiguous.reshape(-1).view(np.uint8))
        chunk_bytes = chunk_rows * contiguous.dtype.itemsize * math.prod(array.shape[1:])
        count = 0 if len(raw) == 0 else -(-len(raw) // chunk_bytes)
        if self._may_hold_chunks:
            self._drop_chunks(location)
        self._may_hold_chunks = True
        digests = []
        stored: list[int] | None = None if compression is None else []
        for number in range(count):
            body: bytes | memoryview = raw[number * chunk_bytes : (number + 1) * chunk_bytes]
            if stored is not None:
                assert compression is not None
                body = _codec._compress(compression, body, compression_level)
                stored.append(len(body))
            _, metadata = self._store(_chunk_location(location, number), body, _chunks.CHUNK_TAG)
            digests.append(metadata[-32:])
        header = _chunks._encode_header(meta, len(raw), chunk_bytes, digests, stored)
        return self._store(location, header, _chunks.CHUNKED_TAG)[0]

    def write_any(
        self,
        obj: Any,
        *,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
    ) -> bool:
        """Store a supported value as type-tagged MessagePack bytes.

        Args:
            obj: Supported scalar, container, array or registered custom value.
            key: Logical storage key.
            index: Optional nonnegative index; None omits it.

        Returns:
            True on a successful write. Existing keys are overwritten, so this
            result does not distinguish insertion from replacement.

        Note:
            Read with ``LMDBReader.read_as_any``. Prefer the dedicated write
            methods for arrays and tensors. Encoding and LMDB errors propagate.
            Values are limited to what every reader decodes by default: 64 MiB
            encoded, 1,000,000 nodes and nesting depth 100; larger values raise
            ValueError here, at write time, rather than becoming unreadable.
        """
        return self._put_tagged(get_loc(key, index), _codec.any_to_bytes(obj), "write_any")

    def write(
        self, obj: Any, *, key: str | Path | tuple[str, ...], index: int | None = None
    ) -> bool:
        """Choose a writer from the value's exact type and store the value.

        Args:
            obj: Supported scalar, array, Tensor or registered custom value.
            key: Caller-selected storage key. No namespace is added automatically.
            index: Optional nonnegative index.

        Returns:
            True on a successful write. Unsupported values and invalid keys raise
            the same errors as the selected typed writer.
        """
        method = _WRITE_METHODS[_codec.storage_kind_for_type(type(obj))]
        return getattr(self, method)(obj, key=key, index=index)

    def write_primitive(
        self,
        obj: bool | int | float | complex,
        *,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
    ) -> bool:
        """Store a boolean or number as type-tagged MessagePack bytes.

        Args:
            obj: Bool, int, float or complex value to store.
            key: Logical storage key.
            index: Optional nonnegative index; None omits it.

        Returns:
            True on success, overwriting any value at the same key.

        Note:
            Read with ``LMDBReader.read_as_primitive``. Encoding and LMDB errors
            propagate without conversion.
        """
        return self._put_tagged(
            get_loc(key, index), _codec.primitive_to_bytes(obj), "write_primitive"
        )

    def write_str(
        self,
        obj: str,
        *,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
    ) -> bool:
        """Store a string as UTF-8 bytes with a string writer tag.

        Args:
            obj: String to store.
            key: Logical storage key.
            index: Optional nonnegative index; None omits it.

        Returns:
            True on success, overwriting any value at the same key.

        Note:
            Read with ``LMDBReader.read_as_str``. Encoding and LMDB errors propagate.
        """
        if not isinstance(obj, str):
            raise TypeError(f"write_str needs a str, got {type(obj).__name__}")
        return self._put_tagged(get_loc(key, index), _codec.str_to_bytes(obj), "write_str")

    def write_sequence(
        self,
        obj: tuple[Any, ...] | list[Any],
        *,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
    ) -> bool:
        """Store a list or tuple as type-tagged MessagePack bytes.

        Args:
            obj: List or tuple to store.
            key: Logical storage key.
            index: Optional nonnegative index; None omits it.

        Returns:
            True on success, overwriting any value at the same key.

        Note:
            Use ``read_as_list`` for stored lists and ``read_as_tuple`` for stored
            tuples. Typed reads reject a different container type instead of converting it.
            The same write-time limits as ``write_any`` apply (64 MiB encoded,
            1,000,000 nodes, depth 100).
        """
        return self._put_tagged(
            get_loc(key, index), _codec.sequence_to_bytes(obj), "write_sequence"
        )

    def write_set(
        self,
        obj: set[Any] | frozenset[Any],
        *,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
    ) -> bool:
        """Store a set or frozenset as type-tagged MessagePack bytes.

        Args:
            obj: Set or frozenset to store.
            key: Logical storage key.
            index: Optional nonnegative index; None omits it.

        Returns:
            True on success, overwriting any value at the same key.

        Note:
            Use ``read_as_set`` for stored sets and ``read_as_frozenset`` for
            stored frozensets. Typed reads do not convert between these types.
            The same write-time limits as ``write_any`` apply (64 MiB encoded,
            1,000,000 nodes, depth 100).
        """
        return self._put_tagged(get_loc(key, index), _codec.set_to_bytes(obj), "write_set")

    def write_dict(
        self,
        obj: dict[Any, Any] | frozendict[Any, Any],
        *,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
    ) -> bool:
        """Store a dict or frozendict as type-tagged MessagePack bytes.

        Args:
            obj: Dict or frozendict to store.
            key: Logical storage key.
            index: Optional nonnegative index; None omits it.

        Returns:
            True on success, overwriting any value at the same key.

        Note:
            Use ``read_as_dict`` for stored dicts and ``read_as_frozendict`` for
            stored frozendicts. Typed reads do not convert between these types.
            The same write-time limits as ``write_any`` apply (64 MiB encoded,
            1,000,000 nodes, depth 100).
        """
        return self._put_tagged(get_loc(key, index), _codec.dict_to_bytes(obj), "write_dict")

    def write_ndarray(
        self,
        obj: NDArray[Any],
        *,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        chunk_rows: int | None = None,
        compression: str | None = None,
        compression_level: int | None = None,
    ) -> bool:
        """Store a NumPy array with a self-describing header.

        Args:
            obj: Array to store; noncontiguous input is copied to contiguous layout.
            key: Logical storage key.
            index: Optional nonnegative index; None omits it.
            chunk_rows: Store the array in chunks of this many rows (axis 0), each
                with its own integrity envelope, so ``read_array_rows`` copies and
                verifies only the chunks a row range touches. Chunks of about
                64-256 KiB suit row windows; full reads cost about the same as
                unchunked arrays. A value above the row count stores one chunk
                and is recorded as the row count. None (default) stores one entry.
            compression: ``"zlib"``, ``"lzma"`` or ``"zstd"`` (Python 3.14+) stores the
                values compressed (every chunk on its own with ``chunk_rows``); reads
                decompress transparently and integrity digests cover the stored bytes.
                Readers older than this format refuse compressed arrays. None (default)
                stores raw values, which suit random reads of incompressible data best.
            compression_level: Level for ``compression``; None uses 6 (zlib, lzma) or 3
                (zstd).

        Returns:
            True on success, overwriting any value at the same key.

        Raises:
            TypeError: ``obj`` is not a NumPy array, is a masked array (its mask would
                be lost), or has an object dtype.

        Note:
            The header retains shape and dtype. Restore values with ``read_as_ndarray``
            or inspect only the header with ``read_as_shape`` or ``read_as_dtype``;
            both work the same for chunked arrays. Chunks live under reserved keys.
        """
        if isinstance(obj, np.ma.MaskedArray):
            raise TypeError(
                "masked arrays would lose their mask; store obj.filled(...) and "
                "np.ma.getmaskarray(obj) as two arrays"
            )
        if isinstance(obj, np.generic):  # a NumPy scalar is stored as a 0-d array
            obj = np.asarray(obj)
        if not isinstance(obj, np.ndarray):
            raise TypeError(
                f"write_ndarray needs a numpy.ndarray, got {type(obj).__name__}; "
                "convert it with np.asarray(...)"
            )
        if compression is not None:
            _codec._check_compression(compression, compression_level)
        resolved = get_loc(key, index)
        if chunk_rows is not None:
            return self._put_chunked(resolved, obj, chunk_rows, compression, compression_level)
        body = _codec.ndarray_to_bytes(
            obj, compression=compression, compression_level=compression_level
        )
        return self._put_tagged(resolved, body, "write_ndarray")

    def write_tensor(
        self,
        obj: Tensor,
        *,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
    ) -> bool:
        """Store a PyTorch tensor with a self-describing header.

        Args:
            obj: Tensor to store. Requires the optional PyTorch dependency.
            key: Logical storage key.
            index: Optional nonnegative index; None omits it.

        Returns:
            True on success, overwriting any value at the same key.

        Note:
            Detach gradient history and use contiguous CPU layout. Shape and dtype
            are retained, but device and ``requires_grad`` are not. Restore with
            ``LMDBReader.read_as_tensor``.
        """
        torch_module = sys.modules.get("torch")
        if torch_module is None or not isinstance(obj, torch_module.Tensor):
            raise TypeError(f"write_tensor needs a torch.Tensor, got {type(obj).__name__}")
        resolved = get_loc(key, index)
        return self._put_tagged(resolved, _codec.tensor_to_bytes(obj), "write_tensor")

    def delete(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        missing_ok: bool = False,
    ) -> bool:
        """Remove one entry's body and integrity envelope from the staging generation.

        Args:
            key: Logical storage key.
            index: Index of the entry; None selects the unindexed entry only.
            missing_ok: Return False instead of raising when the entry is absent.

        Returns:
            True when an entry was removed, False when it was absent and
            ``missing_ok`` is set.

        Raises:
            KeyError: The entry is absent and ``missing_ok`` is False.
            ValueError: The key is in the reserved ``"amber-lmdb"`` namespace.

        Note:
            Applies to entries copied from ``base`` and to entries written earlier
            by this writer. A body without an envelope is removed as well, and so
            are the chunks of a chunked array.
        """
        if type(missing_ok) is not bool:
            raise TypeError("missing_ok must be bool")
        location = get_loc(key, index)
        _refuse_reserved(location)
        transaction = self._get_txn()
        if self._may_hold_chunks:
            self._drop_chunks(location)
        removed = transaction.delete(location, db=self._metadata_db)
        removed = transaction.delete(location) or removed
        if not removed and not missing_ok:
            raise _key_not_found(location)
        return removed

    def delete_key(self, key: str | Path | tuple[str, ...]) -> int:
        """Remove every entry of a logical key: its unindexed entry and all indices.

        Args:
            key: Logical storage key.

        Returns:
            Number of entries removed; 0 when the key has none. Chunks of chunked
            arrays are removed with them and not counted.

        Raises:
            ValueError: The key is in the reserved ``"amber-lmdb"`` namespace.
        """
        prefix = _key_prefix(key)
        _refuse_reserved(prefix)
        transaction = self._get_txn()
        locations: set[bytes] = set()
        for database in (self._metadata_db, None):
            with transaction.cursor(db=database) as cursor:
                found = cursor.set_range(prefix)
                while found:
                    location = bytes(cursor.key())
                    # The prefix fixes kind, component count and components, so only
                    # this key's unindexed and indexed entries can follow it.
                    if (
                        location != prefix + b"\x00"
                        and _index_of(location, prefix + b"\x01") is None
                    ):
                        break
                    locations.add(location)
                    found = cursor.next()
        for location in locations:
            if self._may_hold_chunks:
                self._drop_chunks(location)
            transaction.delete(location, db=self._metadata_db)
            transaction.delete(location)
        return len(locations)

    def commit(self) -> None:
        """Commit the current write transaction without publishing the generation.

        A subsequent write lazily starts a new transaction. Publication happens
        only on successful context exit, which also commits the remaining writes, so
        calling ``commit()`` is optional; use it to bound the size of one transaction.
        With no writes since the last commit it does nothing. LMDB commit errors
        propagate without conversion.
        """
        if self._txn is not None:
            self._txn.commit()
            self._txn = None

    def _acquire_lock(self) -> None:
        """Acquire the neighboring writer lock without waiting.

        Hold the handle until ``_release_lock``. Convert OS lock-acquisition errors
        to BlockingIOError; propagate failures to open the lock file. Keep the lock
        file in place rather than unlinking its inode.
        """
        self._lock_handle = _acquire_writer_lock(self._lock_path)

    def _release_lock(self) -> None:
        """Release the OS writer lock and close its handle.

        Do nothing if no lock is held. Ignore unlock errors during cleanup, but
        always close the handle and clear the stored reference.
        """
        handle, self._lock_handle = self._lock_handle, None
        if handle is not None:
            _release_writer_lock(handle)

    def _remove_temporary(self) -> None:
        """Remove this writer's staging directory within the expected parent.

        Refuse an unexpected resolved parent with RuntimeError. Ignore deletion
        errors and clear the staging-path reference after the attempt.
        """
        temporary = self._tmp_path
        if temporary is None:
            return
        expected_parent = self._path.parent.resolve()
        if temporary.parent.resolve() != expected_parent:
            raise RuntimeError("refusing to remove a temporary directory outside the parent")
        shutil.rmtree(temporary, ignore_errors=True)
        self._tmp_path = None

    def _cleanup_staging(self) -> None:
        """Remove recognized UUID-named staging directories left by earlier writers.

        Preserve symlinks and entries that do not match the expected naming and
        directory rules. Propagate deletion errors.
        """
        prefix = f".{self._path.name}."
        for candidate in self._path.parent.iterdir():
            name = candidate.name
            if not name.startswith(prefix) or not name.endswith(".tmp"):
                continue
            token = name[len(prefix) : -4]
            if _is_version_name(f"v-{token}") and candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate)

    def _cleanup_current_temporaries(self) -> None:
        """Validate and remove temporary ``CURRENT`` files directly below the root.

        Reject symlinks and nonregular files with ValueError; propagate deletion errors.
        """
        for candidate in self._path.glob(".CURRENT.*.tmp"):
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"invalid CURRENT temporary entry: {candidate}")
            candidate.unlink()

    def _recover_incomplete_root(self) -> None:
        """Handle a root without ``CURRENT``: remove only provably unpublished output.

        A generation recorded in ``HISTORY`` became current at least once, so a root
        whose ``CURRENT`` is missing is refused rather than cleaned (restore it with
        ``set_current``). Without ``HISTORY`` nothing can be proven and the root is
        refused too. Generations absent from an existing ``HISTORY`` were left by a
        writer interrupted before its switch; they are removed under their exclusive
        lease through ``.trash`` and logged. Unexpected entries, symlinks and invalid
        generation names or layouts raise ValueError.
        """
        root = self._path
        _validate_dataset_root(root)
        for entry in root.iterdir():
            if entry.name in {"FORMAT", "versions", ".coordination", ".trash", HISTORY_NAME} or (
                entry.name.startswith(".CURRENT.") and entry.name.endswith(".tmp")
            ):
                continue
            raise ValueError(f"unrecognized entry in incomplete LMDB dataset: {entry}")
        self._cleanup_current_temporaries()
        versions = root / "versions"
        if not versions.exists():
            return
        if versions.is_symlink() or not versions.is_dir():
            raise ValueError(f"invalid LMDB versions directory: {versions}")
        candidates = sorted(versions.iterdir())
        for candidate in candidates:
            if not _is_version_name(candidate.name) or not _is_plain_dir(candidate):
                raise ValueError(f"invalid unpublished LMDB version: {candidate}")
        if not candidates:
            return
        names = ", ".join(candidate.name for candidate in candidates)
        history = read_history(root)
        if history is None:
            raise ValueError(
                f"{root} has generations ({names}) but neither CURRENT nor HISTORY, so it "
                "cannot tell which were published; refusing to remove them. Restore one "
                "with set_current() (CLI: amber-lmdb set-current) or delete them explicitly "
                "with delete_generations()"
            )
        published = {item.generation for item in history}
        kept = [candidate.name for candidate in candidates if candidate.name in published]
        if kept:
            raise ValueError(
                f"CURRENT is missing in {root}, but {', '.join(kept)} were published; restore "
                "one with set_current() (CLI: amber-lmdb set-current) before writing"
            )
        for candidate in candidates:
            _remove_unpublished(root, candidate.name)

    def _ensure_dataset_root(self) -> Path:
        """Prepare the dataset root, format marker and generations directory.

        Create and fsync missing entries, validate an existing format marker, and
        refuse to initialize an unrecognized nonempty directory. Reject symlinks
        and paths escaping the root. Propagate filesystem errors and return the
        validated ``versions`` path.
        """
        root = self._path
        created = not root.exists()
        root.mkdir(exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f"LMDB dataset root must be a regular directory: {root}")
        marker = root / "FORMAT"
        if marker.exists():
            _validate_dataset_root(root)
        else:
            if any(root.iterdir()):
                raise ValueError(f"refusing to initialize non-empty directory: {root}")
            with marker.open("xb") as stream:
                stream.write(_FORMAT_MARKER.encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(root)
        if created:
            _fsync_directory(root.parent)
        versions = root / "versions"
        versions.mkdir(exist_ok=True)
        if versions.is_symlink() or not versions.is_dir():
            raise ValueError(f"invalid LMDB versions directory: {versions}")
        if versions.resolve().parent != root.resolve():
            raise ValueError("LMDB versions directory escapes the dataset root")
        if read_history(root) is None and not any(
            _is_version_name(entry.name) for entry in versions.iterdir()
        ):
            _create_history(root)
            _fsync_directory(root)
        return versions

    def _publish(self) -> None:
        """Serialize generation publication with cooperative acquisition and deletion."""
        self._ensure_dataset_root()
        with _root_gate(self._path):
            self._publish_locked()

    def _publish_locked(self) -> None:
        """Publish the staged generation and atomically switch ``CURRENT``.

        Move the generation into ``versions``, fsync it and replace a synced
        temporary pointer file. On failure before the pointer switches, remove
        unpublished output and re-raise. A later root-fsync failure preserves the
        published generation. Retain earlier generations because another process
        may still have one open.
        """
        if self._tmp_path is None or self._version_name is None:
            raise RuntimeError("writer has no completed temporary version")
        root = self._path
        versions = self._ensure_dataset_root()
        try:
            previous: str | None = _read_current(root)
        except FileNotFoundError:
            previous = None
        _record_current(root, previous)
        version_path = versions / self._version_name
        os.replace(self._tmp_path, version_path)
        self._tmp_path = None
        try:
            _fsync_directory(versions)
            _write_current(root, self._version_name)
        except BaseException:
            shutil.rmtree(version_path, ignore_errors=True)
            raise
        self._finalized = True
        _fsync_directory(root)
        _append_history(root, self._version_name)
        # Cleanup remains explicit: delete_generations checks lifetime OS leases.

    def __enter__(self) -> LMDBWriter:
        if self._finalized:
            raise RuntimeError("LMDBWriter instances cannot be reused")
        if self._lock_handle is not None or self._tmp_path is not None:
            raise RuntimeError("LMDBWriter is already active")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._acquire_lock()
        try:
            self._cleanup_staging()
            published = self._path.exists() and (self._path / "CURRENT").exists()
            if self._base is not None and not published:
                raise FileNotFoundError(f"base requires a published LMDB dataset: {self._path}")
            if self._path.exists():
                if self._path.is_symlink() or not self._path.is_dir():
                    raise ValueError(f"LMDB dataset root must be a regular directory: {self._path}")
                if published:
                    _read_current(self._path)
                    self._cleanup_current_temporaries()
                    if not self._overwrite and self._base is None:
                        raise FileExistsError(
                            f"{self._path} already exists. Pass overwrite=True to replace all "
                            'of it, or base="current" to add, replace or delete entries.'
                        )
                elif any(self._path.iterdir()):
                    self._recover_incomplete_root()
            self._version_name = f"v-{uuid.uuid4().hex}"
            self._tmp_path = self._path.parent / (f".{self._path.name}.{uuid.uuid4().hex}.tmp")
            self._tmp_path.mkdir(mode=0o700)
            _log.debug("LMDB path %s (staging %s)", self._path, self._tmp_path.name)
            if self._base is not None:
                self._copy_base(self._base, self._tmp_path)
            return self
        except BaseException:
            self._close_staging_env()
            self._remove_temporary()
            self._release_lock()
            raise

    def _copy_base(self, base: str, staging: Path) -> None:
        """Compactly copy the base generation into the empty staging directory.

        Hold a shared reference (and its generation lease) for the duration of the
        copy, require a matching engine, then drop any copied manifest.
        """
        root_name, _ = _resolve_lmdb_path(self._path)
        root = Path(root_name)
        name = _read_current(root) if base == "current" else base
        generation = root / "versions" / name
        if generation.is_symlink() or not generation.is_dir():
            raise FileNotFoundError(f"base generation not found: {generation}")
        key = _environment_key(
            root,
            readonly=True,
            readahead=False,
            max_readers=512,
            map_size=self._map_size,
            generation=name,
        )
        environment = _acquire_env(key)
        try:
            engine = environment.lib_version()[0]
            if engine not in (0, 1):
                raise ValueError(f"unsupported LMDB engine in base generation: {engine}")
            if self._lib_version is not None and self._lib_version != engine:
                raise ValueError(
                    f"lib_version={self._lib_version} does not match the base generation's "
                    f"engine ({engine}); omit lib_version to use the base engine"
                )
            # copyfd writes through the page cache; LMDB's own copy() opens the target
            # with O_DIRECT on Linux, so the published copy (and any generation derived
            # from it next) would have to be read back from disk.
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            descriptor = os.open(staging / "data.mdb", flags, 0o600)
            try:
                environment.copyfd(descriptor, compact=True)
            finally:
                os.close(descriptor)
        finally:
            _release_env(key)
        self._engine = 0 if engine == 0 else 1
        self._base_name = name
        transaction = self._get_txn()
        manifest = get_loc(_MANIFEST_KEY)
        transaction.delete(manifest, db=self._metadata_db)
        transaction.delete(manifest)
        self.commit()

    def _close_staging_env(self) -> None:
        """Abort any transaction and close the staging environment after a failure."""
        if self._txn is not None:
            self._txn.abort()
            self._txn = None
        if self._env is not None:
            self._env.close()
            self._env = None
            self._metadata_db = None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        try:
            if self._txn is not None:
                if exc_type is None:
                    # A successful context exit commits the remaining writes; that is the
                    # documented way to finish, so it is not worth a warning.
                    _log.debug("committing the remaining writes on context exit")
                    self._txn.commit()
                else:
                    self._txn.abort()
                self._txn = None
            if exc_type is None and self._env is None:
                self._get_env()
            if exc_type is None and self._manifest:
                from .manifest import _write_manifest

                _write_manifest(self, self._manifest_max_groups, self._provenance)
            if self._env is not None:
                if exc_type is None:
                    self._env.sync(force=True)
                self._env.close()
                self._env = None
                self._metadata_db = None
            if exc_type is None:
                if self._tmp_path is None:
                    raise RuntimeError("writer staging directory is missing")
                for child in self._tmp_path.iterdir():
                    if child.is_file():
                        _fsync_file(child)
                _fsync_directory(self._tmp_path)
                if self._verify_on_publish:
                    from .physical import verify_physical

                    report = verify_physical(self._tmp_path)
                    if not report.ok:
                        raise ValueError(f"staging physical validation failed: {report}")
                self._publish()
        finally:
            if self._env is not None:
                self._env.close()
                self._env = None
                self._metadata_db = None
            if not self._finalized:
                self._remove_temporary()
            self._release_lock()

        return False  # Do not suppress exceptions from the context.

    def __getstate__(self) -> dict[str, Any]:
        if self._lock_handle is not None:
            raise TypeError("an active LMDBWriter cannot be pickled")
        state = self.__dict__.copy()
        state["_env"] = None
        state["_txn"] = None
        state["_metadata_db"] = None
        state["_lock_handle"] = None
        return state


# Process-local shared environments and their reference counts.

# root, generation, readonly, readahead, max_readers, map_size, cooperative
_EnvKey = tuple[str, str, bool, bool, int, int, bool]
_ENV_CACHE: dict[_EnvKey, Environment] = {}
_ENV_REF_COUNT: dict[_EnvKey, int] = {}
_ENV_LEASES: dict[str, _FileLock] = {}
_ENV_LOCK = _threading.Lock()


def get_active_envs() -> frozenset[_EnvKey]:
    """Return an immutable snapshot of process-local shared environment keys.

    Returns:
        Frozen set of requested root, generation and connection-setting keys.
        Several keys may reference one physical environment; tuning follows its
        first opener until all references close.
    """
    with _ENV_LOCK:
        return frozenset(_ENV_CACHE)


def _sniff_engine(path: str) -> Literal[-1, 0, 1]:
    # Mirror upstream's 64-byte probe using Unicode-aware Python I/O. py-lmdb
    # 3.0's CPython fopen probe fails on non-ASCII Windows paths and picks 0.9.
    try:
        with (Path(path) / "data.mdb").open("rb") as stream:
            header = stream.read(64)
    except FileNotFoundError:
        return -1
    for offset in range(0, len(header) - 7, 4):
        magic, version = struct.unpack_from("=II", header, offset)
        if magic == 0xBEEFC0DE:
            if version == 1:
                return 0
            if version == 3:
                return 1
            break
    return -1


def _open_env(
    path: str,
    readonly: bool,
    readahead: bool,
    max_readers: int,
    map_size: int,
    *,
    lock: bool = True,
) -> Environment:
    """Open an LMDB environment with the supplied settings.

    ``lock=False`` is used only for read-only access to immutable published generations
    (non-cooperative readers). A missing or unrecognized data file raises ValueError;
    LMDB and OS errors propagate.
    """
    engine = _sniff_engine(path)
    if engine == -1:
        data = Path(path) / "data.mdb"
        if _lstat(data) is None:
            raise ValueError(f"generation has no data.mdb: {path}")
        raise ValueError(f"not an LMDB data file (run doctor --physical): {data}")
    env = lmdb.open(
        path,
        readonly=readonly,
        lock=lock,
        readahead=readahead,
        max_readers=max_readers,
        map_size=map_size,
        subdir=True,
        create=False,  # see LMDBWriter._get_env: never let py-lmdb make directories
        max_dbs=1,
        max_spare_txns=0,
        lib_version=engine,
    )
    if env is None:
        raise RuntimeError(f"lmdb.open returned None for {path}")
    return env


def _validate_env_options(readonly: bool, readahead: bool, max_readers: int, map_size: int) -> None:
    """Require boolean options and strictly positive integer connection limits.

    Invalid boolean types raise TypeError; invalid reader or map-size limits raise ValueError.
    """
    if type(readonly) is not bool or type(readahead) is not bool:
        raise TypeError("readonly and readahead must be bool")
    if type(max_readers) is not int or max_readers <= 0:
        raise ValueError("max_readers must be a positive integer")
    if type(map_size) is not int or map_size <= 0:
        raise ValueError("map_size must be a positive integer")


def _resolve_generation(path: str | Path, name: str) -> tuple[str, str]:
    """Validate a dataset root and resolve one named generation beneath ``versions``.

    ``CURRENT`` is not required. A missing generation raises FileNotFoundError.
    """
    if type(name) is not str or not _is_version_name(name):
        raise ValueError("generation must be a generation name such as v-<32 hex>")
    requested = Path(path).absolute()
    if requested.is_symlink():
        raise ValueError(f"LMDB dataset root cannot be a symlink: {requested}")
    root = requested.resolve()
    _validate_dataset_root(root)
    versions = root / "versions"
    if not _is_plain_dir(versions) or versions.resolve().parent != root:
        raise ValueError("LMDB versions must be a regular directory inside the dataset root")
    candidate = versions / name
    if not _is_plain_dir(candidate):
        raise FileNotFoundError(f"generation not found: {candidate}")
    version_path = candidate.resolve()
    if version_path.parent != versions.resolve():
        raise ValueError("generation points outside the versions directory")
    return str(root), str(version_path)


def _environment_key(
    path: str | Path,
    *,
    readonly: bool,
    readahead: bool,
    max_readers: int,
    map_size: int,
    generation: str | None = None,
    cooperative: bool = True,
) -> _EnvKey:
    """Validate settings and resolve a cache key containing the root and generation.

    ``generation`` selects a named generation instead of ``CURRENT``. Propagate errors
    for invalid options, layout, pointer or generation paths.
    """
    _validate_env_options(readonly, readahead, max_readers, map_size)
    if type(cooperative) is not bool:
        raise TypeError("cooperative must be bool")
    if not cooperative and not readonly:
        raise ValueError("non-cooperative access is read-only")
    if generation is None:
        root, version = _resolve_lmdb_path(path)
    else:
        root, version = _resolve_generation(path, generation)
    return root, version, readonly, readahead, max_readers, map_size, cooperative


def _acquire_env(key: _EnvKey, *, validated: bool = False) -> Environment:
    """Acquire a shared environment by key and increment its reference count.

    ``validated`` skips re-validating the root and its marker when the caller has just
    built ``key`` with ``_environment_key``, which validates them.
    """
    if not validated:
        _validate_dataset_root(Path(key[0]))
    cooperative = key[6]
    # Non-cooperative readers (immutable read-only media) take no coordination locks.
    gate = _root_gate(Path(key[0])) if cooperative else nullcontext()
    with gate, _ENV_LOCK:
        generation = Path(key[1])
        if not _is_plain_dir(generation) or generation.parent != Path(key[0]) / "versions":
            raise FileNotFoundError("generation was removed before acquisition; retry the read")
        if key in _ENV_CACHE:
            _ENV_REF_COUNT[key] += 1
            return _ENV_CACHE[key]
        matching = next((opened for opened in _ENV_CACHE if opened[1] == key[1]), None)
        if matching is not None:
            if matching[2] != key[2]:
                raise ValueError(
                    "generation is already open with a different readonly setting; "
                    "close all references before changing access mode"
                )
            if matching[6] != key[6]:
                raise ValueError(
                    "generation is already open with a different cooperative setting; "
                    "close all references before changing it"
                )
            # LMDB must not be opened twice for one generation in a process.
            # The first opener's tuning applies until every reference is released.
            env = _ENV_CACHE[matching]
        else:
            lease = (
                _generation_lock(Path(key[0]), generation.name, shared=True)
                if cooperative
                else None
            )
            try:
                env = _open_env(
                    key[1],
                    readonly=key[2],
                    readahead=key[3],
                    max_readers=key[4],
                    map_size=key[5],
                    lock=cooperative,
                )
            except BaseException:
                if lease is not None:
                    lease.close()
                raise
            if lease is not None:
                _ENV_LEASES[key[1]] = lease
        _ENV_CACHE[key] = env
        _ENV_REF_COUNT[key] = 1
        return env


def _release_env_locked(key: _EnvKey) -> None:
    """Release one reference under the cache lock; close the environment at zero."""
    if key not in _ENV_REF_COUNT:
        return
    if _ENV_REF_COUNT[key] > 1:
        _ENV_REF_COUNT[key] -= 1
        return
    environment = _ENV_CACHE[key]
    if not any(shared is environment for other, shared in _ENV_CACHE.items() if other != key):
        # A failed native close must retain ownership; the same release can retry.
        environment.close()
        lease = _ENV_LEASES.pop(key[1], None)
        if lease is not None:
            lease.close()
    del _ENV_CACHE[key]
    del _ENV_REF_COUNT[key]


def _release_env(key: _EnvKey) -> None:
    """Release one reference for the exact key under the environment-cache lock."""
    with _ENV_LOCK:
        _release_env_locked(key)


def _clear_env_cache() -> None:
    """Clear and close the process-local environment cache for cleanup."""
    with _ENV_LOCK:
        for key in tuple(_ENV_CACHE):
            _ENV_REF_COUNT[key] = 1
            _release_env_locked(key)


class EnvironmentHandle:
    """Own one exact generation reference; create with ``open_environment``.

    Handles are context managers, cannot be pickled, and must not be reused after
    close or transferred across a fork. Closing a handle always releases the
    generation it acquired, even when ``CURRENT`` changes.
    """

    def __init__(self, key: _EnvKey, environment: Environment) -> None:
        self._key = key
        self._environment: Environment | None = environment
        self._pid = os.getpid()

    @property
    def environment(self) -> Environment:
        """Return the owned environment; fail after close or a PID change."""
        if self._environment is None or self._pid != os.getpid():
            raise RuntimeError("environment handle is closed or belongs to another process")
        return self._environment

    def close(self) -> None:
        """Release the exact reference once; repeated closes are harmless."""
        if self._environment is not None:
            if self._pid == os.getpid():
                _release_env(self._key)
            self._environment = None

    def __enter__(self) -> EnvironmentHandle:
        _ = self.environment
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    def __getstate__(self) -> dict[str, Any]:
        raise TypeError("environment handles cannot be pickled; use LMDBReader")


def open_environment(
    path: str | Path,
    *,
    readonly: bool = True,
    readahead: bool = False,
    max_readers: int = 512,
    map_size: int = 512 * 1024**3,
    generation: str | None = None,
    cooperative: bool = True,
) -> EnvironmentHandle:
    """Acquire an exact, context-managed reference to a published generation.

    Tuning follows the first opener until all references close. Readonly access
    modes must match. ``generation`` opens a named generation instead of ``CURRENT``;
    ``cooperative=False`` is the read-only mode for immutable media (see
    ``LMDBReader``). Use this low-level API only when raw transactions are needed;
    prefer ``LMDBReader`` for typed reads. Do not close the raw environment directly.
    """
    key = _environment_key(
        path,
        readonly=readonly,
        readahead=readahead,
        max_readers=max_readers,
        map_size=map_size,
        generation=generation,
        cooperative=cooperative,
    )
    return EnvironmentHandle(key, _acquire_env(key))


@contextmanager
def _opened_store(
    path: str | Path, *, generation: str | None = None, cooperative: bool = True
) -> Iterator[tuple[Environment, _EnvKey]]:
    """Hold a read-only reference to ``CURRENT`` or a named generation for one scan."""
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
    try:
        yield environment, key
    finally:
        _release_env(key)


def _after_fork() -> None:
    """Discard inherited environment references and replace the cache lock after fork."""
    global _ENV_LOCK
    environments = tuple(dict.fromkeys(_ENV_CACHE.values()))
    _ENV_CACHE.clear()
    _ENV_REF_COUNT.clear()
    _ENV_LOCK = _threading.Lock()
    for environment in environments:
        environment.close()
    for lease in _ENV_LEASES.values():
        lease.close()
    _ENV_LEASES.clear()


register_at_fork: Callable[..., None] | None = getattr(os, "register_at_fork", None)
if register_at_fork is not None:
    register_at_fork(after_in_child=_after_fork)


@dataclass(frozen=True, slots=True)
class LMDBSnapshot:
    """Identify the generation actually opened and its committed transaction ID.

    Attributes:
        root: Resolved absolute dataset root.
        generation: Resolved absolute directory of the opened generation.
        last_txnid: Last committed transaction ID observed in that generation.
    """

    root: str
    generation: str
    last_txnid: int


@dataclass(frozen=True, slots=True)
class KeyGroup:
    """One logical key of a listing, with its unindexed entry and index bounds.

    Indices are not counted, so a listing costs one seek per logical key, not one
    step per entry; ``count(key)`` counts the indices of a single key.

    Attributes:
        key: Logical key as written: a string or a tuple of strings.
        unindexed: Whether an entry without index exists.
        first_index: Smallest stored index, or None without indexed entries.
        last_index: Largest stored index, or None without indexed entries.
    """

    key: str | tuple[str, ...]
    unindexed: bool
    first_index: int | None
    last_index: int | None


class TypedReader:
    """Typed read API shared by ``LMDBReader`` and ``LMDBOverlayReader``.

    Use it in annotations for code that accepts either reader (``RecordTable`` and
    ``ValueView`` do). Subclasses supply owned entries, array header entries and a
    source identity; every public typed read below is built on those operations.
    It is not meant to be subclassed outside amber-lmdb.
    """

    _value_limits: ValueLimits | None = None

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
        raise NotImplementedError

    def _owned_rows(
        self,
        locations: Sequence[bytes],
        ranges: Sequence[tuple[int, int]],
        *,
        max_bytes: int | None,
    ) -> list[tuple[EntryMetadata, _ArrayRows]]:
        raise NotImplementedError

    def _array_header_entries(
        self,
        locations: Sequence[bytes],
        max_header_bytes: int,
        max_bytes: int,
    ) -> list[tuple[str, bytes, int]]:
        raise NotImplementedError

    def get_snapshot(self) -> Any:
        """Return an identity of the opened source; it changes when the source does."""
        raise NotImplementedError

    def _get_bytes(
        self, key: bytes, read_method: str, max_bytes: int | None = None
    ) -> bytes | bytearray:
        """Validate a writer tag and obtain owned bytes in the same transaction.

        Args:
            key: Binary LMDB key encoded by ``get_loc``.
            read_method: Reader method whose compatibility must be checked.
            max_bytes: Copy budget for the body and its integrity metadata, or None.

        Returns:
            Owned bytes that remain valid after the transaction closes.

        Raises:
            KeyError: The key does not exist.
            ValueError: The writer tag is missing or incompatible.
        """
        _validate_max_bytes(max_bytes)
        return _plain(self._owned_entries([key], read_method, max_bytes=max_bytes)[0][2])

    @staticmethod
    def _decode_entry(
        tag: str,
        value: _Body,
        *,
        allow_custom: bool,
        limits: ValueLimits | None = None,
        budget: _codec._NodeBudget | None = None,
    ) -> Any:
        """Decode owned bytes using the codec selected by a validated writer tag.

        ``limits`` and ``budget`` bound MessagePack values only; arrays, strings and
        primitives have no schema nodes beyond their own.
        """
        if isinstance(value, _ArrayRows):
            return value.array()
        if tag in (_chunks.CHUNKED_TAG, _chunks.CHUNK_TAG):
            raise ValueError(f"{tag} entries are decoded only through their array reads")
        if tag == "write_ndarray":
            return _codec.bytes_to_ndarray(_as_bytes(value))
        if tag == "write_tensor":
            return _tensor(value)
        if tag == "write_str":
            return _codec.bytes_to_str(_as_bytes(value))
        if tag == "write_primitive":
            return _codec.bytes_to_primitive(value)
        result = _codec._unpack_value(
            value, allow_custom=allow_custom, limits=limits, budget=budget
        )
        expected = {
            "write_sequence": (list, tuple),
            "write_dict": (dict, frozendict),
            "write_set": (set, frozenset),
        }.get(tag)
        if expected is not None and type(result) not in expected:
            raise ValueError(f"value type does not match writer tag: {tag}")
        return result

    def read(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        allow_custom: bool = False,
        max_bytes: int | None = None,
    ) -> Any:
        """Select a decoder from the stored writer tag and restore the original type.

        Args:
            key: Key used when storing the value.
            index: Optional index used when storing the value.
            allow_custom: Whether registered custom decoders may run.
            max_bytes: Refuse (``ReadLimitError``) to copy a body larger than this,
                counting its integrity metadata, before copying; None: no limit.

        Returns:
            Value that remains valid after the reader closes. NumPy arrays reference
            owned bytes rather than the LMDB mapping.

        Raises:
            TypeError: An option or key has an invalid type.
            KeyError: The key does not exist.
            ValueError: The writer tag or encoded value is invalid.
        """
        if type(allow_custom) is not bool:
            raise TypeError("allow_custom must be bool")
        _validate_max_bytes(max_bytes)
        loc = get_loc(key, index)
        _, metadata, value = self._owned_entries([loc], "read", max_bytes=max_bytes)[0]
        return self._decode_entry(
            metadata.writer_tag, value, allow_custom=allow_custom, limits=self._value_limits
        )

    def read_many(
        self,
        keys: Sequence[str | Path | tuple[str, ...]],
        *,
        indices: Sequence[int | None] | None = None,
        allow_custom: bool = False,
        max_items: int = 1024,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        max_decoded_nodes: int | None = None,
    ) -> list[Any]:
        """Read a bounded batch in one transaction, then decode after it closes.

        Args:
            keys: Stored keys; input order and duplicates are preserved.
            indices: Index for each key; None means every key has no index.
            allow_custom: Whether registered custom decoders may run.
            max_items: Item limit, 1 to 1,048,576 (2**20).
            max_bytes: Aggregate encoded body/integrity metadata copy budget, 64 MiB by
                default; None removes it. Decoder allocation/runtime and retained caller
                results are separate.
            max_decoded_nodes: MessagePack schema nodes the whole batch may decode;
                exceeding it raises ``ReadLimitError(kind="nodes")``. None: unbounded.

        Returns:
            Values in key order. Empty input opens no environment or transaction.
            Decoders run after the transaction closes.

        Raises:
            TypeError: An option or key has an invalid type.
            ValueError: A limit, index, length, writer tag or encoded value is invalid.
            KeyError: Any requested key is missing.
        """
        if type(allow_custom) is not bool:
            raise TypeError("allow_custom must be bool")
        budget = _node_budget(max_decoded_nodes)
        locations = self._batch_locations(keys, indices, max_items, max_bytes)
        if not locations:
            return []
        return [
            self._decode_entry(
                metadata.writer_tag,
                value,
                allow_custom=allow_custom,
                limits=self._value_limits,
                budget=budget,
            )
            for _, metadata, value in self._owned_entries(locations, "read", max_bytes=max_bytes)
        ]

    @staticmethod
    def _batch_locations(
        keys: Sequence[str | Path | tuple[str, ...]],
        indices: Sequence[int | None] | None,
        max_items: int,
        max_bytes: int | None,
    ) -> list[bytes]:
        _validate_limits(max_items, max_bytes)
        if isinstance(keys, (str, bytes, PurePath)):
            raise TypeError("keys must be a sequence of keys, not a single key")
        if not isinstance(keys, Sequence):
            raise TypeError(
                f"keys must be a sequence such as a list, not {type(keys).__name__}; "
                "materialize it with list(...), or stream it with iter_read_pages"
            )
        if len(keys) > max_items:
            raise ReadLimitError(kind="items", limit=max_items, required=len(keys))
        positions = [None] * len(keys) if indices is None else _index_sequence(indices)
        if len(positions) != len(keys):
            raise ValueError(
                f"indices needs one entry per key: got {len(positions)} for {len(keys)} keys"
            )
        return [get_loc(key, index) for key, index in zip(keys, positions, strict=True)]

    def _stream_owned(
        self,
        keys: Iterable[str | Path | tuple[str, ...]],
        indices: Iterable[int | None] | None,
        max_items: int,
        max_bytes: int | None,
        read_method: str,
    ) -> Iterator[list[tuple[bytes, EntryMetadata, _Body]]]:
        from .collections import SourceChangedError

        _validate_limits(max_items, max_bytes)
        if isinstance(keys, (str, bytes, PurePath)):
            raise TypeError("keys must be an iterable of keys, not a single key")
        if isinstance(indices, (str, bytes)):
            raise TypeError("indices must be an iterable of integers or None")
        requests = ((key, None) for key in keys) if indices is None else _pairs(keys, indices)
        snapshot = None
        while True:
            locations = [get_loc(key, index) for key, index in islice(requests, max_items)]
            if not locations:
                return
            if snapshot is None:
                snapshot = self.get_snapshot()
            start = 0
            while start < len(locations):
                if self.get_snapshot() != snapshot:
                    raise SourceChangedError("store changed between owned read pages")
                page = self._owned_entries(
                    locations[start:],
                    read_method,
                    max_bytes=max_bytes,
                    partial=True,
                    snapshot=snapshot,
                )
                start += len(page)
                yield page

    def iter_read_pages(
        self,
        keys: Iterable[str | Path | tuple[str, ...]],
        *,
        indices: Iterable[int | None] | None = None,
        allow_custom: bool = False,
        max_items: int = 1024,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        max_decoded_nodes: int | None = None,
    ) -> Iterator[list[Any]]:
        """Yield owned byte-aware decoded pages in input order, including duplicates.

        Each page closes its transaction before checksum verification/decoding/yield.
        Empty requests open nothing. Keys are validated page-by-page. A single entry
        larger than max_bytes raises ReadLimitError; smaller pages do not silently
        raise the limit. The opened generation/transaction ID must remain unchanged.
        ``max_decoded_nodes`` bounds the MessagePack schema nodes of each page
        (``ReadLimitError(kind="nodes")``); otherwise decoder allocations and results
        accumulated by the caller are not bounded.
        """
        if type(allow_custom) is not bool:
            raise TypeError("allow_custom must be bool")
        _node_budget(max_decoded_nodes)
        for page in self._stream_owned(keys, indices, max_items, max_bytes, "read"):
            budget = _node_budget(max_decoded_nodes)
            yield [
                self._decode_entry(
                    metadata.writer_tag,
                    value,
                    allow_custom=allow_custom,
                    limits=self._value_limits,
                    budget=budget,
                )
                for _, metadata, value in page
            ]

    def iter_dict_pages(
        self,
        keys: Iterable[str | Path | tuple[str, ...]],
        *,
        indices: Iterable[int | None] | None = None,
        allow_custom: bool = False,
        max_items: int = 1024,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        max_decoded_nodes: int | None = None,
    ) -> Iterator[list[dict[Any, Any]]]:
        """Yield byte-aware strict dictionary pages; lifecycle/budgets match iter_read_pages."""
        if type(allow_custom) is not bool:
            raise TypeError("allow_custom must be bool")
        _node_budget(max_decoded_nodes)
        for page in self._stream_owned(keys, indices, max_items, max_bytes, "read_as_dict"):
            budget = _node_budget(max_decoded_nodes)
            yield [self._unpack(_plain(value), dict, allow_custom, budget) for _, _, value in page]

    def _unpack(
        self,
        value: bytes | bytearray,
        expected: type[Any],
        allow_custom: bool,
        budget: _codec._NodeBudget | None = None,
    ) -> Any:
        """Decode one MessagePack value of an exact type under this reader's limits."""
        return _codec._unpack_expected(
            value, expected, allow_custom=allow_custom, limits=self._value_limits, budget=budget
        )

    def read_array_headers_many(
        self,
        keys: Sequence[str | Path | tuple[str, ...]],
        *,
        indices: Sequence[int | None] | None = None,
        max_items: int = 1024,
        max_header_bytes: int = 64 * 1024,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> list[ArrayHeader]:
        """Read bounded array headers in one short transaction without body copies.

        Args:
            keys: NumPy/tensor keys, preserving order and duplicates.
            indices: Optional index for each key; None selects unindexed keys.
            max_items: Item limit, 1 to 1,048,576 (2**20).
            max_header_bytes: Positive per-entry array metadata byte limit.
            max_bytes: Aggregate owned header/integrity metadata copy limit; None
                removes it. No body checksum is verified by this header-only API.

        Returns:
            Owned ArrayHeader values parsed after the transaction ends. Empty input
            opens no environment. Tensor headers do not require PyTorch.

        Raises:
            TypeError: Keys or options have invalid types.
            ValueError: Limits, lengths, tags, headers or array sizes are invalid.
            KeyError: A requested key is missing.
        """
        if type(max_header_bytes) is not int:
            raise TypeError("max_header_bytes must be an integer")
        if max_header_bytes <= 0:
            raise ValueError("max_header_bytes must be a positive integer")
        locations = self._batch_locations(keys, indices, max_items, max_bytes)
        if not locations:
            return []
        entries = self._array_header_entries(
            locations, max_header_bytes, sys.maxsize if max_bytes is None else max_bytes
        )
        result = []
        parsed: dict[tuple[str, bytes, int], ArrayHeader] = {}
        for tag, header, size in entries:
            identity = (tag, header, size)
            info = parsed.get(identity)
            if info is None:
                info = parse_array_header(
                    header, stored_bytes=size, max_header_bytes=max_header_bytes
                )
                _check_array_tag(tag, info)
                parsed[identity] = info
            result.append(info)
        return result

    def read_array_rows[T: np.generic](
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        start: int,
        stop: int,
        dtype: type[T] | None = None,
        casting: str = "same_kind",
        writable: bool = False,
        max_bytes: int | None = None,
    ) -> NDArray[T]:
        """Read rows ``[start, stop)`` along axis 0 of a NumPy array.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the array.
            start: First row, a nonnegative integer.
            stop: Row after the last, a nonnegative integer. Both are clamped to the
                row count like slicing, so a range past the end returns fewer rows.
            dtype: Output dtype; None retains the stored dtype.
            casting: NumPy casting rule for ``dtype``; the default ``"same_kind"``
                refuses lossy kinds such as float to int (pass ``"unsafe"`` for them).
            writable: Return a writable array (in place when possible, else a copy).
            max_bytes: Copy budget (rows or chunks, headers, integrity metadata);
                None: no limit.

        Returns:
            Array of shape ``(rows, *shape[1:])``, read-only unless converted or
            ``writable``; it owns its memory, not the rest of the stored array.

        Note:
            For an array written with ``chunk_rows``, only the chunks the rows touch
            are copied and verified. For an unchunked array, a reader with
            ``verify_checksums=False`` copies only the requested rows; a verifying
            reader copies and verifies the whole body, then keeps the rows. Tensors
            and arrays without dimensions are not supported.
        """
        return self.read_array_rows_many(
            [key],
            indices=[index],
            starts=[start],
            stops=[stop],
            dtype=dtype,
            casting=casting,
            writable=writable,
            max_bytes=max_bytes,
        )[0]

    def read_array_rows_many[T: np.generic](
        self,
        keys: Sequence[str | Path | tuple[str, ...]],
        *,
        indices: Sequence[int | None] | None = None,
        starts: Sequence[int],
        stops: Sequence[int],
        dtype: type[T] | None = None,
        casting: str = "same_kind",
        writable: bool = False,
        max_items: int = 1024,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> list[NDArray[T]]:
        """Read row ranges of several NumPy arrays in one short transaction.

        Args:
            keys: Array keys; order and duplicates are preserved.
            indices: Index for each key; None means every key has no index.
            starts: First row for each key.
            stops: Row after the last for each key; see ``read_array_rows``.
            dtype: Output dtype; None retains the stored dtype.
            casting: NumPy casting rule for ``dtype``; see ``read_array_rows``.
            writable: Return writable arrays; see ``read_array_rows``.
            max_items: Item limit, 1 to 1,048,576 (2**20).
            max_bytes: Budget for copied headers, rows or chunks and their integrity
                metadata, 64 MiB by default; None removes it.

        Returns:
            Arrays in key order, as ``read_array_rows`` returns them. Empty input
            opens no environment.

        Raises:
            TypeError: A key, row bound or option has an invalid type.
            ValueError: A row bound is negative, a limit is invalid, or an entry is
                not an array with at least one dimension.
            KeyError: A requested key is missing.
            ReadLimitError: The copies would exceed ``max_bytes``.
        """
        if type(writable) is not bool:
            raise TypeError("writable must be bool")
        if casting not in _CASTINGS:
            raise ValueError(f"casting must be one of {sorted(_CASTINGS)}")
        locations = self._batch_locations(keys, indices, max_items, max_bytes)
        if len(starts) != len(keys) or len(stops) != len(keys):
            raise ValueError("starts and stops need one value per key")
        for bound in (*starts, *stops):
            if isinstance(bound, bool) or not isinstance(bound, (int, np.integer)):
                raise TypeError("row bounds must be integers")
            if bound < 0:
                raise ValueError("row bounds must be nonnegative")
        if not locations:
            return []
        ranges = [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]
        owned = self._owned_rows(locations, ranges, max_bytes=max_bytes)
        arrays = [_converted(rows.array(), dtype, casting) for _, rows in owned]
        return [_writable(array) for array in arrays] if writable else arrays

    def _get_many_bytes(
        self,
        keys: Sequence[str | Path | tuple[str, ...]],
        indices: Sequence[int | None] | None,
        max_items: int,
        read_method: str,
        max_bytes: int | None,
    ) -> list[bytes | bytearray]:
        """Validate all keys and typed tags, then obtain owned bytes in one transaction."""
        locations = self._batch_locations(keys, indices, max_items, max_bytes)
        if not locations:
            return []
        return [
            _plain(value)
            for _, _, value in self._owned_entries(locations, read_method, max_bytes=max_bytes)
        ]

    def read_as_dict_many(
        self,
        keys: Sequence[str | Path | tuple[str, ...]],
        *,
        indices: Sequence[int | None] | None = None,
        allow_custom: bool = False,
        max_items: int = 1024,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
        max_decoded_nodes: int | None = None,
    ) -> list[dict[Any, Any]]:
        """Read dictionaries in one short transaction and decode after it closes.

        Args:
            keys: Stored keys; input order and duplicates are preserved.
            indices: Index for each key; None means every key has no index.
            allow_custom: Whether registered custom decoders may run.
            max_items: Item limit, 1 to 1,048,576 (2**20).
            max_bytes: Aggregate owned encoded body/integrity metadata copy limit.
                Decoder allocations and runtime are not bounded by this budget.

        Returns:
            Dictionaries with the same strict type checks as ``read_as_dict``.
            Empty input opens no environment or transaction. Stored frozendicts
            are rejected rather than converted to dicts.

        Raises:
            TypeError: An option or key has an invalid type.
            ValueError: A limit, index, length, writer tag or dictionary payload is invalid.
            KeyError: Any requested key is missing.
        """
        if type(allow_custom) is not bool:
            raise TypeError("allow_custom must be bool")
        budget = _node_budget(max_decoded_nodes)
        values = self._get_many_bytes(keys, indices, max_items, "read_as_dict", max_bytes)
        return [self._unpack(value, dict, allow_custom, budget) for value in values]

    def read_as_primitive_many(
        self,
        keys: Sequence[str | Path | tuple[str, ...]],
        *,
        indices: Sequence[int | None] | None = None,
        max_items: int = 1024,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> list[bool | int | float | complex]:
        """Read booleans or numbers in one short transaction and decode afterward.

        Args:
            keys: Stored keys; input order and duplicates are preserved.
            indices: Index for each key; None means every key has no index.
            max_items: Item limit, 1 to 1,048,576 (2**20).
            max_bytes: Aggregate owned encoded body/integrity metadata copy limit.

        Returns:
            Values passing the same strict type checks as ``read_as_primitive``.
            Empty input opens no environment or transaction.

        Raises:
            TypeError: An option or key has an invalid type.
            ValueError: A limit, index, length, writer tag or numeric payload is invalid.
            KeyError: Any requested key is missing.
        """
        values = self._get_many_bytes(keys, indices, max_items, "read_as_primitive", max_bytes)
        return [_codec.bytes_to_primitive(value) for value in values]

    def read_as_any(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        allow_custom: bool = False,
        max_bytes: int | None = None,
    ) -> Any:
        """Restore a type-tagged MessagePack value with the Python type it was stored as.

        Reads values stored by ``write_any``, ``write_primitive``, ``write_sequence``,
        ``write_set`` and ``write_dict``; strings (``write_str``) and arrays have their
        own readers, and ``read`` chooses the reader from the writer tag.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the value.
            allow_custom: Whether registered custom decoders may run.
            max_bytes: Refuse bodies larger than this before copying; None: no limit.

        Returns:
            Stored value, retaining its encoded Python type.
        """
        loc = get_loc(key, index)
        if type(allow_custom) is not bool:
            raise TypeError("allow_custom must be bool")
        return _codec._unpack_value(
            self._get_bytes(loc, "read_as_any", max_bytes),
            allow_custom=allow_custom,
            limits=self._value_limits,
        )

    def read_as_primitive(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        max_bytes: int | None = None,
    ) -> bool | int | float | complex:
        """Restore a boolean or number stored by ``write_primitive``.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the value.
            max_bytes: Refuse bodies larger than this before copying; None: no limit.

        Returns:
            Stored bool, int, float or complex value of the original type.
        """
        loc = get_loc(key, index)
        return _codec.bytes_to_primitive(self._get_bytes(loc, "read_as_primitive", max_bytes))

    def read_as_str(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        max_bytes: int | None = None,
    ) -> str:
        """Restore a UTF-8 string stored by ``write_str``.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the value.
            max_bytes: Refuse bodies larger than this before copying; None: no limit.

        Returns:
            Decoded string.
        """
        loc = get_loc(key, index)
        return _codec.bytes_to_str(_as_bytes(self._get_bytes(loc, "read_as_str", max_bytes)))

    def read_as_tuple(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        allow_custom: bool = False,
        max_bytes: int | None = None,
    ) -> tuple[Any, ...]:
        """Restore a tuple stored by ``write_sequence``.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the value.
            allow_custom: Whether registered custom decoders may run.
            max_bytes: Refuse bodies larger than this before copying; None: no limit.

        Returns:
            Stored tuple. A stored list raises ValueError rather than being converted.
        """
        loc = get_loc(key, index)
        return self._unpack(self._get_bytes(loc, "read_as_tuple", max_bytes), tuple, allow_custom)

    def read_as_list(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        allow_custom: bool = False,
        max_bytes: int | None = None,
    ) -> list[Any]:
        """Restore a list stored by ``write_sequence``.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the value.
            allow_custom: Whether registered custom decoders may run.
            max_bytes: Refuse bodies larger than this before copying; None: no limit.

        Returns:
            Stored list. A stored tuple raises ValueError rather than being converted.
        """
        loc = get_loc(key, index)
        return self._unpack(self._get_bytes(loc, "read_as_list", max_bytes), list, allow_custom)

    def read_as_set(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        allow_custom: bool = False,
        max_bytes: int | None = None,
    ) -> set[Any]:
        """Restore a set stored by ``write_set``.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the value.
            allow_custom: Whether registered custom decoders may run.
            max_bytes: Refuse bodies larger than this before copying; None: no limit.

        Returns:
            Stored set. A stored frozenset raises ValueError rather than being converted.
        """
        loc = get_loc(key, index)
        return self._unpack(self._get_bytes(loc, "read_as_set", max_bytes), set, allow_custom)

    def read_as_frozenset(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        allow_custom: bool = False,
        max_bytes: int | None = None,
    ) -> frozenset[Any]:
        """Restore a frozenset stored by ``write_set``.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the value.
            allow_custom: Whether registered custom decoders may run.
            max_bytes: Refuse bodies larger than this before copying; None: no limit.

        Returns:
            Stored frozenset. A stored set raises ValueError rather than being converted.
        """
        loc = get_loc(key, index)
        return self._unpack(
            self._get_bytes(loc, "read_as_frozenset", max_bytes), frozenset, allow_custom
        )

    def read_as_dict(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        allow_custom: bool = False,
        max_bytes: int | None = None,
    ) -> dict[Any, Any]:
        """Restore a dict stored by ``write_dict``.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the value.
            allow_custom: Whether registered custom decoders may run.
            max_bytes: Refuse bodies larger than this before copying; None: no limit.

        Returns:
            Stored dict. A stored frozendict raises ValueError rather than being converted.
        """
        loc = get_loc(key, index)
        return self._unpack(self._get_bytes(loc, "read_as_dict", max_bytes), dict, allow_custom)

    def read_as_frozendict(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        *,
        allow_custom: bool = False,
        max_bytes: int | None = None,
    ) -> frozendict[Any, Any]:
        """Restore a frozendict stored by ``write_dict``.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the value.
            allow_custom: Whether registered custom decoders may run.
            max_bytes: Refuse bodies larger than this before copying; None: no limit.

        Returns:
            Stored frozendict. A stored dict raises ValueError rather than being converted.
        """
        loc = get_loc(key, index)
        return self._unpack(
            self._get_bytes(loc, "read_as_frozendict", max_bytes), frozendict, allow_custom
        )

    def read_as_ndarray[T: np.generic](
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        dtype: type[T] | None = None,
        *,
        casting: str = "same_kind",
        writable: bool = False,
        max_bytes: int | None = None,
    ) -> NDArray[T]:
        """Restore a NumPy array stored by ``write_ndarray``.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the array.
            dtype: Output dtype; None retains the stored dtype.
            casting: NumPy casting rule for ``dtype``; the default ``"same_kind"``
                refuses lossy kinds such as float to int (pass ``"unsafe"`` for them).
            writable: Return a writable array without a second copy, for in-place
                updates or ``torch.from_numpy``.
            max_bytes: Refuse bodies larger than this before copying; None: no limit.

        Returns:
            NumPy array with the stored shape.

        Note:
            Without a dtype conversion, the array references owned bytes copied
            from LMDB and is read-only unless ``writable=True``. A conversion returns
            a new writable array. Unavailable extension dtypes fall back to void.
        """
        if writable is not False and writable is not True:
            raise TypeError("writable must be bool")
        if casting != "same_kind" and casting not in _CASTINGS:
            raise ValueError(f"casting must be one of {sorted(_CASTINGS)}")
        _validate_max_bytes(max_bytes)
        resolved = get_loc(key, index)
        _, _, value = self._owned_entries(
            [resolved], "read_as_ndarray", max_bytes=max_bytes, mutable=writable
        )[0]
        array = value.array() if isinstance(value, _ArrayRows) else _codec.bytes_to_ndarray(value)
        if dtype is not None:
            array = _converted(array, dtype, casting)
        return _writable(array) if writable else array

    def read_as_ndarray_many[T: np.generic](
        self,
        keys: Sequence[str | Path | tuple[str, ...]],
        *,
        indices: Sequence[int | None] | None = None,
        dtype: type[T] | None = None,
        casting: str = "same_kind",
        writable: bool = False,
        max_items: int = 1024,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> list[NDArray[T]]:
        """Read NumPy arrays (plain or chunked) in one short transaction.

        Args:
            keys: Array keys; input order and duplicates are preserved.
            indices: Index for each key; None means every key has no index.
            dtype: Output dtype for every array; None retains the stored dtypes.
            casting: NumPy casting rule for ``dtype``; see ``read_as_ndarray``.
            writable: Return writable arrays; see ``read_as_ndarray``.
            max_items: Item limit, 1 to 1,048,576 (2**20).
            max_bytes: Aggregate body/integrity metadata copy budget, 64 MiB by
                default; None removes it.

        Returns:
            Arrays in key order, as ``read_as_ndarray`` returns them. Empty input
            opens no environment.
        """
        if type(writable) is not bool:
            raise TypeError("writable must be bool")
        if casting not in _CASTINGS:
            raise ValueError(f"casting must be one of {sorted(_CASTINGS)}")
        locations = self._batch_locations(keys, indices, max_items, max_bytes)
        if not locations:
            return []
        owned = self._owned_entries(
            locations, "read_as_ndarray", max_bytes=max_bytes, mutable=writable
        )
        arrays = []
        for _, _, value in owned:
            if isinstance(value, _ArrayRows):
                array = value.array()
            else:
                array = _codec.bytes_to_ndarray(value)  # a bytearray body stays writable
            array = _converted(array, dtype, casting)
            arrays.append(_writable(array) if writable else array)
        return arrays

    def read_as_str_many(
        self,
        keys: Sequence[str | Path | tuple[str, ...]],
        *,
        indices: Sequence[int | None] | None = None,
        max_items: int = 1024,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> list[str]:
        """Read strings stored by ``write_str`` in one short transaction.

        Args:
            keys: Stored keys; input order and duplicates are preserved.
            indices: Index for each key; None means every key has no index.
            max_items: Item limit, 1 to 1,048,576 (2**20).
            max_bytes: Aggregate body/integrity metadata copy budget, 64 MiB by
                default; None removes it.

        Returns:
            Decoded strings in key order. Empty input opens no environment.
        """
        values = self._get_many_bytes(keys, indices, max_items, "read_as_str", max_bytes)
        return [_codec.bytes_to_str(value) for value in values]

    def read_as_tensor(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
        dtype: torch.dtype | None = None,
        *,
        max_bytes: int | None = None,
    ) -> Tensor:
        """Restore a CPU tensor from a stored tensor or NumPy array (plain or chunked).

        Args:
            key: Logical storage key.
            index: Optional index used when storing the tensor or array.
            dtype: Output dtype; None retains the stored dtype.
            max_bytes: Refuse bodies larger than this before copying; None: no limit.

        Returns:
            Writable CPU tensor with the stored shape that owns its memory (the bytes
            copied from LMDB, without a second copy). Requires PyTorch.

        Raises:
            TypeError: A stored array's dtype has no PyTorch equivalent.

        Note:
            The original device and gradient history are not restored.
        """
        _validate_max_bytes(max_bytes)
        resolved = get_loc(key, index)
        _, metadata, value = self._owned_entries(
            [resolved], "read_as_tensor", max_bytes=max_bytes, mutable=True
        )[0]
        if metadata.writer_tag == "write_tensor":
            result = _tensor(_plain(value))
        else:
            from .tensors import as_tensor

            array = (
                value.array() if isinstance(value, _ArrayRows) else _codec.bytes_to_ndarray(value)
            )
            result = as_tensor(_writable(array))
        return result if dtype is None else result.to(dtype=dtype)

    def read_as_shape(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
    ) -> tuple[int, ...]:
        """Read a stored array or tensor shape without constructing its values.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the array or tensor.

        Returns:
            Stored dimensions.

        Note:
            Inspect the header through a transaction-local view without copying
            the payload or constructing an array or tensor.
        """
        return self.read_array_headers_many([key], indices=[index])[0].shape

    def read_as_dtype(
        self,
        key: str | Path | tuple[str, ...],
        index: int | None = None,
    ) -> Any:
        """Read a stored array or tensor dtype without constructing its values.

        Args:
            key: Logical storage key.
            index: Optional index used when storing the array or tensor.

        Returns:
            NumPy dtype for an array or PyTorch dtype for a tensor.

        Note:
            Inspect only the header through a transaction-local view. Resolving
            a tensor dtype requires PyTorch; ``inspect_store`` can instead show
            its dtype name without importing PyTorch.
        """
        tag, header, size = self._array_header_entries(
            [get_loc(key, index)], 64 * 1024, DEFAULT_MAX_BYTES
        )[0]
        info = parse_array_header(header, stored_bytes=size)
        _check_array_tag(tag, info)
        return _codec.bytes_to_dtype(header)


class LMDBReader(TypedReader):
    """Lazily open a published generation and restore its typed Python values.

    Readers in the same process share one environment per opened generation.
    The first opener's tuning applies until all references close. Each read closes
    its transaction; retaining a reader pins its
    generation but does not keep a long-lived read transaction open.

    Args:
        path: Published LMDB dataset directory to read.
        map_size: Memory-map size limit in bytes for a newly opened environment.
        readahead: Whether to enable OS readahead for a newly opened environment.
        verify_checksums: Verify each body's integrity digest before decoding
            (default). False skips only that digest for this reader, for stores whose
            operator guarantees they are undamaged and unaltered; writer tags, stored
            lengths and decoder validation still apply. ``verify_store``, ``doctor``
            and inspection keep verifying; header-only reads never hash bodies.
        value_limits: Lower decoding limits for MessagePack values (``ValueLimits``),
            for example for stores from untrusted sources; None keeps the built-in
            limits. Arrays are unaffected.
        verify_threads: Verify digests of bodies and chunks of at least 64 KiB in a
            pool of this many threads when a read has two or more of them; hashlib
            releases the GIL, so large batches verify several times faster on idle
            cores. 0 (default) verifies in the calling thread, which suits data-loader
            workers that already use every core. The pool's threads live until
            ``close()``; close the reader before forking (spawn is preferred), though
            a forked child starts its own pool.
        generation: Read this generation (``v-<32 hex>``) instead of the one ``CURRENT``
            selects, for example to compare or roll back; it stays pinned across
            ``refresh()``.
        cooperative: False reads stores on media nothing can change (read-only
            mounts, snapshots, archives) without coordination or LMDB lock files.
            Never use it while a writer, ``set_current`` or deletion may run on the
            store: nothing then keeps a cleanup from removing the open generation.

    Examples:
        Create and read a temporary dataset, then remove it at the end.

        >>> from contextlib import redirect_stdout
        >>> from io import StringIO
        >>> from pathlib import Path
        >>> from tempfile import TemporaryDirectory
        >>> import numpy as np
        >>> output = StringIO()
        >>> with TemporaryDirectory() as tmp, redirect_stdout(output):
        ...     path = Path(tmp) / "dataset.lmdb"
        ...     with LMDBWriter(path) as writer:
        ...         _ = writer.write_ndarray(np.arange(4), key="features", index=0)
        ...         writer.commit()
        ...     with LMDBReader(path) as reader:
        ...         arr = reader.read_as_ndarray("features", index=0)
        ...     values = arr.tolist()
        >>> values
        [0, 1, 2, 3]

    Note:
        The first read opens (and pins) the generation ``CURRENT`` selects at that
        moment; later publications do not affect this reader until ``refresh()``,
        which releases it so the next read follows ``CURRENT`` again. ``close()``
        releases it for good; a closed reader raises ``ReaderClosedError``, and a
        reader dropped without closing releases its reference when collected.

        Pickling (for example into ``DataLoader`` workers) carries the pinned
        generation, so every copy reads the same data as the original; an unopened
        reader carries none and each copy pins on its own first read, so call
        ``get_snapshot()`` before pickling to give all copies one generation. Copies
        reopen their own environments. One reader may be shared by threads.

        Custom decoders run only when reading with ``allow_custom=True`` and must
        validate external input. Missing keys raise ``KeyNotFoundError`` (a
        ``KeyError``); missing or incompatible writer tags raise
        ``ReaderMismatchError`` (a ``ValueError``).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        map_size: int = 512 * 1024**3,
        readahead: bool = False,
        verify_checksums: bool = True,
        value_limits: ValueLimits | None = None,
        verify_threads: int = 0,
        generation: str | None = None,
        cooperative: bool = True,
    ):
        if type(map_size) is not int or map_size <= 0:
            raise ValueError("map_size must be a positive integer number of bytes")
        if type(readahead) is not bool:
            raise TypeError("readahead must be bool")
        if type(verify_checksums) is not bool:
            raise TypeError("verify_checksums must be bool")
        if value_limits is not None and not isinstance(value_limits, ValueLimits):
            raise TypeError("value_limits must be a ValueLimits instance or None")
        if type(verify_threads) is not int or verify_threads < 0:
            raise ValueError("verify_threads must be a nonnegative integer")
        if generation is not None and (
            type(generation) is not str or not _is_version_name(generation)
        ):
            raise ValueError("generation must be a generation name such as v-<32 hex>")
        if type(cooperative) is not bool:
            raise TypeError("cooperative must be bool")
        self._path = Path(path)
        self._map_size = map_size
        self._readahead = readahead
        self._verify_checksums = verify_checksums
        self._value_limits = value_limits
        self._verify_threads = verify_threads
        self._generation = generation  # explicit: kept across refresh()
        self._pinned: str | None = None  # carried over by pickling
        self._cooperative = cooperative
        self._closed = False
        self._lock = _threading.RLock()
        self._finalizer: weakref.finalize[Any, Any] | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._env: Environment | None = None
        self._env_key: _EnvKey | None = None
        self._metadata_db: Any = None
        self._pid = os.getpid()

    @property
    def path(self) -> Path:
        """Return the dataset path supplied when this reader was constructed.

        Returns:
            Configured path, without replacing it with the resolved generation path.
        """
        return self._path

    @property
    def closed(self) -> bool:
        """Return whether ``close()`` was called; a closed reader cannot read again."""
        return self._closed

    @property
    def generation(self) -> str | None:
        """Return the generation this reader is pinned to, if any.

        That is the explicit ``generation=`` argument, the generation carried over by
        pickling, or the generation opened by the first read, in this order.
        """
        if self._generation is not None:
            return self._generation
        if self._env_key is not None:
            return Path(self._env_key[1]).name
        return self._pinned

    @property
    def cooperative(self) -> bool:
        """Return whether this reader takes cooperative locks (False: read-only media)."""
        return self._cooperative

    @property
    def verify_checksums(self) -> bool:
        """Return whether body reads verify integrity digests before decoding."""
        return self._verify_checksums

    @property
    def value_limits(self) -> ValueLimits | None:
        """Return the lowered MessagePack decoding limits, or None for the built-in ones."""
        return self._value_limits

    @property
    def verify_threads(self) -> int:
        """Return the digest verification thread count; 0 verifies in the calling thread."""
        return self._verify_threads

    def __repr__(self) -> str:
        state = "closed" if self._closed else ("open" if self._env is not None else "unopened")
        return f"LMDBReader({str(self._path)!r}, generation={self.generation!r}, {state})"

    def _verify(self, checks: Sequence[tuple[bytes, EntryMetadata, Any]]) -> None:
        """Verify digests in input order; large bodies may run in the thread pool.

        The first failing entry in input order raises, as in a sequential check.
        """
        if not self._verify_checksums:
            return
        if not self._verify_threads or len(checks) < 2:
            for location, metadata, value in checks:
                _verify_checksum(location, metadata, value)
            return
        large = [len(value) >= _PARALLEL_DIGEST_BYTES for _, _, value in checks]
        if sum(large) < 2:
            for location, metadata, value in checks:
                _verify_checksum(location, metadata, value)
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._verify_threads, thread_name_prefix="amber-lmdb-verify"
            )
        pool = self._pool
        futures = {
            position: pool.submit(_verify_checksum, *checks[position])
            for position, parallel in enumerate(large)
            if parallel
        }
        try:
            for position, (location, metadata, value) in enumerate(checks):
                future = futures.get(position)
                if future is None:
                    _verify_checksum(location, metadata, value)
                else:
                    future.result()
        finally:
            for future in futures.values():
                future.cancel()

    def _get_env(self) -> Environment:
        """Lazily acquire the reader environment and open its writer-tag database.

        Thread-safe: concurrent first reads acquire one reference. After a PID change,
        inherited references are discarded and the current process acquires its own.
        A pinned generation (explicit or carried over by pickling) is opened instead of
        ``CURRENT``. Raises ReaderClosedError after ``close()``.
        """
        environment = self._env
        if environment is not None and self._pid == os.getpid():
            return environment
        with self._lock:
            if self._closed:
                raise ReaderClosedError(
                    "LMDBReader is closed; create a new reader, or use refresh() to release "
                    "a reader without closing it"
                )
            if self._pid != os.getpid():
                if self._finalizer is not None:
                    self._finalizer.detach()  # the parent's reference is not ours to release
                self._finalizer = None
                self._env = None
                self._env_key = None
                self._metadata_db = None
                self._pool = None  # An inherited pool has no threads in this process.
                self._pid = os.getpid()
            if self._env is None:
                key = _environment_key(
                    self._path,
                    readonly=True,
                    readahead=self._readahead,
                    max_readers=512,
                    map_size=self._map_size,
                    generation=self._generation or self._pinned,
                    cooperative=self._cooperative,
                )
                env = _acquire_env(key, validated=True)
                try:
                    self._metadata_db = env.open_db(_METADATA_DB_NAME, create=False)
                except Exception:
                    _release_env(key)
                    raise
                self._env_key = key
                # A dropped reader releases its reference (and generation lease).
                self._finalizer = weakref.finalize(self, _release_in_process, key, self._pid)
                self._env = env
            return self._env

    def _owned_entries(
        self,
        locations: Sequence[bytes],
        read_method: str,
        *,
        max_bytes: int | None,
        partial: bool = False,
        snapshot: LMDBSnapshot | None = None,
        mutable: bool = False,
    ) -> list[tuple[bytes, EntryMetadata, _Body]]:
        """Check aggregate sizes before any body copy; verify digests after closure.

        Digest verification is skipped only for a reader built with
        ``verify_checksums=False``; tag and length checks always run. Tensor bodies
        (every body with ``mutable``) are copied into bytearrays so tensors and
        writable arrays can use them without a second copy; chunked arrays are
        assembled from their chunks in the same transaction; every other body is
        owned bytes.
        """
        views: list[tuple[bytes, EntryMetadata, bytes | memoryview | _ArrayRows]] = []
        used = 0
        has_rows = False
        with self._get_env().begin(write=False, buffers=True) as transaction:
            if snapshot is not None:
                self._check_snapshot(transaction, snapshot, "owned read pages")
            for location in locations:
                try:
                    metadata, value = self._get_checked_entry(
                        transaction,
                        location,
                        read_method,
                        used_bytes=used,
                        max_bytes=max_bytes,
                    )
                    if metadata.writer_tag == _chunks.CHUNKED_TAG:
                        value = self._copy_rows(
                            transaction, location, metadata, value, None, used, max_bytes
                        )
                        has_rows = True
                except ReadLimitError:
                    if partial and views:
                        break
                    raise
                used += len(value) + metadata.metadata_bytes
                views.append((location, metadata, value))
            owned: list[tuple[bytes, EntryMetadata, _Body]] = [
                (
                    location,
                    metadata,
                    value
                    if isinstance(value, _ArrayRows)
                    else bytearray(value)
                    if mutable or metadata.writer_tag == "write_tensor"
                    else bytes(value),
                )
                for location, metadata, value in views
            ]
        if not self._verify_checksums:
            return owned
        if has_rows:
            checks: list[tuple[bytes, EntryMetadata, Any]] = []
            for location, metadata, value in owned:
                if isinstance(value, _ArrayRows):
                    checks.extend(_row_checks(location, metadata, value))
                else:
                    checks.append((location, metadata, value))
            self._verify(checks)
        elif self._verify_threads:
            self._verify(owned)
        else:  # The common case, without a method call per read.
            for location, metadata, value in owned:
                _verify_checksum(location, metadata, cast(bytes | bytearray, value))
        return owned

    def _copy_rows(
        self,
        transaction: Transaction,
        location: bytes,
        metadata: EntryMetadata,
        value: bytes | memoryview,
        rows: tuple[int, int] | None,
        used: int,
        max_bytes: int | None,
    ) -> _ArrayRows:
        """Copy the rows ``[start, stop)`` (None: all) of an array inside ``transaction``.

        A chunked array copies only the chunks those rows touch, after checking each
        chunk's envelope against its header; an unchunked array copies its whole body
        when digests are verified, otherwise only the requested rows. The header is
        parsed here only to plan the copy. Digests are verified after the transaction.
        """
        start, stop = (0, _MAX_INDEX) if rows is None else rows
        if metadata.writer_tag == _chunks.CHUNKED_TAG:
            layout = _chunks._layout(value)
            header = bytes(value)
            if layout.rows is None:  # No chunks: nothing to copy, rows are clamped later.
                first, numbers = 0, range(0)
            else:
                start = min(start, layout.rows)
                stop = min(max(stop, start), layout.rows)
                first = start // layout.chunk_rows
                last = min(layout.count, -(-stop // layout.chunk_rows))
                numbers = range(first, last) if stop > start else range(0)
            needed = sum(layout.stored_size(number) for number in numbers)
            envelopes = len(numbers) * _chunks._CHUNK_ENVELOPE_BYTES
            required = used + len(header) + metadata.metadata_bytes + needed + envelopes
            if max_bytes is not None and required > max_bytes:
                raise ReadLimitError(
                    kind="bytes", limit=max_bytes, required=required, location=location
                )
            data = np.empty(needed, dtype=np.uint8)
            target = memoryview(data)
            checks: list[tuple[bytes, EntryMetadata, int, int]] = []
            offset = 0
            for number in numbers:
                chunk = _chunk_location(location, number)
                raw = transaction.get(chunk, db=self._metadata_db)
                body = transaction.get(chunk)
                if raw is None or body is None:
                    from .integrity import IntegrityError

                    raise IntegrityError(
                        f"chunk {number} of {describe_location(location)} is missing"
                    )
                chunk_metadata = _parse_metadata(raw)
                size = layout.stored_size(number)
                if (
                    chunk_metadata.writer_tag != _chunks.CHUNK_TAG
                    or chunk_metadata.size_bytes != size
                    or len(body) != size
                    or chunk_metadata.digest != layout.digest(number)
                ):
                    from .integrity import IntegrityError

                    raise IntegrityError(
                        f"chunk {number} of {describe_location(location)} does not match its header"
                    )
                target[offset : offset + size] = body
                checks.append((chunk, chunk_metadata, offset, offset + size))
                offset += size
            return _ArrayRows(
                header=header,
                chunked=True,
                data=data,
                # Without copied chunks the requested (empty) range starts the data.
                first_row=first * layout.chunk_rows if numbers else start,
                start=start,
                stop=stop,
                copied=len(header) + needed + envelopes,
                checks=checks,
                whole=False,
            )
        header = bytes(_header_view(value, 64 * 1024))
        info = _chunks._plain_info(header)
        if info.compression is None and len(value) != len(header) + info.shape[0] * info.row_bytes:
            raise ValueError("array shape and payload size do not match")
        start = min(start, info.shape[0])
        stop = min(max(stop, start), info.shape[0])
        # A compressed body decompresses only as a whole.
        if self._verify_checksums or info.compression is not None:
            required = used + metadata.metadata_bytes + len(value)
            if max_bytes is not None and required > max_bytes:
                raise ReadLimitError(
                    kind="bytes", limit=max_bytes, required=required, location=location
                )
            return _ArrayRows(
                header=header,
                chunked=False,
                data=bytes(value),
                first_row=0,
                start=start,
                stop=stop,
                copied=len(value),
                checks=[],
                whole=True,
            )
        begin = len(header) + start * info.row_bytes
        end = len(header) + stop * info.row_bytes
        required = used + metadata.metadata_bytes + len(header) + end - begin
        if max_bytes is not None and required > max_bytes:
            raise ReadLimitError(
                kind="bytes", limit=max_bytes, required=required, location=location
            )
        return _ArrayRows(
            header=header,
            chunked=False,
            data=bytes(value[begin:end]),
            first_row=start,
            start=start,
            stop=stop,
            copied=len(header) + end - begin,
            checks=[],
            whole=False,
        )

    def _owned_rows(
        self,
        locations: Sequence[bytes],
        ranges: Sequence[tuple[int, int]],
        *,
        max_bytes: int | None,
    ) -> list[tuple[EntryMetadata, _ArrayRows]]:
        """Copy row ranges of several arrays in one short transaction, then verify them."""
        entries: list[tuple[bytes, EntryMetadata, _ArrayRows]] = []
        used = 0
        with self._get_env().begin(write=False, buffers=True) as transaction:
            for location, rows in zip(locations, ranges, strict=True):
                metadata, value = self._get_checked_entry(
                    transaction,
                    location,
                    "read_array_rows",
                    used_bytes=used,
                    max_bytes=max_bytes,
                    header_limit=64 * 1024,
                )
                copied = self._copy_rows(
                    transaction, location, metadata, value, rows, used, max_bytes
                )
                used += len(copied) + metadata.metadata_bytes
                entries.append((location, metadata, copied))
        checks: list[tuple[bytes, EntryMetadata, Any]] = []
        for location, metadata, copied in entries:
            checks.extend(_row_checks(location, metadata, copied))
        self._verify(checks)
        return [(metadata, copied) for _, metadata, copied in entries]

    def _get_checked_entry(
        self,
        transaction: Transaction,
        loc: bytes,
        read_method: str,
        *,
        used_bytes: int = 0,
        max_bytes: int | None = None,
        header_limit: int | None = None,
    ) -> tuple[EntryMetadata, bytes | memoryview]:
        """Fetch a value and validate its tag within a caller-owned transaction.

        Args:
            transaction: Read transaction whose lifetime the caller manages.
            loc: Binary key of the value to check.
            read_method: Requested reader method.

        Returns:
            Writer tag and value. With ``buffers=True``, the view is valid only
            within the transaction.

        Raises:
            KeyError: The key does not exist.
            ValueError: The tag is missing, unknown or incompatible with the reader.
        """
        raw = transaction.get(loc, db=self._metadata_db)
        if raw is None:
            if transaction.get(loc) is None:
                raise _key_not_found(loc)
            raise ValueError(f"LMDB value has no writer tag: {describe_location(loc)}")
        value = transaction.get(loc)
        if value is None:
            raise _key_not_found(loc)
        copied_size = len(value) if header_limit is None else len(_header_view(value, header_limit))
        required = used_bytes + len(raw) + copied_size
        if max_bytes is not None and required > max_bytes:
            raise ReadLimitError(kind="bytes", limit=max_bytes, required=required, location=loc)
        metadata = _parse_metadata(raw)
        if len(value) != metadata.size_bytes:
            from .integrity import IntegrityError

            raise IntegrityError("encoded value length does not match integrity metadata")
        writer_tag = metadata.writer_tag
        valid_readers = _VALID_READERS.get(writer_tag, frozenset())
        generic = read_method == "read" and writer_tag != _chunks.CHUNK_TAG
        if not valid_readers or (not generic and read_method not in valid_readers):
            expected = tuple(sorted(valid_readers)) if valid_readers else ()
            hint = ", ".join(expected) if expected else "(unknown writer tag)"
            raise ReaderMismatchError(
                f"Invalid write/read method pair for {describe_location(loc)}: stored by "
                f"{writer_tag}(), so {read_method}() cannot read it; use {hint}, or read() "
                f"for any value",
                location=loc,
                writer_tag=writer_tag,
                read_method=read_method,
                valid_readers=expected,
            )
        return metadata, value

    def _array_header_entries(
        self,
        locations: Sequence[bytes],
        max_header_bytes: int,
        max_bytes: int,
    ) -> list[tuple[str, bytes, int]]:
        return self._header_entries_used(locations, max_header_bytes, max_bytes)[0]

    def _header_entries_used(
        self,
        locations: Sequence[bytes],
        max_header_bytes: int,
        max_bytes: int,
    ) -> tuple[list[tuple[str, bytes, int]], int]:
        """Return owned header entries and the header/metadata bytes they used."""
        used = 0
        with self._get_env().begin(write=False, buffers=True) as transaction:
            views = []
            for location in locations:
                metadata, value = self._get_checked_entry(
                    transaction,
                    location,
                    "read_as_shape",
                    used_bytes=used,
                    max_bytes=max_bytes,
                    header_limit=max_header_bytes,
                )
                header = _header_view(value, max_header_bytes)
                used += len(header) + metadata.metadata_bytes
                views.append((metadata.writer_tag, header, len(value)))
            return [(tag, bytes(header), size) for tag, header, size in views], used

    def get_lib_version(self) -> tuple[int, int, int]:
        """Return the LMDB engine version backing the actually opened generation.

        Returns:
            Native engine major, minor and patch version, independently of the
            py-lmdb package version. Opening existing files auto-detects the engine.
        """
        return self._get_env().lib_version()

    def get_snapshot(self) -> LMDBSnapshot:
        """Return the actually opened generation identity for external index validation.

        Lazily open on first use. An already-open reader does not resolve
        ``CURRENT`` again; after ``refresh()``, the next call opens the newly
        selected generation. Fetch no feature values and start no explicit read
        transaction. This identity is not a retained transaction snapshot.

        Returns:
            Immutable root, opened generation and last committed transaction ID.

        Raises:
            RuntimeError: The opened environment has no recorded generation identity.
        """
        environment = self._get_env()
        key = self._env_key
        if key is None:
            raise RuntimeError("LMDBReader has no opened generation")
        return LMDBSnapshot(key[0], key[1], environment.info()["last_txnid"])

    def manifest(self) -> StoreManifest | None:
        """Return the writer manifest of the opened generation, or None without one.

        The manifest is an ordinary dictionary entry under a reserved key, read and
        verified like any body (subject to ``verify_checksums``).
        """
        from .manifest import _manifest_from_dict

        try:
            raw = self.read_as_dict(_MANIFEST_KEY)
        except KeyError:
            return None
        return _manifest_from_dict(raw)

    def _check_snapshot(self, transaction: Transaction, snapshot: LMDBSnapshot, what: str) -> None:
        """Raise SourceChangedError unless the transaction still sees ``snapshot``."""
        if (
            self._env_key is None
            or self._env_key[:2] != (snapshot.root, snapshot.generation)
            or transaction.id() != snapshot.last_txnid
        ):
            from .collections import SourceChangedError

            raise SourceChangedError(f"store changed between {what}")

    def contains(self, key: str | Path | tuple[str, ...], index: int | None = None) -> bool:
        """Return whether a typed entry exists, without copying or hashing its body.

        Only the entry's integrity metadata is looked up. A body without metadata
        counts as absent here; reading it reports that corruption.
        """
        return self.contains_many([key], indices=[index])[0]

    def contains_many(
        self,
        keys: Sequence[str | Path | tuple[str, ...]],
        *,
        indices: Sequence[int | None] | None = None,
        max_items: int = 1024,
    ) -> list[bool]:
        """Check several keys in one short transaction; order and duplicates are kept.

        Empty input opens no environment. Like ``contains``, no body is read.
        """
        locations = self._batch_locations(keys, indices, max_items, DEFAULT_MAX_BYTES)
        return self._present(locations) if locations else []

    def _present(self, locations: Sequence[bytes]) -> list[bool]:
        """Look up integrity metadata for encoded locations in one short transaction."""
        with self._get_env().begin(write=False, buffers=True) as transaction:
            return [
                transaction.get(location, db=self._metadata_db) is not None
                for location in locations
            ]

    def _iter_locations(self, page_size: int = 4096) -> Iterator[list[bytes]]:
        """Yield every encoded location with integrity metadata, in pages and key order."""
        snapshot = self.get_snapshot()
        position: bytes | None = None
        while True:
            page: list[bytes] = []
            with (
                self._get_env().begin(write=False, buffers=True) as transaction,
                transaction.cursor(db=self._metadata_db) as cursor,
            ):
                self._check_snapshot(transaction, snapshot, "location pages")
                found = cursor.first() if position is None else cursor.set_range(position)
                while found and len(page) < page_size:
                    page.append(bytes(cursor.key()))
                    found = cursor.next()
            if not page:
                return
            yield page
            if not found:
                return
            position = page[-1] + b"\x00"

    def index_range(self, key: str | Path | tuple[str, ...]) -> tuple[int, int] | None:
        """Return the smallest and largest stored index of a key with two seeks.

        Returns None when the key has no indexed entries. Gaps are not detected;
        use ``count`` or ``iter_indices`` for that.
        """
        prefix = _key_prefix(key)
        with (
            self._get_env().begin(write=False, buffers=True) as transaction,
            transaction.cursor(db=self._metadata_db) as cursor,
        ):
            first = _boundary_index(cursor, prefix, last=False)
            if first is None:
                return None
            last = _boundary_index(cursor, prefix, last=True)
        if last is None:
            raise RuntimeError("index range lost its last entry")
        return first, last

    def count(self, key: str | Path | tuple[str, ...]) -> int:
        """Count the indexed entries of a key; the unindexed entry is not included.

        This walks the key's index range in bounded pages without reading bodies,
        so it takes time proportional to the count.
        """
        return sum(len(page) for page in self.iter_indices(key, page_size=65536))

    def iter_indices(
        self,
        key: str | Path | tuple[str, ...],
        *,
        start: int = 0,
        page_size: int = 4096,
    ) -> Iterator[list[int]]:
        """Yield the stored indices of a key in ascending pages, starting at ``start``.

        Each page uses its own short transaction; the opened generation must not
        change between pages (``SourceChangedError``). No body is read.
        """
        _validate_page_size(page_size)
        if type(start) is not int or not 0 <= start <= _MAX_INDEX:
            raise ValueError("start must be an integer index in [0, 2**64)")
        prefix = _key_prefix(key)
        low = prefix + b"\x01"
        position = low + struct.pack(">Q", start)
        snapshot = self.get_snapshot()
        while True:
            page: list[int] = []
            with (
                self._get_env().begin(write=False, buffers=True) as transaction,
                transaction.cursor(db=self._metadata_db) as cursor,
            ):
                self._check_snapshot(transaction, snapshot, "index pages")
                found = cursor.set_range(position)
                while found and len(page) < page_size:
                    index = _index_of(bytes(cursor.key()), low)
                    if index is None:
                        break
                    page.append(index)
                    found = cursor.next()
            if not page:
                return
            yield page
            if len(page) < page_size or page[-1] == _MAX_INDEX:
                return
            position = low + struct.pack(">Q", page[-1] + 1)

    def iter_keys(
        self, *, prefix: tuple[str, ...] = (), page_size: int = 1024
    ) -> Iterator[list[KeyGroup]]:
        """Yield the logical keys of the opened generation in encoded order, in pages.

        Encoded order lists string keys first, then tuple keys by component count;
        each component compares by UTF-8 byte length before content.

        Args:
            prefix: Leading components of tuple keys to select. The default lists
                every key, string keys first; a nonempty prefix lists only tuple keys.
            page_size: Maximum groups per page.

        Each page uses its own short transaction; the opened generation must not
        change between pages (``SourceChangedError``). Bodies are not read, and the
        cost grows with the number of logical keys rather than entries. The
        reserved ``"amber-lmdb"`` namespace (manifest, array chunks) is not listed.
        """
        if type(prefix) is not tuple or not all(isinstance(part, str) for part in prefix):
            raise TypeError("prefix must be a tuple of strings")
        _validate_page_size(page_size)
        tuple_keys = _KEY_MAGIC + b"\x01"
        encoded = b"".join(
            struct.pack(">Q", len(raw)) + raw for raw in (part.encode("utf-8") for part in prefix)
        )

        def region(count: int) -> bytes:
            return tuple_keys + struct.pack(">I", count) + encoded

        position = region(len(prefix)) if prefix else None
        snapshot = self.get_snapshot()
        exhausted = False
        while not exhausted:
            groups: list[KeyGroup] = []
            with (
                self._get_env().begin(write=False, buffers=True) as transaction,
                transaction.cursor(db=self._metadata_db) as cursor,
            ):
                self._check_snapshot(transaction, snapshot, "key pages")
                found = cursor.first() if position is None else cursor.set_range(position)
                while found and len(groups) < page_size:
                    location = bytes(cursor.key())
                    if prefix:
                        # The component count precedes the components, so each count
                        # holds its own contiguous region for the prefix.
                        if (
                            not location.startswith(tuple_keys)
                            or len(location) < len(tuple_keys) + 4
                        ):
                            found = False
                            break
                        count = struct.unpack_from(">I", location, len(tuple_keys))[0]
                        wanted = region(count)
                        if not location.startswith(wanted):
                            if location < wanted:
                                found = cursor.set_range(wanted)
                            elif count == 0xFFFFFFFF:
                                found = False
                            else:
                                found = cursor.set_range(tuple_keys + struct.pack(">I", count + 1))
                            continue
                    if _is_reserved(location):
                        # Skip this component count's reserved namespace in one seek;
                        # component lengths stay far below 2**56, so they start with 0.
                        found = cursor.set_range(location[:10] + _RESERVED_MARK + b"\x01")
                        continue
                    key, index = decode_key(location)
                    group = location[: len(location) - (1 if index is None else 9)]
                    unindexed = (
                        index is None
                        or transaction.get(group + b"\x00", db=self._metadata_db) is not None
                    )
                    first = (
                        index if index is not None else _boundary_index(cursor, group, last=False)
                    )
                    last = _boundary_index(cursor, group, last=True) if first is not None else None
                    groups.append(KeyGroup(key, unindexed, first, last))
                    position = group + b"\x02"
                    found = cursor.set_range(position)
                exhausted = not found
            if groups:
                yield groups

    def close(self) -> None:
        """Release this reader's environment reference and close it for good.

        Repeated calls are harmless. Later reads raise ReaderClosedError; use
        ``refresh()`` to release a reader that should read again. A reader that is
        dropped without ``close()`` releases its reference when it is collected.
        """
        with self._lock:
            self._release()
            self._closed = True

    def refresh(self) -> None:
        """Release the pinned generation so the next read opens ``CURRENT`` again.

        An explicit ``generation=`` stays pinned; a generation carried over by pickling
        is dropped. The reader stays usable.
        """
        with self._lock:
            self._release()
            self._pinned = None

    def _release(self) -> None:
        """Shut down the verification pool and release the environment reference."""
        pool, self._pool = self._pool, None
        if pool is not None and self._pid == os.getpid():
            pool.shutdown(wait=True)
        finalizer = self._finalizer
        key = self._env_key
        # Detach first; only the process that acquired the reference releases it.
        if (
            finalizer is not None
            and finalizer.detach() is not None
            and key is not None
            and self._pid == os.getpid()
        ):
            try:
                _release_env(key)
            except BaseException:
                # A failed native close keeps the reference; close() can retry.
                self._finalizer = weakref.finalize(self, _release_in_process, key, self._pid)
                raise
        self._finalizer = None
        self._env = None
        self._env_key = None
        self._metadata_db = None

    def __enter__(self) -> LMDBReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.close()
        return False

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # A copy reads the generation this reader opened, so worker processes stay
        # consistent with the process that created them.
        if self._generation is None and self._env_key is not None:
            state["_pinned"] = Path(self._env_key[1]).name
        state["_env"] = None
        state["_env_key"] = None
        state["_metadata_db"] = None
        state["_pool"] = None
        state["_finalizer"] = None
        state["_lock"] = None
        state["_pid"] = os.getpid()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = _threading.RLock()


def _release_in_process(key: _EnvKey, pid: int) -> None:
    """Release a reader's environment reference, only in the process that acquired it."""
    if os.getpid() == pid:
        _release_env(key)


__all__ = [
    "EnvironmentHandle",
    "KeyGroup",
    "LMDBReader",
    "LMDBSnapshot",
    "LMDBWriter",
    "TypedReader",
    "decode_key",
    "get_active_envs",
    "get_loc",
    "open_environment",
]
