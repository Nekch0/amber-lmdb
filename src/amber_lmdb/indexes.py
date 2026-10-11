"""Caller-identified derived caches with OS locks and checksummed atomic publication."""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import math
import os
import re
import stat
import struct
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from .progress import CancelEvent, OperationCancelled

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

type Identity = Mapping[str, Any] | DataclassInstance

_DIGEST = re.compile(r"[0-9a-f]{64}")
_DEFAULT_STRIPES = 64
_STRIPES_NAME = "STRIPES"
# Stripe lock files held by the current thread, so nested acquisitions reenter.
_HELD_STRIPES = threading.local()


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _canonical(value: object, where: str, *, tuples: bool) -> Any:
    """Return ``value`` as JSON types, refusing what JSON would change silently."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _canonical(getattr(value, item.name), f"{where}.{item.name}", tuples=True)
            for item in dataclasses.fields(value)
        }
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where} must be a finite number")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in cast(Mapping[object, object], value).items():
            if type(key) is not str:
                raise TypeError(
                    f"{where} has a {type(key).__name__} key; identity keys must be str "
                    "(JSON would turn them into strings)"
                )
            result[key] = _canonical(item, f"{where}[{key!r}]", tuples=tuples)
        return result
    if isinstance(value, list) or (tuples and isinstance(value, tuple)):
        items = cast(list[object] | tuple[object, ...], value)
        return [_canonical(item, f"{where}[{i}]", tuples=tuples) for i, item in enumerate(items)]
    if isinstance(value, tuple):
        raise TypeError(f"{where} is a tuple; use a list (JSON stores tuples as lists)")
    raise TypeError(
        f"{where} has unsupported type {type(value).__name__}; use None, bool, int, "
        "float, str, lists, str-keyed mappings or dataclass instances"
    )


def _canonical_identity(identity: object) -> dict[str, Any]:
    """Validate an identity and return its JSON form: a new, independent dict.

    Mappings need str keys and JSON values; tuples are refused because JSON would
    store them as lists. Dataclass instances (such as ``LMDBSnapshot``) are converted
    field by field, their tuple fields as lists.
    """
    if dataclasses.is_dataclass(identity) and not isinstance(identity, type):
        return _canonical(identity, "identity", tuples=True)
    if not isinstance(identity, Mapping):
        raise TypeError("identity must be a str-keyed mapping or a dataclass instance")
    return _canonical(identity, "identity", tuples=False)


def _read_stripes(record: Path) -> int | None:
    """Return the persisted stripe count, None if absent; wait out a concurrent creator."""
    for _ in range(200):
        try:
            text = record.read_text(encoding="ascii")
        except FileNotFoundError:
            return None
        if text.endswith("\n"):
            try:
                count = int(text)
            except ValueError:
                break
            if 1 <= count <= 4096:
                return count
            break
        time.sleep(0.005)  # created but not yet written by another builder
    raise ValueError(f"invalid cache lock stripe record: {record}")


def _create_stripes(record: Path, count: int) -> None:
    """Persist the stripe count unless another builder did it first."""
    try:
        descriptor = os.open(
            record, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
    except FileExistsError:
        return
    try:
        os.write(descriptor, f"{count}\n".encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _namespaces(directory: Path, suffix: str) -> tuple[str, ...]:
    """Return the namespaces of cache files with ``suffix`` in ``directory``, sorted."""
    found: set[str] = set()
    for path in directory.iterdir():
        name = path.name
        if not name.endswith(suffix) or len(name) < 64 + len(suffix):
            continue
        stem = name[: len(name) - len(suffix)]
        if not _DIGEST.fullmatch(stem[-64:]):
            continue
        prefix = stem[:-64]
        if not prefix:
            found.add("")
        elif prefix.endswith("-") and all(
            char.isascii() and (char.isalnum() or char in "._-") for char in prefix[:-1]
        ):
            found.add(prefix[:-1])
    return tuple(sorted(found))


def _check_seconds(value: object, name: str) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite, nonnegative, or None")


def _check_cancel(cancel: object) -> None:
    if cancel is not None and not callable(getattr(cancel, "is_set", None)):
        raise TypeError("cancel must provide is_set(), such as threading.Event")


@contextmanager
def index_lock(
    path: str | Path,
    *,
    timeout: float | None = None,
    cancel: CancelEvent | None = None,
) -> Iterator[None]:
    """Acquire a retained lock file; optionally stop waiting with TimeoutError.

    None waits indefinitely, including behind a live stuck builder. A set ``cancel``
    event stops waiting with ``OperationCancelled``. The lock inode is never unlinked.
    Process exit releases the OS lock. Normal local filesystem locking is assumed;
    this does not establish cross-host or NFS safety.
    """
    _check_seconds(timeout, "timeout")
    _check_cancel(cancel)
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"index lock must be a regular file: {path}")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "r+b") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("index lock must be a regular file")
        deadline = None if timeout is None else time.monotonic() + timeout
        polling = deadline is not None or cancel is not None
        if os.name == "nt":
            import msvcrt

            def acquire() -> None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)

            def release() -> None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire() -> None:
                flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if polling else 0)
                fcntl.flock(stream.fileno(), flags)

            def release() -> None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

        while True:
            if cancel is not None and cancel.is_set():
                raise OperationCancelled("index_lock")
            try:
                acquire()
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring index lock: {path}") from error
                time.sleep(0.05)
        try:
            if os.name == "nt" and os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"\x00")
                stream.flush()
            yield
        finally:
            release()


@dataclass(frozen=True, slots=True)
class CacheResult[T]:
    """Explicit cache hit/miss; value=None can be a successful hit.

    Check hit before using value. A miss always carries None; hit values follow
    the caller's decoder type, including None when that type permits it.
    """

    hit: bool
    value: T | None = None


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """One cache file as listed by ``DerivedIndexCache.entries``.

    Only the magic and the bounded header are read, never the payload or its
    checksum; ``valid`` therefore means a well-formed envelope of the right size,
    not a verified payload (``lookup`` verifies).

    Attributes:
        path: Cache file.
        identity: Identity stored in the header, or None when unreadable.
        size_bytes: File size.
        modified_ns: Modification time: publication, or the last refreshed hit
            with ``touch_on_hit``.
        valid: Whether magic, header and sizes are consistent.
    """

    path: Path
    identity: dict[str, Any] | None
    size_bytes: int
    modified_ns: int
    valid: bool


@dataclass(frozen=True, slots=True)
class PruneReport:
    """What ``DerivedIndexCache.prune`` removed, or would remove with ``dry_run``.

    Attributes:
        dry_run: Whether nothing was deleted.
        removed: Entries removed (or removable in a dry run), selection order.
        in_use: Selected entries skipped because a builder or reader held them.
        kept: Entries left in place, including skipped ones.
        freed_bytes: Bytes of ``removed``.
        reasons: Count of removed entries per reason: ``invalid``, ``predicate``,
            ``expired`` or ``capacity``.
        temporaries: Abandoned temporary files removed (or removable).
    """

    dry_run: bool
    removed: tuple[Path, ...]
    in_use: tuple[Path, ...]
    kept: int
    freed_bytes: int
    reasons: dict[str, int] = field(default_factory=lambda: dict[str, int]())
    temporaries: tuple[Path, ...] = ()


class DerivedIndexCache[T]:
    """Persist caller-built values using an explicit, JSON-compatible source identity.

    Identity must include the opened source snapshot, builder/schema/codec versions
    and all inputs affecting the result. It is a str-keyed mapping of JSON values
    (None, bool, int, finite float, str, lists, mappings) or a dataclass instance;
    dataclass values such as ``LMDBSnapshot`` are converted field by field. Tuples
    and non-str keys are refused (``TypeError``) because JSON would change them
    silently. This class knows nothing about datasets, Record fields, class maps or
    LMDB keys. Encoder, decoder and validator are supplied by the caller; do not use
    executable untrusted deserialization.

    The empty namespace uses hash-only filenames; nonempty namespaces prefix
    filenames. No files are written with directory=None.

    Builders coordinate through lock stripe files in ``<directory>/.locks``; an
    entry uses the stripe its identity digest selects, so builders of different
    identities rarely wait for each other and the number of lock files stays fixed.
    The first builder records the stripe count in ``.locks/STRIPES``; later caches
    on the directory use it, so every process maps an identity to the same lock.
    Lock files are never unlinked. ``prune`` removes entries under their stripe lock.

    Args:
        lock_stripes: Number of builder lock files, 1 to 4096. None (default) uses
            the directory's recorded count, or 64 for a new directory. A value that
            differs from the recorded count raises ValueError at the first lock.
        touch_on_hit: Refresh a file's modification time on a hit, at most once per
            ``touch_interval`` seconds, so ``prune(older_than=...)`` measures time since
            last use. Off by default: reads then never write.
        touch_interval: Minimum seconds between two refreshes of one file.
    """

    def __init__(
        self,
        directory: str | Path | None,
        *,
        encode: Callable[[T], bytes],
        decode: Callable[[bytes], T],
        namespace: str = "",
        suffix: str = ".typed-index",
        magic: bytes = b"AMBER-LMDB-INDEX\x00\x01",
        max_header_bytes: int = 64 * 1024,
        max_payload_bytes: int = 256 * 1024**2,
        lock_timeout: float | None = None,
        lock_stripes: int | None = None,
        touch_on_hit: bool = False,
        touch_interval: float = 3600.0,
    ) -> None:
        if not callable(encode) or not callable(decode):
            raise TypeError("encode and decode must be callable")
        if not isinstance(namespace, str) or any(
            not (char.isascii() and (char.isalnum() or char in "._-")) for char in namespace
        ):
            raise ValueError("namespace must contain only ASCII filename characters")
        if (
            not isinstance(suffix, str)
            or not suffix.startswith(".")
            or any(not (char.isascii() and (char.isalnum() or char in "._-")) for char in suffix)
        ):
            raise ValueError("suffix must be a safe filename suffix beginning with '.'")
        if suffix.lower().endswith(".lock"):
            raise ValueError("suffix must not end in '.lock'; it is reserved for builder locks")
        if not isinstance(magic, bytes) or not magic:
            raise ValueError("magic must be nonempty bytes")
        if any(
            type(size) is not int or size <= 0 for size in (max_header_bytes, max_payload_bytes)
        ):
            raise ValueError("cache byte limits must be positive integers")
        if max_header_bytes > 2**32 - 1:
            raise ValueError("header limit exceeds the envelope length field")
        try:
            _check_seconds(lock_timeout, "lock_timeout")
        except ValueError:
            raise ValueError("lock_timeout must be finite, nonnegative, or None") from None
        if lock_stripes is not None and (
            type(lock_stripes) is not int or not 1 <= lock_stripes <= 4096
        ):
            raise ValueError("lock_stripes must be None or an integer between 1 and 4096")
        if type(touch_on_hit) is not bool:
            raise TypeError("touch_on_hit must be bool")
        if touch_interval is None:
            raise ValueError("touch_interval must be finite and nonnegative")
        _check_seconds(touch_interval, "touch_interval")
        self.directory: Path | None = (
            None if directory is None else Path(directory).expanduser().resolve()
        )
        self.encode = encode
        self.decode = decode
        self.namespace = namespace
        self.suffix = suffix
        self.magic = magic
        self.max_header_bytes = max_header_bytes
        self.max_payload_bytes = max_payload_bytes
        self.lock_timeout = lock_timeout
        self.lock_stripes = lock_stripes
        self.touch_on_hit = touch_on_hit
        self.touch_interval = touch_interval
        self._stripes: int | None = None

    def __repr__(self) -> str:
        directory = None if self.directory is None else str(self.directory)
        return f"DerivedIndexCache({directory!r}, namespace={self.namespace!r})"

    @property
    def _prefix(self) -> str:
        return f"{self.namespace}-" if self.namespace else ""

    @property
    def _stem_tail(self) -> str:
        """The part of the suffix that ``Path.stem`` keeps (all but its last extension)."""
        return self.suffix[: self.suffix.rfind(".")]

    def path_for(self, identity: Identity) -> Path:
        """Return the stable cache path without creating directories or files."""
        return self._path_of(_canonical_identity(identity))

    def _path_of(self, source: dict[str, Any]) -> Path:
        if self.directory is None:
            raise RuntimeError("cache persistence is disabled")
        digest = hashlib.sha256(_json_bytes(source)).hexdigest()
        return self.directory / f"{self._prefix}{digest}{self.suffix}"

    def _stripe_count(self, locks: Path) -> int:
        """Resolve the stripe count against the directory's record, creating it once."""
        if self._stripes is None:
            record = locks / _STRIPES_NAME
            stored = _read_stripes(record)
            if stored is None:
                _create_stripes(record, self.lock_stripes or _DEFAULT_STRIPES)
                stored = _read_stripes(record)
                assert stored is not None
            if self.lock_stripes is not None and self.lock_stripes != stored:
                raise ValueError(
                    f"cache directory uses {stored} lock stripes ({record}); "
                    f"lock_stripes={self.lock_stripes} would break mutual exclusion "
                    "with other builders; pass None or the recorded count"
                )
            self._stripes = stored
        return self._stripes

    def _digest_of(self, name: str, suffix: str) -> str | None:
        """Return the identity digest of a file name of this cache, else None."""
        if not name.startswith(self._prefix) or not name.endswith(suffix):
            return None
        digest = name[len(self._prefix) : len(name) - len(suffix)]
        return digest if _DIGEST.fullmatch(digest) else None

    @contextmanager
    def _stripe_lock(
        self,
        digest: str,
        *,
        timeout: float | None,
        cancel: CancelEvent | None = None,
    ) -> Iterator[None]:
        """Hold the builder lock stripe of an identity digest."""
        assert self.directory is not None
        locks = self.directory / ".locks"
        locks.mkdir(parents=True, mode=0o700, exist_ok=True)
        if locks.is_symlink() or not locks.is_dir():
            raise ValueError(f"cache lock directory must be a regular directory: {locks}")
        path = locks / f"stripe-{int(digest[:8], 16) % self._stripe_count(locks)}.lock"
        held: set[str] | None = getattr(_HELD_STRIPES, "paths", None)
        if held is None:
            held = _HELD_STRIPES.paths = set()
        key = str(path)
        if key in held:
            # This thread already holds the stripe (publish inside get_or_build or a
            # builder); a second OS lock on the same file would wait for itself.
            yield
            return
        with index_lock(path, timeout=timeout, cancel=cancel):
            held.add(key)
            try:
                yield
            finally:
                held.discard(key)

    def _header(self, stream: Any) -> tuple[dict[str, Any], int] | None:
        """Read magic and the bounded header; return it and the payload offset."""
        if stream.read(len(self.magic)) != self.magic:
            return None
        length = stream.read(4)
        if len(length) != 4:
            return None
        header_size = struct.unpack(">I", length)[0]
        if header_size > self.max_header_bytes:
            return None
        header = json.loads(stream.read(header_size))
        if not isinstance(header, dict):
            return None
        return cast(dict[str, Any], header), len(self.magic) + 4 + header_size

    def lookup(
        self,
        identity: Identity,
        *,
        validate: Callable[[T], None] | None = None,
    ) -> CacheResult[T]:
        """Return an explicit hit, including cached None, or a missing/corrupt miss.

        Check bounded lengths and the checksum before the caller's decoder runs.
        Decoder/validator ValueError or TypeError marks corruption. Filesystem
        permission and I/O failures propagate. Validators also receive None hits.
        With ``touch_on_hit``, a hit refreshes the file's modification time.
        """
        miss: CacheResult[T] = CacheResult(False)
        source = _canonical_identity(identity)
        if self.directory is None:
            return miss
        path = self._path_of(source)
        if path.is_symlink():
            return miss
        try:
            with path.open("rb") as stream:
                parsed = self._header(stream)
                if parsed is None:
                    return miss
                header = parsed[0]
                if header.get("source") != source:
                    return miss
                payload_size = header.get("payload_size")
                if type(payload_size) is not int or not 0 <= payload_size <= self.max_payload_bytes:
                    return miss
                payload = stream.read(payload_size + 1)
                if len(payload) != payload_size:
                    return miss
                if hashlib.sha256(payload).hexdigest() != header.get("sha256"):
                    return miss
                modified_ns = os.fstat(stream.fileno()).st_mtime_ns
            value = self.decode(payload)
            if validate is not None:
                validate(value)
            if self.touch_on_hit and time.time_ns() - modified_ns >= self.touch_interval * 1e9:
                # A read-only cache directory still serves hits.
                with suppress(OSError):
                    os.utime(path)
            return CacheResult(True, value)
        except FileNotFoundError:
            return miss
        except (ValueError, TypeError, UnicodeError, struct.error):
            return miss

    def publish(self, identity: Identity, value: T) -> None:
        """Atomically publish an encoded value after file fsync and then sync its directory.

        Takes the identity's builder lock stripe while writing (reentrant within a
        thread, so ``get_or_build`` and its builders may call it), so ``prune`` never
        mistakes the temporary file for an abandoned one. Prefer get_or_build for
        competing builders. No partial output is published. Directory-fsync failure
        after replacement propagates and leaves the complete file in place.
        """
        source = _canonical_identity(identity)
        if self.directory is None:
            return
        digest = self._path_of(source).name[len(self._prefix) :][:64]
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        with self._stripe_lock(digest, timeout=self.lock_timeout):
            self._publish_unlocked(source, value)

    def _publish_unlocked(self, source: dict[str, Any], value: T) -> None:
        """Publish with the builder lock stripe already held by the caller."""
        if self.directory is None:
            return
        path = self._path_of(source)
        payload = self.encode(value)
        if not isinstance(payload, bytes):
            raise TypeError("cache encoder must return bytes")
        if len(payload) > self.max_payload_bytes:
            raise ValueError("index payload exceeds the configured byte limit")
        header = _json_bytes(
            {
                "source": source,
                "payload_size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        if len(header) > self.max_header_bytes:
            raise ValueError("index header exceeds the configured byte limit")
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.directory, prefix=f".{path.stem}.", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(self.magic)
                stream.write(struct.pack(">I", len(header)))
                stream.write(header)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_flag = getattr(os, "O_DIRECTORY", None)
            if directory_flag is not None:
                descriptor = os.open(self.directory, os.O_RDONLY | directory_flag)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def get_or_build(
        self,
        identity: Identity,
        build: Callable[[], T],
        *,
        validate: Callable[[T], None] | None = None,
        source_check: Callable[[], None] | None = None,
        cancel: CancelEvent | None = None,
    ) -> T:
        """Read a ready hit directly; on a miss, lock, recheck, build and publish.

        source_check may reject a changed source before use/publication. Validators
        run on both hits and new output. Failed builders publish nothing. All
        callbacks run outside LMDB transactions.

        Three different stops: ``lock_timeout`` stops waiting for another builder
        (``TimeoutError``); a set ``cancel`` event stops this caller while it waits,
        before it builds and before it publishes (``OperationCancelled``; the builder
        itself should check the same event to stop long work); a runaway builder can
        only be stopped by running it in a separate process. Nothing is published
        after a stop.
        """
        _check_cancel(cancel)
        # A new, independent copy: a callback cannot mutate the selected cache key.
        source = _canonical_identity(identity)

        def check() -> None:
            if source_check is not None:
                source_check()

        def stop() -> None:
            if cancel is not None and cancel.is_set():
                raise OperationCancelled("get_or_build")

        def create() -> T:
            check()
            stop()
            value = build()
            stop()
            if validate is not None:
                validate(value)
            check()
            self.publish(source, value)  # The stripe is held; publish reenters it.
            check()
            return value

        stop()
        check()
        if self.directory is None:
            return create()
        result = self.lookup(source, validate=validate)
        if result.hit:
            check()
            return cast(T, result.value)
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        digest = self._path_of(source).name[len(self._prefix) :][:64]
        with self._stripe_lock(digest, timeout=self.lock_timeout, cancel=cancel):
            check()
            result = self.lookup(source, validate=validate)
            if not result.hit:
                return create()
            check()
            return cast(T, result.value)

    def entries(self) -> Iterator[CacheEntry]:
        """List this cache's files (its namespace and suffix only), reading headers only.

        Foreign files, symlinks, directories, temporaries and lock files are skipped.
        Yields nothing when persistence is disabled or the directory does not exist.
        """
        if self.directory is None or not self.directory.is_dir():
            return
        for path in sorted(self.directory.iterdir()):
            if self._digest_of(path.name, self.suffix) is None:
                continue
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            identity = None
            valid = False
            try:
                with path.open("rb") as stream:
                    parsed = self._header(stream)
                if parsed is not None:
                    header, offset = parsed
                    source = header.get("source")
                    identity = cast(dict[str, Any], source) if isinstance(source, dict) else None
                    payload_size = header.get("payload_size")
                    valid = (
                        identity is not None
                        and type(payload_size) is int
                        and 0 <= payload_size <= self.max_payload_bytes
                        and info.st_size == offset + payload_size
                    )
            except FileNotFoundError:
                continue
            except (ValueError, UnicodeError, struct.error):
                pass
            yield CacheEntry(path, identity, info.st_size, info.st_mtime_ns, valid)

    def prune(
        self,
        *,
        remove: Callable[[CacheEntry], bool] | None = None,
        older_than: float | None = None,
        max_total_bytes: int | None = None,
        max_entries: int | None = None,
        dry_run: bool = True,
        temporary_grace: float = 3600.0,
    ) -> PruneReport:
        """Remove invalid, unwanted, expired and excess entries; a dry run by default.

        Selection order: invalid entries; valid entries for which ``remove`` returns
        True; entries not modified for ``older_than`` seconds; then the oldest
        remaining entries until the cache totals at most ``max_total_bytes`` and
        ``max_entries``. Each selected
        entry is removed under its builder lock stripe, taken without waiting: a
        stripe held by a builder, an entry changed since it was listed, or (on
        Windows) a file open in a reader leaves the entry in place as ``in_use``.
        A reader that opens it afterwards sees a miss and rebuilds.

        Abandoned temporaries of crashed builders are removed when older than
        ``temporary_grace`` seconds and their stripe is free. Lock stripe files are
        never removed. Same-host scope: no NFS or cross-host guarantee.
        """
        if remove is not None and not callable(remove):
            raise TypeError("remove must be callable or None")
        _check_seconds(older_than, "older_than")
        _check_seconds(temporary_grace, "temporary_grace")
        if temporary_grace is None:
            raise ValueError("temporary_grace must be finite and nonnegative")
        for name, limit in (("max_total_bytes", max_total_bytes), ("max_entries", max_entries)):
            if limit is not None and (type(limit) is not int or limit < 0):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        if type(dry_run) is not bool:
            raise TypeError("dry_run must be bool")
        entries = list(self.entries())
        now = time.time_ns()
        selected: list[tuple[CacheEntry, str]] = []
        remaining: list[CacheEntry] = []
        for entry in entries:
            if not entry.valid:
                selected.append((entry, "invalid"))
            elif remove is not None and remove(entry):
                selected.append((entry, "predicate"))
            elif older_than is not None and now - entry.modified_ns > older_than * 1e9:
                selected.append((entry, "expired"))
            else:
                remaining.append(entry)
        remaining.sort(key=lambda entry: (entry.modified_ns, entry.path.name))
        total = sum(entry.size_bytes for entry in remaining)
        while remaining and (
            (max_total_bytes is not None and total > max_total_bytes)
            or (max_entries is not None and len(remaining) > max_entries)
        ):
            oldest = remaining.pop(0)
            total -= oldest.size_bytes
            selected.append((oldest, "capacity"))
        removed: list[Path] = []
        in_use: list[Path] = []
        reasons: dict[str, int] = {}
        freed = 0
        for entry, reason in selected:
            digest = self._digest_of(entry.path.name, self.suffix)
            assert digest is not None
            if self._remove_locked(entry.path, digest, entry, dry_run=dry_run):
                removed.append(entry.path)
                reasons[reason] = reasons.get(reason, 0) + 1
                freed += entry.size_bytes
            else:
                in_use.append(entry.path)
        temporaries = self._prune_temporaries(now, temporary_grace, dry_run=dry_run)
        return PruneReport(
            dry_run,
            tuple(removed),
            tuple(in_use),
            len(entries) - len(removed),
            freed,
            reasons,
            tuple(temporaries),
        )

    def _remove_locked(
        self, path: Path, digest: str, listed: CacheEntry | None, *, dry_run: bool
    ) -> bool:
        """Remove one file under its stripe lock taken without waiting; False if busy."""
        try:
            with self._stripe_lock(digest, timeout=0):
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    return False
                if listed is not None and (
                    info.st_size != listed.size_bytes or info.st_mtime_ns != listed.modified_ns
                ):
                    return False  # Republished since it was listed.
                if dry_run:
                    return True
                try:
                    path.unlink()
                except FileNotFoundError:
                    return False
                except PermissionError:
                    if os.name == "nt":
                        return False  # Open in a reader; Windows refuses the deletion.
                    raise
                return True
        except TimeoutError:
            return False

    def _prune_temporaries(self, now: int, grace: float, *, dry_run: bool) -> list[Path]:
        assert self.directory is not None
        if not self.directory.is_dir():
            return []
        found = []
        for path in sorted(self.directory.iterdir()):
            name = path.name
            if not name.startswith(".") or not name.endswith(".tmp"):
                continue
            stem, _, _ = name[1:-4].rpartition(".")  # Random parts contain no dots.
            digest = self._digest_of(stem, self._stem_tail)
            if digest is None:
                continue
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode) or now - info.st_mtime_ns <= grace * 1e9:
                continue
            if self._remove_locked(path, digest, None, dry_run=dry_run):
                found.append(path)
        return found


__all__ = ["CacheEntry", "CacheResult", "DerivedIndexCache", "PruneReport", "index_lock"]
