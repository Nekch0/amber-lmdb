"""Generation inventory, publication order, rollback and explicit cooperative deletion."""

from __future__ import annotations

import contextlib
import os
import shutil
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ._ownership import _generation_lock, _root_gate
from .collections import SourceChangedError
from .history import Publication, _append_history, _record_current, read_history
from .lmdb import (
    _acquire_writer_lock,
    _fsync_directory,
    _is_plain_dir,
    _is_version_name,
    _read_current,
    _release_writer_lock,
    _validate_dataset_root,
    _write_current,
    _writer_lock_path,
)


@dataclass(frozen=True, slots=True)
class GenerationInfo:
    """Filesystem and publication information for one generation directory.

    ``size_bytes`` sums the logical lengths of ``data.mdb`` and ``lock.mdb``.
    ``modified_ns`` is the modification time of ``data.mdb`` alone (readers update
    ``lock.mdb``). ``sequence`` and ``published_ns`` come from the root's ``HISTORY``:
    the latest time the generation became current, or None if it never did (or the
    root predates ``HISTORY``). ``error`` describes a damaged generation, which is
    still listed. Non-current does not mean unused or removable.
    """

    name: str
    path: str
    current: bool
    size_bytes: int
    modified_ns: int
    sequence: int | None = None
    published_ns: int | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class GenerationInventory:
    """Everything under a dataset root that an operator needs to decide on cleanup.

    Attributes:
        root: Resolved dataset root.
        current: Generation ``CURRENT`` selects, or None when it is missing or invalid.
        current_error: Why ``CURRENT`` could not be read, or None.
        generations: Generations in publication order (never-published ones first).
        trash: Generation names left in ``.trash`` by an interrupted deletion.
        staging: Staging directories of writers that did not finish, beside the root.
        history: Every publication recorded in ``HISTORY``, or None without it.
    """

    root: str
    current: str | None
    current_error: str | None
    generations: tuple[GenerationInfo, ...]
    trash: tuple[str, ...]
    staging: tuple[str, ...]
    history: tuple[Publication, ...] | None


def _generation_info(
    directory: Path, current: str | None, published: dict[str, Publication]
) -> GenerationInfo:
    sizes = 0
    modified = 0
    error = None
    try:
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("generation is not a regular directory")
        for filename in ("data.mdb", "lock.mdb"):
            candidate = directory / filename
            try:
                entry = candidate.lstat()
            except FileNotFoundError:
                if filename == "data.mdb":
                    raise ValueError("generation has no data.mdb") from None
                continue
            if not stat.S_ISREG(entry.st_mode):
                raise ValueError(f"generation file must be a regular file: {candidate}")
            sizes += entry.st_size
            if filename == "data.mdb":
                modified = entry.st_mtime_ns
    except (OSError, ValueError) as exc:
        error = str(exc)
    publication = published.get(directory.name)
    return GenerationInfo(
        directory.name,
        str(directory),
        directory.name == current,
        sizes,
        modified,
        None if publication is None else publication.sequence,
        None if publication is None else publication.published_ns,
        error,
    )


def _sort_key(item: GenerationInfo) -> tuple[int, int, str]:
    return (0 if item.sequence is None else 1, item.sequence or 0, item.name)


def inspect_generations(path: str | Path) -> GenerationInventory:
    """Inventory a dataset root without mutating it, even when it is damaged.

    A missing or invalid ``CURRENT`` and damaged generations are reported, not raised.
    A ``CURRENT`` switch during the inventory raises SourceChangedError. Inventory alone
    is not removal permission; see ``delete_generations`` and ``prune_generations``.
    """
    requested = Path(path).absolute()
    if requested.is_symlink():
        raise ValueError(f"LMDB dataset root cannot be a symlink: {requested}")
    root = requested.resolve()
    _validate_dataset_root(root)
    current: str | None
    current_error = None
    try:
        current = _read_current(root)
    except (OSError, ValueError, UnicodeError) as exc:
        current, current_error = None, f"{type(exc).__name__}: {exc}"
    try:
        history = read_history(root)
    except ValueError:
        history = None
    published: dict[str, Publication] = {}
    for publication in history or ():
        published[publication.generation] = publication  # the latest wins
    versions = root / "versions"
    generations = (
        [
            _generation_info(directory, current, published)
            for directory in versions.iterdir()
            if _is_version_name(directory.name)
        ]
        if _is_plain_dir(versions)
        else []
    )
    generations.sort(key=_sort_key)
    trash_dir = root / ".trash"
    trash = (
        tuple(sorted(entry.name for entry in trash_dir.iterdir() if _is_version_name(entry.name)))
        if _is_plain_dir(trash_dir)
        else ()
    )
    prefix = f".{root.name}."
    staging = tuple(
        sorted(
            entry.name
            for entry in root.parent.iterdir()
            if entry.name.startswith(prefix)
            and entry.name.endswith(".tmp")
            and _is_version_name(f"v-{entry.name[len(prefix) : -4]}")
        )
    )
    if current is not None:
        try:
            now = _read_current(root)
        except (OSError, ValueError, UnicodeError):
            now = None
        if now != current:
            raise SourceChangedError("CURRENT changed during generation inventory")
    return GenerationInventory(
        str(root), current, current_error, tuple(generations), trash, staging, history
    )


def list_generations(path: str | Path) -> tuple[GenerationInfo, ...]:
    """List generation directories in publication order without mutations.

    Damaged generations are listed with ``error`` set, and a missing or invalid
    ``CURRENT`` marks none as current; ``inspect_generations`` also reports why.
    Inventory alone is not removal permission.
    """
    return inspect_generations(path).generations


def set_current(path: str | Path, name: str, *, verify: bool = False) -> Publication:
    """Point ``CURRENT`` at an existing generation: roll back, or restore a lost pointer.

    Takes the root's writer lock (BlockingIOError while a writer is active) and the
    root gate, writes ``CURRENT`` atomically and records the switch in ``HISTORY``.
    Readers already open stay on their generation; new readers see the selected one.

    Args:
        path: Dataset root.
        name: Generation name (``v-<32 hex>``) under ``versions``.
        verify: Run ``verify_store`` on the generation first and refuse unless it is ok.

    Returns:
        The recorded publication.
    """
    if type(name) is not str or not _is_version_name(name):
        raise ValueError("name must be a generation name such as v-<32 hex>")
    if type(verify) is not bool:
        raise TypeError("verify must be bool")
    requested = Path(path).absolute()
    if requested.is_symlink():
        raise ValueError(f"LMDB dataset root cannot be a symlink: {requested}")
    root = requested.resolve()
    _validate_dataset_root(root)
    info = _generation_info(root / "versions" / name, None, {})
    if not (root / "versions" / name).exists():
        raise FileNotFoundError(f"generation not found: {root / 'versions' / name}")
    if info.error is not None:
        raise ValueError(f"generation {name} is damaged: {info.error}")
    if verify:
        from .scanning import verify_store

        report = verify_store(root, generation=name)
        if not report.ok:
            raise ValueError(
                f"generation {name} failed verification: {report.error_count} errors, "
                f"{report.skipped_values} unverified values"
            )
    handle = _acquire_writer_lock(_writer_lock_path(root))
    try:
        with _root_gate(root):
            try:
                previous: str | None = _read_current(root)
            except (OSError, ValueError, UnicodeError):
                previous = None
            _record_current(root, previous)
            _write_current(root, name)
            _fsync_directory(root)
            return _append_history(root, name)
    finally:
        _release_writer_lock(handle)


@dataclass(frozen=True, slots=True)
class GenerationDeletion:
    """One explicit candidate's result; eligible is a transient dry-run observation.

    ``unexpected_files`` means the generation directory holds files other than
    ``data.mdb`` and ``lock.mdb``; nothing was removed and ``message`` names them.
    """

    name: str
    status: Literal["eligible", "deleted", "current", "in_use", "missing", "unexpected_files"]
    message: str | None = None


def _check_deletable_directory(directory: Path) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"generation must be a regular directory: {directory}")
    for entry in directory.iterdir():
        if entry.name not in {"data.mdb", "lock.mdb"} or not stat.S_ISREG(entry.lstat().st_mode):
            raise ValueError(f"unexpected or nonregular generation entry: {entry}")


def delete_generations(
    path: str | Path,
    names: Sequence[str],
    *,
    dry_run: bool = True,
    same_host: bool = False,
) -> tuple[GenerationDeletion, ...]:
    """Delete only named non-current generations unused by cooperating local clients.

    dry_run defaults to True and never renames, removes or creates anything. Applying
    requires same_host=True: the caller affirms that every access uses this version's
    shared lifetime locks on this host. Raw lmdb opens, older clients, cross-host/NFS
    access and malicious filesystem changes are outside this guarantee. No PID,
    time-based expiry or process-local reference count is treated as permission.

    A retained root gate serializes acquisition, CURRENT publication and rename to
    private .trash. An exclusive generation OS lock rejects live or paused holders.
    Physical removal follows the rename with that lock still held; process death
    releases it. Repeating an explicit request resumes its interrupted trash removal.
    Once a generation is fully removed, its coordination lock file is removed under
    the root gate too (no new acquisition can reach it). A generation directory with
    unexpected files is reported as ``unexpected_files`` and kept. I/O failures
    propagate and may leave recoverable trash.
    """
    if type(dry_run) is not bool or type(same_host) is not bool:
        raise TypeError("dry_run and same_host must be bool")
    if not dry_run and not same_host:
        raise ValueError("deletion requires same_host=True and only cooperating amber-lmdb clients")
    if isinstance(names, (str, bytes)):
        raise ValueError("names must be a sequence of recognized generation names")
    selected = tuple(names)
    if any(type(name) is not str or not _is_version_name(name) for name in selected):
        raise ValueError("names must be a sequence of recognized generation names")
    if len(set(selected)) != len(selected):
        raise ValueError("generation names must be unique")
    requested = Path(path).absolute()
    if requested.is_symlink():
        raise ValueError(f"LMDB dataset root cannot be a symlink: {requested}")
    root = requested.resolve()
    _validate_dataset_root(root)
    return tuple(_delete_one(root, name, dry_run=dry_run) for name in selected)


def _current_or_none(root: Path) -> str | None:
    try:
        return _read_current(root)
    except (OSError, ValueError, UnicodeError):
        return None


def _delete_one(root: Path, name: str, *, dry_run: bool) -> GenerationDeletion:
    trash = root / ".trash"
    retired = trash / name
    directory = root / "versions" / name
    lock_file = root / ".coordination" / f"{name}.lock"
    if dry_run:
        # Never create anything: a missing lock file means nobody holds the lease.
        if _current_or_none(root) == name:
            return GenerationDeletion(name, "current")
        if not directory.exists() and not retired.exists():
            return GenerationDeletion(name, "missing")
        try:
            _check_deletable_directory(directory if directory.exists() else retired)
        except ValueError as exc:
            return GenerationDeletion(name, "unexpected_files", str(exc))
        if lock_file.exists():
            try:
                lease = _generation_lock(root, name, shared=False, blocking=False)
            except BlockingIOError:
                return GenerationDeletion(name, "in_use")
            lease.close()
        return GenerationDeletion(name, "eligible")
    lease = None
    with _root_gate(root):
        if _current_or_none(root) == name:
            return GenerationDeletion(name, "current")
        if trash.is_symlink() or (trash.exists() and not trash.is_dir()):
            raise ValueError("generation trash must be a regular private directory")
        if (
            not directory.exists()
            and not directory.is_symlink()
            and not retired.exists()
            and not retired.is_symlink()
        ):
            return GenerationDeletion(name, "missing")
        try:
            lease = _generation_lock(root, name, shared=False, blocking=False)
        except BlockingIOError:
            return GenerationDeletion(name, "in_use")
        try:
            try:
                if directory.exists() or directory.is_symlink():
                    _check_deletable_directory(directory)
                    if retired.exists() or retired.is_symlink():
                        raise ValueError("generation exists in both versions and trash")
                else:
                    _check_deletable_directory(retired)
            except ValueError as exc:
                lease.close()
                return GenerationDeletion(name, "unexpected_files", str(exc))
            if directory.exists():
                trash.mkdir(mode=0o700, exist_ok=True)
                os.replace(directory, retired)
                _fsync_directory(root / "versions")
                _fsync_directory(trash)
        except BaseException:
            lease.close()
            raise
    try:
        shutil.rmtree(retired)
        _fsync_directory(trash)
    finally:
        lease.close()
    with _root_gate(root):
        # The generation is gone from versions and trash, so no acquisition can reach
        # its lease file any more; a file still open elsewhere (Windows) is kept.
        if not directory.exists() and not retired.exists():
            with contextlib.suppress(PermissionError):
                lock_file.unlink(missing_ok=True)
    return GenerationDeletion(name, "deleted")


def prune_generations(
    path: str | Path,
    *,
    keep_last: int,
    dry_run: bool = True,
    same_host: bool = False,
) -> tuple[GenerationDeletion, ...]:
    """Delete all but the newest ``keep_last`` non-current generations.

    ``CURRENT`` is always kept in addition. Order follows ``HISTORY`` (generations that
    never became current count as oldest), so the result does not depend on random
    names or on read access times. Damaged generations are not selected. Deletion
    follows ``delete_generations`` (dry run by default, ``same_host`` required).
    """
    if type(keep_last) is not int or keep_last < 0:
        raise ValueError("keep_last must be a nonnegative integer")
    inventory = inspect_generations(path)
    candidates = [item for item in inventory.generations if not item.current and item.error is None]
    newest_first = sorted(candidates, key=_sort_key, reverse=True)
    selected = [item.name for item in reversed(newest_first[keep_last:])]  # oldest first
    if not selected:
        return ()
    return delete_generations(inventory.root, selected, dry_run=dry_run, same_host=same_host)


__all__ = [
    "GenerationDeletion",
    "GenerationInfo",
    "GenerationInventory",
    "delete_generations",
    "inspect_generations",
    "list_generations",
    "prune_generations",
    "set_current",
]
