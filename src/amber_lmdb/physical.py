"""py-lmdb 3 engine capabilities and offline physical verification."""

from __future__ import annotations

import importlib.metadata
import stat
from dataclasses import dataclass
from pathlib import Path

import lmdb
from lmdb import verify as _verifier


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Binding version and available native engines, including single-engine builds."""

    binding_version: str
    engines: tuple[tuple[int, int, int], ...]


def get_backend_info() -> BackendInfo:
    """Report py-lmdb and native engine versions without opening a database.

    Probe both engine selectors. System-library builds may provide only one;
    unavailable engines are omitted rather than assumed to be bundled.
    """
    versions = []
    for selector in (0, 1):
        try:
            versions.append(lmdb.version(lib_version=selector))
        except (ValueError, lmdb.Error):
            continue
    return BackendInfo(importlib.metadata.version("lmdb"), tuple(versions))


@dataclass(frozen=True, slots=True)
class PhysicalReport:
    """Offline page-structure coverage, separate from typed-value validation.

    error describes why the verifier could not begin. problems contains bounded
    detail while problem_count counts every returned problem. This checks neither
    application values, supplied lock files, authenticity nor later file changes.
    """

    data_file: str
    problem_count: int
    problems: tuple[str, ...]
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the offline verifier completed without detected problems."""
        return self.error is None and self.problem_count == 0


def verify_physical(path: str | Path, *, max_issues: int = 1000) -> PhysicalReport:
    """Verify native LMDB page structures without passing the file to its C engine.

    Accept a typed store root, raw LMDB directory or data file. A typed root selects
    its CURRENT generation through Python-only path validation. The py-lmdb 3
    byte walker validates tree/free-page relationships, ordering and counters.
    Invalid/unsupported files yield report.error; filesystem failures propagate.

    Verify a private, immutable copy, or an exclusively owned closed staging file.
    A before/after file-identity/size/mtime check detects ordinary races but cannot
    establish protection from malicious replacement. Only native-endian, 64-bit
    files are supported by upstream. Supplied lock files are never read or removed;
    discard them only on an inactive private copy before a later trusted open.

    max_issues caps returned details, not upstream verification memory/time. This
    neither repairs pages nor enables writemap, decoder callbacks or NFS safety.
    """
    if type(max_issues) is not int or max_issues < 0:
        raise ValueError("max_issues must be a nonnegative integer")
    requested = Path(path).absolute()
    if requested.is_symlink():
        raise ValueError("physical verification input cannot be a symlink")
    if requested.is_dir() and (requested / "CURRENT").exists():
        from .lmdb import _resolve_lmdb_path

        _, selected = _resolve_lmdb_path(requested)
        requested = Path(selected)
    data = requested / "data.mdb" if requested.is_dir() else requested
    before = data.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("physical verification requires a regular data file")
    error = None
    problems: list[str] = []
    try:
        problems = _verifier.verify(str(data), subdir=False)
    except _verifier.VerifyError as exc:
        error = f"{type(exc).__name__}: {exc}"
    after = data.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        from .collections import SourceChangedError

        raise SourceChangedError("data file changed during offline physical verification")
    return PhysicalReport(str(data.resolve()), len(problems), tuple(problems[:max_issues]), error)


__all__ = ["BackendInfo", "PhysicalReport", "get_backend_info", "verify_physical"]
