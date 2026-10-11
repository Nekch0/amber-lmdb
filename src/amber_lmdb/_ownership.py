"""Retained same-host OS locks for cooperative acquisition/publication/deletion."""

from __future__ import annotations

import errno
import os
import stat
import time
from pathlib import Path


class _FileLock:
    """One shared/exclusive kernel lock; inherited descriptors close without unlocking."""

    def __init__(self, path: Path, *, shared: bool = False, blocking: bool = True):
        self._pid = os.getpid()
        self._descriptor: int | None = None
        if path.is_symlink():
            raise ValueError(f"coordination lock cannot be a symlink: {path}")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("coordination lock requires a regular file")
            self._descriptor = fd
            while True:
                try:
                    self._lock(shared)
                    break
                except BlockingIOError:
                    if not blocking:
                        raise
                    time.sleep(0.01)
        except BaseException:
            os.close(fd)
            self._descriptor = None
            raise

    def _lock(self, shared: bool) -> None:
        descriptor = self._descriptor
        assert descriptor is not None
        if os.name == "nt":
            import ctypes
            import msvcrt
            from ctypes import wintypes

            class Overlapped(ctypes.Structure):
                _fields_ = [
                    ("internal", ctypes.c_size_t),
                    ("internal_high", ctypes.c_size_t),
                    ("offset", wintypes.DWORD),
                    ("offset_high", wintypes.DWORD),
                    ("event", wintypes.HANDLE),
                ]

            self._overlapped = Overlapped()
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            acquire = kernel.LockFileEx
            acquire.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(Overlapped),
            ]
            acquire.restype = wintypes.BOOL
            release = kernel.UnlockFileEx
            release.argtypes = [
                wintypes.HANDLE,
                wintypes.DWORD,
                wintypes.DWORD,
                wintypes.DWORD,
                ctypes.POINTER(Overlapped),
            ]
            release.restype = wintypes.BOOL
            handle = msvcrt.get_osfhandle(descriptor)
            if not acquire(
                handle, 1 | (0 if shared else 2), 0, 1, 0, ctypes.byref(self._overlapped)
            ):
                error = ctypes.get_last_error()
                if error == 33:
                    raise BlockingIOError(errno.EAGAIN, "generation ownership lock is held")
                raise ctypes.WinError(error)
            self._release = lambda: release(handle, 0, 1, 0, ctypes.byref(self._overlapped))
        else:
            import fcntl

            fcntl.flock(descriptor, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
            self._release = lambda: fcntl.flock(descriptor, fcntl.LOCK_UN)

    def close(self) -> None:
        fd = self._descriptor
        if fd is not None:
            try:
                if self._pid == os.getpid():
                    self._release()
            finally:
                os.close(fd)
                self._descriptor = None

    def __enter__(self) -> _FileLock:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def _coordination(root: Path) -> Path:
    directory = root / ".coordination"
    try:
        info = os.lstat(directory)
    except FileNotFoundError:
        directory.mkdir(mode=0o700, exist_ok=True)  # Usually exists; create on first use.
        info = os.lstat(directory)
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("coordination directory must be regular")
    return directory


def _root_gate(root: Path) -> _FileLock:
    return _FileLock(_coordination(root) / "gate.lock")


def _generation_lock(root: Path, name: str, *, shared: bool, blocking: bool = True) -> _FileLock:
    return _FileLock(_coordination(root) / f"{name}.lock", shared=shared, blocking=blocking)
