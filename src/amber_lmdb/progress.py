"""Progress reports and cooperative cancellation for long whole-store operations."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from .errors import AmberLMDBError


@dataclass(frozen=True, slots=True)
class Progress:
    """One report, delivered between pages and never inside an LMDB transaction.

    Attributes:
        operation: Running operation, such as ``"verify_store"``.
        entries_done: Entries processed so far.
        entries_total: Expected entries when known: the stored entries, including
            the chunks of chunked arrays and the manifest, which some scans skip.
        bytes_done: Bytes copied so far (bodies, headers or metadata).
        elapsed_seconds: Seconds since the operation started.
        finished: True only for the final report of a completed operation.
    """

    operation: str
    entries_done: int
    entries_total: int | None
    bytes_done: int
    elapsed_seconds: float
    finished: bool = False


class CancelEvent(Protocol):
    """Anything with ``is_set()``, typically a ``threading.Event``."""

    def is_set(self) -> bool: ...


class OperationCancelled(AmberLMDBError):
    """The caller's cancel event was set; ``partial`` holds a result so far, if any.

    Only ``verify_store`` provides a partial result, a report marked cancelled and
    therefore incomplete; other operations leave ``partial`` as None because a
    prefix of their result would be misleading.
    """

    def __init__(self, operation: str, partial: Any = None) -> None:
        super().__init__(f"{operation} was cancelled")
        self.operation = operation
        self.partial = partial


type ProgressCallback = Callable[[Progress], None]

_INTERVAL = 0.2  # seconds between progress callbacks


class _Tracker:
    """Count work, throttle callbacks and check cancellation between pages."""

    def __init__(
        self,
        operation: str,
        progress: ProgressCallback | None,
        cancel: CancelEvent | None,
        *,
        total: int | None = None,
        interval: float | None = None,
    ) -> None:
        if progress is not None and not callable(progress):
            raise TypeError("progress must be callable or None")
        if cancel is not None and not callable(getattr(cancel, "is_set", None)):
            raise TypeError("cancel must provide is_set(), such as threading.Event")
        self.operation = operation
        self.progress = progress
        self.cancel = cancel
        self.total = total
        self.interval = _INTERVAL if interval is None else interval
        self.entries = 0
        self.bytes = 0
        self.started = time.monotonic()
        self.reported = self.started

    def check(self, partial: Callable[[], Any] | None = None) -> None:
        """Raise OperationCancelled if cancellation was requested."""
        if self.cancel is not None and self.cancel.is_set():
            raise OperationCancelled(self.operation, None if partial is None else partial())

    def page(self, entries: int, nbytes: int = 0, partial: Callable[[], Any] | None = None) -> None:
        """Record one finished page, report at most every interval, then check cancel."""
        self.entries += entries
        self.bytes += nbytes
        now = time.monotonic()
        if self.progress is not None and now - self.reported >= self.interval:
            self.reported = now
            self.progress(self._report(now, finished=False))
        self.check(partial)

    def finish(self) -> None:
        """Deliver the final report of a completed operation."""
        if self.progress is not None:
            self.progress(self._report(time.monotonic(), finished=True))

    def _report(self, now: float, *, finished: bool) -> Progress:
        return Progress(
            self.operation, self.entries, self.total, self.bytes, now - self.started, finished
        )


__all__ = ["CancelEvent", "OperationCancelled", "Progress", "ProgressCallback"]
