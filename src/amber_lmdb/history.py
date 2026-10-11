"""Publication history: which generations became ``CURRENT``, in order and when.

Each dataset root holds ``HISTORY``, one JSON object per line
(``{"generation": ..., "sequence": ..., "published_ns": ...}``), appended after
``CURRENT`` switches to a generation. A generation recorded here was published at
least once, so recovery never removes it automatically. The file is created empty
with the root; roots created before it existed get it on their next publication,
starting with the generation ``CURRENT`` selected then.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

HISTORY_NAME = "HISTORY"


@dataclass(frozen=True, slots=True)
class Publication:
    """One switch of ``CURRENT``.

    Attributes:
        generation: Generation name that became current.
        sequence: Position in the root's history, starting at 1.
        published_ns: Wall-clock time of the switch in nanoseconds since the epoch.
    """

    generation: str
    sequence: int
    published_ns: int


def read_history(root: str | Path) -> tuple[Publication, ...] | None:
    """Return a root's publications in order, or None when it has no ``HISTORY`` file.

    A final line torn by a crash during an append is ignored; any other malformed line
    raises ValueError.
    """
    path = Path(root) / HISTORY_NAME
    if path.is_symlink():
        raise ValueError(f"LMDB HISTORY must be a regular file: {path}")
    try:
        text = path.read_bytes().decode("utf-8")
    except FileNotFoundError:
        return None
    lines = text.split("\n")
    publications = []
    for number, line in enumerate(lines):
        if not line:
            continue
        try:
            item = json.loads(line)
            publication = Publication(
                str(item["generation"]), int(item["sequence"]), int(item["published_ns"])
            )
        except (ValueError, KeyError, TypeError) as exc:
            if number == len(lines) - 1:  # torn last line without its newline
                break
            raise ValueError(f"malformed LMDB HISTORY line {number + 1}: {path}") from exc
        publications.append(publication)
    return tuple(publications)


def _create_history(root: Path) -> None:
    """Create an empty ``HISTORY`` file if it is missing, durably."""
    path = root / HISTORY_NAME
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_history(root: Path, generation: str) -> Publication:
    """Record that ``generation`` became current; the caller holds the root's writer lock."""
    existing = read_history(root) or ()
    sequence = (existing[-1].sequence if existing else 0) + 1
    publication = Publication(generation, sequence, time.time_ns())
    line = json.dumps(
        {
            "generation": publication.generation,
            "sequence": publication.sequence,
            "published_ns": publication.published_ns,
        },
        separators=(",", ":"),
    )
    path = root / HISTORY_NAME
    with path.open("ab") as stream:
        if stream.tell() and not path.read_bytes().endswith(b"\n"):
            stream.write(b"\n")  # terminate a torn line left by a crash
        stream.write(line.encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    return publication


def _record_current(root: Path, current: str | None) -> None:
    """Make sure ``HISTORY`` exists and records the generation ``CURRENT`` selects now."""
    history = read_history(root)
    if history is None:
        _create_history(root)
        history = ()
    if current is not None and current not in {item.generation for item in history}:
        _append_history(root, current)


__all__ = ["Publication", "read_history"]
