"""Structured limits for owned batch copies, separate from decoder resource usage."""

from __future__ import annotations

from typing import Literal

from .errors import AmberLMDBError
from .keys import describe_location

DEFAULT_MAX_BYTES = 64 * 1024**2
#: Largest ``max_items`` a batch read accepts (2**20).
MAX_BATCH_ITEMS = 1 << 20


class ReadLimitError(AmberLMDBError, ValueError):
    """A requested batch or single streaming entry exceeds its item/copy-byte budget.

    required counts requested entries, encoded body plus integrity metadata bytes,
    or (kind ``"nodes"``, from ``max_decoded_nodes``) MessagePack schema nodes decoded
    so far. Copy budgets do not bound decoded Python/native allocations,
    hashing/decoding time, transaction overhead, or values the caller retains across
    streaming pages; a node budget bounds the decoded MessagePack objects.
    """

    def __init__(
        self,
        *,
        kind: Literal["items", "bytes", "nodes"],
        limit: int,
        required: int,
        location: bytes | None = None,
    ) -> None:
        label = {"items": "item", "bytes": "byte", "nodes": "decoded-node"}[kind]
        where = "" if location is None else f" at {describe_location(location)}"
        remedy = {
            "items": "split the request or use iter_read_pages",
            "bytes": "raise max_bytes or use iter_read_pages, which splits pages by bytes",
            "nodes": "raise max_decoded_nodes or read fewer values per batch",
        }[kind]
        super().__init__(
            f"batch exceeds the configured {label} limit{where}: "
            f"required={required}, limit={limit}; {remedy}"
        )
        self.kind = kind
        self.limit = limit
        self.required = required
        self.location = location


def _validate_limits(max_items: int, max_bytes: int | None) -> None:
    if type(max_items) is not int:
        raise TypeError("max_items must be an integer")
    if not 1 <= max_items <= MAX_BATCH_ITEMS:
        raise ValueError(f"max_items must be between 1 and {MAX_BATCH_ITEMS}")
    _validate_max_bytes(max_bytes)


def _validate_max_bytes(max_bytes: int | None) -> None:
    """Accept a positive byte budget, or None for no budget."""
    if max_bytes is not None and (type(max_bytes) is not int or max_bytes <= 0):
        raise ValueError("max_bytes must be a positive integer or None")


__all__ = ["DEFAULT_MAX_BYTES", "MAX_BATCH_ITEMS", "ReadLimitError"]
