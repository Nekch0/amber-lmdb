"""Namespaced field views and indexed records without dataset or padding policy."""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, overload

from .budgets import DEFAULT_MAX_BYTES, _validate_limits
from .codec import StorageKind, storage_kind_for_type
from .errors import AmberLMDBError
from .lmdb import LMDBReader, TypedReader, get_loc
from .schema import RecordSchema, ValueSchema, validate_schema, validate_value

if TYPE_CHECKING:
    from .overlay import LMDBOverlayReader

type LogicalKey = str | Path | tuple[str, ...]
type FieldKeyFactory = Callable[[str, StorageKind], LogicalKey]


class SourceChangedError(AmberLMDBError, ValueError):
    """The opened source identity changed while a multi-batch result was built."""


class ValueView:
    """Read named fields from an explicit schema and namespace at one optional index.

    By default, keys are ``(*namespace, storage_kind, field_name)``. A key factory
    may override the layout without changing the binary key format. An absent
    index selects an unindexed value, never an implicit padding/default value.
    The caller owns the reader. No connection is opened during construction.
    """

    def __init__(
        self,
        reader: LMDBReader | LMDBOverlayReader,
        schema: ValueSchema,
        *,
        namespace: tuple[str, ...] = (),
        index: int | None = None,
        key_factory: FieldKeyFactory | None = None,
    ) -> None:
        _check_reader(reader)
        if isinstance(schema, RecordSchema):
            raise TypeError(
                "ValueView needs a ValueSchema (field name -> exact type); a RecordSchema "
                "validates dictionary rows, so pass it to RecordTable(..., schema=...)"
            )
        if not isinstance(schema, ValueSchema):
            raise TypeError(f"schema must be a ValueSchema, got {type(schema).__name__}")
        if type(namespace) is not tuple or any(not isinstance(part, str) for part in namespace):
            raise TypeError("namespace must be a tuple of strings")
        get_loc(namespace, index)
        if key_factory is not None and not callable(key_factory):
            raise TypeError("key_factory must be callable")
        self.reader = reader
        self.schema = schema
        self.namespace = namespace
        self.index = index
        self._key_factory = key_factory
        self._types = schema.type_map

    def __repr__(self) -> str:
        fields = len(self._types)
        return f"ValueView(namespace={self.namespace!r}, index={self.index!r}, fields={fields})"

    def location(self, name: str) -> LogicalKey:
        """Return a validated logical field key; raise KeyError for unknown fields."""
        if name not in self._types:
            raise KeyError(f"unknown field: {name!r}")
        kind = storage_kind_for_type(self._types[name])
        key = (
            (*self.namespace, kind, name)
            if self._key_factory is None
            else self._key_factory(name, kind)
        )
        get_loc(key, self.index)
        return key

    def get(self, name: str, *, allow_custom: bool = False) -> Any:
        """Read a field with its exact declared type and recursive value checks.

        Custom decoding requires explicit opt-in on both the schema and this call.
        Decode and validation occur after the reader's short transaction closes.
        Values remain valid after the reader closes; no feature padding is inferred.
        """
        if type(allow_custom) is not bool:
            raise TypeError("allow_custom must be bool")
        if allow_custom and not self.schema.allow_custom:
            raise ValueError("schema does not allow custom values")
        value = self.reader.read(self.location(name), self.index, allow_custom=allow_custom)
        expected = self._types[name]
        if type(value) is not expected:
            raise TypeError(f"field {name!r}: expected {expected!r}, got {type(value)!r}")
        validate_value(value, allow_custom=allow_custom, max_depth=self.schema.max_depth, name=name)
        return value

    def get_as[T](self, name: str, expected: type[T], *, allow_custom: bool = False) -> T:
        """Read a field and require an exact Python type, without implicit conversion."""
        value = self.get(name, allow_custom=allow_custom)
        if type(value) is not expected:
            raise TypeError(f"field {name!r} is not exactly {expected!r}")
        return cast(T, value)


def _check_reader(reader: object) -> None:
    if not isinstance(reader, TypedReader):
        raise TypeError(
            f"reader must be an LMDBReader or LMDBOverlayReader, got {type(reader).__name__}"
        )


class RecordTable[T]:
    """Materialize indexed dictionaries using a caller-selected factory and schema.

    Rows share one logical key and use indices ``0..count-1``. The schema is
    optional; factories perform any domain-specific validation. No dynamic import,
    random state, filtering, cache or default row is used. The reader remains
    caller-owned. Factories run outside LMDB transactions.

    A table is a read-only sequence: ``len(table)``, ``table[i]`` (negative ``i``
    counts from the end), ``table[a:b]`` (a list, read in bounded batches) and
    iteration in bounded pages. ``get`` is the strict form that refuses negative
    indices.

    Args:
        reader: Caller-owned ``LMDBReader`` or ``LMDBOverlayReader``.
        key: Logical key of the rows.
        factory: Builds a record from a row dictionary; None returns the dictionaries.
        count: Number of rows. None counts the key's indices now (a key-only scan)
            and requires them to be exactly ``0..count-1``.
        schema: Optional ``RecordSchema`` or ``ValueSchema`` every row must satisfy.
        batch_size: Rows per batch read, 1 to 1,048,576.
        max_bytes: Copy budget of one batch read; None removes it.
    """

    @overload
    def __init__(
        self: RecordTable[dict[str, Any]],
        reader: LMDBReader | LMDBOverlayReader,
        key: LogicalKey,
        factory: None = None,
        *,
        count: int | None = None,
        schema: ValueSchema | RecordSchema | None = None,
        batch_size: int = 512,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> None: ...

    @overload
    def __init__(
        self,
        reader: LMDBReader | LMDBOverlayReader,
        key: LogicalKey,
        factory: Callable[[dict[str, Any]], T],
        *,
        count: int | None = None,
        schema: ValueSchema | RecordSchema | None = None,
        batch_size: int = 512,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> None: ...

    def __init__(
        self,
        reader: LMDBReader | LMDBOverlayReader,
        key: LogicalKey,
        factory: Callable[[dict[str, Any]], T] | None = None,
        *,
        count: int | None = None,
        schema: ValueSchema | RecordSchema | None = None,
        batch_size: int = 512,
        max_bytes: int | None = DEFAULT_MAX_BYTES,
    ) -> None:
        _check_reader(reader)
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("count must be a nonnegative integer or None")
        _validate_limits(batch_size, max_bytes)
        if factory is not None and not callable(factory):
            raise TypeError("factory must be callable or None")
        if schema is not None and not isinstance(schema, (ValueSchema, RecordSchema)):
            raise TypeError(
                f"schema must be a RecordSchema, ValueSchema or None, got {type(schema).__name__}"
            )
        get_loc(key)
        if count is None:
            count = reader.count(key)
            if count and reader.index_range(key) != (0, count - 1):
                raise ValueError(
                    f"indices of {key!r} are not 0..{count - 1}; pass count= explicitly "
                    "and read only existing rows"
                )
        self.reader = reader
        self.key = key
        self.count = count
        self.factory = factory
        self.schema = schema
        self.batch_size = batch_size
        self.max_bytes = max_bytes

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return f"RecordTable(key={self.key!r}, count={self.count})"

    @overload
    def __getitem__(self, item: int) -> T: ...

    @overload
    def __getitem__(self, item: slice) -> list[T]: ...

    def __getitem__(self, item: int | slice) -> T | list[T]:
        """Read one row (negative indices count from the end) or a slice of rows."""
        if isinstance(item, slice):
            return self.get_many(range(self.count)[item])
        position = self._position(item)
        if position < 0:
            position += self.count
        if not 0 <= position < self.count:
            raise IndexError("record index is out of range")
        return self.get(position)

    def __iter__(self) -> Iterator[T]:
        return self.iter_records()

    @staticmethod
    def _position(index: object) -> int:
        if isinstance(index, bool):
            raise TypeError("record index must be an integer, not bool")
        try:
            return operator.index(index)  # type: ignore[arg-type]
        except TypeError:
            raise TypeError(
                f"record index must be an integer, got {type(index).__name__}"
            ) from None

    def _check_index(self, index: object) -> int:
        position = self._position(index)
        if not 0 <= position < self.count:
            raise IndexError("record index is out of range")
        return position

    def _materialize(self, row: dict[str, Any], index: int) -> T:
        if self.schema is not None:
            validate_schema(row, self.schema, sample=str(index))
        return cast(T, row) if self.factory is None else self.factory(row)

    def get(self, index: int) -> T:
        """Read and materialize one row; out-of-range indices raise IndexError."""
        position = self._check_index(index)
        return self._materialize(self.reader.read_as_dict(self.key, position), position)

    def get_many(self, indices: Sequence[int]) -> list[T]:
        """Read bounded batches in input order, retaining duplicate indices.

        Validate every index before opening the reader. Empty input opens nothing.
        Reject a source snapshot change between batches rather than returning a
        mixed result. Byte-aware pages use max_bytes for encoded bodies/integrity
        metadata; the returned list and decoded allocations are not memory-bounded.
        """
        positions = tuple(self._check_index(index) for index in indices)
        if not positions:
            return []
        snapshot = self.reader.get_snapshot()
        result: list[T] = []
        start = 0
        for rows in self.reader.iter_dict_pages(
            (self.key for _ in positions),
            indices=positions,
            max_items=self.batch_size,
            max_bytes=self.max_bytes,
        ):
            if self.reader.get_snapshot() != snapshot:
                raise SourceChangedError("record source changed during batch reading")
            batch = positions[start : start + len(rows)]
            result.extend(
                self._materialize(row, index) for row, index in zip(rows, batch, strict=True)
            )
            start += len(rows)
        if self.reader.get_snapshot() != snapshot:
            raise SourceChangedError("record source changed during materialization")
        return result

    def iter_records(self, *, offset: int = 0, limit: int | None = None) -> Iterator[T]:
        """Yield rows in bounded pages from a validated offset and optional count limit.

        Check one source identity across all pages and retain no read transaction
        across a yield. Refreshing the reader between pages raises SourceChangedError.
        """
        if type(offset) is not int or not 0 <= offset <= self.count:
            raise ValueError("offset must be within the record table")
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit must be a nonnegative integer or None")
        end = self.count if limit is None else min(self.count, offset + limit)
        if offset == end:
            return
        snapshot = self.reader.get_snapshot()
        index = offset
        for rows in self.reader.iter_dict_pages(
            (self.key for _ in range(offset, end)),
            indices=range(offset, end),
            max_items=self.batch_size,
            max_bytes=self.max_bytes,
        ):
            if self.reader.get_snapshot() != snapshot:
                raise SourceChangedError("record source changed during iteration")
            for row in rows:
                value = self._materialize(row, index)
                index += 1
                if self.reader.get_snapshot() != snapshot:
                    raise SourceChangedError("record source changed during materialization")
                yield value


__all__ = ["FieldKeyFactory", "LogicalKey", "RecordTable", "SourceChangedError", "ValueView"]
