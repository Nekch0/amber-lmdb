"""Store summaries and their rich views, customizable through :class:`StoreProfile`.

Stores are dataset roots named ``*.lmdb``. Names of the form ``{type}_{hash8}.lmdb``
are split into a type and a hash for listings.

Feature-store convention (optional): a ``meta`` dictionary entry with ``names`` (feature
names), ``type_map`` (name to type), ``dtype_map`` (name to dtype), ``shape_map`` (name to
per-dimension ``(mins, maxes)``) and ``count_map`` (name to entry count). When ``names``
is present the views show a feature table; when the store directory holds the profile's
``source_file`` (``code.log`` by default) the views show it as source code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

from ..codec import storage_kind_for_type
from ..inspection import inspect_store
from ._common import TreeNode
from .messages import ENGLISH, Messages

SummaryRow = tuple[str, str, Any]
"""``(group, key, value)``; an empty group continues the previous group."""

Bounds = tuple[list[int], list[int]]

FEATURE_COLUMNS = ("FEATURE", "TYPE", "DTYPE", "KIND", "SHAPE", "ENTRIES")
FEATURE_META_KEYS = frozenset({"names", "type_map", "dtype_map", "shape_map", "count_map"})
ENVIRONMENT_KEYS = (
    ("python", "python_version"),
    ("torch", "torch_version"),
    ("cuda", "cuda_version"),
    ("cudnn", "cudnn_version"),
)


@dataclass(frozen=True)
class StoreSummary:
    """Listing and detail information of one store."""

    path: Path
    typename: str
    digest: str
    version: str
    n_versions: int
    entries: int | None
    used_bytes: int | None
    size_bytes: int
    modified: float
    meta: dict[str, Any] | None = None
    shapes: dict[str, Bounds] | None = None
    counts: dict[str, int] | None = None
    meta_error: str | None = None

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def listed_size(self) -> int:
        """Used bytes when the store was opened, otherwise the file size."""
        return self.used_bytes if self.used_bytes is not None else self.size_bytes

    @property
    def modified_text(self) -> str:
        return datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")


def split_store_name(name: str) -> tuple[str, str]:
    """Split ``{typename}_{digest8}.lmdb`` into the type and the hash (``-`` if absent)."""
    stem = name.removesuffix(".lmdb")
    typename, separator, digest = stem.rpartition("_")
    if not separator or len(digest) != 8:
        return stem, "-"
    return typename, digest


def find_stores(directory: Path, name: str | None = None) -> list[Path]:
    """Sorted ``*.lmdb`` stores of a directory, optionally only names containing ``name``."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.name.endswith(".lmdb") and (name is None or name in path.name)
    )


def _current_version(store: Path) -> str:
    try:
        return (store / "CURRENT").read_text(encoding="ascii").strip()
    except OSError:
        return "-"


def parse_shape_map(raw: Any) -> dict[str, Bounds] | None:
    """Normalize ``{name: (mins, maxes)}``; None when malformed or absent."""
    if not isinstance(raw, Mapping):
        return None
    bounds: dict[str, Bounds] = {}
    try:
        for name, pair in cast("Mapping[Any, Any]", raw).items():
            mins, maxes = pair
            bounds[str(name)] = ([int(v) for v in mins], [int(v) for v in maxes])
    except (TypeError, ValueError):
        return None
    return bounds


def parse_count_map(raw: Any) -> dict[str, int] | None:
    """Normalize ``{name: count}``; None when malformed or absent."""
    if not isinstance(raw, Mapping):
        return None
    try:
        return {str(name): int(count) for name, count in cast("Mapping[Any, Any]", raw).items()}
    except (TypeError, ValueError):
        return None


def describe_store(
    store: Path,
    *,
    with_entries: bool = True,
    with_meta: bool = False,
    messages: Messages = ENGLISH,
) -> StoreSummary:
    """Collect the summary of one store, opening its current generation at most once."""
    typename, digest = split_store_name(store.name)
    version = _current_version(store)
    versions_dir = store / "versions"
    n_versions = (
        sum(1 for child in versions_dir.iterdir() if child.is_dir()) if versions_dir.is_dir() else 0
    )
    entries: int | None = None
    used: int | None = None
    meta: dict[str, Any] | None = None
    meta_error: str | None = None
    if version == "-":
        if with_meta:
            meta_error = messages.current_unreadable
    elif with_entries or with_meta:
        try:
            info = inspect_store(store, with_meta=with_meta)
            if Path(info.snapshot.generation) != (versions_dir / version).resolve():
                raise ValueError("CURRENT changed during inspection; retry the command")
            entries, used, meta_error = info.entries, info.used_bytes, info.meta_error
            meta = cast("dict[str, Any] | None", info.meta)
        except Exception as exc:
            if with_meta:
                meta_error = f"{type(exc).__name__}: {exc}"
        if not with_entries:
            entries = used = None

    if used is not None and not with_meta:
        # Listings approximate the size with one stat of data.mdb.
        try:
            stat = (versions_dir / version / "data.mdb").stat()
            size, modified = stat.st_size, stat.st_mtime
        except OSError:
            size, modified = used, store.stat().st_mtime
    else:
        # Unreadable stores and detail views walk the files for exact values.
        files = [path for path in store.rglob("*") if path.is_file()]
        size = sum(path.stat().st_size for path in files)
        modified = max((path.stat().st_mtime for path in files), default=store.stat().st_mtime)
    return StoreSummary(
        path=store,
        typename=typename,
        digest=digest,
        version=version,
        n_versions=n_versions,
        entries=entries,
        used_bytes=used,
        size_bytes=size,
        modified=modified,
        meta=meta,
        shapes=parse_shape_map(meta.get("shape_map")) if meta else None,
        counts=parse_count_map(meta.get("count_map")) if meta else None,
        meta_error=meta_error,
    )


def describe_stores(stores: Sequence[Path], *, with_entries: bool = True) -> list[StoreSummary]:
    """Summaries of several stores, read in parallel because the work is I/O bound."""
    if not stores:
        return []
    describe = partial(describe_store, with_entries=with_entries)
    with ThreadPoolExecutor(max_workers=min(8, len(stores))) as pool:
        return list(pool.map(describe, stores))


# Profile helpers -------------------------------------------------------------------


def meta_value(meta: Mapping[str, Any], key: str) -> Any:
    """``meta[key]``, or ``-`` when absent or None."""
    value = meta.get(key)
    return "-" if value is None else value


def meta_rows(
    meta: Mapping[str, Any], group: str, keys: Iterable[str | tuple[str, str]]
) -> list[SummaryRow]:
    """Rows of one group; a key may be ``(label, meta_key)``."""
    rows: list[SummaryRow] = []
    for index, key in enumerate(keys):
        label, meta_key = key if isinstance(key, tuple) else (key, key)
        rows.append((group if index == 0 else "", label, meta_value(meta, meta_key)))
    return rows


def environment_rows(meta: Mapping[str, Any]) -> list[SummaryRow]:
    """``environment`` group: Python, torch, CUDA, cuDNN and the device of the writer."""
    device = meta.get("device")
    if device is None:
        device = meta.get("gpu")
    return [
        *meta_rows(meta, "environment", ENVIRONMENT_KEYS),
        ("", "device", "-" if device is None else device),
    ]


def type_name(value: Any) -> str:
    """Display name of a ``type_map`` value; builtins without the module."""
    if isinstance(value, type):
        module, qualname = value.__module__, value.__qualname__
        return qualname if module == "builtins" else f"{module}.{qualname}"
    return str(value)


def storage_kind(value: Any) -> str:
    """Storage kind of a declared type, ``-`` for non-types."""
    return storage_kind_for_type(value) if isinstance(value, type) else "-"


def feature_nodes(
    meta: Mapping[str, Any], *, limit: int = 2, placeholder: str = "<feature>"
) -> list[TreeNode]:
    """Contents-tree nodes of the first ``limit`` features of the convention."""
    names = [str(name) for name in meta.get("names") or ()]
    type_map: Mapping[str, Any] = meta.get("type_map") or {}
    if not names:
        return [(placeholder, "feature value", ())]
    nodes: list[TreeNode] = [
        (name, f"feature value (kind: {storage_kind(type_map.get(name))})", ())
        for name in names[:limit]
    ]
    if len(names) > limit:
        nodes.append((f"... ({len(names) - limit} more)", "", ()))
    return nodes


def default_summary(summary: StoreSummary) -> list[SummaryRow]:
    """Scalar ``meta`` values outside the feature convention, then the environment."""
    meta = summary.meta or {}
    environment = {key for _, key in ENVIRONMENT_KEYS} | {"device", "gpu"}
    scalars = [
        (key, value)
        for key, value in meta.items()
        if key not in FEATURE_META_KEYS
        and key not in environment
        and isinstance(value, (str, int, float, bool))
    ]
    rows: list[SummaryRow] = [
        ("meta" if index == 0 else "", key, value) for index, (key, value) in enumerate(scalars)
    ]
    if environment & meta.keys():
        rows.extend(environment_rows(meta))
    return rows


def default_usage(summary: StoreSummary) -> str:
    """Snippet reading the metadata of the store with :class:`amber_lmdb.LMDBReader`."""
    return (
        "from amber_lmdb import LMDBReader\n"
        "\n"
        f'with LMDBReader(r"{summary.path}") as reader:\n'
        '    meta = reader.read_as_dict("meta")'
    )


def feature_usage_comment(summary: StoreSummary) -> tuple[str, str]:
    """First feature name and its ``type dtype`` comment, for usage snippets."""
    meta = summary.meta or {}
    names = meta.get("names") or ()
    feature = str(names[0]) if names else "<feature>"
    declared = (meta.get("type_map") or {}).get(feature)
    dtype = (meta.get("dtype_map") or {}).get(feature)
    comment = type_name(declared) if isinstance(declared, type) else ""
    if dtype is not None:
        comment = f"{comment} {dtype}".strip()
    return feature, comment


@dataclass(frozen=True)
class StoreProfile:
    """How stores of one kind are summarized.

    ``summary`` returns the rows below the file group. ``contents`` returns a tree
    describing the stored layout (omitted when None). ``usage`` returns example code
    (omitted when None or when it returns None). ``source_file`` names a file in the
    store directory to show as source code. ``title`` names the stores in headings.
    """

    summary: Callable[[StoreSummary], Iterable[SummaryRow]] = default_summary
    contents: Callable[[StoreSummary], Sequence[TreeNode]] | None = None
    usage: Callable[[StoreSummary], str | None] | None = default_usage
    source_file: str | None = "code.log"
    title: str = "Stores"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def usage_code(self, summary: StoreSummary) -> str | None:
        return None if self.usage is None else self.usage(summary)

    def source_code(self, summary: StoreSummary) -> str | None:
        if self.source_file is None:
            return None
        path = summary.path / self.source_file
        try:
            return path.read_text(encoding="utf-8") or None
        except OSError:
            return None


def shape_range(bounds: Bounds) -> str:
    """Per-dimension bounds as ``(80, 33..296)``."""
    mins, maxes = bounds
    parts = [str(lo) if lo == hi else f"{lo}..{hi}" for lo, hi in zip(mins, maxes, strict=True)]
    return f"({', '.join(parts)})"


def feature_rows(summary: StoreSummary) -> list[tuple[str, ...]]:
    """FEATURE, TYPE, DTYPE, KIND, SHAPE and ENTRIES of every feature in ``meta``."""
    meta = summary.meta or {}
    type_map: Mapping[str, Any] = meta.get("type_map") or {}
    dtype_map: Mapping[str, Any] = meta.get("dtype_map") or {}
    shapes = summary.shapes or {}
    counts = summary.counts or {}
    rows: list[tuple[str, ...]] = []
    for name in meta.get("names") or ():
        declared = type_map.get(name)
        rows.append(
            (
                str(name),
                type_name(declared) if declared is not None else "-",
                str(dtype_map.get(name, "-")),
                storage_kind(declared),
                shape_range(shapes[name]) if name in shapes else "-",
                str(counts[name]) if name in counts else "-",
            )
        )
    return rows


def has_features(summary: StoreSummary) -> bool:
    return bool((summary.meta or {}).get("names"))


__all__ = [
    "ENVIRONMENT_KEYS",
    "FEATURE_COLUMNS",
    "FEATURE_META_KEYS",
    "StoreProfile",
    "StoreSummary",
    "SummaryRow",
    "default_summary",
    "default_usage",
    "describe_store",
    "describe_stores",
    "environment_rows",
    "feature_nodes",
    "feature_rows",
    "feature_usage_comment",
    "find_stores",
    "has_features",
    "meta_rows",
    "meta_value",
    "parse_count_map",
    "parse_shape_map",
    "shape_range",
    "split_store_name",
    "storage_kind",
    "type_name",
]
