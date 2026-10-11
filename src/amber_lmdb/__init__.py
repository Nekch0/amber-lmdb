"""Typed LMDB storage with generation publication and short read transactions.

The top-level namespace holds the task-level API: readers and writers, keys, scans,
generation management, records and schemas, caches, limits and errors. Value codecs
live in ``amber_lmdb.codec`` and result dataclasses also in their modules.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import codec
from .arrays import ArrayHeader, parse_array_header
from .budgets import DEFAULT_MAX_BYTES, MAX_BATCH_ITEMS, ReadLimitError
from .codec import ValueLimits, register_bytes_codec, unregister_bytes_codec
from .collections import RecordTable, SourceChangedError, ValueView
from .errors import (
    AmberLMDBError,
    KeyNotFoundError,
    ReaderClosedError,
    ReaderMismatchError,
    StoreNotFoundError,
)
from .fingerprints import StoreDiff, StoreFingerprint, diff_stores, fingerprint_store
from .generations import (
    GenerationDeletion,
    GenerationInfo,
    GenerationInventory,
    delete_generations,
    inspect_generations,
    list_generations,
    prune_generations,
    set_current,
)
from .history import Publication, read_history
from .indexes import CacheEntry, CacheResult, DerivedIndexCache, PruneReport, index_lock
from .inspection import EntryInfo, StoreInfo, inspect_store
from .integrity import IntegrityError
from .keys import decode_key, describe_location, get_loc
from .lmdb import (
    EnvironmentHandle,
    KeyGroup,
    LMDBReader,
    LMDBSnapshot,
    LMDBWriter,
    TypedReader,
    get_active_envs,
    open_environment,
)
from .manifest import ManifestArray, ManifestGroup, StoreManifest
from .merging import merge_stores
from .overlay import LMDBOverlayReader, OverlayConflictError
from .physical import BackendInfo, PhysicalReport, get_backend_info, verify_physical
from .progress import CancelEvent, OperationCancelled, Progress, ProgressCallback
from .scanning import (
    ArrayGroup,
    ArrayStatistics,
    ValidationIssue,
    ValidationReport,
    array_statistics,
    verify_store,
)
from .schema import (
    Dimension,
    FieldSpec,
    RecordSchema,
    SchemaConstraintError,
    ValueSchema,
    ValueTypeError,
    ValueValidationError,
    infer_schema,
    validate_schema,
    validate_value,
)
from .tensors import as_tensor

try:
    __version__ = _version("amber-lmdb")
except PackageNotFoundError:  # pragma: no cover - running from an unbuilt checkout
    __version__ = "0+unknown"

__all__ = [
    "DEFAULT_MAX_BYTES",
    "MAX_BATCH_ITEMS",
    "AmberLMDBError",
    "ArrayGroup",
    "ArrayHeader",
    "ArrayStatistics",
    "BackendInfo",
    "CacheEntry",
    "CacheResult",
    "CancelEvent",
    "DerivedIndexCache",
    "Dimension",
    "EntryInfo",
    "EnvironmentHandle",
    "FieldSpec",
    "GenerationDeletion",
    "GenerationInfo",
    "GenerationInventory",
    "IntegrityError",
    "KeyGroup",
    "KeyNotFoundError",
    "LMDBOverlayReader",
    "LMDBReader",
    "LMDBSnapshot",
    "LMDBWriter",
    "ManifestArray",
    "ManifestGroup",
    "OperationCancelled",
    "OverlayConflictError",
    "PhysicalReport",
    "Progress",
    "ProgressCallback",
    "PruneReport",
    "Publication",
    "ReadLimitError",
    "ReaderClosedError",
    "ReaderMismatchError",
    "RecordSchema",
    "RecordTable",
    "SchemaConstraintError",
    "SourceChangedError",
    "StoreDiff",
    "StoreFingerprint",
    "StoreInfo",
    "StoreManifest",
    "StoreNotFoundError",
    "TypedReader",
    "ValidationIssue",
    "ValidationReport",
    "ValueLimits",
    "ValueSchema",
    "ValueTypeError",
    "ValueValidationError",
    "ValueView",
    "__version__",
    "array_statistics",
    "as_tensor",
    "codec",
    "decode_key",
    "delete_generations",
    "describe_location",
    "diff_stores",
    "fingerprint_store",
    "get_active_envs",
    "get_backend_info",
    "get_loc",
    "index_lock",
    "infer_schema",
    "inspect_generations",
    "inspect_store",
    "list_generations",
    "merge_stores",
    "open_environment",
    "parse_array_header",
    "prune_generations",
    "read_history",
    "register_bytes_codec",
    "set_current",
    "unregister_bytes_codec",
    "validate_schema",
    "validate_value",
    "verify_physical",
    "verify_store",
]
