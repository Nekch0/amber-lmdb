"""Validate supported values and define ordered, exact-type record schemas."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

import numpy as np

from . import codec
from .codec import StorageKind, storage_kind_for_type
from .constraints import Dimension, FieldSpec, RecordSchema, SchemaConstraintError
from .errors import AmberLMDBError

if TYPE_CHECKING:
    import torch


type ValidationReason = Literal["unsupported_type", "object_dtype", "cycle", "depth"]


class ValueTypeError(AmberLMDBError, TypeError):
    """A value at a known field/path has an unsupported storage type."""

    def __init__(
        self,
        message: str,
        *,
        name: str | None,
        path: str,
        value_type: type[Any],
        reason: ValidationReason,
    ) -> None:
        super().__init__(message)
        self.name = name
        self.path = path
        self.value_type = value_type
        self.reason = reason


class ValueValidationError(AmberLMDBError, ValueError):
    """A value at a known field/path violates a structural storage constraint."""

    def __init__(
        self,
        message: str,
        *,
        name: str | None,
        path: str,
        reason: ValidationReason,
    ) -> None:
        super().__init__(message)
        self.name = name
        self.path = path
        self.reason = reason


def validate_value(
    value: Any,
    *,
    allow_custom: bool = False,
    max_depth: int = 100,
    name: str | None = None,
) -> StorageKind:
    """Validate a value's supported types, object dtypes, nesting and cycles.

    Return its storage category. Shared acyclic containers are accepted. Custom
    classes require explicit opt-in and registration; validation does not execute
    their encoders. Encoding still checks the encoder output and byte/item budgets.
    ``name`` supplies a field label for structured errors. Limits are per value.
    """
    if type(allow_custom) is not bool:
        raise TypeError("allow_custom must be bool")
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    active: set[int] = set()

    def walk(item: Any, path: str, depth: int) -> None:
        label = f"field {name!r} at {path}" if name is not None else path
        if depth > max_depth:
            raise ValueValidationError(
                f"{label} exceeds maximum nesting depth {max_depth}",
                name=name,
                path=path,
                reason="depth",
            )
        value_type = type(item)
        kind = codec._STORAGE_KINDS.get(value_type)
        if codec._is_tensor_type(value_type):
            kind = "tensor"
        if kind is None:
            if allow_custom and codec.is_registered_bytes_type(value_type):
                return
            fail_type(value_type, path, label)
        if kind == "ndarray" and item.dtype.hasobject:
            raise ValueTypeError(
                f"{label} contains object dtype",
                name=name,
                path=path,
                value_type=value_type,
                reason="object_dtype",
            )
        if isinstance(item, type) and not (
            item in codec._VALUE_TYPES.values()
            or codec._is_tensor_type(item)
            or (allow_custom and codec.is_registered_bytes_type(item))
        ):
            fail_type(item, path, label)
        if kind not in {"dict", "frozendict", "list", "tuple", "set", "frozenset"}:
            return
        identity = id(item)
        if identity in active:
            raise ValueValidationError(
                f"{label} contains a cycle",
                name=name,
                path=path,
                reason="cycle",
            )
        active.add(identity)
        try:
            if kind in {"dict", "frozendict"}:
                for index, (key, child) in enumerate(cast(Mapping[Any, Any], item).items()):
                    walk(key, f"{path}.keys()[{index}]", depth + 1)
                    child_path = (
                        f"{path}[{key!r}]"
                        if type(key) in {str, bool, int, float, complex}
                        else f"{path}.values()[{index}]"
                    )
                    walk(child, child_path, depth + 1)
            else:
                for index, child in enumerate(cast(Iterable[Any], item)):
                    child_path = (
                        f"{path}[{index}]"
                        if kind in {"list", "tuple"}
                        else f"{path}.items()[{index}]"
                    )
                    walk(child, child_path, depth + 1)
        finally:
            active.remove(identity)

    def fail_type(value_type: type[Any], path: str, label: str) -> NoReturn:
        qualified = f"{value_type.__module__}.{value_type.__qualname__}"
        raise ValueTypeError(
            f"{label}: unsupported type {qualified}; register_bytes_codec() and "
            "allow_custom=True are required for custom values",
            name=name,
            path=path,
            value_type=value_type,
            reason="unsupported_type",
        )

    walk(value, "$", 0)
    return storage_kind_for_type(type(value))


@dataclass(frozen=True, slots=True)
class ValueSchema:
    """Exact field types and storage kinds for an application-defined mapping.

    ``ordered`` decides whether field order must match. Empty schemas are valid.
    Array dtypes and shapes are not fixed by this schema. Construct explicitly or
    use ``infer_schema``; application-specific constraints belong in a validator.
    """

    names: tuple[str, ...]
    types: tuple[type[Any], ...]
    storage_kinds: tuple[StorageKind, ...]
    allow_custom: bool = False
    ordered: bool = True
    max_depth: int = 100

    def __post_init__(self) -> None:
        if any(type(items) is not tuple for items in (self.names, self.types, self.storage_kinds)):
            raise TypeError("schema names, types and storage kinds must be tuples")
        if not (len(self.names) == len(self.types) == len(self.storage_kinds)):
            raise ValueError("schema names, types and storage kinds must have equal lengths")
        if any(not isinstance(name, str) for name in self.names):
            raise TypeError("schema field names must be strings")
        if len(set(self.names)) != len(self.names):
            raise ValueError("schema field names must be unique")
        if any(not isinstance(value_type, type) for value_type in self.types):
            raise TypeError("schema types must be Python classes")
        if type(self.allow_custom) is not bool or type(self.ordered) is not bool:
            raise TypeError("schema options must be bool")
        if type(self.max_depth) is not int or self.max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        if self.storage_kinds != tuple(storage_kind_for_type(cls) for cls in self.types):
            raise ValueError("schema storage kinds do not match its types")

    @property
    def type_map(self) -> dict[str, type[Any]]:
        """Return a fresh field-name to exact-type mapping."""
        return dict(zip(self.names, self.types, strict=True))


def infer_schema(
    values: Mapping[str, Any],
    *,
    allow_custom: bool = False,
    ordered: bool = True,
    max_depth: int = 100,
) -> ValueSchema:
    """Infer a schema from one mapping while recursively validating its values.

    No field names, namespaces, completion markers or dataset conventions are
    imposed. Empty mappings are accepted. Subsequent values must be checked with
    ``validate_schema``; inference is not a statistical dtype/shape inference.
    """
    names = tuple(values)
    if any(not isinstance(name, str) for name in names):
        raise TypeError("schema field names must be strings")
    kinds: tuple[StorageKind, ...] = tuple(
        validate_value(value, allow_custom=allow_custom, max_depth=max_depth, name=name)
        for name, value in values.items()
    )
    return ValueSchema(
        names, tuple(type(values[name]) for name in names), kinds, allow_custom, ordered, max_depth
    )


def validate_schema(
    values: Mapping[str, Any],
    schema: ValueSchema | RecordSchema,
    *,
    sample: str = "",
) -> None:
    """Check names, exact field types and recursive value constraints against a schema.

    Reject missing/extra fields, order changes when ordered, unsupported nested
    values and cycles. Do not execute custom encoders or infer padding/defaults.
    ``sample`` identifies the caller's record in name/type mismatch diagnostics.
    """
    if isinstance(schema, RecordSchema):
        schema.validate(values, sample=sample)
        return
    actual = tuple(values)
    if (schema.ordered and actual != schema.names) or set(actual) != set(schema.names):
        raise ValueError(
            f"schema fields differ for {sample!r}: expected {schema.names!r}, got {actual!r}"
        )
    for name, expected in zip(schema.names, schema.types, strict=True):
        if type(values[name]) is not expected:
            raise TypeError(
                f"field {name!r} in {sample!r}: expected {expected!r}, got {type(values[name])!r}"
            )
        validate_value(
            values[name], allow_custom=schema.allow_custom, max_depth=schema.max_depth, name=name
        )


def dtype_name(value: Any) -> str | None:
    """Return a NumPy dtype string, a Tensor dtype name, or None for other values.

    NumPy strings retain byte order. Tensor names omit the ``torch.`` prefix.
    This helper does not import PyTorch to detect an already-loaded Tensor type.
    """
    if codec._is_tensor_type(type(value)):
        return str(value.dtype).removeprefix("torch.")
    if isinstance(value, np.ndarray):
        return value.dtype.str
    return None


def torch_dtype_from_name(name: str) -> torch.dtype:
    """Resolve a dtype name with an optional ``torch.`` prefix; import PyTorch on demand.

    Raise TypeError for a non-string or a name that does not resolve to torch.dtype.
    """
    if not isinstance(name, str):
        raise TypeError("dtype name must be a string")
    import torch

    candidate = getattr(torch, name.removeprefix("torch."), None)
    if not isinstance(candidate, torch.dtype):
        raise TypeError(f"unsupported torch dtype: {name!r}")
    return candidate


__all__ = [
    "Dimension",
    "FieldSpec",
    "RecordSchema",
    "SchemaConstraintError",
    "ValidationReason",
    "ValueSchema",
    "ValueTypeError",
    "ValueValidationError",
    "dtype_name",
    "infer_schema",
    "torch_dtype_from_name",
    "validate_schema",
    "validate_value",
]
