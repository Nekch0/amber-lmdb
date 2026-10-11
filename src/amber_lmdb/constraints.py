"""Explicit record-field constraints beyond exact storage types."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sized
from dataclasses import dataclass
from typing import Any, NoReturn, cast

import numpy as np

from . import codec
from .errors import AmberLMDBError


class SchemaConstraintError(AmberLMDBError, ValueError):
    """A named record field violates a declared constraint; callbacks propagate separately."""

    def __init__(self, message: str, *, field: str | None, constraint: str) -> None:
        super().__init__(message)
        self.field = field
        self.constraint = constraint


@dataclass(frozen=True, slots=True)
class Dimension:
    """Inclusive variable-axis bounds; maximum=None leaves its upper bound open."""

    minimum: int = 0
    maximum: int | None = None

    def __post_init__(self) -> None:
        if type(self.minimum) is not int or self.minimum < 0:
            raise ValueError("dimension minimum must be a nonnegative integer")
        if self.maximum is not None and (
            type(self.maximum) is not int or self.maximum < self.minimum
        ):
            raise ValueError("dimension maximum must be an integer >= minimum or None")


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Required/nullable exact-type union with array, numeric and length constraints.

    types is a nonempty tuple of exact classes. shape has fixed integer axes,
    None wildcards or Dimension ranges. dtypes accepts NumPy dtype names/strings
    and torch-prefixed names. Numeric bounds apply to every numeric array element;
    they reject NaN. finite additionally rejects infinities. Length is Python len,
    not bytes or array element count. validator is explicit trusted caller code.
    """

    name: str
    types: tuple[type[Any], ...]
    required: bool = True
    nullable: bool = False
    shape: tuple[int | Dimension | None, ...] | None = None
    dtypes: tuple[str, ...] | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    finite: bool = False
    min_length: int | None = None
    max_length: int | None = None
    validator: Callable[[Any], None] | None = None

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name:
            raise ValueError("field name must be a nonempty string")
        if (
            type(self.types) is not tuple
            or not self.types
            or any(not isinstance(kind, type) for kind in self.types)
            or len(set(self.types)) != len(self.types)
        ):
            raise TypeError("field types must be a nonempty tuple of distinct Python classes")
        for kind in self.types:
            if issubclass(kind, np.generic):
                raise TypeError(
                    f"field {self.name!r}: numpy.{kind.__name__} is a NumPy scalar type, which "
                    "is not stored; use the Python type (int, float, bool, complex) or "
                    "numpy.ndarray with dtypes=(...)"
                )
        arrays = np.ndarray in self.types or any(codec._is_tensor_type(kind) for kind in self.types)
        if (self.shape is not None or self.dtypes is not None) and not arrays:
            raise ValueError(
                f"field {self.name!r}: shape and dtypes constrain arrays; add numpy.ndarray "
                "or torch.Tensor to types"
            )
        if any(type(flag) is not bool for flag in (self.required, self.nullable, self.finite)):
            raise TypeError("field flags must be bool")
        if self.shape is not None and (
            type(self.shape) is not tuple
            or any(
                axis is not None
                and not isinstance(axis, Dimension)
                and (type(axis) is not int or axis < 0)
                for axis in self.shape
            )
        ):
            raise ValueError("shape must contain nonnegative integers, Dimension or None")
        if self.dtypes is not None and (
            type(self.dtypes) is not tuple
            or not self.dtypes
            or any(type(name) is not str or not name for name in self.dtypes)
        ):
            raise ValueError("dtypes must be a nonempty tuple of dtype strings")
        if self.dtypes is not None:
            for name in self.dtypes:
                if not self._known_dtype(name):
                    raise ValueError(f"field {self.name!r}: unknown dtype {name!r}")
        for bound in (self.minimum, self.maximum):
            if bound is not None and (
                type(bound) not in (int, float)
                or (type(bound) is float and not math.isfinite(bound))
            ):
                raise ValueError("numeric bounds must be finite int/float values or None")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        for bound in (self.min_length, self.max_length):
            if bound is not None and (type(bound) is not int or bound < 0):
                raise ValueError("length bounds must be nonnegative integers or None")
        if (
            self.min_length is not None
            and self.max_length is not None
            and self.min_length > self.max_length
        ):
            raise ValueError("min_length must not exceed max_length")
        if self.validator is not None and not callable(self.validator):
            raise TypeError("field validator must be callable")

    def _known_dtype(self, name: str) -> bool:
        """Whether a dtype name names a NumPy or (for Tensor fields) a PyTorch dtype."""
        if (
            np.ndarray in self.types
            and not name.startswith("torch.")
            and codec._resolve_named_numpy_dtype(name) is not None
        ):
            return True
        tensor = next((kind for kind in self.types if codec._is_tensor_type(kind)), None)
        if tensor is not None:
            import sys

            torch = sys.modules["torch"]  # loaded: its Tensor class is in types
            return isinstance(getattr(torch, name.removeprefix("torch."), None), torch.dtype)
        return False

    def _fail(self, constraint: str, message: str) -> NoReturn:
        raise SchemaConstraintError(
            f"field {self.name!r}: {message}", field=self.name, constraint=constraint
        )

    def validate(self, value: Any, *, allow_custom: bool = False, max_depth: int = 100) -> None:
        """Validate one present value; custom registration/consent follows validate_value.

        No input is converted. Nullable None skips shape/dtype/range/length checks
        but still reaches the caller validator. Tensor numeric checks can synchronize
        their device; this is explicit full-value validation, not header inspection.
        """
        from .schema import validate_value

        if type(allow_custom) is not bool:
            raise TypeError("allow_custom must be bool")
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        if value is None:
            if not (self.nullable or type(None) in self.types):
                self._fail("type", "None requires nullable=True or an explicit NoneType union")
            if self.validator is not None:
                self.validator(value)
            return
        if type(value) not in self.types:
            self._fail("type", f"expected exact types {self.types!r}, got {type(value)!r}")
        validate_value(value, allow_custom=allow_custom, max_depth=max_depth, name=self.name)
        numpy = type(value) is np.ndarray
        tensor = codec._is_tensor_type(type(value))
        if self.shape is not None:
            if not (numpy or tensor):
                self._fail("shape", "shape requires an array or Tensor")
            actual = tuple(value.shape)
            if len(actual) != len(self.shape):
                self._fail("shape", f"expected rank {len(self.shape)}, got {actual!r}")
            for index, (axis, size) in enumerate(zip(self.shape, actual, strict=True)):
                valid = (
                    axis is None
                    or (type(axis) is int and size == axis)
                    or (
                        isinstance(axis, Dimension)
                        and size >= axis.minimum
                        and (axis.maximum is None or size <= axis.maximum)
                    )
                )
                if not valid:
                    self._fail("shape", f"axis {index} expected {axis!r}, got {size}")
        if self.dtypes is not None:
            if not (numpy or tensor):
                self._fail("dtype", "dtype requires an array or Tensor")
            actual_dtype = str(value.dtype)
            valid_dtype = False
            for name in self.dtypes:
                if tensor:
                    valid_dtype |= actual_dtype == (
                        name if name.startswith("torch.") else f"torch.{name}"
                    )
                else:
                    try:
                        valid_dtype |= value.dtype == np.dtype(name)
                    except TypeError:
                        valid_dtype |= actual_dtype == name
            if not valid_dtype:
                self._fail("dtype", f"expected {self.dtypes!r}, got {actual_dtype!r}")
        if self.finite or self.minimum is not None or self.maximum is not None:
            if numpy:
                extension_numeric = value.dtype.type.__module__.split(".")[0] == "ml_dtypes"
                if (
                    value.dtype.kind not in "biuf"
                    and not extension_numeric
                    and not (
                        self.finite
                        and value.dtype.kind == "c"
                        and self.minimum is None
                        and self.maximum is None
                    )
                ):
                    self._fail("range", "numeric constraints require real numeric values")
                if self.finite and not np.all(np.isfinite(value)):
                    self._fail("finite", "values must be finite")
                if (self.minimum is not None or self.maximum is not None) and np.any(
                    np.isnan(value)
                ):
                    self._fail("range", "NaN does not satisfy numeric bounds")
                if self.minimum is not None and np.any(value < self.minimum):
                    self._fail("minimum", f"value is below {self.minimum}")
                if self.maximum is not None and np.any(value > self.maximum):
                    self._fail("maximum", f"value is above {self.maximum}")
            elif tensor:
                import torch

                if value.is_complex() and (self.minimum is not None or self.maximum is not None):
                    self._fail("range", "numeric bounds require real numeric values")
                if self.finite and not bool(torch.all(torch.isfinite(value))):
                    self._fail("finite", "values must be finite")
                if (self.minimum is not None or self.maximum is not None) and bool(
                    torch.any(torch.isnan(value))
                ):
                    self._fail("range", "NaN does not satisfy numeric bounds")
                if self.minimum is not None and bool(torch.any(value < self.minimum)):
                    self._fail("minimum", f"value is below {self.minimum}")
                if self.maximum is not None and bool(torch.any(value > self.maximum)):
                    self._fail("maximum", f"value is above {self.maximum}")
            else:
                if type(value) not in (int, float):
                    self._fail("range", "numeric constraints require int, float or numeric arrays")
                if self.finite and type(value) is float and not math.isfinite(value):
                    self._fail("finite", "value must be finite")
                if type(value) is float and math.isnan(value):
                    self._fail("range", "NaN does not satisfy numeric bounds")
                if self.minimum is not None and value < self.minimum:
                    self._fail("minimum", f"value is below {self.minimum}")
                if self.maximum is not None and value > self.maximum:
                    self._fail("maximum", f"value is above {self.maximum}")
        if self.min_length is not None or self.max_length is not None:
            try:
                length = len(cast(Sized, value))
            except TypeError:
                self._fail("length", "length constraints require a sized value")
            if self.min_length is not None and length < self.min_length:
                self._fail("min_length", f"length is below {self.min_length}")
            if self.max_length is not None and length > self.max_length:
                self._fail("max_length", f"length is above {self.max_length}")
        if self.validator is not None:
            self.validator(value)


@dataclass(frozen=True, slots=True)
class RecordSchema:
    """Field rules plus optional trusted cross-field validation, with no coercion.

    Unknown fields are rejected by default. ordered=True checks relative order of
    present declared fields; absent optional fields have no default inserted.
    Cross-field validation runs after all structural/field checks and propagates
    the callback's exceptions. Schema construction never opens storage or encodes.
    """

    fields: tuple[FieldSpec, ...]
    ordered: bool = False
    allow_extra: bool = False
    allow_custom: bool = False
    max_depth: int = 100
    validator: Callable[[Mapping[str, Any]], None] | None = None

    def __post_init__(self) -> None:
        if type(self.fields) is not tuple or any(
            not isinstance(field, FieldSpec) for field in self.fields
        ):
            raise TypeError("fields must be a tuple of FieldSpec instances")
        if len({field.name for field in self.fields}) != len(self.fields):
            raise ValueError("record field names must be unique")
        if any(
            type(flag) is not bool for flag in (self.ordered, self.allow_extra, self.allow_custom)
        ):
            raise TypeError("record schema flags must be bool")
        if type(self.max_depth) is not int or self.max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        if self.validator is not None and not callable(self.validator):
            raise TypeError("record validator must be callable")

    def validate(self, values: Mapping[str, Any], *, sample: str = "") -> None:
        """Validate mapping keys/fields, then explicit cross-field caller code."""
        from .schema import validate_value

        if not isinstance(values, Mapping) or any(type(name) is not str for name in values):
            raise TypeError("record schema requires a mapping with string keys")
        names = tuple(field.name for field in self.fields)
        actual = tuple(values)
        extra = set(actual) - set(names)
        if extra and not self.allow_extra:
            raise SchemaConstraintError(
                f"unexpected fields in {sample!r}: {sorted(extra)!r}",
                field=None,
                constraint="extra",
            )
        if self.ordered and tuple(name for name in actual if name in names) != tuple(
            name for name in names if name in values
        ):
            raise SchemaConstraintError(
                f"field order differs in {sample!r}", field=None, constraint="order"
            )
        for field in self.fields:
            if field.name not in values:
                if field.required:
                    raise SchemaConstraintError(
                        f"required field {field.name!r} is missing in {sample!r}",
                        field=field.name,
                        constraint="required",
                    )
            else:
                field.validate(
                    values[field.name], allow_custom=self.allow_custom, max_depth=self.max_depth
                )
        for name in extra:
            validate_value(
                values[name], allow_custom=self.allow_custom, max_depth=self.max_depth, name=name
            )
        if self.validator is not None:
            self.validator(values)


__all__ = ["Dimension", "FieldSpec", "RecordSchema", "SchemaConstraintError"]
