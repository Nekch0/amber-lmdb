"""Explicit field constraints, cross-field callbacks and persistent None hits."""

import numpy as np
import pytest

from amber_lmdb import (
    CacheResult,
    DerivedIndexCache,
    Dimension,
    FieldSpec,
    LMDBReader,
    LMDBWriter,
    RecordSchema,
    RecordTable,
    SchemaConstraintError,
    ValueTypeError,
    register_bytes_codec,
    validate_schema,
)
from amber_lmdb.codec import any_to_bytes, bytes_to_any


def _schema(callback=None):
    return RecordSchema(
        (
            FieldSpec("id", (int, str), min_length=None),
            FieldSpec(
                "values",
                (np.ndarray,),
                shape=(Dimension(1, 4), 2),
                dtypes=("float32",),
                minimum=0,
                maximum=10,
                finite=True,
            ),
            FieldSpec("description", (str,), required=False, nullable=True, max_length=12),
        ),
        validator=callback,
    )


def test_optional_nullable_union_shape_dtype_range_and_cross_field_callbacks(tmp_path):
    calls = []
    schema = _schema(lambda row: calls.append(row["id"]))
    record = {"id": "one", "values": np.ones((2, 2), dtype=np.float32), "description": None}
    validate_schema(record, schema)
    validate_schema({"id": 2, "values": np.zeros((1, 2), dtype=np.float32)}, schema)
    assert calls == ["one", 2]
    path = tmp_path / "schema"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_dict(record, key="records", index=0)
    with LMDBReader(path) as reader:
        table = RecordTable(reader, "records", lambda row: row["id"], count=1, schema=schema)
        assert table.get(0) == "one"
    assert calls == ["one", 2, "one"]


@pytest.mark.parametrize(
    "field,value,constraint",
    [
        ("id", True, "type"),
        ("values", np.ones((0, 2), dtype=np.float32), "shape"),
        ("values", np.ones((5, 2), dtype=np.float32), "shape"),
        ("values", np.ones((2, 3), dtype=np.float32), "shape"),
        ("values", np.ones((2, 2), dtype=np.float64), "dtype"),
        ("values", np.full((2, 2), -1, dtype=np.float32), "minimum"),
        ("values", np.full((2, 2), 11, dtype=np.float32), "maximum"),
        ("values", np.full((2, 2), np.nan, dtype=np.float32), "finite"),
        ("description", "x" * 13, "max_length"),
    ],
)
def test_constraints_reject_wrong_values_before_cross_field_callback(field, value, constraint):
    calls = []
    row = {"id": 1, "values": np.ones((2, 2), dtype=np.float32)}
    row[field] = value
    with pytest.raises(SchemaConstraintError) as error:
        validate_schema(row, _schema(lambda record: calls.append(record)))
    assert error.value.field == field and error.value.constraint == constraint
    assert calls == []


def test_missing_extra_order_fixed_scalar_and_empty_shapes():
    with pytest.raises(SchemaConstraintError, match="required field"):
        validate_schema({}, _schema())
    with pytest.raises(SchemaConstraintError, match="unexpected fields"):
        validate_schema(
            {"id": 1, "values": np.ones((2, 2), dtype=np.float32), "extra": 1}, _schema()
        )
    schema = RecordSchema(
        (FieldSpec("a", (int,)), FieldSpec("b", (str,), required=False)), ordered=True
    )
    with pytest.raises(SchemaConstraintError, match="order"):
        validate_schema({"b": "b", "a": 1}, schema)
    validate_schema({"a": 1}, schema)
    FieldSpec("scalar", (np.ndarray,), shape=()).validate(np.array(1))
    FieldSpec("empty", (np.ndarray,), shape=(0, None), finite=True, minimum=0).validate(
        np.empty((0, 3))
    )
    schema = RecordSchema((), allow_extra=True)
    validate_schema({"supported": [1, None]}, schema)
    with pytest.raises(ValueTypeError):
        validate_schema({"unsupported": object()}, schema)


@pytest.mark.parametrize(
    "options",
    [
        {"types": ()},
        {"types": [int]},
        {"types": (int, int)},
        {"required": 1},
        {"shape": (True,)},
        {"shape": (-1,)},
        {"dtypes": ()},
        {"minimum": float("nan")},
        {"minimum": 3, "maximum": 2},
        {"min_length": True},
        {"min_length": 3, "max_length": 2},
        {"validator": 1},
    ],
)
def test_invalid_field_declarations_fail_before_io(options):
    with pytest.raises((TypeError, ValueError)):
        FieldSpec("field", **{"types": (int,), **options})


@pytest.mark.parametrize("options", [{"allow_custom": 1}, {"max_depth": -1}, {"max_depth": True}])
def test_nullable_validation_still_checks_public_options(options):
    with pytest.raises((TypeError, ValueError)):
        FieldSpec("nullable", (str,), nullable=True).validate(None, **options)


@pytest.mark.parametrize("bounds", [(True, None), (2, 1), (-1, None)])
def test_dimension_declarations_are_strict(bounds):
    with pytest.raises(ValueError):
        Dimension(*bounds)


def test_custom_validation_requires_consent_without_running_encoder():
    class Custom:
        pass

    calls = []
    register_bytes_codec(
        "completion.schema.custom",
        Custom,
        lambda value: calls.append(value) or None,
        lambda raw: Custom(),
    )
    schema = RecordSchema((FieldSpec("custom", (Custom,)),))
    with pytest.raises(ValueTypeError):
        schema.validate({"custom": Custom()})
    RecordSchema(schema.fields, allow_custom=True).validate({"custom": Custom()})
    assert calls == []


def test_callback_exceptions_propagate_and_only_follow_built_in_validation():
    def validate(row):
        if row["end"] < row["start"]:
            raise RuntimeError("end precedes start")

    schema = RecordSchema(
        (FieldSpec("start", (int,), minimum=0), FieldSpec("end", (int,), minimum=0)),
        validator=validate,
    )
    with pytest.raises(RuntimeError, match="precedes"):
        schema.validate({"start": 2, "end": 1})
    with pytest.raises(SchemaConstraintError):
        schema.validate({"start": -1, "end": 1})


def test_tensor_constraints_are_optional():
    torch = pytest.importorskip("torch")
    field = FieldSpec(
        "tensor", (torch.Tensor,), shape=(None, 2), dtypes=("float32",), minimum=0, finite=True
    )
    field.validate(torch.ones((3, 2), dtype=torch.float32))
    with pytest.raises(SchemaConstraintError):
        field.validate(torch.full((3, 2), float("nan")))


def test_extension_numeric_dtype_constraints_when_installed():
    ml_dtypes = pytest.importorskip("ml_dtypes")
    field = FieldSpec(
        "extension", (np.ndarray,), dtypes=("bfloat16",), minimum=0, maximum=2, finite=True
    )
    field.validate(np.ones(2, dtype=ml_dtypes.bfloat16))
    with pytest.raises(SchemaConstraintError):
        field.validate(np.full(2, 3, dtype=ml_dtypes.bfloat16))


def test_persisted_none_is_an_explicit_hit_not_a_rebuild(tmp_path):
    cache = DerivedIndexCache(tmp_path, encode=any_to_bytes, decode=bytes_to_any)
    identity = {"source": "none", "builder": 1}
    assert cache.lookup(identity) == CacheResult(False)
    assert cache.get_or_build(identity, lambda: None) is None
    assert cache.lookup(identity) == CacheResult(True, None)
    assert cache.get_or_build(identity, lambda: pytest.fail("None hit was rebuilt")) is None
    assert cache.lookup(identity).value is None
    calls = []
    assert cache.lookup(identity, validate=lambda value: calls.append(value)).hit
    assert calls == [None]
    damaged = cache.path_for(identity)
    damaged.write_bytes(damaged.read_bytes()[:-1])
    assert cache.lookup(identity) == CacheResult(False)
    assert cache.get_or_build(identity, lambda: 7) == 7


def test_none_validator_and_memory_only_building(tmp_path):
    cache = DerivedIndexCache(tmp_path, encode=any_to_bytes, decode=bytes_to_any)
    cache.publish({}, None)

    def reject(value):
        if value is None:
            raise ValueError("None forbidden by this caller")

    assert not cache.lookup({}, validate=reject).hit
    assert cache.get_or_build({}, lambda: 8, validate=reject) == 8
    memory = DerivedIndexCache(None, encode=any_to_bytes, decode=bytes_to_any)
    calls = []
    assert memory.get_or_build({}, lambda: calls.append(1)) is None
    assert memory.get_or_build({}, lambda: calls.append(2)) is None
    assert calls == [1, 2] and not memory.lookup({}).hit
