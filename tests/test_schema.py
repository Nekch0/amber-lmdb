"""Generic schemas work without a feature-store or dataset metadata convention."""

from dataclasses import dataclass

import numpy as np
import pytest

from amber_lmdb import (
    ValueSchema,
    ValueTypeError,
    ValueValidationError,
    infer_schema,
    register_bytes_codec,
    validate_schema,
    validate_value,
)
from amber_lmdb.codec import StorageKind
from amber_lmdb.schema import dtype_name


def test_plain_record_schema_and_empty_schema():
    values = {"invoice": 12, "paid": False, "lines": [("book", 2)]}
    schema = infer_schema(values)
    assert schema.type_map == {"invoice": int, "paid": bool, "lines": list}
    validate_schema(values, schema)
    assert infer_schema({}).names == ()
    assert StorageKind is not None


@pytest.mark.parametrize("values", [{"invoice": True}, {}, {"invoice": 1, "extra": 2}])
def test_wrong_type_missing_and_extra_fields(values):
    with pytest.raises((ValueError, TypeError)):
        validate_schema(values, infer_schema({"invoice": 12}))


def test_explicit_schema_and_configurable_order():
    schema = ValueSchema(("x", "y"), (int, str), ("primitive", "str"), ordered=False)
    validate_schema({"y": "a", "x": 2}, schema)
    with pytest.raises(ValueError, match="fields differ"):
        validate_schema({"y": "a", "x": 2}, infer_schema({"x": 1, "y": "b"}))


@pytest.mark.parametrize(
    "options",
    [
        {"names": ("x", "x"), "types": (int, int), "storage_kinds": ("primitive", "primitive")},
        {"names": ("x",), "types": (), "storage_kinds": ()},
        {"names": ("x",), "types": (int,), "storage_kinds": ("dict",)},
    ],
)
def test_invalid_explicit_schema(options):
    with pytest.raises(ValueError):
        ValueSchema(**options)


def test_recursion_paths_cycles_and_shared_containers():
    shared = [1]
    assert validate_value([shared, shared]) == "list"
    shared.append(shared)
    with pytest.raises(ValueValidationError) as caught:
        validate_value({"nested": shared}, name="payload")
    assert caught.value.path == "$['nested'][1]"
    assert caught.value.name == "payload"
    assert caught.value.reason == "cycle"
    with pytest.raises(ValueValidationError, match="depth"):
        validate_value([[1]], max_depth=1)


def test_unsupported_values_report_mapping_key_paths_and_object_dtype():
    class Unregistered:
        pass

    with pytest.raises(ValueTypeError) as caught:
        validate_value({Unregistered(): 1})
    assert caught.value.path == "$.keys()[0]"
    with pytest.raises(ValueTypeError) as caught:
        validate_value(np.asarray([object()], dtype=object))
    assert caught.value.reason == "object_dtype"


def test_custom_types_need_explicit_opt_in_without_executing_encoder():
    @dataclass
    class Order:
        number: int

    def encoder(_value):
        pytest.fail("validation must not execute an encoder")

    register_bytes_codec("schema.order.no-encode", Order, encoder, lambda data: Order(1))
    with pytest.raises(ValueTypeError):
        infer_schema({"order": Order(1)})
    schema = infer_schema({"order": Order(1)}, allow_custom=True)
    validate_schema({"order": Order(2)}, schema)
    assert schema.storage_kinds == ("any",)


def test_dtype_helpers_keep_numpy_byte_order():
    assert dtype_name(np.zeros(2, dtype=">f4")) == ">f4"
    assert dtype_name(1) is None


def test_torch_dtype_helpers_are_optional():
    torch = pytest.importorskip("torch")
    from amber_lmdb.schema import torch_dtype_from_name

    assert dtype_name(torch.zeros(2)) == "float32"
    assert torch_dtype_from_name("torch.float32") is torch.float32
    with pytest.raises(TypeError):
        torch_dtype_from_name("zeros")
