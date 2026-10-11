"""Concrete invoice records and namespaced values built from generic APIs."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from amber_lmdb import (
    LMDBReader,
    LMDBWriter,
    RecordTable,
    SourceChangedError,
    ValueView,
    get_active_envs,
    infer_schema,
    register_bytes_codec,
)


@dataclass(frozen=True)
class Invoice:
    number: int
    paid: bool


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "invoices"
    with LMDBWriter(path) as writer:
        for index in range(4):
            writer.write_dict(
                {"number": index + 10, "paid": bool(index % 2)}, key="invoices", index=index
            )
        writer.write_ndarray(np.arange(4), key=("account", "ndarray", "vector"), index=2)
        writer.write_str("customer", key="custom-name")
        writer.commit()
    return path


def test_namespaced_view_and_custom_key_layout(store):
    schema = infer_schema({"vector": np.zeros(4, dtype=np.int64)})
    with LMDBReader(store) as reader:
        view = ValueView(reader, schema, namespace=("account",), index=2)
        array = view.get_as("vector", np.ndarray)
        np.testing.assert_array_equal(array, np.arange(4))
        custom = ValueView(
            reader, infer_schema({"name": ""}), key_factory=lambda name, kind: "custom-name"
        )
        assert custom.get_as("name", str) == "customer"
        with pytest.raises(KeyError):
            view.get("unknown")
        with pytest.raises(TypeError):
            view.get_as("vector", list)
    np.testing.assert_array_equal(array, np.arange(4))


def test_custom_view_requires_explicit_decode_opt_in(tmp_path):
    @dataclass(frozen=True)
    class Account:
        number: int

    register_bytes_codec(
        "collections.account.v1",
        Account,
        lambda value: {"number": value.number},
        lambda value: Account(**value),
    )
    path = tmp_path / "accounts"
    with LMDBWriter(path) as writer:
        writer.write(Account(7), key=("any", "account"))
        writer.commit()
    with LMDBReader(path) as reader:
        schema = infer_schema({"account": Account(0)}, allow_custom=True)
        view = ValueView(reader, schema)
        with pytest.raises(ValueError):
            view.get("account")
        assert view.get_as("account", Account, allow_custom=True) == Account(7)


def test_view_has_no_implicit_padding(store):
    with LMDBReader(store) as reader:
        view = ValueView(reader, infer_schema({"vector": np.arange(4)}), namespace=("account",))
        with pytest.raises(KeyError):
            view.get("vector")


def test_wrong_declared_type_is_rejected(store):
    with LMDBReader(store) as reader:
        view = ValueView(
            reader, infer_schema({"name": 1}), key_factory=lambda name, kind: "custom-name"
        )
        with pytest.raises(TypeError, match="expected"):
            view.get("name")


def test_record_factory_schema_batches_and_duplicates(store, monkeypatch):
    with LMDBReader(store) as reader:
        batches = []
        original = reader.iter_dict_pages
        inside = False

        def many(keys, **options):
            nonlocal inside
            pages = original(keys, **options)
            while True:
                inside = True
                try:
                    page = next(pages)
                except StopIteration:
                    return
                finally:
                    inside = False
                batches.append(len(page))
                yield page

        def factory(row):
            assert not inside
            return Invoice(**row)

        monkeypatch.setattr(reader, "iter_dict_pages", many)
        table = RecordTable(
            reader,
            "invoices",
            factory,
            count=4,
            schema=infer_schema({"number": 1, "paid": False}),
            batch_size=2,
        )
        assert len(table) == 4
        assert table.get_many([3, 1, 3]) == [
            Invoice(13, True),
            Invoice(11, True),
            Invoice(13, True),
        ]
        assert batches == [2, 1]
        assert list(table.iter_records(offset=1, limit=2)) == [
            Invoice(11, True),
            Invoice(12, False),
        ]
        assert table.get(0) == Invoice(10, False)


def test_invalid_indices_and_empty_input_open_nothing(tmp_path):
    reader = LMDBReader(tmp_path / "missing")
    table = RecordTable(reader, "invoices", lambda row: row, count=1)
    assert table.get_many([]) == []
    assert list(table.iter_records(limit=0)) == []
    for index in [-1, 1]:
        with pytest.raises(IndexError):
            table.get(index)
    with pytest.raises(TypeError):
        table.get(True)
    with pytest.raises(IndexError):
        table.get_many([0, 1])
    assert not get_active_envs()


def test_schema_failure_does_not_execute_factory(store):
    with LMDBReader(store) as reader:
        table = RecordTable(
            reader,
            "invoices",
            lambda row: pytest.fail("factory ran"),
            count=4,
            schema=infer_schema({"number": "", "paid": False}),
        )
        with pytest.raises(TypeError):
            table.get(0)


def test_refresh_between_pages_rejects_mixed_generations(store):
    with LMDBReader(store) as reader:
        table = RecordTable(reader, "invoices", lambda row: row, count=4, batch_size=1)
        iterator = table.iter_records()
        assert next(iterator)["number"] == 10
        with LMDBWriter(store, overwrite=True) as writer:
            writer.write_dict({"number": 99, "paid": True}, key="invoices", index=0)
            writer.commit()
        reader.refresh()
        with pytest.raises(SourceChangedError):
            next(iterator)


@pytest.mark.parametrize("namespace,index", [("not-a-tuple", None), ((1,), None), ((), -1)])
def test_invalid_view_configuration(namespace, index):
    with pytest.raises((TypeError, ValueError)):
        ValueView(LMDBReader(Path("missing")), infer_schema({}), namespace=namespace, index=index)
