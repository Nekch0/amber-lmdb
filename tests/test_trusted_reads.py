"""Readers built with verify_checksums=False skip only the body digest."""

import pickle

import numpy as np
import pytest

import amber_lmdb.lmdb as storage
from amber_lmdb import (
    IntegrityError,
    LMDBReader,
    LMDBWriter,
    RecordTable,
    ValueView,
    get_loc,
    infer_schema,
    open_environment,
    verify_store,
)
from amber_lmdb.lmdb import _METADATA_DB_NAME

CHECKSUMS = ["blake2b-256", "sha256"]


def _write(path, checksum="blake2b-256"):
    with LMDBWriter(path, map_size=1024**2, checksum=checksum) as writer:
        writer.write_ndarray(np.arange(3, dtype=np.int32), key="values")
        writer.write_str("text", key="label")
        writer.write_primitive(7, key="count")
        for index in range(3):
            writer.write_dict({"number": index}, key="rows", index=index)
        writer.write_primitive(5, key=("fields", "primitive", "size"))


def _flip_last_byte(path, key):
    with (
        open_environment(path, readonly=False) as handle,
        handle.environment.begin(write=True) as txn,
    ):
        raw = txn.get(get_loc(key))
        txn.put(get_loc(key), raw[:-1] + bytes([raw[-1] ^ 1]))


def test_readers_verify_by_default(tmp_path):
    path = tmp_path / "store"
    _write(path)
    _flip_last_byte(path, "values")
    with LMDBReader(path) as reader:
        assert reader.verify_checksums is True
        with pytest.raises(IntegrityError, match="checksum mismatch"):
            reader.read_as_ndarray("values")


@pytest.mark.parametrize("checksum", CHECKSUMS)
def test_trusted_reader_returns_same_length_changes_unverified(tmp_path, checksum):
    path = tmp_path / "store"
    _write(path, checksum)
    _flip_last_byte(path, "values")
    with LMDBReader(path, verify_checksums=False) as reader:
        assert reader.verify_checksums is False
        # The change is not detected: the caller has taken responsibility for trust.
        assert reader.read_as_ndarray("values").tolist() == [0, 1, 2 ^ (1 << 24)]
    # Store-level verification does not depend on any reader option.
    assert verify_store(path).error_count == 1


def _body_reads():
    schema = infer_schema({"size": 5})
    return {
        "read": lambda r: r.read("values"),
        "read_many": lambda r: r.read_many(["values", "label"]),
        "read_as_ndarray": lambda r: r.read_as_ndarray("values"),
        "read_as_str": lambda r: r.read_as_str("label"),
        "read_as_primitive": lambda r: r.read_as_primitive("count"),
        "read_as_primitive_many": lambda r: r.read_as_primitive_many(["count"]),
        "read_as_dict": lambda r: r.read_as_dict("rows", index=0),
        "read_as_dict_many": lambda r: r.read_as_dict_many(["rows"] * 3, indices=[0, 1, 2]),
        "iter_read_pages": lambda r: list(r.iter_read_pages(["values", "label"])),
        "iter_dict_pages": lambda r: list(r.iter_dict_pages(["rows"] * 3, indices=[0, 1, 2])),
        "RecordTable.get_many": lambda r: RecordTable(r, "rows", dict, count=3).get_many([0, 1, 2]),
        "ValueView.get": lambda r: ValueView(r, schema, namespace=("fields",)).get("size"),
    }


@pytest.mark.parametrize("name", list(_body_reads()))
@pytest.mark.parametrize("verify", [True, False])
def test_option_applies_to_every_body_read_path(tmp_path, monkeypatch, name, verify):
    path = tmp_path / "store"
    _write(path)
    calls = []
    real = storage._verify_checksum
    monkeypatch.setattr(
        storage, "_verify_checksum", lambda *args: calls.append(args[0]) or real(*args)
    )
    with LMDBReader(path, verify_checksums=verify) as reader:
        _body_reads()[name](reader)
    assert bool(calls) is verify


@pytest.mark.parametrize("damage", ["length", "missing-tag", "wrong-reader"])
def test_trusted_reader_keeps_tag_and_length_checks(tmp_path, damage):
    path = tmp_path / "store"
    _write(path)
    if damage != "wrong-reader":
        with open_environment(path, readonly=False) as handle:
            tags = handle.environment.open_db(_METADATA_DB_NAME)
            with handle.environment.begin(write=True) as txn:
                if damage == "length":
                    txn.put(get_loc("label"), b"longer text")
                else:
                    txn.delete(get_loc("label"), db=tags)
    with LMDBReader(path, verify_checksums=False) as reader:
        if damage == "length":
            with pytest.raises(IntegrityError, match="length"):
                reader.read_as_str("label")
        elif damage == "missing-tag":
            with pytest.raises(ValueError, match="no writer tag"):
                reader.read_as_str("label")
        else:
            with pytest.raises(ValueError, match="Invalid write/read method pair"):
                reader.read_as_ndarray("label")


def test_trusted_option_survives_pickle_and_refresh(tmp_path):
    path = tmp_path / "store"
    _write(path)
    _flip_last_byte(path, "values")
    with LMDBReader(path, verify_checksums=False) as reader:
        reader.read_as_str("label")
        clone = pickle.loads(pickle.dumps(reader))
        try:
            assert clone.verify_checksums is False
            assert clone.read_as_ndarray("values")[-1] == 2 ^ (1 << 24)
        finally:
            clone.close()
        reader.refresh()
        assert reader.verify_checksums is False
        assert reader.read_as_ndarray("values")[-1] == 2 ^ (1 << 24)


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_verify_checksums_must_be_bool(tmp_path, value):
    with pytest.raises(TypeError, match="verify_checksums must be bool"):
        LMDBReader(tmp_path / "store", verify_checksums=value)
