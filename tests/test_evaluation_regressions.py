"""Reproduced empty-publication defect and request-local immutable header reuse."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import amber_lmdb.lmdb as storage
from amber_lmdb import (
    LMDBReader,
    LMDBWriter,
    get_loc,
    inspect_store,
    open_environment,
    verify_physical,
    verify_store,
)


@pytest.mark.parametrize("engine", [0, 1])
@pytest.mark.parametrize("verify", [False, True])
def test_empty_successful_writer_publishes_readable_valid_store(tmp_path, engine, verify):
    path = tmp_path / "empty"
    with LMDBWriter(path, map_size=1024**2, lib_version=engine, verify_on_publish=verify):
        pass
    assert inspect_store(path).data_entries == 0
    assert verify_store(path).ok
    assert verify_physical(path).ok
    with LMDBReader(path) as reader:
        assert reader.get_lib_version()[0] == engine
        with pytest.raises(KeyError):
            reader.read("missing")


def test_header_reuse_is_request_local_and_retains_order(tmp_path, monkeypatch):
    path = tmp_path / "headers"
    with LMDBWriter(path) as writer:
        writer.write_ndarray(np.zeros((2, 3), dtype=np.float32), key="a")
        writer.write_ndarray(np.ones((2, 3), dtype=np.float32), key="b")
        writer.write_ndarray(np.ones((3, 2), dtype=np.float32), key="c")
        writer.write_ndarray(np.ones((2, 3), dtype=np.float64), key="d")
    parse = storage.parse_array_header
    calls = []

    def count(header, **options):
        calls.append(header)
        return parse(header, **options)

    monkeypatch.setattr(storage, "parse_array_header", count)
    with LMDBReader(path) as reader:
        first = reader.read_array_headers_many(["a", "c", "b", "d", "a"])
        assert len(calls) == 3
        assert [header.shape for header in first] == [(2, 3), (3, 2), (2, 3), (2, 3), (2, 3)]
        assert [header.dtype for header in first] == [
            "float32",
            "float32",
            "float32",
            "float64",
            "float32",
        ]
        with pytest.raises(FrozenInstanceError):
            first[0].shape = (99,)
        assert reader.read_array_headers_many(["a", "b"]) == [first[0], first[2]]
        assert len(calls) == 4
    assert first[0].payload_bytes == 24


@pytest.mark.parametrize("fault", ["length", "tag"])
def test_header_reuse_does_not_hide_length_or_kind_corruption(tmp_path, fault):
    path = tmp_path / "corrupt"
    with LMDBWriter(path) as writer:
        for key in ("good", "bad"):
            writer.write_ndarray(np.zeros(3, dtype=np.float32), key=key)
    with open_environment(path, readonly=False) as handle:
        tags = handle.environment.open_db(storage._METADATA_DB_NAME, create=False)
        with handle.environment.begin(write=True) as transaction:
            if fault == "length":
                transaction.put(get_loc("bad"), transaction.get(get_loc("bad"))[:-1])
            else:
                transaction.put(get_loc("bad"), b"write_tensor", db=tags)
    with LMDBReader(path) as reader, pytest.raises(ValueError):
        reader.read_array_headers_many(["good", "bad"])
