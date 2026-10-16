"""Readers with read_threads fetch the bodies of a batch concurrently before copying them."""

import pickle

import numpy as np
import pytest

import amber_lmdb.lmdb as storage
from amber_lmdb import LMDBReader, LMDBWriter, get_loc


def _write(path, count=6):
    with LMDBWriter(path, map_size=16 * 1024**2) as writer:
        for index in range(count):
            writer.write_ndarray(np.full(5000, index, dtype=np.int64), key="values", index=index)
        writer.commit()


def _record(monkeypatch):
    fetched = []
    touch = storage._touch_body

    def record(environment, location):
        fetched.append(location)
        touch(environment, location)

    monkeypatch.setattr(storage, "_touch_body", record)
    return fetched


def test_batch_reads_fetch_every_body_and_return_the_same_values(tmp_path, monkeypatch):
    path = tmp_path / "store.lmdb"
    _write(path)
    fetched = _record(monkeypatch)
    indices = [4, 0, 2, 2]
    with LMDBReader(path) as plain, LMDBReader(path, read_threads=3) as threaded:
        expected = plain.read_as_ndarray_many(["values"] * 4, indices=indices)
        assert fetched == []
        actual = threaded.read_as_ndarray_many(["values"] * 4, indices=indices)
        assert threaded.read_threads == 3
    assert sorted(fetched) == sorted(get_loc("values", index) for index in indices)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)


def test_single_reads_do_not_use_the_pool(tmp_path, monkeypatch):
    path = tmp_path / "store.lmdb"
    _write(path)
    fetched = _record(monkeypatch)
    with LMDBReader(path, read_threads=2) as reader:
        assert reader.read_as_ndarray("values", index=1)[0] == 1
        assert reader._read_pool is None
    assert fetched == []


def test_missing_keys_are_reported_by_the_read(tmp_path):
    path = tmp_path / "store.lmdb"
    _write(path, count=2)
    with LMDBReader(path, read_threads=2) as reader, pytest.raises(KeyError):
        reader.read_as_ndarray_many(["values"] * 2, indices=[0, 9])


def test_pool_is_closed_and_not_pickled(tmp_path):
    path = tmp_path / "store.lmdb"
    _write(path)
    reader = LMDBReader(path, read_threads=2)
    reader.read_as_ndarray_many(["values"] * 2, indices=[0, 1])
    pool = reader._read_pool
    assert pool is not None
    copy = pickle.loads(pickle.dumps(reader))
    assert copy._read_pool is None
    assert copy.read_threads == 2
    assert copy.read_as_ndarray_many(["values"] * 2, indices=[3, 5])[1][0] == 5
    copy.close()
    reader.close()
    assert reader._read_pool is None
    with pytest.raises(RuntimeError):
        pool.submit(int)


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_read_threads_must_be_a_nonnegative_integer(tmp_path, value):
    with pytest.raises(ValueError, match="read_threads"):
        LMDBReader(tmp_path / "store.lmdb", read_threads=value)
