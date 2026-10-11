"""Aggregate owned copies, byte-aware pages and transaction-free callbacks/yields."""

import multiprocessing as mp
from contextlib import contextmanager

import numpy as np
import pytest

import amber_lmdb.lmdb as storage
from amber_lmdb import (
    LMDBReader,
    LMDBWriter,
    ReadLimitError,
    RecordTable,
    SourceChangedError,
    get_loc,
    inspect_store,
    open_environment,
)


@pytest.mark.parametrize(
    "method",
    ["read_many", "read_as_dict_many", "read_as_primitive_many", "read_array_headers_many"],
)
@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_byte_limits_validate_before_open_even_for_empty_requests(tmp_path, method, limit):
    reader = LMDBReader(tmp_path / "absent")
    with pytest.raises((ValueError, TypeError)):
        getattr(reader, method)([], max_bytes=limit)
    assert reader._env is None


@pytest.mark.parametrize(
    "method,values",
    [
        ("read_many", ["abcd", "efgh"]),
        ("read_as_dict_many", [{"a": 1}, {"b": 2}]),
        ("read_as_primitive_many", [1, 2]),
    ],
)
def test_batch_budget_is_checked_before_any_body_copy_and_counts_duplicates(
    tmp_path, monkeypatch, method, values
):
    path = tmp_path / "budget"
    with LMDBWriter(path, map_size=1024**2) as writer:
        for index, value in enumerate(values):
            writer.write(value, key="row", index=index)
    with LMDBReader(path) as reader:
        checked = reader._get_checked_entry
        copies = []
        sizes = []

        class Observed:
            def __init__(self, value):
                self.value = value

            def __len__(self):
                return len(self.value)

            def __bytes__(self):
                copies.append(len(self.value))
                return bytes(self.value)

        def observe(*args, **kwargs):
            metadata, value = checked(*args, **kwargs)
            sizes.append(metadata.metadata_bytes + len(value))
            return metadata, Observed(value)

        monkeypatch.setattr(reader, "_get_checked_entry", observe)
        # Determine one entry's canonical encoded cost without copying its body.
        env = reader._get_env()
        with env.begin(buffers=True) as txn:
            metadata, body = checked(txn, get_loc("row", 0), "read")
            cost = len(body) + metadata.metadata_bytes
        read = getattr(reader, method)
        with pytest.raises(ReadLimitError) as error:
            read(["row", "row"], indices=[0, 0], max_bytes=cost)
        assert error.value.kind == "bytes" and error.value.limit == cost
        assert error.value.required == 2 * cost and copies == []
        assert read(["row", "row"], indices=[0, 0], max_bytes=2 * cost) == [values[0], values[0]]
        assert len(copies) == 2


def test_header_budget_counts_integrity_metadata_and_never_copies_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "headers"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_ndarray(np.arange(64, dtype=np.float32), key="array")
    with LMDBReader(path) as reader:
        with reader._get_env().begin(buffers=True) as txn:
            metadata, value = reader._get_checked_entry(txn, get_loc("array"), "read_as_shape")
            cost = metadata.metadata_bytes + len(storage._header_view(value, 64 * 1024))
        with pytest.raises(ReadLimitError):
            reader.read_array_headers_many(["array", "array"], max_bytes=cost)
        assert len(reader.read_array_headers_many(["array", "array"], max_bytes=2 * cost)) == 2
    with pytest.raises(ReadLimitError):
        inspect_store(path, entry_limit=1, max_batch_bytes=cost - 1)
    assert len(inspect_store(path, entry_limit=1, max_batch_bytes=cost).samples) == 1
    with pytest.raises(ReadLimitError) as error:
        LMDBReader(path).read_array_headers_many(["array"] * 2, max_items=1)
    assert error.value.kind == "items"


def test_stream_pages_and_checksums_decode_after_transaction_and_survive_closure(
    tmp_path, monkeypatch
):
    path = tmp_path / "pages"
    with LMDBWriter(path, map_size=1024**2) as writer:
        for index in range(4):
            writer.write_str("x" * 20, key="row", index=index)
    with LMDBReader(path) as reader:
        env = reader._get_env()
        active = []
        verify = storage._verify_checksum
        decode = reader._decode_entry

        class Proxy:
            def __getattr__(self, name):
                return getattr(env, name)

            @contextmanager
            def begin(self, **options):
                assert options["buffers"]
                with env.begin(**options) as txn:
                    active.append(True)
                    try:
                        yield txn
                    finally:
                        active.pop()

        def checked_verify(*args):
            assert not active
            verify(*args)

        def checked_decode(*args, **kwargs):
            assert not active
            return decode(*args, **kwargs)

        monkeypatch.setattr(reader, "_get_env", lambda: Proxy())
        monkeypatch.setattr(storage, "_verify_checksum", checked_verify)
        monkeypatch.setattr(reader, "_decode_entry", checked_decode)
        pages = list(reader.iter_read_pages(["row"] * 5, indices=[3, 1, 1, 0, 2], max_bytes=150))
        assert [len(page) for page in pages] == [2, 2, 1]
        assert not active
    assert [value for page in pages for value in page] == ["x" * 20] * 5


def test_stream_empty_opens_nothing_and_oversized_single_entry_fails(tmp_path):
    assert list(LMDBReader(tmp_path / "absent").iter_read_pages([])) == []
    path = tmp_path / "oversized"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_str("large", key="value")
    with LMDBReader(path) as reader, pytest.raises(ReadLimitError):
        list(reader.iter_read_pages(["value"], max_bytes=1))


def test_stream_refresh_and_in_place_identity_changes_abort(tmp_path, monkeypatch):
    path = tmp_path / "replace"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_primitive(1, key="value")
    with LMDBReader(path) as reader:
        pages = reader.iter_read_pages(["value"] * 2, max_items=1)
        assert next(pages) == [1]
        with LMDBWriter(path, overwrite=True, map_size=1024**2) as writer:
            writer.write_primitive(2, key="value")
        reader.refresh()
        with pytest.raises(SourceChangedError):
            next(pages)
        pages = reader.iter_read_pages(["value"] * 2, max_items=1)
        assert next(pages) == [2]
        snapshot = reader.get_snapshot()
        from dataclasses import replace

        monkeypatch.setattr(
            reader, "get_snapshot", lambda: replace(snapshot, last_txnid=snapshot.last_txnid + 1)
        )
        with pytest.raises(SourceChangedError):
            next(pages)


def _commit_same_body(path):
    with (
        open_environment(path, readonly=False) as handle,
        handle.environment.begin(write=True) as transaction,
    ):
        location = get_loc("row", 0)
        transaction.put(location, transaction.get(location))


def test_actual_in_place_commit_between_pages_is_detected(tmp_path):
    path = tmp_path / "in-place"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_primitive(1, key="row", index=0)
    with LMDBReader(path) as reader:
        pages = reader.iter_read_pages(["row"] * 2, indices=[0, 0], max_items=1)
        assert next(pages) == [1]
        process = mp.get_context("spawn").Process(target=_commit_same_body, args=(str(path),))
        try:
            process.start()
            process.join(20)
            assert process.exitcode == 0
            with pytest.raises(SourceChangedError):
                next(pages)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(10)


def test_record_table_uses_byte_aware_pages_with_order_and_factory_outside_transactions(tmp_path):
    path = tmp_path / "records"
    with LMDBWriter(path, map_size=1024**2) as writer:
        for index in range(4):
            writer.write_dict({"id": index, "text": "x" * 60}, key="records", index=index)
    with LMDBReader(path) as reader:
        table = RecordTable(reader, "records", lambda row: row["id"], count=4, max_bytes=200)
        assert table.get_many([3, 1, 1, 0]) == [3, 1, 1, 0]
        assert list(table.iter_records(offset=1, limit=2)) == [1, 2]
