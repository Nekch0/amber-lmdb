"""Threaded digest verification and the coverage summary of whole-store validation."""

import json
import os
import pickle
import threading
import warnings

import numpy as np
import pytest

import amber_lmdb.lmdb as lmdb_module
from amber_lmdb import (
    IntegrityError,
    LMDBOverlayReader,
    LMDBReader,
    LMDBWriter,
    get_loc,
    open_environment,
    register_bytes_codec,
    verify_store,
)
from amber_lmdb.cli import main

LARGE = [np.full(70_000, index, np.float32) for index in range(8)]  # 280 KB bodies


def _store(path):
    with LMDBWriter(path, map_size=64 * 1024**2) as writer:
        for index, value in enumerate(LARGE):
            writer.write_ndarray(value, key="large", index=index)
        writer.write_ndarray(
            np.arange(4096 * 64, dtype=np.float32).reshape(4096, 64), key="rows", chunk_rows=256
        )
        for index in range(4):
            writer.write_dict({"id": index}, key="small", index=index)
    return path


def test_option_validation():
    for threads in (-1, True, 1.5):
        with pytest.raises(ValueError, match="verify_threads"):
            LMDBReader("x", verify_threads=threads)


def test_threaded_results_match_and_run_in_the_pool(tmp_path, monkeypatch):
    path = _store(tmp_path / "store.lmdb")
    names = set()
    original = lmdb_module._verify_checksum

    def recording(*args):
        names.add(threading.current_thread().name)
        return original(*args)

    monkeypatch.setattr(lmdb_module, "_verify_checksum", recording)
    with LMDBReader(path) as plain, LMDBReader(path, verify_threads=4) as threaded:
        assert threaded.verify_threads == 4
        keys, indices = ["large"] * 8 + ["small"] * 4, list(range(8)) + list(range(4))
        expected = plain.read_many(keys, indices=indices)
        names.clear()
        got = threaded.read_many(keys, indices=indices)
        assert all(
            np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
            for a, b in zip(got, expected, strict=True)
        )
        assert any(name.startswith("amber-lmdb-verify") for name in names)
        rows = threaded.read_array_rows("rows", start=100, stop=900)
        assert np.array_equal(rows, plain.read_array_rows("rows", start=100, stop=900))
        assert np.array_equal(threaded.read_as_ndarray("rows"), plain.read_as_ndarray("rows"))
        names.clear()
        threaded.read_many(["large", "small"], indices=[0, 0])  # one large body: no pool
        assert names == {threading.current_thread().name}
        pool = threaded._pool
        assert pool is not None
        restored = pickle.loads(pickle.dumps(threaded))
        assert restored._pool is None and restored.verify_threads == 4
        restored.close()
    assert threaded._pool is None
    with LMDBReader(path, verify_threads=4, verify_checksums=False) as trusted:
        trusted.read_many(["large"] * 8, indices=range(8))
        assert trusted._pool is None
    with LMDBOverlayReader([path], verify_threads=2) as overlay:
        assert all(layer.verify_threads == 2 for layer in overlay._layers)
        assert len(overlay.read_many(["large"] * 8, indices=range(8))) == 8


def test_first_corrupt_entry_in_request_order_raises(tmp_path):
    path = _store(tmp_path / "store.lmdb")
    with (
        open_environment(path, readonly=False) as handle,
        handle.environment.begin(write=True) as txn,
    ):
        for index in (5, 2):
            location = get_loc("large", index)
            body = bytearray(txn.get(location))
            body[-1] ^= 1
            txn.put(location, bytes(body))
    with LMDBReader(path, verify_threads=4) as reader:
        with pytest.raises(IntegrityError, match=get_loc("large", 2).hex()):
            reader.read_many(["large"] * 8, indices=range(8))
        with pytest.raises(IntegrityError, match=get_loc("large", 5).hex()):
            reader.read_many(["large"] * 3, indices=[0, 5, 2])
        assert len(reader.read_many(["large"] * 2, indices=[0, 1])) == 2


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
def test_forked_child_gets_its_own_pool(tmp_path):
    path = _store(tmp_path / "store.lmdb")
    with LMDBReader(path, verify_threads=2) as reader:
        reader.read_many(["large"] * 8, indices=range(8))
        with warnings.catch_warnings():
            # Forking with live pool threads is what this test checks; Python warns.
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:  # pragma: no cover - runs in the child
            try:
                values = reader.read_many(["large"] * 8, indices=range(8))
                os._exit(0 if len(values) == 8 else 1)
            except BaseException:
                os._exit(2)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0


class _Point:
    def __init__(self, x):
        self.x = x


register_bytes_codec("test.verify_threads.point", _Point, lambda point: point.x, _Point)


def test_coverage_summary_counts_reasons_and_tags(tmp_path, monkeypatch, capsys):
    path = tmp_path / "coverage.lmdb"
    with LMDBWriter(path, map_size=64 * 1024**2) as writer:
        writer.write_any(_Point(1), key="custom")
        writer.write_any(_Point(2), key="custom", index=1)
        writer.write_str("x" * 5000, key="long")
        writer.write_dict({"a": 1}, key="meta")
        writer.write_sequence([1, 2], key="sequence")
    report = verify_store(path, max_value_bytes=1000)
    assert not report.complete
    assert report.unverified_by_reason == {
        "custom decoding not allowed": 2,
        "value exceeds the copy limit": 1,
    }
    assert report.entries_by_tag == {
        "write_any": 2,
        "write_dict": 1,
        "write_sequence": 1,
        "write_str": 1,
    }
    original = LMDBReader._decode_entry

    def missing(tag, value, **options):
        if tag == "write_sequence":
            error = ImportError("No module named 'torch'")
            error.name = "torch"
            raise error
        return original(tag, value, **options)

    monkeypatch.setattr(LMDBReader, "_decode_entry", staticmethod(missing))
    report = verify_store(path, allow_custom=True)
    assert report.unverified_by_reason == {"optional dependency unavailable: torch": 1}
    monkeypatch.undo()
    assert main(["doctor", str(path), "--full", "--max-value-bytes", "1000"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["unverified_by_reason"]["value exceeds the copy limit"] == 1
    assert output["entries_by_tag"]["write_dict"] == 1
