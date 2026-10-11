"""Behaviour added after the usability evaluation: recovery, generations, readers, values."""

import gc
import json
import logging
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import amber_lmdb.lmdb as storage
from amber_lmdb import (
    MAX_BATCH_ITEMS,
    DerivedIndexCache,
    FieldSpec,
    KeyNotFoundError,
    LMDBOverlayReader,
    LMDBReader,
    LMDBWriter,
    OperationCancelled,
    OverlayConflictError,
    ReaderClosedError,
    ReaderMismatchError,
    ReadLimitError,
    RecordSchema,
    RecordTable,
    StoreNotFoundError,
    ValueSchema,
    ValueView,
    __version__,
    array_statistics,
    fingerprint_store,
    get_active_envs,
    inspect_generations,
    inspect_store,
    merge_stores,
    prune_generations,
    read_history,
    set_current,
    unregister_bytes_codec,
    verify_store,
)
from amber_lmdb.cli import main
from amber_lmdb.codec import any_to_bytes, bytes_to_any, register_bytes_codec

ARRAY = np.arange(600, dtype=np.float32).reshape(200, 3)


def _write(path, value, overwrite=False, **options):
    with LMDBWriter(path, map_size=16 * 1024**2, overwrite=overwrite, **options) as writer:
        writer.write_primitive(value, key="value")
    return (path / "CURRENT").read_text().strip()


# --------------------------------------------------------------- history, recovery


def test_history_records_every_switch_in_order(tmp_path):
    path = tmp_path / "store"
    first = _write(path, 1)
    second = _write(path, 2, overwrite=True)
    history = read_history(path)
    assert history is not None
    assert [item.generation for item in history] == [first, second]
    assert [item.sequence for item in history] == [1, 2]
    assert history[0].published_ns <= history[1].published_ns
    assert read_history(tmp_path / "absent") is None


def test_missing_current_is_refused_and_nothing_is_deleted(tmp_path):
    path = tmp_path / "store"
    old = _write(path, 1)
    current = _write(path, 2, overwrite=True)
    (path / "CURRENT").unlink()
    with pytest.raises(ValueError, match="set_current"):
        _write(path, 3, overwrite=True)
    assert (path / "versions" / old).is_dir() and (path / "versions" / current).is_dir()
    publication = set_current(path, current)
    assert publication.generation == current and publication.sequence == 3
    with LMDBReader(path) as reader:
        assert reader.read("value") == 2


def test_root_without_history_is_refused(tmp_path):
    path = tmp_path / "store"
    old = _write(path, 1)
    (path / "CURRENT").unlink()
    (path / "HISTORY").unlink()
    with pytest.raises(ValueError, match="neither CURRENT nor HISTORY"):
        _write(path, 2, overwrite=True)
    assert (path / "versions" / old).is_dir()


def test_generation_left_by_an_interrupted_first_write_is_removed(tmp_path):
    path = tmp_path / "store"
    unpublished = _write(path, 1)
    (path / "CURRENT").unlink()
    (path / "HISTORY").write_bytes(b"")  # the switch never happened
    current = _write(path, 2, overwrite=True)
    assert not (path / "versions" / unpublished).exists()
    assert [item.generation for item in read_history(path) or ()] == [current]


def test_set_current_rolls_back_and_verifies(tmp_path):
    path = tmp_path / "store"
    old = _write(path, 1)
    _write(path, 2, overwrite=True)
    with LMDBReader(path) as pinned:
        assert pinned.read("value") == 2
        set_current(path, old, verify=True)
        assert pinned.read("value") == 2  # an open reader keeps its generation
    with LMDBReader(path) as reader:
        assert reader.read("value") == 1
    with pytest.raises(ValueError, match="generation name"):
        set_current(path, "latest")
    with pytest.raises(FileNotFoundError):
        set_current(path, "v-" + "0" * 32)
    with LMDBWriter(path, map_size=2**20, overwrite=True), pytest.raises(BlockingIOError):
        set_current(path, old)


def test_prune_generations_keeps_current_and_the_newest(tmp_path):
    path = tmp_path / "store"
    names = [_write(path, value, overwrite=value > 0) for value in range(4)]
    dry = prune_generations(path, keep_last=1)
    assert [row.name for row in dry] == names[:2] and {row.status for row in dry} == {"eligible"}
    with pytest.raises(ValueError, match="same_host"):
        prune_generations(path, keep_last=1, dry_run=False)
    applied = prune_generations(path, keep_last=1, dry_run=False, same_host=True)
    assert [row.status for row in applied] == ["deleted", "deleted"]
    assert sorted(item.name for item in inspect_generations(path).generations) == sorted(names[2:])
    assert prune_generations(path, keep_last=5) == ()


def test_inventory_reports_a_missing_current_instead_of_raising(tmp_path):
    path = tmp_path / "store"
    name = _write(path, 1)
    (path / "CURRENT").unlink()
    inventory = inspect_generations(path)
    assert inventory.current is None and inventory.current_error is not None
    assert [item.name for item in inventory.generations] == [name]
    assert not inventory.generations[0].current


# --------------------------------------------------------------- reader lifecycle


def test_pickled_reader_pins_the_opened_generation(tmp_path):
    path = tmp_path / "store"
    first = _write(path, 1)
    reader = LMDBReader(path)
    unopened = pickle.loads(pickle.dumps(reader))
    assert reader.read("value") == 1 and reader.generation == first
    _write(path, 2, overwrite=True)
    copy = pickle.loads(pickle.dumps(reader))
    assert copy.generation == first and copy.read("value") == 1
    assert unopened.generation is None and unopened.read("value") == 2
    copy.refresh()
    assert copy.read("value") == 2
    for item in (reader, copy, unopened):
        item.close()
    assert not get_active_envs()


def test_explicit_generation_survives_refresh(tmp_path):
    path = tmp_path / "store"
    old = _write(path, 1)
    _write(path, 2, overwrite=True)
    with LMDBReader(path, generation=old) as reader:
        assert reader.read("value") == 1
        reader.refresh()
        assert reader.read("value") == 1 and reader.generation == old
    with pytest.raises(ValueError, match="generation"):
        LMDBReader(path, generation="v-1")


def test_closed_reader_refuses_reads_and_refresh_does_not_close(tmp_path):
    path = tmp_path / "store"
    _write(path, 1)
    reader = LMDBReader(path)
    assert reader.read("value") == 1 and not reader.closed
    reader.refresh()
    assert reader.read("value") == 1
    reader.close()
    assert reader.closed and "closed" in repr(reader)
    with pytest.raises(ReaderClosedError):
        reader.read("value")
    reader.close()


def test_concurrent_first_reads_open_one_reference(tmp_path):
    path = tmp_path / "store"
    _write(path, 7)
    reader = LMDBReader(path)
    barrier = threading.Barrier(8)

    def work(_):
        barrier.wait()
        return reader.read("value")

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(work, range(8))) == [7] * 8
    assert reader._env_key is not None
    assert storage._ENV_REF_COUNT[reader._env_key] == 1
    reader.close()
    assert not get_active_envs()


def test_dropped_reader_releases_its_reference(tmp_path):
    path = tmp_path / "store"
    _write(path, 1)
    reader = LMDBReader(path)
    assert reader.read("value") == 1
    assert get_active_envs()
    del reader
    gc.collect()
    assert not get_active_envs()


def test_non_cooperative_reader_takes_no_locks(tmp_path):
    path = tmp_path / "store"
    name = _write(path, 1)
    lease = path / ".coordination" / f"{name}.lock"
    with LMDBReader(path, cooperative=False) as reader:
        assert reader.read("value") == 1 and not reader.cooperative
        assert not lease.exists()
        with LMDBReader(path) as other, pytest.raises(ValueError, match="cooperative"):
            other.read("value")
    assert verify_store(path, cooperative=False).ok


# --------------------------------------------------------------- errors


def test_error_types_carry_decoded_keys(tmp_path):
    path = tmp_path / "store"
    with LMDBWriter(path, map_size=2**20) as writer:
        writer.write_str("text", key=("s01", "name"), index=3)
    with LMDBReader(path) as reader:
        with pytest.raises(KeyNotFoundError) as missing:
            reader.read(("s01", "name"), 4)
        assert isinstance(missing.value, KeyError)
        assert missing.value.key == ("s01", "name") and missing.value.index == 4
        assert "('s01', 'name') index 4" in str(missing.value)
        with pytest.raises(ReaderMismatchError) as mismatch:
            reader.read_as_dict(("s01", "name"), 3)
        assert isinstance(mismatch.value, ValueError)
        assert mismatch.value.writer_tag == "write_str"
        assert "read_as_str" in mismatch.value.valid_readers
    with pytest.raises(StoreNotFoundError) as absent, LMDBReader(tmp_path / "absent") as reader:
        reader.read("value")
    assert isinstance(absent.value, FileNotFoundError)


# --------------------------------------------------------------- overlay and merge


def _layer(path, **values):
    with LMDBWriter(path, map_size=16 * 1024**2) as writer:
        for key, indices in values.items():
            for index in indices:
                writer.write_primitive(index, key=key, index=index)
    return path


def test_overlay_conflicts_name_the_layers_that_hold_the_entry(tmp_path):
    first = _layer(tmp_path / "a", rows=[0, 1])
    second = _layer(tmp_path / "b", other=[0])
    third = _layer(tmp_path / "c", rows=[1])
    with LMDBOverlayReader([first, second, third], verify_threads=2) as overlay:
        for operation in (
            lambda: list(overlay.iter_indices("rows")),
            lambda: list(overlay.iter_keys()),
            overlay.check_disjoint,
        ):
            with pytest.raises(OverlayConflictError) as conflict:
                operation()
            assert conflict.value.layers == (first, third)
        assert overlay.verify_threads == 2 and overlay.value_limits is None
        assert overlay.cooperative and not overlay.closed
        assert len(overlay.generations) == 3 and "LMDBOverlayReader" in repr(overlay)
    assert overlay.closed


def test_merge_stores_copies_every_entry_and_refuses_conflicts(tmp_path):
    base = tmp_path / "base"
    with LMDBWriter(base, map_size=16 * 1024**2) as writer:
        writer.write_ndarray(ARRAY, key="chunked", chunk_rows=32)
        writer.write_dict({"a": 1}, key="meta")
    feature = _layer(tmp_path / "feature", pitch=[0, 1, 2])
    merged = tmp_path / "merged"
    assert merge_stores([base, feature], merged, manifest=True) == 5
    with LMDBReader(merged) as reader:
        assert np.array_equal(reader.read_as_ndarray("chunked"), ARRAY)
        assert reader.read_as_dict("meta") == {"a": 1}
        assert reader.read("pitch", 2) == 2
        manifest = reader.manifest()
        assert manifest is not None and manifest.entries == 5
    assert verify_store(merged).ok
    with LMDBOverlayReader([base, feature]) as overlay:
        assert fingerprint_store(merged).entries == overlay.check_disjoint()
    with pytest.raises(ValueError, match="distinct"):
        merge_stores([feature, feature.parent / "feature"], tmp_path / "x")
    clash = _layer(tmp_path / "clash", pitch=[1])
    with pytest.raises(OverlayConflictError):
        merge_stores([feature, clash], tmp_path / "never")
    assert not (tmp_path / "never").exists()
    event = threading.Event()
    event.set()
    with pytest.raises(OperationCancelled):
        merge_stores([base, feature], tmp_path / "cancelled", cancel=event)
    assert not (tmp_path / "cancelled" / "CURRENT").exists()


# --------------------------------------------------------------- typed reads


def test_read_as_any_decodes_every_messagepack_writer(tmp_path):
    path = tmp_path / "store"
    values = {"dict": {"a": (1, 2)}, "list": [1, "x"], "set": {1, 2}, "number": 2.5}
    with LMDBWriter(path, map_size=2**20) as writer:
        writer.write_dict(values["dict"], key="dict")
        writer.write_sequence(values["list"], key="list")
        writer.write_set(values["set"], key="set")
        writer.write_primitive(values["number"], key="number")
        writer.write_str("text", key="text")
    with LMDBReader(path) as reader:
        for key, value in values.items():
            assert reader.read_as_any(key) == value
        with pytest.raises(ReaderMismatchError):
            reader.read_as_any("text")


def test_batch_arguments_are_validated_with_clear_messages(tmp_path):
    path = tmp_path / "store"
    _layer(path, rows=range(4))
    with LMDBReader(path) as reader:
        with pytest.raises(TypeError, match="list"):
            reader.read_many(key for key in ["rows"])
        with pytest.raises(ValueError, match="one entry per key"):
            reader.read_many(["rows", "rows"], indices=[0])
        with pytest.raises(ValueError, match="different lengths"):
            list(reader.iter_read_pages(["rows"] * 3, indices=[0, 1]))
        assert reader.read_many(["rows"] * 4, indices=np.arange(4)) == [0, 1, 2, 3]
        assert reader.read_many(["rows"], indices=[np.int64(2)], max_bytes=None) == [2]
        assert reader.contains_many(["rows"], indices=[1], max_items=MAX_BATCH_ITEMS) == [True]
        with pytest.raises(ValueError, match="max_items"):
            reader.read_many(["rows"], indices=[0], max_items=MAX_BATCH_ITEMS + 1)
        with pytest.raises(ReadLimitError):
            reader.read("rows", 0, max_bytes=8)
        with pytest.raises(ReadLimitError):
            reader.read_as_primitive("rows", 0, max_bytes=8)
        assert reader.read_as_primitive("rows", 0, max_bytes=4096) == 0


def test_array_reads_cast_safely_and_can_be_writable(tmp_path):
    path = tmp_path / "store"
    with LMDBWriter(path, map_size=16 * 1024**2) as writer:
        writer.write_ndarray(ARRAY, key="plain")
        writer.write_ndarray(ARRAY, key="chunked", chunk_rows=16)
        writer.write_ndarray(np.float32(2.5), key="scalar")
        writer.write_str("a", key="names", index=0)
        writer.write_str("b", key="names", index=1)
    with LMDBReader(path) as reader:
        for key in ("plain", "chunked"):
            assert not reader.read_as_ndarray(key).flags.writeable
            writable = reader.read_as_ndarray(key, writable=True)
            writable[0, 0] = -1
            assert reader.read_as_ndarray(key)[0, 0] == 0
            with pytest.raises(TypeError, match="casting='unsafe'"):
                reader.read_as_ndarray(key, dtype=np.int32)
            assert reader.read_as_ndarray(key, dtype=np.int32, casting="unsafe")[1, 0] == 3
            assert reader.read_as_ndarray(key, dtype=np.float64).dtype == np.float64
        arrays = reader.read_as_ndarray_many(["plain", "chunked"], writable=True)
        assert all(array.flags.writeable for array in arrays)
        assert reader.read_as_ndarray("scalar").shape == ()
        assert reader.read_as_str_many(["names", "names"], indices=[1, 0]) == ["b", "a"]
        with pytest.raises(ValueError, match="casting"):
            reader.read_as_ndarray("plain", casting="lossy")


@pytest.mark.parametrize("compression", ["zlib", "lzma", "zstd"])
def test_compressed_arrays_round_trip(tmp_path, compression):
    if compression == "zstd" and sys.version_info < (3, 14):
        with pytest.raises(ValueError, match=r"3\.14"):
            LMDBWriter(tmp_path / "x").write_ndarray(ARRAY, key="a", compression="zstd")
        return
    path = tmp_path / "store"
    with LMDBWriter(path, map_size=16 * 1024**2) as writer:
        writer.write_ndarray(ARRAY, key="plain", compression=compression)
        writer.write_ndarray(ARRAY, key="chunked", chunk_rows=16, compression=compression)
        writer.write_ndarray(ARRAY[:0], key="empty", chunk_rows=4, compression=compression)
    with LMDBReader(path) as reader:
        for key in ("plain", "chunked"):
            assert np.array_equal(reader.read_as_ndarray(key), ARRAY)
            assert np.array_equal(reader.read_array_rows(key, start=30, stop=70), ARRAY[30:70])
            header = reader.read_array_headers_many([key])[0]
            assert header.compression == compression and header.payload_bytes == ARRAY.nbytes
        assert reader.read_as_ndarray("empty").shape == (0, 3)
    assert verify_store(path).ok
    with LMDBReader(path, verify_checksums=False) as reader:
        assert np.array_equal(reader.read_array_rows("plain", start=5, stop=9), ARRAY[5:9])
    samples = inspect_store(path, entry_limit=10).samples
    assert {sample.error for sample in samples} == {None}


def test_compression_options_are_validated(tmp_path):
    with LMDBWriter(tmp_path / "store", map_size=2**20) as writer:
        with pytest.raises(ValueError, match="compression must be one of"):
            writer.write_ndarray(ARRAY, key="a", compression="gzip")
        with pytest.raises(ValueError, match="level"):
            writer.write_ndarray(ARRAY, key="a", compression="zlib", compression_level=42)
        with pytest.raises(TypeError, match="compression_level"):
            writer.write_ndarray(ARRAY, key="a", compression="lzma", compression_level="9")


def test_damaged_compressed_payload_is_detected(tmp_path):
    from amber_lmdb import codec

    body = bytearray(codec.ndarray_to_bytes(ARRAY, compression="zlib"))
    body[-5] ^= 0xFF
    with pytest.raises(ValueError, match="compressed array payload"):
        codec.bytes_to_ndarray(bytes(body))
    raw = codec.ndarray_to_bytes(ARRAY)
    assert raw[4] == 2 and codec.ndarray_to_bytes(ARRAY, compression="zlib")[4] == 3


def test_writers_reject_values_that_would_change_silently(tmp_path):
    with LMDBWriter(tmp_path / "store", map_size=2**20) as writer:
        with pytest.raises(TypeError, match="mask"):
            writer.write_ndarray(np.ma.masked_array([1, 2], mask=[0, 1]), key="m")
        with pytest.raises(TypeError, match=r"np\.asarray"):
            writer.write_ndarray([1, 2], key="l")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="write_str needs a str"):
            writer.write_str(b"bytes", key="s")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match=r"torch\.Tensor"):
            writer.write_tensor(ARRAY, key="t")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match=r"\.item\(\)"):
            writer.write_any({"x": np.float32(1)}, key="a")
        with pytest.raises(TypeError, match=r"\.item\(\)"):
            writer.write_primitive(np.int64(1), key="p")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            writer.write_str("x", "positional-key")  # type: ignore[misc]


def test_value_writer_limits_apply_at_write_time(tmp_path):
    with LMDBWriter(tmp_path / "store", map_size=2**20) as writer:
        deep: list[object] = []
        node = deep
        for _ in range(101):
            node.append([])
            node = node[0]  # type: ignore[assignment]
        with pytest.raises(ValueError, match="depth"):
            writer.write_sequence(deep, key="deep")


# --------------------------------------------------------------- records and schemas


def test_record_table_is_a_bounded_sequence(tmp_path):
    path = tmp_path / "store"
    with LMDBWriter(path, map_size=2**20) as writer:
        for index in range(5):
            writer.write_dict({"id": index}, key="rows", index=index)
        writer.write_dict({"id": 9}, key="gappy", index=0)
        writer.write_dict({"id": 9}, key="gappy", index=2)
    with LMDBReader(path) as reader:
        table = RecordTable(reader, "rows", batch_size=2)
        assert len(table) == 5 and repr(table) == "RecordTable(key='rows', count=5)"
        assert table[0] == {"id": 0} and table[-1] == {"id": 4}
        assert table[np.int64(3)] == {"id": 3}
        assert table[1:4] == [{"id": 1}, {"id": 2}, {"id": 3}]
        assert [row["id"] for row in table] == [0, 1, 2, 3, 4]
        with pytest.raises(IndexError):
            table[5]
        with pytest.raises(IndexError):
            table.get(-1)
        with pytest.raises(TypeError):
            table[True]
        ids = RecordTable(reader, "rows", lambda row: row["id"], count=2)
        assert list(ids) == [0, 1]
        with pytest.raises(ValueError, match="pass count="):
            RecordTable(reader, "gappy")
        with pytest.raises(TypeError, match="LMDBReader"):
            RecordTable(object(), "rows")  # type: ignore[call-overload]


def test_schemas_are_checked_where_they_are_built(tmp_path):
    with pytest.raises(TypeError, match="NumPy scalar"):
        FieldSpec("x", (np.float32,))
    with pytest.raises(ValueError, match="constrain arrays"):
        FieldSpec("x", (int,), shape=(2,))
    with pytest.raises(ValueError, match="unknown dtype"):
        FieldSpec("x", (np.ndarray,), dtypes=("float33",))
    FieldSpec("x", (np.ndarray,), dtypes=("float32", "<i8"))
    path = tmp_path / "store"
    _write(path, 1)
    record = RecordSchema((FieldSpec("x", (int,)),))
    with LMDBReader(path) as reader, pytest.raises(TypeError, match="RecordTable"):
        ValueView(reader, record)  # type: ignore[arg-type]
    with LMDBReader(path) as reader:
        view = ValueView(reader, ValueSchema(("x",), (int,), ("primitive",)))
        assert "fields=1" in repr(view)


# --------------------------------------------------------------- caches


def test_cache_identities_refuse_what_json_would_change(tmp_path):
    cache = DerivedIndexCache(tmp_path, encode=any_to_bytes, decode=bytes_to_any)
    with pytest.raises(TypeError, match="tuple"):
        cache.publish({"shape": (1, 2)}, 1)
    with pytest.raises(TypeError, match="keys must be str"):
        cache.publish({1: "x"}, 1)  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="finite"):
        cache.lookup({"x": float("nan")})
    with pytest.raises(TypeError, match="unsupported type"):
        cache.lookup({"x": np.int64(1)})
    path = tmp_path / "store"
    _write(path, 1)
    with LMDBReader(path) as reader:
        snapshot = reader.get_snapshot()
    cache.publish(snapshot, "built")
    assert cache.path_for(snapshot) == cache.path_for(asdict(snapshot))
    assert cache.lookup(asdict(snapshot)).value == "built"
    assert "DerivedIndexCache" in repr(cache)


def test_codec_registration_can_be_undone():
    class Point:
        def __init__(self, x):
            self.x = x

    register_bytes_codec("tests.usability.point", Point, lambda p: p.x, Point)
    unregister_bytes_codec("tests.usability.point")
    with pytest.raises(TypeError, match="register"):
        any_to_bytes(Point(1))
    with pytest.raises(KeyError):
        unregister_bytes_codec("tests.usability.point")


# --------------------------------------------------------------- scans


def test_scans_read_named_generations(tmp_path):
    path = tmp_path / "store"
    with LMDBWriter(path, map_size=16 * 1024**2) as writer:
        writer.write_ndarray(ARRAY, key="array")
    old = (path / "CURRENT").read_text().strip()
    before = fingerprint_store(path)
    _write(path, 2, overwrite=True)
    assert fingerprint_store(path, generation=old).digest == before.digest
    assert verify_store(path, generation=old).ok
    assert array_statistics(path, generation=old).array_entries == 1
    assert array_statistics(path).array_entries == 0
    assert Path(inspect_store(path, generation=old).snapshot.generation).name == old


def test_verify_issues_carry_keys_and_progress_counts_entries(tmp_path):
    path = tmp_path / "store"
    _layer(path, rows=range(5))
    reports = []
    report = verify_store(path, progress=reports.append)
    assert report.ok and reports[-1].finished
    assert reports[-1].entries_total == reports[-1].entries_done == 5
    from amber_lmdb import ValidationIssue
    from amber_lmdb.keys import get_loc

    issue = ValidationIssue(get_loc(("a", "b"), 3), "error", "broken")
    assert issue.key == ("a", "b") and issue.index == 3
    assert ValidationIssue(None, "error", "x").key is None
    assert ValidationIssue(b"junk", "error", "x").index is None


# --------------------------------------------------------------- writer


def test_writer_exit_commit_is_quiet_and_staging_is_separate(tmp_path, caplog):
    path = tmp_path / "store"
    with caplog.at_level(logging.WARNING, logger="amber_lmdb"), LMDBWriter(path) as writer:
        assert writer.path == path.absolute() and writer.staging_path is not None
        writer.write_primitive(1, key="value")
        writer.commit()
        writer.commit()  # nothing to commit: silent
        writer.write_primitive(2, key="value")
    assert not caplog.records
    assert writer.staging_path is None and "published" in repr(writer)


# --------------------------------------------------------------- command line


def test_cli_generation_management_and_errors(tmp_path, capsys):
    path = tmp_path / "store"
    old = _write(path, 1)
    _write(path, 2, overwrite=True)
    assert main(["set-current", str(path), old]) == 0
    assert json.loads(capsys.readouterr().out)["generation"] == old
    assert main(["delete-generations", str(path), "--keep-last", "0"]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"]
    assert main(["manifest", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["manifest"] is None
    assert main(["fingerprint", str(path), "--generation", old, "--read-only-media"]) == 0
    capsys.readouterr()
    assert main(["set-current", str(path), "v-" + "0" * 32]) == 3
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["type"] == "FileNotFoundError"
    assert "FileNotFoundError" in captured.err
    with pytest.raises(SystemExit) as version:
        main(["--version"])
    assert version.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_cli_reports_decoded_keys(tmp_path, capsys):
    path = tmp_path / "store"
    with LMDBWriter(path, map_size=16 * 1024**2) as writer:
        writer.write_ndarray(ARRAY, key=("s01", "f0"), index=2)
    assert main(["stats", str(path)]) == 0
    group = json.loads(capsys.readouterr().out)["groups"][0]
    assert group["smallest_key"] == {"key": ["s01", "f0"], "index": 2}
    assert main(["doctor", str(path), "--full", "--max-value-bytes", "100"]) == 1
    issue = json.loads(capsys.readouterr().out)["issues"][0]
    assert issue["key"] == ["s01", "f0"] and issue["index"] == 2


# --------------------------------------------------------------- PyTorch


def test_arrays_read_as_tensors_without_warnings(tmp_path, recwarn):
    torch = pytest.importorskip("torch")
    from amber_lmdb import as_tensor

    path = tmp_path / "store"
    with LMDBWriter(path, map_size=16 * 1024**2) as writer:
        writer.write_ndarray(ARRAY, key="plain")
        writer.write_ndarray(ARRAY, key="chunked", chunk_rows=16, compression="zlib")
        writer.write_ndarray(np.zeros(2, dtype=[("a", "<i4")]), key="structured")
        writer.write_tensor(torch.arange(4), key="tensor")
    with LMDBReader(path) as reader:
        for key in ("plain", "chunked"):
            tensor = reader.read_as_tensor(key)
            assert tensor.shape == ARRAY.shape and torch.equal(tensor, torch.from_numpy(ARRAY))
            tensor[0, 0] = -1  # writable and private
            assert reader.read_as_ndarray(key)[0, 0] == 0
        assert reader.read_as_tensor("tensor", dtype=torch.float64).dtype == torch.float64
        with pytest.raises(TypeError, match="PyTorch"):
            reader.read_as_tensor("structured")
        frozen = reader.read_as_ndarray("plain")
        copied = as_tensor(frozen)
        copied[0, 0] = 5
        assert frozen[0, 0] == 0
        shared = reader.read_as_ndarray("plain", writable=True)
        as_tensor(shared)[0, 0] = 5
        assert shared[0, 0] == 5
        assert as_tensor(np.arange(3, dtype=">i4")).tolist() == [0, 1, 2]
    assert not [warning for warning in recwarn if "not writable" in str(warning.message)]
