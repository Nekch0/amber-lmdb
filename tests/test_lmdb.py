"""Tests for typed storage, publication and reader lifecycle behavior."""

import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from frozendict import frozendict

from amber_lmdb import register_bytes_codec
from amber_lmdb.lmdb import (
    LMDBReader,
    LMDBWriter,
    get_active_envs,
    get_loc,
    open_environment,
)


@dataclass(frozen=True)
class _LMDBPoint:
    x: int
    y: int


@pytest.fixture
def ds_path(tmp_path) -> Path:
    return tmp_path / "dataset.lmdb"


@pytest.fixture(autouse=True)
def _clean_env_cache():
    """Close cached environments after each test to prevent connection leaks."""
    yield
    import amber_lmdb.lmdb as db

    db._clear_env_cache()


def test_public_api_is_owned_by_lmdb_module():
    assert get_loc.__module__ == "amber_lmdb.keys"
    assert LMDBWriter.__module__ == "amber_lmdb.lmdb"
    assert LMDBReader.__module__ == "amber_lmdb.lmdb"
    assert open_environment.__module__ == "amber_lmdb.lmdb"


class TestGetLoc:
    def test_str(self):
        assert isinstance(get_loc("features"), bytes)

    def test_str_with_index(self):
        assert get_loc("features", 3) != get_loc(("features", "3"))

    def test_path(self):
        assert get_loc(Path("a/b")) == get_loc("a/b")  # POSIX form on every OS

    def test_tuple(self):
        assert get_loc(("split", "train", "audio")) != get_loc("split/train/audio")

    def test_tuple_with_index(self):
        assert get_loc(("split", "train", "audio"), index=3) != get_loc(
            ("split", "train", "audio", "3")
        )

    def test_component_boundaries_and_unicode_are_bijective(self):
        keys = [
            ("a/b", "c"),
            ("a", "b/c"),
            ("", "日本語/🚀"),
            ("\x00", ""),
        ]
        assert len({get_loc(key) for key in keys}) == len(keys)

    @pytest.mark.parametrize("index", [-1, True, 1.5])
    def test_invalid_index_raises(self, index):
        with pytest.raises((TypeError, ValueError)):
            get_loc("key", index=index)

    def test_invalid_tuple_component_raises(self):
        with pytest.raises(TypeError):
            get_loc(("valid", 1))  # type: ignore[arg-type]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            get_loc(123)  # type: ignore[arg-type]


class TestWriterLifecycle:
    def test_writes_to_tmp_then_renames(self, ds_path):
        with LMDBWriter(ds_path) as w:
            temporary = w.staging_path
            assert temporary is not None and w.path == ds_path
            assert temporary.parent == ds_path.parent
            assert temporary != ds_path.with_name(ds_path.name + ".tmp")
            assert temporary.exists()
            w.write_str("x", key="k")
            w.commit()
        assert w.path == ds_path and w.staging_path is None
        assert ds_path.exists()
        assert not temporary.exists()

    def test_auto_commit_on_exit(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_str("hello", key="k")  # Context exit must commit without an explicit call.
        with LMDBReader(ds_path) as r:
            assert r.read_as_str("k") == "hello"

    def test_existing_path_raises_without_overwrite(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_str("x", key="k")
        with pytest.raises(FileExistsError), LMDBWriter(ds_path) as w:
            w.write_str("y", key="k")

    def test_overwrite_true_replaces(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_str("old", key="k")
        with LMDBWriter(ds_path, overwrite=True) as w:
            w.write_str("new", key="k2")
        with LMDBReader(ds_path) as r:
            assert r.read_as_str("k2") == "new"

    def test_exception_aborts_and_cleans_tmp(self, ds_path):
        temporary = None
        with pytest.raises(RuntimeError), LMDBWriter(ds_path) as w:
            temporary = w.staging_path
            w.write_str("x", key="k")
            raise RuntimeError("boom")
        assert not ds_path.exists()  # A failed write must not create the final dataset.
        assert temporary is not None and not temporary.exists()

    def test_concurrent_writer_is_rejected_without_touching_first(self, ds_path):
        first = LMDBWriter(ds_path)
        second = LMDBWriter(ds_path)
        first.__enter__()
        first_temporary = first.staging_path
        try:
            with pytest.raises(BlockingIOError):
                second.__enter__()
            assert first_temporary is not None and first_temporary.exists()
            assert second.staging_path is None
        finally:
            first.__exit__(RuntimeError, RuntimeError("cleanup"), None)

    def test_publish_failure_preserves_previous_version(self, ds_path, monkeypatch):
        with LMDBWriter(ds_path) as writer:
            writer.write_str("old", key="key")

        import amber_lmdb.lmdb as db

        real_replace = db.os.replace

        def fail_current(source, destination):
            if Path(destination).name == "CURRENT":
                raise OSError("publish failed")
            return real_replace(source, destination)

        monkeypatch.setattr(db.os, "replace", fail_current)
        with (
            pytest.raises(OSError, match="publish failed"),
            LMDBWriter(ds_path, overwrite=True) as writer,
        ):
            writer.write_str("new", key="key")
        with LMDBReader(ds_path) as reader:
            assert reader.read_as_str("key") == "old"

    @pytest.mark.parametrize("stage", ["versions_sync", "mkstemp", "write", "pointer_sync"])
    def test_prepublication_failure_removes_unpublished_output(self, ds_path, monkeypatch, stage):
        """Failures before publication must leave only the old generation available."""
        import amber_lmdb.lmdb as db

        with LMDBWriter(ds_path) as writer:
            writer.write_str("old", key="key")
        previous = set((ds_path / "versions").iterdir())
        original_sync = db._fsync_directory

        def fail(*args, **kwargs):
            raise OSError("injected publication failure")

        def fail_versions(path):
            if Path(path).name == "versions":
                fail()
            return original_sync(path)

        with (
            pytest.raises(OSError, match="injected publication failure"),
            LMDBWriter(ds_path, overwrite=True) as writer,
        ):
            writer.write_str("new", key="key")
            if stage == "versions_sync":
                monkeypatch.setattr(db, "_fsync_directory", fail_versions)
            elif stage == "mkstemp":
                monkeypatch.setattr(db.tempfile, "mkstemp", fail)
            elif stage == "write":
                monkeypatch.setattr(db.os, "write", fail)
            else:
                original_mkstemp = db.tempfile.mkstemp

                def fail_pointer_sync(*args, **kwargs):
                    result = original_mkstemp(*args, **kwargs)
                    monkeypatch.setattr(db.os, "fsync", fail)
                    return result

                monkeypatch.setattr(db.tempfile, "mkstemp", fail_pointer_sync)
        assert set((ds_path / "versions").iterdir()) == previous
        assert not list(ds_path.glob(".CURRENT.*.tmp"))
        with LMDBReader(ds_path) as reader:
            assert reader.read_as_str("key") == "old"

    def test_fsync_failure_after_pointer_switch_keeps_published_version(self, ds_path, monkeypatch):
        with LMDBWriter(ds_path) as writer:
            writer.write_str("old", key="key")

        import amber_lmdb.lmdb as db

        real_fsync_directory = db._fsync_directory

        def fail_root(path):
            if Path(path) == ds_path:
                raise OSError("directory fsync failed")
            return real_fsync_directory(path)

        monkeypatch.setattr(db, "_fsync_directory", fail_root)
        with (
            pytest.raises(OSError, match="directory fsync failed"),
            LMDBWriter(ds_path, overwrite=True) as writer,
        ):
            writer.write_str("new", key="key")
        with LMDBReader(ds_path) as reader:
            assert reader.read_as_str("key") == "new"

    def test_initial_publish_failure_is_recoverable(self, ds_path, monkeypatch):
        import amber_lmdb.lmdb as db

        real_replace = db.os.replace

        def fail_current(source, destination):
            if Path(destination).name == "CURRENT":
                raise OSError("publish failed")
            return real_replace(source, destination)

        monkeypatch.setattr(db.os, "replace", fail_current)
        with pytest.raises(OSError, match="publish failed"), LMDBWriter(ds_path) as writer:
            writer.write_str("unpublished", key="key")

        monkeypatch.setattr(db.os, "replace", real_replace)
        with LMDBWriter(ds_path) as writer:
            writer.write_str("recovered", key="key")
        with LMDBReader(ds_path) as reader:
            assert reader.read_as_str("key") == "recovered"

    def test_orphan_staging_is_recovered(self, ds_path):
        orphan = ds_path.parent / f".{ds_path.name}.{'0' * 32}.tmp"
        orphan.mkdir()
        (orphan / "partial").write_bytes(b"partial")

        with LMDBWriter(ds_path) as writer:
            assert not orphan.exists()
            writer.write_str("complete", key="key")

    def test_unrecognized_existing_directory_is_preserved(self, ds_path):
        ds_path.mkdir()
        sentinel = ds_path / "keep.txt"
        sentinel.write_text("keep", encoding="utf-8")

        with pytest.raises(ValueError, match="FORMAT marker"), LMDBWriter(ds_path):
            pass
        assert sentinel.read_text(encoding="utf-8") == "keep"

    def test_writer_instance_cannot_be_reused(self, ds_path):
        writer = LMDBWriter(ds_path)
        with writer:
            writer.write_str("value", key="key")
        with pytest.raises(RuntimeError, match="cannot be reused"):
            writer.__enter__()

    def test_reader_rejects_unknown_dataset_format(self, ds_path):
        with LMDBWriter(ds_path) as writer:
            writer.write_str("value", key="key")
        (ds_path / "FORMAT").write_text("unknown\n", encoding="ascii")

        with (
            pytest.raises(ValueError, match="unsupported LMDB dataset format"),
            LMDBReader(ds_path) as reader,
        ):
            reader.read_as_str("key")

    def test_commit_then_write_starts_new_txn(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_str("a", key="k1")
            w.commit()
            w.write_str("b", key="k2")  # This write starts a new transaction.
            w.commit()
        with LMDBReader(ds_path) as r:
            assert r.read_as_str("k1") == "a"
            assert r.read_as_str("k2") == "b"


class TestPrimitiveRoundTrips:
    def _roundtrip(self, ds_path, write, read):
        with LMDBWriter(ds_path) as w:
            write(w)
            w.commit()
        with LMDBReader(ds_path) as r:
            return read(r)

    def test_any(self, ds_path):
        obj = {"a": [1, 2, (3, 4)]}
        out = self._roundtrip(
            ds_path,
            lambda w: w.write_any(obj, key="k"),
            lambda r: r.read_as_any("k"),
        )
        assert out == obj

    def test_registered_class_requires_explicit_read_permission(self, ds_path):
        register_bytes_codec(
            "tests.lmdb-point.v1",
            _LMDBPoint,
            lambda point: [point.x, point.y],
            lambda value: _LMDBPoint(*value),
        )
        with LMDBWriter(ds_path) as writer:
            writer.write_any(_LMDBPoint(1, 2), key="point")
        with LMDBReader(ds_path) as reader:
            with pytest.raises(ValueError, match="allow_custom=True"):
                reader.read_as_any("point")
            assert reader.read_as_any("point", allow_custom=True) == _LMDBPoint(1, 2)

    @pytest.mark.parametrize("v", [True, 42, 3.14, 2 + 3j])
    def test_primitive(self, ds_path, v):
        out = self._roundtrip(
            ds_path,
            lambda w: w.write_primitive(v, key="k"),
            lambda r: r.read_as_primitive("k"),
        )
        assert out == v

    def test_str(self, ds_path):
        out = self._roundtrip(
            ds_path,
            lambda w: w.write_str("こんにちは🌟", key="k"),
            lambda r: r.read_as_str("k"),
        )
        assert out == "こんにちは🌟"

    def test_sequence_as_list_and_tuple(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_sequence([1, 2, 3], key="lst")
            w.write_sequence((4, 5), key="tpl")
            w.commit()
        with LMDBReader(ds_path) as r:
            assert r.read_as_list("lst") == [1, 2, 3]
            assert r.read_as_tuple("tpl") == (4, 5)

    def test_set_and_frozenset(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_set({1, 2, 3}, key="s")
            w.write_set(frozenset({4, 5}), key="fs")
            w.commit()
        with LMDBReader(ds_path) as r:
            assert r.read_as_set("s") == {1, 2, 3}
            assert r.read_as_frozenset("fs") == frozenset({4, 5})

    def test_dict_and_frozendict(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_dict({"a": 1}, key="d")
            w.write_dict(frozendict({"b": 2}), key="fd")
            w.commit()
        with LMDBReader(ds_path) as r:
            assert r.read_as_dict("d") == {"a": 1}
            fd = r.read_as_frozendict("fd")
            assert fd == frozendict({"b": 2}) and isinstance(fd, frozendict)


class TestArrayRoundTrips:
    @pytest.mark.parametrize("shape", [(), (5,), (2, 3), (0,), (2, 0, 3)])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint8", "complex64", "bool"])
    def test_ndarray(self, ds_path, shape, dtype):
        n = 1 if shape == () else int(np.prod(shape))
        a = np.arange(n).astype(dtype).reshape(shape)
        with LMDBWriter(ds_path) as w:
            w.write_ndarray(a, key="x")
            w.commit()
        with LMDBReader(ds_path) as r:
            out = r.read_as_ndarray("x")
            assert out.dtype == a.dtype
            assert out.shape == a.shape
            assert out.tobytes() == a.tobytes()

    def test_ndarray_structured(self, ds_path):
        dt = np.dtype([("x", "<f4"), ("y", ">i8")])
        a = np.array([(1.5, 7), (2.5, 9)], dtype=dt)
        with LMDBWriter(ds_path) as w:
            w.write_ndarray(a, key="x")
            w.commit()
        with LMDBReader(ds_path) as r:
            out = r.read_as_ndarray("x")
            assert out.dtype == dt
            assert out.tobytes() == a.tobytes()

    def test_ndarray_ml_dtypes(self, ds_path):
        ml = pytest.importorskip("ml_dtypes")
        a = np.array([1.5, -2.0, 0.5], dtype=ml.bfloat16)
        with LMDBWriter(ds_path) as w:
            w.write_ndarray(a, key="x")
            w.commit()
        with LMDBReader(ds_path) as r:
            out = r.read_as_ndarray("x")
            assert out.dtype == np.dtype(ml.bfloat16)
            assert np.array_equal(out, a)


class TestShapeDtypeReaders:
    def test_no_separate_meta_keys_written(self, ds_path):
        a = np.zeros((2, 3), dtype=np.float32)
        with LMDBWriter(ds_path) as w:
            w.write_ndarray(a, key="x")
            w.commit()
        with LMDBReader(ds_path) as r:
            with pytest.raises(KeyError):
                r.read_as_shape("x.__shape__")
            with pytest.raises(KeyError):
                r.read_as_dtype("x.__dtype__")

    def test_ndarray_shape_and_dtype(self, ds_path):
        a = np.zeros((4, 5), dtype=np.float64)
        with LMDBWriter(ds_path) as w:
            w.write_ndarray(a, key="x")
            w.commit()
        with LMDBReader(ds_path) as r:
            assert r.read_as_shape("x") == (4, 5)
            dt = r.read_as_dtype("x")
            assert dt == np.dtype(np.float64) and isinstance(dt, np.dtype)

    def test_ndarray_0d_shape(self, ds_path):
        a = np.array(3.5, dtype=np.float32)
        with LMDBWriter(ds_path) as w:
            w.write_ndarray(a, key="x")
            w.commit()
        with LMDBReader(ds_path) as r:
            assert r.read_as_shape("x") == ()

    def test_shape_dtype_no_warning_for_array_writers(self, ds_path):
        a = np.zeros((2, 2), dtype=np.float32)
        with LMDBWriter(ds_path) as w:
            w.write_ndarray(a, key="x")
            w.commit()
        with LMDBReader(ds_path) as r, warnings.catch_warnings():
            warnings.simplefilter("error")  # Unexpected warnings fail the test.
            r.read_as_shape("x")
            r.read_as_dtype("x")


class TestKeyForms:
    def test_tuple_key_with_index(self, ds_path):
        with LMDBWriter(ds_path) as w:
            for i in range(3):
                w.write_ndarray(np.full(2, i, dtype=np.int64), key=("split", "train"), index=i)
            w.commit()
        with LMDBReader(ds_path) as r:
            for i in range(3):
                out = r.read_as_ndarray(("split", "train"), index=i)
                assert out.tolist() == [i, i]

    def test_adversarial_keys_do_not_overwrite_each_other(self, ds_path):
        values = {
            "a/b": "single",
            ("a", "b"): "tuple",
            ("a/b", "c"): "left-slash",
            ("a", "b/c"): "right-slash",
            "x.__writer__": "reserved-looking",
        }
        with LMDBWriter(ds_path) as writer:
            for key, value in values.items():
                writer.write_str(value, key=key)
        with LMDBReader(ds_path) as reader:
            for key, value in values.items():
                assert reader.read_as_str(key) == value

    def test_writer_tags_are_in_a_named_metadata_database(self, ds_path):
        with LMDBWriter(ds_path) as writer:
            writer.write_str("value", key="key")
            transaction = writer._get_txn()
            assert transaction.get(get_loc("key")) == b"value"
            from amber_lmdb.integrity import _parse_metadata, _verify_checksum

            metadata = _parse_metadata(transaction.get(get_loc("key"), db=writer._metadata_db))
            assert metadata.writer_tag == "write_str"
            _verify_checksum(get_loc("key"), metadata, b"value")
            assert transaction.get(get_loc("key") + b".__writer__") is None


class TestWriterTag:
    def test_mismatch_is_rejected(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_sequence([1, 2, 3], key="x")  # Sequence tag; strict container type.
            w.commit()
        with LMDBReader(ds_path) as r, pytest.raises(ValueError, match="Invalid write/read"):
            r.read_as_set("x")  # A sequence writer tag is incompatible with set reads.

    def test_correct_pair_no_warning(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_dict({"a": 1}, key="x")
            w.commit()
        with LMDBReader(ds_path) as r, warnings.catch_warnings():
            warnings.simplefilter("error")
            assert r.read_as_dict("x") == {"a": 1}

    def test_missing_tag_is_rejected(self, ds_path):
        from amber_lmdb import codec as proc

        with LMDBWriter(ds_path) as w:
            w._put(get_loc("raw"), proc.dict_to_bytes({"k": 9}))
            w.commit()
        with LMDBReader(ds_path) as r, pytest.raises(ValueError, match="no writer tag"):
            r.read_as_dict("raw")

    def test_missing_key_raises(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_str("x", key="present")
            w.commit()
        with LMDBReader(ds_path) as r, pytest.raises(KeyError):
            r.read_as_str("absent")


class TestEnvCache:
    def _make_dataset(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_str("x", key="k")
            w.commit()
        return str(ds_path)

    def test_shared_env_and_refcount(self, ds_path):
        self._make_dataset(ds_path)
        root = str(ds_path.resolve())
        r1 = LMDBReader(ds_path)
        r2 = LMDBReader(ds_path)
        try:
            assert r1.read_as_str("k") == "x"  # env open (refcount 1)
            assert any(config[0] == root for config in get_active_envs())
            assert r2.read_as_str("k") == "x"  # Shared environment, two references.
            assert r1._get_env() is r2._get_env()
        finally:
            r1.close()
            assert any(config[0] == root for config in get_active_envs())
            r2.close()
            assert not any(config[0] == root for config in get_active_envs())

    def test_handles_balance_references(self, ds_path):
        key = self._make_dataset(ds_path)
        config = {"readonly": True, "readahead": False, "max_readers": 8, "map_size": 2**20}
        first = open_environment(key, **config)
        second = open_environment(key, **config)
        assert first.environment is second.environment
        first.close()
        assert get_active_envs()
        second.close()
        assert not get_active_envs()

    def test_handles_release_their_generation_after_current_changes(self, ds_path):
        key = self._make_dataset(ds_path)
        config = {"readonly": True, "readahead": False, "max_readers": 8, "map_size": 2**20}
        old = open_environment(key, **config)
        with LMDBWriter(ds_path, overwrite=True) as writer:
            writer.write_str("new", key="k")
        new = open_environment(key, **config)
        assert old.environment is not new.environment
        old.close()
        new.close()
        assert not get_active_envs()

    def test_active_environments_are_an_immutable_snapshot(self, ds_path):
        key = self._make_dataset(ds_path)
        with open_environment(key, map_size=2**20) as handle:
            active = get_active_envs()
            with pytest.raises(TypeError):
                active["injected"] = handle.environment  # type: ignore[index]

    def test_cache_key_includes_normalized_open_configuration(self, ds_path, monkeypatch):
        self._make_dataset(ds_path)
        import amber_lmdb.lmdb as db

        class FakeEnvironment:
            def close(self):
                return None

        monkeypatch.setattr(db, "_open_env", lambda *args, **kwargs: FakeEnvironment())
        first = open_environment(ds_path, max_readers=8, map_size=2**20)
        second = open_environment(ds_path, readahead=True, max_readers=8, map_size=2**20)
        assert first.environment is second.environment
        assert len(get_active_envs()) == 2
        first.close()
        second.close()


class TestReaderCloseAndPickle:
    def _make_dataset(self, ds_path):
        with LMDBWriter(ds_path) as w:
            w.write_ndarray(np.arange(6, dtype=np.float32).reshape(2, 3), key="x")
            w.commit()

    def test_close_idempotent(self, ds_path):
        self._make_dataset(ds_path)
        r = LMDBReader(ds_path)
        r.read_as_ndarray("x")
        r.close()
        r.close()  # Repeated close must be harmless.

    def test_refresh_switches_to_the_new_atomic_version(self, ds_path):
        with LMDBWriter(ds_path) as writer:
            writer.write_str("old", key="key")
        reader = LMDBReader(ds_path)
        try:
            assert reader.read_as_str("key") == "old"
            with LMDBWriter(ds_path, overwrite=True) as writer:
                writer.write_str("new", key="key")
            assert reader.read_as_str("key") == "old"
            reader.refresh()
            assert reader.read_as_str("key") == "new"
        finally:
            reader.close()

    def test_pickle_before_open(self, ds_path):
        self._make_dataset(ds_path)
        r = LMDBReader(ds_path)
        r2 = pickle.loads(pickle.dumps(r))  # Pickle before the reader opens an environment.
        out = r2.read_as_ndarray("x")
        assert out.shape == (2, 3)
        r2.close()

    def test_pickle_after_open_drops_env(self, ds_path):
        self._make_dataset(ds_path)
        r = LMDBReader(ds_path)
        r.read_as_ndarray("x")  # Open the environment and perform a read transaction.
        state = r.__getstate__()
        assert state["_env"] is None and state["_env_key"] is None
        r2 = pickle.loads(pickle.dumps(r))  # Simulate transferring the reader to a worker.
        out = r2.read_as_ndarray("x")  # The restored reader opens its own connection.
        assert out.shape == (2, 3)
        r.close()
        r2.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize(
    "keys, options, error",
    [
        ("abc", {}, TypeError),
        (["a"], {"max_items": True}, TypeError),
        (["a"], {"allow_custom": 1}, TypeError),
        (["a"], {"max_items": 0}, ValueError),
        (["a", "b"], {"max_items": 1}, ValueError),
        (["a"], {"indices": []}, ValueError),
        (["a"], {"indices": [-1]}, ValueError),
    ],
)
def test_read_many_rejects_invalid_requests_before_open(tmp_path, keys, options, error):
    with LMDBReader(tmp_path / "missing") as reader:
        with pytest.raises(error):
            reader.read_many(keys, **options)
        assert reader.read_many([]) == []


def test_read_many_uses_one_transaction_and_decodes_after_close(ds_path, monkeypatch):
    from unittest.mock import MagicMock

    with LMDBWriter(ds_path, map_size=2 * 1024**2) as writer:
        writer.write(1, key="a")
        writer.write(2, key="b")
    with LMDBReader(ds_path) as reader:
        environment = reader._get_env()
        calls = []
        active = []
        original_decode = reader._decode_entry

        class Transaction:
            def __enter__(self):
                self.transaction = environment.begin(write=False, buffers=True)
                active.append(True)
                return self.transaction

            def __exit__(self, *exc):
                self.transaction.abort()
                active.pop()

        def begin(**kwargs):
            calls.append(kwargs)
            return Transaction()

        def decode(*args, **kwargs):
            assert not active
            return original_decode(*args, **kwargs)

        monkeypatch.setattr(reader, "_get_env", lambda: MagicMock(begin=begin))
        monkeypatch.setattr(reader, "_decode_entry", decode)
        assert reader.read_many(["b", "a", "b"]) == [2, 1, 2]
        assert calls == [{"write": False, "buffers": True}]
        with pytest.raises(KeyError):
            reader.read_many(["a", "missing"])
        assert not active


def test_registered_types_in_metadata_do_not_execute_decoder(ds_path):
    @dataclass(frozen=True)
    class Point:
        x: int

    decoded = []

    def decode(payload):
        decoded.append(payload)
        return Point(payload)

    register_bytes_codec("test.lmdb.generic.point", Point, lambda p: p.x, decode)
    with LMDBWriter(ds_path, map_size=2 * 1024**2) as writer:
        writer.write({"type_map": {"point": Point}}, key="meta")
        writer.write({"nested": Point(7)}, key="value")
    with LMDBReader(ds_path) as reader:
        assert reader.read("meta")["type_map"]["point"] is Point
        assert decoded == []
        with pytest.raises(ValueError, match="allow_custom"):
            reader.read("value")
        assert decoded == []
        assert reader.read_many(["value"], allow_custom=True) == [{"nested": Point(7)}]
    assert decoded == [7]
