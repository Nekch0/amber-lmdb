"""Derived writers copy a published generation, then add, replace and delete entries."""

import numpy as np
import pytest

import amber_lmdb.lmdb as lmdb_module
from amber_lmdb import (
    LMDBReader,
    LMDBWriter,
    delete_generations,
    diff_stores,
    fingerprint_store,
    get_active_envs,
    list_generations,
    verify_store,
)


def _base(path, *, engine=1, checksum="blake2b-256", manifest=False):
    with LMDBWriter(
        path, map_size=8 * 1024**2, lib_version=engine, checksum=checksum, manifest=manifest
    ) as writer:
        writer.write_str("kept", key="text")
        writer.write_primitive(1, key="count")
        for index in range(4):
            writer.write_ndarray(np.full(3, index, np.float32), key=("s01", "pitch"), index=index)
        writer.write_ndarray(np.zeros(2), key=("s01", "old"), index=0)
        writer.write_ndarray(np.zeros(2), key=("s01", "old"), index=1)
        writer.write_str("unindexed", key=("s01", "old"))
    return path


def _current(path):
    return (path / "CURRENT").read_text().strip()


def _staging(path):
    return [entry for entry in path.parent.iterdir() if entry.name.endswith(".tmp")]


def test_derived_generation_is_base_plus_changes(tmp_path):
    path = _base(tmp_path / "store.lmdb")
    before = _current(path)
    with LMDBWriter(path, map_size=8 * 1024**2, base="current") as writer:
        writer.write_ndarray(np.arange(5), key=("s01", "energy"), index=0)
        writer.write_primitive(2, key="count")
        assert writer.delete(("s01", "pitch"), index=3)
        assert writer.delete_key(("s01", "old")) == 3
    after = _current(path)
    assert after != before
    difference = diff_stores(path, path, before_generation=before)
    assert difference.added == ((("s01", "energy"), 0),)
    assert difference.changed == (("count", None),)
    assert set(difference.removed) == {
        (("s01", "pitch"), 3),
        (("s01", "old"), None),
        (("s01", "old"), 0),
        (("s01", "old"), 1),
    }
    assert difference.unchanged == 4
    with LMDBReader(path) as reader:
        assert reader.read_as_str("text") == "kept"
        assert reader.read_as_primitive("count") == 2
        assert reader.count(("s01", "pitch")) == 3
        assert reader.read_as_ndarray(("s01", "pitch"), index=2).tolist() == [2, 2, 2]
        assert not reader.contains(("s01", "old"))
    assert verify_store(path).ok
    assert {item.name for item in list_generations(path)} == {before, after}
    assert not get_active_envs()


def test_pinned_reader_keeps_reading_the_base(tmp_path):
    path = _base(tmp_path / "store.lmdb")
    with LMDBReader(path) as reader:
        assert reader.read_as_primitive("count") == 1
        with LMDBWriter(path, map_size=8 * 1024**2, base="current") as writer:
            writer.write_primitive(5, key="count")
        assert reader.read_as_primitive("count") == 1
        reader.refresh()
        assert reader.read_as_primitive("count") == 5


@pytest.mark.parametrize("engine", [0, 1])
def test_base_engine_is_kept_and_a_mismatch_is_refused(tmp_path, engine):
    path = _base(tmp_path / f"engine-{engine}.lmdb", engine=engine)
    before = _current(path)
    with LMDBWriter(path, map_size=8 * 1024**2, base="current") as writer:
        writer.write_str("new", key="added")
    with LMDBReader(path) as reader:
        assert reader.get_lib_version()[0] == engine
        assert reader.read_as_str("added") == "new"
    with LMDBWriter(path, map_size=8 * 1024**2, base="current", lib_version=engine) as writer:
        writer.write_str("again", key="added")
    published = _current(path)
    with (
        pytest.raises(ValueError, match="does not match the base"),
        LMDBWriter(path, map_size=8 * 1024**2, base="current", lib_version=1 - engine),
    ):
        pass
    assert _current(path) == published != before
    assert not _staging(path)
    assert not get_active_envs()


def test_named_base_generation(tmp_path):
    path = _base(tmp_path / "store.lmdb")
    first = _current(path)
    with LMDBWriter(path, map_size=8 * 1024**2, overwrite=True) as writer:
        writer.write_str("unrelated", key="text")
    with LMDBWriter(path, map_size=8 * 1024**2, base=first) as writer:
        writer.write_str("from first", key="note")
    with LMDBReader(path) as reader:
        assert reader.read_as_str("text") == "kept"
        assert reader.read_as_str("note") == "from first"
        assert reader.count(("s01", "pitch")) == 4


def test_missing_base_and_invalid_options(tmp_path):
    with (
        pytest.raises(FileNotFoundError, match="published"),
        LMDBWriter(tmp_path / "absent.lmdb", base="current"),
    ):
        pass
    assert not (tmp_path / "absent.lmdb").exists() or not any((tmp_path / "absent.lmdb").iterdir())
    path = _base(tmp_path / "store.lmdb")
    missing = "v-" + "0" * 32
    with (
        pytest.raises(FileNotFoundError, match="base generation not found"),
        LMDBWriter(path, base=missing),
    ):
        pass
    for invalid in ("latest", "v-123", "CURRENT"):
        with pytest.raises(ValueError, match="base must be"):
            LMDBWriter(path, base=invalid)
    with pytest.raises(TypeError, match="base must be"):
        LMDBWriter(path, base=1)
    assert not _staging(path)
    with LMDBWriter(path, map_size=8 * 1024**2, base="current") as writer:
        writer.write_str("lock was released", key="note")


def test_delete_semantics(tmp_path):
    path = tmp_path / "fresh.lmdb"
    with LMDBWriter(path, map_size=8 * 1024**2) as writer:
        writer.write_str("a", key="first")
        writer.write_str("b", key="second")
        assert writer.delete("first")
        with pytest.raises(KeyError):
            writer.delete("first")
        assert writer.delete("first", missing_ok=True) is False
        assert writer.delete(("never",), index=3, missing_ok=True) is False
        with pytest.raises(TypeError):
            writer.delete("second", missing_ok=1)
        with pytest.raises(ValueError, match="reserved"):
            writer.delete(("amber-lmdb", "manifest"))
        with pytest.raises(ValueError, match="reserved"):
            writer.delete_key(("amber-lmdb", "manifest"))
    with LMDBReader(path) as reader:
        assert not reader.contains("first")
        assert reader.read_as_str("second") == "b"
    assert verify_store(path).ok


def test_delete_key_leaves_keys_with_shared_prefixes(tmp_path):
    path = tmp_path / "prefixes.lmdb"
    with LMDBWriter(path, map_size=8 * 1024**2) as writer:
        writer.write_primitive(0, key="s1")
        writer.write_primitive(1, key="s1", index=0)
        writer.write_primitive(2, key="s1", index=2**64 - 1)
        writer.write_primitive(3, key="s1longer", index=0)
        writer.write_primitive(4, key=("s1",), index=0)
        writer.write_primitive(5, key=("s1", "child"), index=0)
        assert writer.delete_key("s1") == 3
        assert writer.delete_key("s1") == 0
        assert writer.delete_key(("absent", "key")) == 0
    with LMDBReader(path) as reader:
        assert reader.contains_many(
            ["s1", "s1", "s1longer", ("s1",), ("s1", "child")],
            indices=[None, 0, 0, 0, 0],
        ) == [False, False, True, True, True]


def test_delete_removes_orphan_bodies(tmp_path):
    path = tmp_path / "orphan.lmdb"
    location = lmdb_module.get_loc("orphan", 0)
    with LMDBWriter(path, map_size=8 * 1024**2) as writer:
        writer.write_primitive(1, key="orphan", index=0)
        writer.write_primitive(2, key="orphan", index=1)
        writer._get_txn().delete(location, db=writer._metadata_db)
        assert writer.delete_key("orphan") == 2
    with LMDBReader(path) as reader:
        assert reader.count("orphan") == 0
    assert verify_store(path).ok


def test_cooperative_deletion_is_blocked_during_the_copy(tmp_path, monkeypatch):
    path = _base(tmp_path / "store.lmdb")
    base = _current(path)
    with LMDBWriter(path, map_size=8 * 1024**2, overwrite=True) as writer:
        writer.write_str("newer", key="text")
    observed = []
    acquire = lmdb_module._acquire_env

    class Probe:
        def __init__(self, environment):
            self._environment = environment

        def lib_version(self):
            return self._environment.lib_version()

        def copyfd(self, *args, **kwargs):
            observed.extend(delete_generations(path, [base], dry_run=False, same_host=True))
            return self._environment.copyfd(*args, **kwargs)

    monkeypatch.setattr(lmdb_module, "_acquire_env", lambda key: Probe(acquire(key)))
    with LMDBWriter(path, map_size=8 * 1024**2, base=base) as writer:
        writer.write_str("derived", key="note")
    monkeypatch.undo()
    assert [item.status for item in observed] == ["in_use"]
    assert [
        item.status for item in delete_generations(path, [base], dry_run=False, same_host=True)
    ] == ["deleted"]
    with LMDBReader(path) as reader:
        assert reader.read_as_str("text") == "kept"
        assert reader.read_as_str("note") == "derived"


def test_failures_during_copy_or_writes_publish_nothing(tmp_path, monkeypatch):
    path = _base(tmp_path / "store.lmdb")
    before = _current(path)
    acquire = lmdb_module._acquire_env

    class Failing:
        def __init__(self, environment):
            self._environment = environment

        def lib_version(self):
            return self._environment.lib_version()

        def copyfd(self, *args, **kwargs):
            raise OSError("disk full during copy")

    monkeypatch.setattr(lmdb_module, "_acquire_env", lambda key: Failing(acquire(key)))
    with pytest.raises(OSError, match="disk full"), LMDBWriter(path, base="current"):
        pass
    monkeypatch.undo()
    assert _current(path) == before and not _staging(path) and not get_active_envs()
    with (
        pytest.raises(RuntimeError, match="boom"),
        LMDBWriter(path, map_size=8 * 1024**2, base="current") as writer,
    ):
        writer.delete_key(("s01", "pitch"))
        raise RuntimeError("boom")
    assert _current(path) == before and not _staging(path)
    with LMDBReader(path) as reader:
        assert reader.count(("s01", "pitch")) == 4
    assert len(list_generations(path)) == 1


def test_copied_manifest_is_dropped_or_rebuilt(tmp_path):
    path = _base(tmp_path / "store.lmdb", manifest=True)
    base = _current(path)
    with LMDBReader(path) as reader:
        assert reader.manifest() is not None and reader.manifest().base is None
    with LMDBWriter(path, map_size=8 * 1024**2, base="current") as writer:
        writer.write_str("x", key="note")
    with LMDBReader(path) as reader:
        assert reader.manifest() is None
    with LMDBWriter(path, map_size=8 * 1024**2, base=base, manifest=True) as writer:
        writer.delete("text")
    with LMDBReader(path) as reader:
        manifest = reader.manifest()
    assert manifest is not None and manifest.base == base
    assert manifest.fingerprint == fingerprint_store(path).digest
    assert manifest.entries == fingerprint_store(path).entries == 8


def test_copied_envelopes_keep_their_algorithm(tmp_path):
    path = _base(tmp_path / "store.lmdb", checksum="sha256")
    with LMDBWriter(path, map_size=8 * 1024**2, base="current", verify_on_publish=True) as writer:
        writer.write_str("blake", key="new")
    assert fingerprint_store(path).algorithms == ("blake2b-256", "sha256")
    assert verify_store(path).ok
