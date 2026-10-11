"""Real connection settings, exact releases and cross-process generation retention."""

import multiprocessing as mp
import pickle
from pathlib import Path

import pytest

from amber_lmdb import (
    LMDBReader,
    LMDBWriter,
    get_active_envs,
    inspect_store,
    open_environment,
)


def _write(path, value, overwrite=False):
    with LMDBWriter(path, map_size=8 * 1024**2, overwrite=overwrite) as writer:
        writer.write_primitive(value, key="value")
        writer.commit()


def test_different_reader_settings_and_inspection_share_one_environment(tmp_path):
    path = tmp_path / "store"
    _write(path, 1)
    with (
        LMDBReader(path, map_size=16 * 1024**2, readahead=True) as first,
        LMDBReader(path, map_size=32 * 1024**2) as second,
    ):
        assert first.read("value") == second.read("value") == 1
        assert first._get_env() is second._get_env()
        assert first._get_env().flags()["readahead"]
        assert inspect_store(path).snapshot == first.get_snapshot()
        first.close()
        assert second.read("value") == 1
    assert not get_active_envs()


def test_handle_releases_exact_generation_and_rejects_pickle(tmp_path):
    path = tmp_path / "store"
    _write(path, 1)
    with open_environment(path) as old:
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(old)
        _write(path, 2, overwrite=True)
        with open_environment(path) as new:
            assert new.environment is not old.environment
            old.close()
            old.close()
            with pytest.raises(RuntimeError, match="closed"):
                _ = old.environment
            assert new.environment.stat()["entries"] > 0
    assert not get_active_envs()


def test_conflicting_access_modes_fail_without_leaking(tmp_path):
    path = tmp_path / "store"
    _write(path, 1)
    with open_environment(path):
        with pytest.raises(ValueError, match="readonly"):
            open_environment(path, readonly=False)
        assert len(get_active_envs()) == 1
    assert not get_active_envs()


def test_publication_retains_old_generation_without_local_readers(tmp_path):
    path = tmp_path / "store"
    _write(path, 1)
    old = (path / "CURRENT").read_text().strip()
    _write(path, 2, overwrite=True)
    assert (path / "versions" / old / "data.mdb").is_file()
    assert len(list((path / "versions").iterdir())) == 2


def _pinned_child(path, connection):
    try:
        with LMDBReader(path) as reader:
            assert reader.read("value") == 1
            connection.send(reader.get_snapshot().generation)
            connection.recv()
            # Another reader of this retained generation can still open its files.
            assert Path(reader.get_snapshot().generation, "data.mdb").is_file()
            connection.send(reader.read("value"))
    finally:
        connection.close()


def test_spawned_reader_retains_old_generation_during_publication(tmp_path):
    path = tmp_path / "store"
    _write(path, 1)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_pinned_child, args=(str(path), child))
    process.start()
    child.close()
    try:
        assert parent.poll(30)
        old = parent.recv()
        _write(path, 2, overwrite=True)
        assert Path(old, "data.mdb").is_file()
        parent.send(None)
        assert parent.poll(30)
        assert parent.recv() == 1
        process.join(30)
        assert process.exitcode == 0
        with LMDBReader(path) as reader:
            assert reader.read("value") == 2
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
