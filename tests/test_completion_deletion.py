"""Retained kernel ownership, explicit deletion gates and interrupted-trash recovery."""

import json
import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

import amber_lmdb.generations as generations
import amber_lmdb.lmdb as storage
from amber_lmdb import LMDBReader, LMDBWriter, delete_generations, get_active_envs, open_environment
from amber_lmdb._ownership import _FileLock, _generation_lock
from amber_lmdb.cli import main


def _write(path, value, overwrite=False):
    with LMDBWriter(path, map_size=1024**2, overwrite=overwrite) as writer:
        writer.write_primitive(value, key="value")
    return (path / "CURRENT").read_text().strip()


def _holder(path, connection):
    try:
        with LMDBReader(path) as reader:
            connection.send((Path(reader.get_snapshot().generation).name, reader.read("value")))
            connection.recv()
            connection.send(reader.read("value"))
    finally:
        connection.close()


def _tree(path):
    return sorted(str(item.relative_to(path)) for item in path.rglob("*"))


def test_delete_is_explicit_dry_run_default_never_current_and_removes_lease_files(tmp_path, capsys):
    path = tmp_path / "store"
    old = _write(path, 1)
    current = _write(path, 2, overwrite=True)
    with LMDBReader(path, generation=old) as reader:
        assert reader.read("value") == 1
    lease = path / ".coordination" / f"{old}.lock"
    assert lease.exists()
    before = _tree(path)
    result = delete_generations(path, [old, current])
    assert [row.status for row in result] == ["eligible", "current"]
    assert _tree(path) == before  # a dry run creates and removes nothing
    assert (path / "versions" / old).exists()
    with pytest.raises(ValueError, match="same_host"):
        delete_generations(path, [old], dry_run=False)
    assert main(["delete-generations", str(path), old]) == 0
    assert json.loads(capsys.readouterr().out)["dry_run"]
    with pytest.raises(SystemExit):
        main(["delete-generations", str(path), old, "--apply"])
    with pytest.raises(SystemExit):
        main(["delete-generations", str(path), old, "--keep-last", "1"])
    assert main(["delete-generations", str(path), current]) == 1  # refused: current
    capsys.readouterr()
    assert main(["delete-generations", str(path), old, "--apply", "--same-host"]) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["status"] == "deleted"
    assert not lease.exists()  # removed with its generation, under the acquisition gate
    assert delete_generations(path, [old], dry_run=False, same_host=True)[0].status == "missing"
    assert delete_generations(path, [current], dry_run=False, same_host=True)[0].status == "current"
    with LMDBReader(path) as reader:
        assert reader.read("value") == 2


def test_shared_environment_lifetime_blocks_deletion_until_exact_last_release(tmp_path):
    path = tmp_path / "shared"
    old = _write(path, 1)
    reader = LMDBReader(path)
    assert reader.read("value") == 1
    handle = open_environment(path, readahead=True, map_size=2 * 1024**2)
    assert handle.environment is reader._env
    _write(path, 2, overwrite=True)
    assert delete_generations(path, [old], dry_run=False, same_host=True)[0].status == "in_use"
    reader.close()
    assert delete_generations(path, [old])[0].status == "in_use"
    handle.close()
    assert delete_generations(path, [old], dry_run=False, same_host=True)[0].status == "deleted"
    assert not get_active_envs()


@pytest.mark.parametrize("killed", [False, True])
def test_spawned_idle_or_dead_reader_uses_kernel_ownership_not_expiry(tmp_path, killed):
    path = tmp_path / "process"
    old = _write(path, 1)
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_holder, args=(str(path), child))
    try:
        process.start()
        child.close()
        assert parent.poll(20)
        assert parent.recv() == (old, 1)
        _write(path, 2, overwrite=True)
        assert delete_generations(path, [old], dry_run=False, same_host=True)[0].status == "in_use"
        if killed:
            process.terminate()
        else:
            parent.send("close")
            assert parent.poll(20) and parent.recv() == 1
        process.join(20)
        assert not process.is_alive()
        if not killed:
            assert process.exitcode == 0
        assert delete_generations(path, [old], dry_run=False, same_host=True)[0].status == "deleted"
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()


def test_acquisition_gate_prevents_deletion_before_shared_lifetime_lock(tmp_path, monkeypatch):
    path = tmp_path / "race"
    old = _write(path, 1)
    key = storage._environment_key(
        path, readonly=True, readahead=False, max_readers=512, map_size=1024**2
    )
    _write(path, 2, overwrite=True)
    entered, release = Event(), Event()
    original = storage._open_env

    def paused_open(*args, **options):
        entered.set()
        assert release.wait(20)
        return original(*args, **options)

    monkeypatch.setattr(storage, "_open_env", paused_open)
    with ThreadPoolExecutor(max_workers=2) as pool:
        acquisition = pool.submit(storage._acquire_env, key)
        try:
            assert entered.wait(20)
            # This is a lock assertion, not a timing guess about another thread.
            with pytest.raises(BlockingIOError):
                _FileLock(path / ".coordination" / "gate.lock", blocking=False)
            deletion = pool.submit(delete_generations, path, [old], dry_run=False, same_host=True)
            release.set()
            assert acquisition.result(timeout=20) is not None
            assert deletion.result(timeout=20)[0].status == "in_use"
        finally:
            release.set()
            storage._release_env(key)
    assert delete_generations(path, [old], dry_run=False, same_host=True)[0].status == "deleted"


def test_publication_uses_the_same_acquisition_gate(tmp_path, monkeypatch):
    path = tmp_path / "publish-race"
    old = _write(path, 1)
    entered, release = Event(), Event()
    original = storage._open_env

    def paused_open(*args, **options):
        entered.set()
        assert release.wait(20)
        return original(*args, **options)

    monkeypatch.setattr(storage, "_open_env", paused_open)
    reader = LMDBReader(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        acquisition = pool.submit(reader.get_snapshot)
        try:
            assert entered.wait(20)
            with pytest.raises(BlockingIOError):
                _FileLock(path / ".coordination" / "gate.lock", blocking=False)
            publication = pool.submit(_write, path, 2, True)
            release.set()
            snapshot = acquisition.result(timeout=20)
            current = publication.result(timeout=20)
            assert Path(snapshot.generation).name == old and current != old
            assert reader.read("value") == 1
        finally:
            release.set()
            reader.close()


def test_failed_native_close_retains_ownership_until_successful_retry(tmp_path):
    path = tmp_path / "close-failure"
    old = _write(path, 1)
    reader = LMDBReader(path)
    assert reader.read("value") == 1
    _write(path, 2, overwrite=True)
    key, env = reader._env_key, reader._env

    class FailedClose:
        def close(self):
            raise RuntimeError("injected native close failure")

    storage._ENV_CACHE[key] = FailedClose()
    try:
        with pytest.raises(RuntimeError, match="injected"):
            reader.close()
        assert delete_generations(path, [old])[0].status == "in_use"
    finally:
        storage._ENV_CACHE[key] = env
        reader.close()
    assert delete_generations(path, [old], dry_run=False, same_host=True)[0].status == "deleted"


def test_removed_generation_cannot_be_reopened_by_stale_acquisition_key(tmp_path):
    path = tmp_path / "stale"
    old = _write(path, 1)
    key = storage._environment_key(
        path, readonly=True, readahead=False, max_readers=512, map_size=1024**2
    )
    _write(path, 2, overwrite=True)
    delete_generations(path, [old], dry_run=False, same_host=True)
    with pytest.raises(FileNotFoundError, match="removed before acquisition"):
        storage._acquire_env(key)
    assert not (path / "versions" / old).exists()
    assert not get_active_envs()


def test_native_open_failure_releases_generation_lock(tmp_path, monkeypatch):
    path = tmp_path / "failure"
    current = _write(path, 1)

    def fail(*args, **options):
        raise RuntimeError("native open failed")

    monkeypatch.setattr(storage, "_open_env", fail)
    with LMDBReader(path) as reader, pytest.raises(RuntimeError):
        reader.read("value")
    with _generation_lock(path, current, shared=False, blocking=False):
        pass
    assert not get_active_envs()


def test_removal_failure_is_recoverable_and_competing_remover_cannot_steal_lock(
    tmp_path, monkeypatch
):
    path = tmp_path / "trash"
    old = _write(path, 1)
    current = _write(path, 2, overwrite=True)
    original = generations.shutil.rmtree

    def fail(target, *args, **kwargs):
        if Path(target) == path / ".trash" / old:
            raise OSError("injected removal failure")
        return original(target, *args, **kwargs)

    monkeypatch.setattr(generations.shutil, "rmtree", fail)
    with pytest.raises(OSError, match="injected"):
        delete_generations(path, [old], dry_run=False, same_host=True)
    assert not (path / "versions" / old).exists()
    assert (path / ".trash" / old).exists()
    assert (path / "CURRENT").read_text().strip() == current
    monkeypatch.setattr(generations.shutil, "rmtree", original)
    entered, release = Event(), Event()

    def paused(target, *args, **kwargs):
        if Path(target) == path / ".trash" / old:
            entered.set()
            assert release.wait(20)
        return original(target, *args, **kwargs)

    monkeypatch.setattr(generations.shutil, "rmtree", paused)
    with ThreadPoolExecutor(max_workers=1) as pool:
        removal = pool.submit(delete_generations, path, [old], dry_run=False, same_host=True)
        try:
            assert entered.wait(20)
            assert (
                delete_generations(path, [old], dry_run=False, same_host=True)[0].status == "in_use"
            )
        finally:
            release.set()
        assert removal.result(timeout=20)[0].status == "deleted"
    assert delete_generations(path, [old])[0].status == "missing"


@pytest.mark.parametrize(
    "names", [["../outside"], ["CURRENT"], "v-" + "0" * 32, ["v-" + "0" * 32] * 2]
)
def test_invalid_selection_is_rejected_before_coordination_or_mutation(tmp_path, names):
    with pytest.raises(ValueError):
        delete_generations(tmp_path / "missing", names)
    assert not list(tmp_path.iterdir())


def test_unrecognized_generation_contents_are_never_removed(tmp_path):
    path = tmp_path / "unknown"
    old = _write(path, 1)
    _write(path, 2, overwrite=True)
    unexpected = path / "versions" / old / "foreign.txt"
    unexpected.write_text("keep")
    assert delete_generations(path, [old])[0].status == "unexpected_files"
    result = delete_generations(path, [old], dry_run=False, same_host=True)[0]
    assert result.status == "unexpected_files" and "foreign.txt" in result.message
    assert unexpected.read_text() == "keep"


def _fork_holder(path, inherited, connection):
    try:
        with LMDBReader(path) as fresh:
            assert fresh.read("value") == 1
            inherited.close()
            connection.send(fresh.read("value"))
    finally:
        connection.close()


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="fork unavailable")
def test_inherited_reader_close_cannot_steal_child_reference_or_parent_lease(tmp_path):
    path = tmp_path / "fork"
    old = _write(path, 1)
    with LMDBReader(path) as reader:
        assert reader.read("value") == 1
        context = mp.get_context("fork")
        parent, child = context.Pipe()
        process = context.Process(target=_fork_holder, args=(str(path), reader, child))
        try:
            process.start()
            child.close()
            assert parent.poll(20) and parent.recv() == 1
            process.join(20)
            assert process.exitcode == 0
            _write(path, 2, overwrite=True)
            assert delete_generations(path, [old])[0].status == "in_use"
        finally:
            if process.is_alive():
                process.terminate()
                process.join(10)
            parent.close()
            child.close()
    assert delete_generations(path, [old], dry_run=False, same_host=True)[0].status == "deleted"


@pytest.mark.skipif(os.name == "nt", reason="SIGSTOP/SIGCONT are POSIX-only")
def test_actually_stopped_process_keeps_lease(tmp_path):
    import signal

    path = tmp_path / "paused"
    old = _write(path, 1)
    parent, child = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=_holder, args=(str(path), child))
    try:
        process.start()
        child.close()
        assert parent.poll(20) and parent.recv() == (old, 1)
        os.kill(process.pid, signal.SIGSTOP)
        _write(path, 2, overwrite=True)
        assert delete_generations(path, [old], dry_run=False, same_host=True)[0].status == "in_use"
        os.kill(process.pid, signal.SIGCONT)
        parent.send("close")
        assert parent.poll(20) and parent.recv() == 1
        process.join(20)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            os.kill(process.pid, signal.SIGCONT)
            process.terminate()
            process.join(10)
        parent.close()
        child.close()
