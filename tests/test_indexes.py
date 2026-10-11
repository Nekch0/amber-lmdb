"""Derived indexes also work for ordinary text sources, without CEJC or LMDB."""

import hashlib
import json
import multiprocessing as mp
import os
import struct
from pathlib import Path

import pytest

from amber_lmdb import DerivedIndexCache, index_lock
from amber_lmdb.codec import any_to_bytes, bytes_to_any


def _cache(directory, **options):
    return DerivedIndexCache(directory, encode=any_to_bytes, decode=bytes_to_any, **options)


def test_hit_no_persistence_namespace_and_identity(tmp_path):
    identity = {"source": "words", "version": "sha256:one", "builder": 1}
    cache = _cache(tmp_path, namespace="words")
    assert cache.get_or_build(identity, lambda: {"count": 2}) == {"count": 2}
    assert cache.get_or_build(identity, lambda: pytest.fail("unexpected rebuild")) == {"count": 2}
    assert cache.path_for(identity).name.startswith("words-")
    assert cache.path_for({**identity, "builder": 2}) != cache.path_for(identity)
    assert _cache(tmp_path, namespace="other").path_for(identity) != cache.path_for(identity)
    disabled = _cache(None)
    assert disabled.get_or_build(identity, lambda: 7) == 7
    with pytest.raises(RuntimeError):
        disabled.path_for(identity)


@pytest.mark.parametrize("damage", ["truncated", "checksum", "magic", "oversized-header"])
def test_corrupt_cache_rebuilds(tmp_path, damage):
    cache = _cache(tmp_path)
    identity = {"source": "a", "builder": 1}
    cache.get_or_build(identity, lambda: [1, 2])
    path = cache.path_for(identity)
    data = path.read_bytes()
    if damage == "truncated":
        path.write_bytes(data[:-1])
    elif damage == "checksum":
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    elif damage == "magic":
        path.write_bytes(b"invalid" + data[7:])
    else:
        path.write_bytes(cache.magic + struct.pack(">I", cache.max_header_bytes + 1))
    assert cache.get_or_build(identity, lambda: [3, 4]) == [3, 4]
    assert cache.lookup(identity).value == [3, 4]


def test_validator_rejects_corrupt_semantics_and_invalid_builder(tmp_path):
    cache = _cache(tmp_path)
    identity = {"version": 1}

    def validate(value):
        if type(value) is not list or len(value) != 2:
            raise ValueError("wrong rows")

    cache.publish(identity, [1])
    assert cache.get_or_build(identity, lambda: [2, 3], validate=validate) == [2, 3]
    other = {"version": 2}
    with pytest.raises(ValueError, match="wrong rows"):
        cache.get_or_build(other, lambda: [1], validate=validate)
    assert not cache.path_for(other).exists()


def test_changed_source_and_failed_replacement_publish_nothing(tmp_path, monkeypatch):
    cache = _cache(tmp_path)
    changed = False

    def build():
        nonlocal changed
        changed = True
        return [1]

    def check():
        if changed:
            raise ValueError("source changed")

    with pytest.raises(ValueError, match="source changed"):
        cache.get_or_build({"source": 1}, build, source_check=check)
    assert not list(tmp_path.glob("*.typed-index"))

    def fail(*args):
        raise KeyboardInterrupt("interrupted")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(KeyboardInterrupt):
        cache.get_or_build({"source": 2}, lambda: [1])
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.typed-index"))


def test_lock_timeout_and_retained_inode(tmp_path):
    path = tmp_path / "index.lock"
    with index_lock(path):
        inode = path.stat().st_ino
        with pytest.raises(TimeoutError), index_lock(path, timeout=0):
            pytest.fail("acquired twice")
    assert path.is_file() and path.stat().st_ino == inode
    with index_lock(path, timeout=0):
        pass


@pytest.mark.parametrize("timeout", [-1, True, float("nan"), float("inf")])
def test_invalid_lock_timeouts(tmp_path, timeout):
    with pytest.raises(ValueError), index_lock(tmp_path / "lock", timeout=timeout):
        pass
    with pytest.raises(ValueError):
        _cache(tmp_path, lock_timeout=timeout)


@pytest.mark.parametrize("suffix", [".lock", ".records.lock", ".LOCK"])
def test_payload_suffix_cannot_replace_the_active_lock_inode(tmp_path, suffix):
    with pytest.raises(ValueError, match="reserved for builder locks"):
        _cache(tmp_path, suffix=suffix)
    assert not list(tmp_path.iterdir())


def test_limits_and_io_errors_are_not_silenced(tmp_path, monkeypatch):
    cache = _cache(tmp_path, max_payload_bytes=2)
    with pytest.raises(ValueError, match="payload"):
        cache.publish({}, ["larger than two bytes"])
    cache = _cache(tmp_path, max_header_bytes=2)
    with pytest.raises(ValueError, match="header"):
        cache.publish({}, 1)

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(PermissionError):
        _cache(tmp_path).lookup({})


def test_legacy_envelope_bytes_are_readable(tmp_path):
    magic = b"CEJC-RECORD-INDEX\x00\x01"
    cache = _cache(tmp_path, magic=magic, suffix=".cejc-records")
    identity = {"builder": 1, "session_name": "legacy"}
    payload = any_to_bytes({"records": [{"id": 0}]})
    header = json.dumps(
        {
            "source": identity,
            "payload_size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ).encode()
    cache.path_for(identity).write_bytes(magic + struct.pack(">I", len(header)) + header + payload)
    assert cache.lookup(identity).value == {"records": [{"id": 0}]}


def _competing_builder(directory, barrier, queue):
    builds = 0

    def build():
        nonlocal builds
        builds += 1
        return {"words": ["apple", "book"]}

    barrier.wait(timeout=30)
    value = _cache(directory).get_or_build({"text_digest": "fixed", "builder": 1}, build)
    queue.put((builds, value))


def test_competing_builders_recheck_under_lock(tmp_path):
    context = mp.get_context("spawn")
    barrier = context.Barrier(3)
    queue = context.Queue()
    processes = [
        context.Process(target=_competing_builder, args=(str(tmp_path), barrier, queue))
        for _ in range(3)
    ]
    try:
        for process in processes:
            process.start()
        results = [queue.get(timeout=40) for _ in processes]
        for process in processes:
            process.join(20)
            assert process.exitcode == 0
        assert sorted(builds for builds, _ in results) == [0, 0, 1]
        assert all(value == {"words": ["apple", "book"]} for _, value in results)
        assert len(list(tmp_path.glob("*.typed-index"))) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
        queue.close()
        queue.join_thread()
