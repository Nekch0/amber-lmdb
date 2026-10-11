"""Bounded, generation-aware inspection and the standalone CLI."""

import json
import pickle
import struct
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import lmdb
import numpy as np
import pytest

import amber_lmdb.inspection as inspection
from amber_lmdb import (
    LMDBReader,
    LMDBWriter,
    codec,
    decode_key,
    get_active_envs,
    get_loc,
    inspect_store,
    register_bytes_codec,
)
from amber_lmdb.cli import main


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "sample.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_dict({"name": "日本語", "type_map": {"array": np.ndarray}}, key="meta")
        writer.write_ndarray(np.arange(8, dtype=np.float32).reshape(2, 4), key="array", index=0)
        writer.write_primitive(5, key=("session", "count"))
        writer.write_dict({"text": "hello"}, key=("session", "record"), index=0)
    return path


@pytest.mark.parametrize("key", ["", "a/b", ("a/b", "c"), (), ("", "日本語", "\x00")])
@pytest.mark.parametrize("index", [None, 0, 2**64 - 1])
def test_key_roundtrip(key, index):
    assert decode_key(get_loc(key, index)) == (key, index)


@pytest.mark.parametrize("raw", [b"", b"MMFK\x01", get_loc("x") + b"x", get_loc("x")[:-1]])
def test_invalid_key(raw):
    with pytest.raises(ValueError):
        decode_key(raw)


def test_custom_metadata_key(tmp_path):
    path = tmp_path / "custom-meta"
    with LMDBWriter(path) as writer:
        writer.write_dict({"schema": "invoice/1"}, key=("application", "description"))
        writer.commit()
    result = inspect_store(path, with_meta=True, metadata_key=("application", "description"))
    assert result.meta == {"schema": "invoice/1"}
    assert result.meta_error is None
    assert inspect_store(path, with_meta=True).meta_error is not None


def test_invalid_metadata_key_does_not_open_environment(tmp_path):
    with pytest.raises(TypeError):
        inspect_store(tmp_path / "absent", metadata_key=(1,))
    assert not get_active_envs()


def test_path_key_roundtrip():
    from pathlib import PurePosixPath, PureWindowsPath

    path = Path("a/b")
    assert decode_key(get_loc(path)) == ("a/b", None)
    # Paths encode identically on every operating system.
    assert get_loc(PureWindowsPath("a\\b")) == get_loc(PurePosixPath("a/b")) == get_loc("a/b")


def test_huge_component_count_does_not_allocate():
    raw = b"MMFK\x01\x01" + struct.pack(">I", 2**32 - 1) + b"\x00"
    with pytest.raises(ValueError, match="truncated"):
        decode_key(raw)


def test_default_reads_stats_without_values(store, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("default inspection decoded a value")

    monkeypatch.setattr(codec, "bytes_to_dict", forbidden)
    monkeypatch.setattr(codec, "bytes_to_ndarray", forbidden)
    result = inspect_store(store)
    assert result.entries == 5
    assert result.data_entries == 4
    assert result.meta is None and result.samples == ()
    assert result.used_bytes >= result.page_size
    assert Path(result.snapshot.generation).name == (store / "CURRENT").read_text().strip()
    assert not get_active_envs()


def test_array_headers_and_meta_without_array_decode(store, monkeypatch):
    monkeypatch.setattr(codec, "bytes_to_ndarray", lambda *args: pytest.fail("array decode"))
    result = inspect_store(store, with_meta=True, entry_limit=20)
    assert result.meta == {"name": "日本語", "type_map": {"array": np.ndarray}}
    array = next(entry for entry in result.samples if entry.key == "array")
    assert array.shape == (2, 4) and array.dtype == "float32"
    assert array.writer_tag == "write_ndarray" and array.error is None
    assert result.next_after is None and len(result.samples) == 4
    assert not get_active_envs()


def test_pagination_is_ordered_bounded_and_nonduplicating(store):
    all_entries = inspect_store(store, entry_limit=20).samples
    after = None
    seen = []
    for _ in range(10):
        page = inspect_store(store, entry_limit=1, after=after)
        assert len(page.samples) <= 1
        seen.extend(page.samples)
        after = page.next_after
        if after is None:
            break
    assert tuple(seen) == all_entries


def test_inspection_reuses_reader_environment_and_leaves_it_open(store):
    with LMDBReader(store) as reader:
        before = reader.get_snapshot()
        active = get_active_envs()
        assert inspect_store(store).snapshot == before
        assert get_active_envs() == active
        assert reader.read_as_primitive(("session", "count")) == 5
    assert not get_active_envs()


def test_pinned_reader_survives_generation_switch(store):
    with LMDBReader(store) as reader:
        old = reader.get_snapshot()
        with LMDBWriter(store, overwrite=True) as writer:
            writer.write_primitive(9, key=("session", "count"))
        new = inspect_store(store)
        assert new.snapshot.generation != old.generation
        assert reader.get_snapshot() == old
        assert reader.read_as_primitive(("session", "count")) == 5
        with pickle.loads(pickle.dumps(reader)) as restored:
            assert restored.get_snapshot() == old  # a copy stays on the opened generation
            restored.refresh()
            assert restored.get_snapshot() == new.snapshot
    assert not get_active_envs()


@pytest.mark.parametrize(
    "options,error",
    [
        ({"entry_limit": True}, TypeError),
        ({"entry_limit": -1}, ValueError),
        ({"entry_limit": 1025}, ValueError),
        ({"with_meta": 1}, TypeError),
        ({"after": "abc"}, TypeError),
        ({"max_meta_bytes": 0}, ValueError),
        ({"max_meta_bytes": True}, TypeError),
    ],
)
def test_invalid_options_before_open(tmp_path, options, error):
    with pytest.raises(error):
        inspect_store(tmp_path / "absent", **options)
    assert not get_active_envs()


def test_metadata_size_limit(store):
    result = inspect_store(store, with_meta=True, max_meta_bytes=1)
    assert result.meta is None and "limit" in result.meta_error


def test_non_typed_entry_reports_error_and_missing_metadata_db(tmp_path):
    path = tmp_path / "raw.lmdb"
    generation = path / "versions" / ("v-" + "a" * 32)
    generation.mkdir(parents=True)
    (path / "FORMAT").write_text("amber-lmdb-v1\n")
    (path / "CURRENT").write_text(generation.name + "\n")
    with lmdb.open(str(generation)) as env, env.begin(write=True) as txn:
        txn.put(b"raw", b"value")
    result = inspect_store(path, with_meta=True, entry_limit=1)
    assert result.entries == result.data_entries == 1
    assert result.meta_error is not None
    assert result.samples[0].error is not None
    assert not get_active_envs()


def test_custom_decoder_is_never_enabled(tmp_path):
    @dataclass
    class Point:
        x: int

    called = []
    register_bytes_codec("test.inspection.point", Point, lambda p: p.x, lambda p: called.append(p))
    path = tmp_path / "custom.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_dict({"point": Point(1)}, key="meta")
    result = inspect_store(path, with_meta=True)
    assert result.meta is None and result.meta_error is not None
    assert called == []


def test_cli_json_and_error_status(store, capsys):
    assert main(["inspect", str(store), "--meta", "--entries", "2"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["meta"]["type_map"]["array"] == "numpy.ndarray"
    assert len(result["samples"]) == 2
    assert main(["doctor", str(store)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["sample_ok"] and "ok" not in report
    assert report["full_value_validation"] is False and "--full" in report["hint"]
    assert main(["inspect", str(store / "absent")]) == 3
    captured = capsys.readouterr()
    assert "Error" in captured.err
    assert json.loads(captured.out)["error"]["type"] == "StoreNotFoundError"


def test_cli_metadata_with_nonstring_keys_arrays_and_nonfinite_values(tmp_path, capsys):
    path = tmp_path / "metadata.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_dict(
            {
                ("tuple", 3): frozenset({np.ndarray}),
                1: b"\x00",
                "array": np.arange(3),
                "nan": float("nan"),
            },
            key="meta",
        )
    assert main(["inspect", str(path), "--meta", "--entries", "0"]) == 0
    result = json.loads(capsys.readouterr().out)
    entries = result["meta"]["mapping"]
    assert entries[0] == [["tuple", 3], ["numpy.ndarray"]]
    assert entries[1] == [1, {"bytes_hex": "00"}]
    assert entries[2][1]["shape"] == [3]
    assert entries[3] == ["nan", {"float": "nan"}]


def test_cli_list_empty_and_bad_store(tmp_path, capsys):
    assert main(["list", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == []
    (tmp_path / "broken.lmdb").mkdir()
    assert main(["list", str(tmp_path)]) == 1
    assert "error" in json.loads(capsys.readouterr().out)[0]


def test_cli_invalid_limit_is_usage_error(store, capsys):
    with pytest.raises(SystemExit) as result:
        main(["inspect", str(store), "--entries", "1025"])
    assert result.value.code == 2
    with pytest.raises(SystemExit) as result:
        main(["inspect", str(store), "--max-copy-bytes", "many"])
    assert result.value.code == 2
    assert "invalid positive integer value" in capsys.readouterr().err


def test_standalone_import_without_torch():
    program = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith(('torch.', 'mmfabric')):
        raise AssertionError('unwanted dependency import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import amber_lmdb
assert amber_lmdb.LMDBReader is not None
"""
    subprocess.run([sys.executable, "-c", program], check=True)


def test_headers_and_metadata_decode_after_transaction_end(store, monkeypatch):
    active = False
    real_acquire = inspection._acquire_env
    original_header = codec._unpack
    original_dict = codec.bytes_to_dict

    def acquire(key, **options):
        env = real_acquire(key, **options)
        proxy = MagicMock(wraps=env)

        @contextmanager
        def begin(*args, **kwargs):
            nonlocal active
            with env.begin(*args, **kwargs) as transaction:
                active = True
                try:
                    yield transaction
                finally:
                    active = False

        proxy.begin.side_effect = begin
        return proxy

    def header(raw):
        assert not active
        # Only the header was copied, never the array body.
        metadata, body = original_header(raw)
        assert len(body) == 0
        return metadata, body

    def metadata(raw, **kwargs):
        assert not active
        return original_dict(raw, **kwargs)

    monkeypatch.setattr(inspection, "_acquire_env", acquire)
    monkeypatch.setattr(codec, "_unpack", header)
    monkeypatch.setattr(codec, "bytes_to_dict", metadata)
    result = inspect_store(store, with_meta=True, entry_limit=20)
    assert result.meta is not None
    assert next(entry for entry in result.samples if entry.key == "array").shape == (2, 4)
    assert not get_active_envs()


def test_inspection_identity_and_release_follow_opened_generation(store, monkeypatch):
    old_generation = (store / "CURRENT").read_text().strip()
    real_acquire = inspection._acquire_env

    def acquire(key, **options):
        env = real_acquire(key, **options)
        with LMDBWriter(store, overwrite=True) as writer:
            writer.write_str("replacement", key="new")
        return env

    monkeypatch.setattr(inspection, "_acquire_env", acquire)
    result = inspect_store(store, with_meta=True, entry_limit=20)
    assert Path(result.snapshot.generation).name == old_generation
    assert result.meta["name"] == "日本語"
    assert (store / "CURRENT").read_text().strip() != old_generation
    assert not get_active_envs()


def test_oversized_array_header_is_reported_without_reading_payload(tmp_path):
    path = tmp_path / "broken.lmdb"
    with LMDBWriter(path) as writer:
        raw = codec._HEAD.pack(codec._MAGIC, codec._VERSION, 2**32 - 1)
        writer._put_tagged(get_loc("array"), raw, "write_ndarray")
    result = inspect_store(path, entry_limit=1)
    assert "limit" in result.samples[0].error
    assert main(["doctor", str(path)]) == 1
    assert not get_active_envs()


def test_tensor_header_inspection_without_torch(tmp_path):
    path = tmp_path / "tensor.lmdb"
    with LMDBWriter(path) as writer:
        raw = codec._pack({"kind": "torch", "dtype": "float32", "shape": (2, 3)}, bytes(24))
        writer._put_tagged(get_loc("tensor"), raw, "write_tensor")
    program = """
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('torch imported for header inspection')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from amber_lmdb import inspect_store
info = inspect_store(sys.argv[1], entry_limit=1)
assert info.samples[0].shape == (2, 3)
assert info.samples[0].dtype == 'torch.float32'
assert info.samples[0].error is None
"""
    subprocess.run([sys.executable, "-c", program, str(path)], check=True)


def test_module_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "amber_lmdb", "--help"], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0 and "doctor" in result.stdout
