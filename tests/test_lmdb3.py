"""Both py-lmdb 3 native engines, descriptor normalization and offline verification."""

import json

import lmdb
import numpy as np
import pytest

import amber_lmdb.physical as physical
from amber_lmdb import (
    LMDBReader,
    LMDBWriter,
    SourceChangedError,
    array_statistics,
    get_active_envs,
    get_backend_info,
    inspect_store,
    open_environment,
    verify_physical,
    verify_store,
)
from amber_lmdb.cli import main


@pytest.mark.parametrize("engine", [0, 1])
def test_native_engine_roundtrip_and_all_inspection_paths(tmp_path, engine, capsys):
    path = tmp_path / "日本語" / f"engine-{engine}.lmdb"
    with LMDBWriter(path, lib_version=engine, verify_on_publish=True) as writer:
        writer.write_dict({"description": "standalone"}, key="meta")
        writer.write_ndarray(np.arange(6, dtype=np.float32).reshape(2, 3), key="array", index=0)
    with LMDBReader(path) as reader:
        assert reader.get_lib_version()[0] == engine
        assert reader.read_as_dict("meta") == {"description": "standalone"}
        assert reader.read_array_headers_many(["array"], indices=[0])[0].shape == (2, 3)
    info = inspect_store(path, with_meta=True, entry_limit=20)
    assert info.lib_version[0] == engine
    assert info.data_entries == len(info.samples) == 2
    assert not any(entry.error for entry in info.samples)
    assert info.meta_error is None
    assert array_statistics(path).scanned_entries == 2
    assert verify_store(path, batch_size=1).ok
    assert verify_physical(path).ok
    assert main(["doctor", str(path), "--physical"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert main(["inspect", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["lib_version"][0] == engine
    assert not get_active_envs()


def test_default_engine_is_explicit_and_ignores_process_default(tmp_path, monkeypatch):
    monkeypatch.setenv("LMDB_DEFAULT_LIB_VERSION", "0")
    path = tmp_path / "default"
    with LMDBWriter(path) as writer:
        writer.write_primitive(1, key="value")
    with LMDBReader(path) as reader:
        assert reader.get_lib_version()[0] == 1


@pytest.mark.parametrize("engine", [-1, 2, True, "1"])
def test_invalid_engine_rejected_before_any_output(tmp_path, engine):
    with pytest.raises(ValueError):
        LMDBWriter(tmp_path / "invalid", lib_version=engine)
    assert not list(tmp_path.iterdir())


def test_verify_on_publish_is_strict_bool(tmp_path):
    with pytest.raises(TypeError):
        LMDBWriter(tmp_path / "invalid", verify_on_publish=1)


def test_failed_private_staging_check_never_replaces_current(tmp_path, monkeypatch):
    path = tmp_path / "store"
    with LMDBWriter(path) as writer:
        writer.write_primitive(1, key="value")
    previous = (path / "CURRENT").read_bytes()
    monkeypatch.setattr(physical._verifier, "verify", lambda *args, **kwargs: ["bad free page"])
    with (
        pytest.raises(ValueError, match="physical validation failed"),
        LMDBWriter(path, overwrite=True, verify_on_publish=True) as writer,
    ):
        writer.write_primitive(2, key="value")
    assert (path / "CURRENT").read_bytes() == previous
    assert len(list((path / "versions").iterdir())) == 1
    assert not list(tmp_path.glob(".store.*.tmp"))
    with LMDBReader(path) as reader:
        assert reader.read("value") == 1


def test_empty_verified_writer_creates_a_valid_native_file(tmp_path):
    path = tmp_path / "empty"
    with LMDBWriter(path, verify_on_publish=True):
        pass
    assert verify_physical(path).ok
    assert verify_store(path).checked_entries == 0


def test_offline_verification_never_opens_native_engine_or_touches_lock(tmp_path, monkeypatch):
    path = tmp_path / "store"
    with LMDBWriter(path) as writer:
        writer.write_primitive(7, key="value")
    generation = path / "versions" / (path / "CURRENT").read_text().strip()
    lock = generation / "lock.mdb"
    lock.write_bytes(b"untrusted supplied lock file")

    def forbidden(*args, **kwargs):
        pytest.fail("offline verifier opened native engine")

    monkeypatch.setattr(lmdb, "open", forbidden)
    result = verify_physical(path)
    assert result.ok
    assert lock.read_bytes() == b"untrusted supplied lock file"
    assert not get_active_envs()


def test_invalid_file_is_reported_without_native_open(tmp_path, monkeypatch, capsys):
    file = tmp_path / "untrusted.mdb"
    file.write_bytes(b"not an LMDB database")
    monkeypatch.setattr(lmdb, "open", lambda *a, **kw: pytest.fail("native engine"))
    result = verify_physical(file)
    assert not result.ok and result.error
    assert main(["doctor", str(file), "--physical"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize("engine", [0, 1])
def test_truncated_native_file_is_rejected_without_engine_open(tmp_path, monkeypatch, engine):
    path = tmp_path / f"corrupt-{engine}"
    with LMDBWriter(path, lib_version=engine) as writer:
        writer.write_primitive(1, key="value")
    data = path / "versions" / (path / "CURRENT").read_text().strip() / "data.mdb"
    with data.open("r+b") as stream:
        stream.truncate(64)
    monkeypatch.setattr(lmdb, "open", lambda *a, **kw: pytest.fail("native engine"))
    assert not verify_physical(data).ok


def test_offline_detail_cap_and_source_change_detection(tmp_path, monkeypatch):
    file = tmp_path / "copy.mdb"
    file.write_bytes(b"owned")
    monkeypatch.setattr(physical._verifier, "verify", lambda *a, **kw: ["one", "two"])
    result = verify_physical(file, max_issues=1)
    assert result.problem_count == 2 and result.problems == ("one",)

    def mutate(*a, **kw):
        file.write_bytes(b"changed owned file")
        return []

    monkeypatch.setattr(physical._verifier, "verify", mutate)
    with pytest.raises(SourceChangedError):
        verify_physical(file)


def test_backend_capabilities_and_cli(capsys):
    info = get_backend_info()
    assert info.binding_version.startswith("3.")
    assert {item[0] for item in info.engines} == {0, 1}
    assert main(["backend"]) == 0
    assert json.loads(capsys.readouterr().out)["binding_version"] == info.binding_version


def test_system_single_engine_capability_reporting(monkeypatch):
    def available(*, lib_version):
        if lib_version == 1:
            raise ValueError("system build has no LMDB 1.0")
        return (0, 9, 36)

    monkeypatch.setattr(lmdb, "version", available)
    assert get_backend_info().engines == ((0, 9, 36),)


@pytest.mark.parametrize("engine", [0, 1])
def test_structured_readonly_errors_pass_through(tmp_path, engine):
    path = tmp_path / f"readonly-{engine}"
    with LMDBWriter(path, lib_version=engine) as writer:
        writer.write_primitive(1, key="value")
    with open_environment(path) as handle:
        with pytest.raises(lmdb.ReadonlyError) as error:
            handle.environment.begin(write=True)
        assert isinstance(error.value.what, str)
        assert isinstance(error.value.code, int)
        assert isinstance(error.value.reason, str)


def test_cli_physical_and_full_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["doctor", str(tmp_path), "--physical", "--full"])
    assert error.value.code == 2
