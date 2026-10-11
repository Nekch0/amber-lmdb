"""Generation inventory reports filesystem data, never permission to delete."""

import json

import pytest

import amber_lmdb.generations as generations
from amber_lmdb import LMDBWriter, SourceChangedError, list_generations
from amber_lmdb.cli import main


def _write(path, overwrite=False):
    with LMDBWriter(path, overwrite=overwrite) as writer:
        writer.write_primitive(1, key="value")


def test_generations_count_sizes_order_and_cli(tmp_path, capsys):
    path = tmp_path / "store"
    _write(path)
    previous = (path / "CURRENT").read_text().strip()
    _write(path, overwrite=True)
    listing = list_generations(path)
    assert len(listing) == 2 and sum(item.current for item in listing) == 1
    current = (path / "CURRENT").read_text().strip()
    assert [item.name for item in listing] == [previous, current]  # publication order
    assert [item.sequence for item in listing] == [1, 2]
    assert next(item for item in listing if item.name == previous).current is False
    for item in listing:
        assert item.size_bytes > 0 and item.modified_ns > 0
    assert main(["generations", str(path)]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["current"] == current and len(inventory["generations"]) == 2
    assert [row["generation"] for row in inventory["history"]] == [previous, current]


def test_generation_inventory_ignores_staging_and_rejects_current_switch(tmp_path, monkeypatch):
    path = tmp_path / "store"
    _write(path)
    (path / "versions" / "staging").mkdir()
    assert len(list_generations(path)) == 1
    real = generations._read_current
    calls = []

    def switching(root):
        calls.append(root)
        return real(root) if len(calls) == 1 else "v-" + "0" * 32

    monkeypatch.setattr(generations, "_read_current", switching)
    with pytest.raises(SourceChangedError):
        list_generations(path)


def test_generation_inventory_reports_nonregular_files(tmp_path):
    path = tmp_path / "store"
    _write(path)
    current = path / "versions" / (path / "CURRENT").read_text().strip()
    (current / "data.mdb").unlink()
    (current / "data.mdb").mkdir()
    (info,) = list_generations(path)
    assert info.current and info.error is not None and "regular file" in info.error
