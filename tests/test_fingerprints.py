"""Fingerprints and differences come from integrity envelopes, never from bodies."""

import json
from contextlib import contextmanager
from dataclasses import replace

import numpy as np
import pytest

import amber_lmdb.fingerprints as fingerprints
from amber_lmdb import (
    LMDBWriter,
    diff_stores,
    fingerprint_store,
    get_loc,
    open_environment,
    verify_store,
)
from amber_lmdb.cli import main
from amber_lmdb.collections import SourceChangedError

ITEMS = [
    ("text", None, "hello"),
    ("count", None, 3),
    ("values", 0, np.arange(4, dtype=np.float32)),
    ("values", 1, np.ones((2, 2), np.int64)),
    (("nested", "key"), 7, "deep"),
]


def _write(path, items=ITEMS, *, reverse=False, engine=1, checksum="blake2b-256"):
    with LMDBWriter(path, map_size=1024**2, lib_version=engine, checksum=checksum) as writer:
        for key, index, value in reversed(items) if reverse else items:
            if isinstance(value, np.ndarray):
                writer.write_ndarray(value, key=key, index=index)
            elif isinstance(value, str):
                writer.write_str(value, key=key, index=index)
            else:
                writer.write_primitive(value, key=key, index=index)
    return path


def test_fingerprint_ignores_write_order_and_engine(tmp_path):
    first = fingerprint_store(_write(tmp_path / "a.lmdb"))
    reordered = fingerprint_store(_write(tmp_path / "b.lmdb", reverse=True, engine=0))
    assert first.digest == reordered.digest
    assert first.entries == len(ITEMS)
    assert first.algorithms == ("blake2b-256",)


def test_named_generations_of_one_root_can_be_compared(tmp_path):
    root = _write(tmp_path / "a.lmdb")
    old = (root / "CURRENT").read_text().strip()
    original = fingerprint_store(root)
    with LMDBWriter(root, map_size=1024**2, overwrite=True) as writer:
        writer.write_str("only", key="new")
    assert fingerprint_store(root, generation=old).digest == original.digest
    assert fingerprint_store(root).digest != original.digest
    difference = diff_stores(root, root, before_generation=old)
    assert difference.added == (("new", None),) and difference.removed_count == len(ITEMS)
    with pytest.raises(ValueError, match="generation"):
        fingerprint_store(root, generation="../a")
    with pytest.raises(FileNotFoundError):
        fingerprint_store(root, generation="v-" + "0" * 32)


def test_content_tag_and_algorithm_changes_change_the_fingerprint(tmp_path):
    base = fingerprint_store(_write(tmp_path / "base.lmdb")).digest
    edited = [*ITEMS[:-1], (("nested", "key"), 7, "deeP")]
    assert fingerprint_store(_write(tmp_path / "edited.lmdb", edited)).digest != base
    retagged = [*ITEMS[:1], ("count", None, "3"), *ITEMS[2:]]
    assert fingerprint_store(_write(tmp_path / "retag.lmdb", retagged)).digest != base
    sha = fingerprint_store(_write(tmp_path / "sha.lmdb", checksum="sha256"))
    assert sha.digest != base and sha.algorithms == ("sha256",)


def test_fingerprint_reads_envelopes_only(tmp_path):
    path = _write(tmp_path / "a.lmdb")
    before = fingerprint_store(path).digest
    with (
        open_environment(path, readonly=False) as handle,
        handle.environment.begin(write=True) as txn,
    ):
        raw = txn.get(get_loc("text"))
        txn.put(get_loc("text"), raw[:-1] + bytes([raw[-1] ^ 1]))
    # The body no longer matches its envelope: the fingerprint still describes the
    # recorded content, and verify_store reports the damaged body.
    assert fingerprint_store(path).digest == before
    assert verify_store(path).error_count == 1


def test_fingerprint_rejects_a_generation_changed_in_place(tmp_path, monkeypatch):
    path = _write(tmp_path / "a.lmdb")
    real = fingerprints._opened

    @contextmanager
    def shifted(target, **options):
        with real(target, **options) as (env, tags, snapshot):
            yield env, tags, replace(snapshot, last_txnid=snapshot.last_txnid + 1)

    monkeypatch.setattr(fingerprints, "_opened", shifted)
    with pytest.raises(SourceChangedError):
        fingerprint_store(path)


def test_diff_classifies_entries(tmp_path):
    before = _write(tmp_path / "before.lmdb")
    changed = [
        ("text", None, "hello"),
        ("count", None, 4),
        ("values", 0, np.arange(4, dtype=np.float32)),
        (("nested", "key"), 8, "deep"),
        ("added", None, "new"),
    ]
    after = _write(tmp_path / "after.lmdb", changed)
    difference = diff_stores(before, after)
    assert (difference.unchanged, difference.changed_count) == (2, 1)
    assert set(difference.added) == {("added", None), (("nested", "key"), 8)}
    assert set(difference.removed) == {("values", 1), (("nested", "key"), 7)}
    assert difference.changed == (("count", None),)
    assert not difference.identical
    capped = diff_stores(before, after, max_listed=1)
    assert len(capped.added) == 1 and capped.added_count == 2


def test_identical_generations(tmp_path):
    a = _write(tmp_path / "a.lmdb")
    b = _write(tmp_path / "b.lmdb", reverse=True)
    difference = diff_stores(a, b)
    assert difference.identical and difference.unchanged == len(ITEMS)


def test_algorithm_change_is_undetermined_until_bodies_are_compared(tmp_path):
    blake = _write(tmp_path / "blake.lmdb")
    sha = _write(tmp_path / "sha.lmdb", checksum="sha256")
    difference = diff_stores(blake, sha)
    assert difference.undetermined_count == len(ITEMS) and not difference.identical
    assert diff_stores(blake, sha, compare_bodies=True).identical
    edited = [*ITEMS[:-1], (("nested", "key"), 7, "deeP")]
    sha_edited = _write(tmp_path / "sha-edited.lmdb", edited, checksum="sha256")
    resolved = diff_stores(blake, sha_edited, compare_bodies=True)
    assert resolved.changed == ((("nested", "key"), 7),)
    assert resolved.unchanged == len(ITEMS) - 1


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda p: fingerprint_store(p, page_size=0), ValueError),
        (lambda p: diff_stores(p, p, max_listed=-1), ValueError),
        (lambda p: diff_stores(p, p, compare_bodies=1), TypeError),
        (lambda p: diff_stores(p, p, max_value_bytes=0), ValueError),
    ],
)
def test_arguments_are_validated(tmp_path, call, error):
    path = _write(tmp_path / "a.lmdb")
    with pytest.raises(error):
        call(path)


def test_cli_fingerprint_and_diff(tmp_path, capsys):
    a = _write(tmp_path / "a.lmdb")
    b = _write(tmp_path / "b.lmdb", [*ITEMS, ("extra", None, "x")])
    assert main(["fingerprint", str(a), "--progress"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["digest"] == fingerprint_store(a).digest
    assert "fingerprint_store:" in output.err
    assert main(["diff", str(a), str(a)]) == 0
    assert json.loads(capsys.readouterr().out)["identical"] is True
    assert main(["diff", str(a), str(b)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["added_count"] == 1 and report["added"] == [["extra", None]]
