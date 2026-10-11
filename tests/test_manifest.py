"""Opt-in writer manifests summarize a generation without changing its content identity."""

from datetime import datetime

import numpy as np
import pytest

import amber_lmdb.manifest as manifest_module
from amber_lmdb import (
    LMDBOverlayReader,
    LMDBReader,
    LMDBWriter,
    array_statistics,
    codec,
    diff_stores,
    fingerprint_store,
    get_loc,
    inspect_store,
    verify_store,
)
from amber_lmdb.lmdb import _MANIFEST_KEY

SHAPES = [(2, 3), (5, 1), (4, 4)]


def _write(path, **options):
    with LMDBWriter(path, map_size=1024**2, **options) as writer:
        writer.write_str("hello", key="title")
        for index, shape in enumerate(SHAPES):
            writer.write_ndarray(np.zeros(shape, np.float32), key=("s1", "feat"), index=index * 2)
        writer.write_ndarray(np.zeros(3, np.int16), key=("s1", "feat"))
        writer.write_primitive(1, key="count", index=4)
        writer.write_primitive(2, key="count", index=9)
    return path


def test_no_manifest_by_default(tmp_path):
    with LMDBReader(_write(tmp_path / "plain.lmdb")) as reader:
        assert reader.manifest() is None


def test_manifest_contents(tmp_path):
    path = _write(tmp_path / "m.lmdb", manifest=True, provenance={"source": "unit", "n": 3})
    with LMDBReader(path) as reader:
        manifest = reader.manifest()
        engine = reader.get_lib_version()
        keys = [group.key for page in reader.iter_keys() for group in page]
    assert manifest is not None
    assert manifest.entries == 7 and manifest.fingerprint == fingerprint_store(path).digest
    assert manifest.checksums == ("blake2b-256",) and manifest.engine == engine
    assert manifest.provenance == {"source": "unit", "n": 3}
    assert not manifest.groups_truncated and manifest.body_bytes > 0
    assert datetime.fromisoformat(manifest.created_utc).tzinfo is not None
    assert manifest.amber_lmdb_version and manifest.py_lmdb_version
    groups = {group.key: group for group in manifest.groups}
    assert list(groups) == keys
    feat = groups[("s1", "feat")]
    assert feat.tags == (("write_ndarray", 4),)
    assert (feat.unindexed, feat.indexed, feat.first_index, feat.last_index) == (True, 3, 0, 4)
    (floats,) = [item for item in feat.arrays if item.dtype == "float32"]
    assert (floats.rank, floats.count, floats.min_shape, floats.max_shape) == (2, 3, (2, 1), (5, 4))
    (ints,) = [item for item in feat.arrays if item.dtype == "int16"]
    assert (ints.rank, ints.count, ints.min_shape, ints.max_shape) == (1, 1, (3,), (3,))
    count = groups["count"]
    assert (count.unindexed, count.indexed, count.first_index, count.last_index) == (
        False,
        2,
        4,
        9,
    )


def test_manifest_does_not_change_content_identity(tmp_path):
    plain = _write(tmp_path / "plain.lmdb")
    with_manifest = _write(tmp_path / "m.lmdb", manifest=True)
    assert fingerprint_store(plain).digest == fingerprint_store(with_manifest).digest
    assert diff_stores(plain, with_manifest).identical


def test_stores_with_manifests_stay_readable_by_every_scan(tmp_path):
    path = _write(tmp_path / "m.lmdb", manifest=True)
    assert verify_store(path).ok
    assert array_statistics(path).array_entries == 4
    assert inspect_store(path, entry_limit=20).data_entries == 8  # seven entries plus the manifest


def test_overlay_layers_may_each_have_a_manifest(tmp_path):
    def other(writer):
        writer.write_str("other", key="different")

    first = _write(tmp_path / "a.lmdb", manifest=True)
    with LMDBWriter(tmp_path / "b.lmdb", map_size=1024**2, manifest=True) as writer:
        other(writer)
    with LMDBOverlayReader([first, tmp_path / "b.lmdb"]) as reader:
        assert reader.check_disjoint() == 8
        manifests = reader.manifests()
        assert [m.entries for m in manifests if m is not None] == [7, 1]
        assert len([g for page in reader.iter_keys() for g in page]) == 4


def test_manifest_groups_can_be_truncated(tmp_path):
    path = _write(tmp_path / "m.lmdb", manifest=True, manifest_max_groups=2)
    with LMDBReader(path) as reader:
        manifest = reader.manifest()
    assert manifest is not None and manifest.groups_truncated
    assert len(manifest.groups) == 2 and manifest.entries == 7


@pytest.mark.parametrize("index", [None, 0])
def test_reserved_key_cannot_be_written(tmp_path, index):
    with LMDBWriter(tmp_path / "x.lmdb") as writer, pytest.raises(ValueError, match="reserved"):
        writer.write_str("mine", key=_MANIFEST_KEY, index=index)


@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"provenance": {"a": 1}}, ValueError),
        ({"manifest": 1}, TypeError),
        ({"manifest": True, "provenance": {1: "a"}}, TypeError),
        ({"manifest": True, "provenance": {"a": object()}}, TypeError),
        ({"manifest": True, "manifest_max_groups": 0}, ValueError),
    ],
)
def test_manifest_options_are_validated_before_writing(tmp_path, options, error):
    with pytest.raises(error):
        LMDBWriter(tmp_path / "x.lmdb", **options)
    assert not (tmp_path / "x.lmdb").exists()


def test_failed_manifest_publishes_nothing(tmp_path, monkeypatch):
    path = _write(tmp_path / "m.lmdb")
    before = (path / "CURRENT").read_text()

    def fail(*args, **kwargs):
        raise RuntimeError("manifest failed")

    monkeypatch.setattr(manifest_module, "_build", fail)
    with (
        pytest.raises(RuntimeError, match="manifest failed"),
        LMDBWriter(path, overwrite=True, manifest=True) as writer,
    ):
        writer.write_str("new", key="title")
    assert (path / "CURRENT").read_text() == before


def test_malformed_manifest_is_reported(tmp_path):
    path = tmp_path / "bad.lmdb"
    with LMDBWriter(path) as writer:
        writer._put_tagged(
            get_loc(_MANIFEST_KEY),
            codec.dict_to_bytes({"format": "other"}),
            "write_dict",
            reserved=True,
        )
    with LMDBReader(path) as reader, pytest.raises(ValueError, match="manifest format"):
        reader.manifest()
