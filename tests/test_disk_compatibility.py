"""Historical key/body evidence is preserved; obsolete store envelopes are rejected."""

import json
from pathlib import Path

import lmdb
import numpy as np
import pytest

from amber_lmdb import LMDBReader, LMDBWriter, decode_key
from amber_lmdb.integrity import _parse_metadata, _verify_checksum
from amber_lmdb.lmdb import _METADATA_DB_NAME


def _fixture():
    return json.loads((Path(__file__).parent / "fixtures/mmfabric_0_2_2.json").read_text())


def _create_frozen_store(path):
    fixture = _fixture()
    generation = path / "versions" / ("v-" + "a" * 32)
    generation.mkdir(parents=True)
    (path / "FORMAT").write_text(fixture["format"], encoding="ascii")
    (path / "CURRENT").write_text(generation.name + "\n", encoding="ascii")
    with lmdb.open(str(generation), max_dbs=1) as env:
        tags = env.open_db(b"mmfabric.metadata.v1", create=True)
        with env.begin(write=True) as txn:
            for row in fixture["rows"]:
                key = bytes.fromhex(row["key"])
                txn.put(key, bytes.fromhex(row["value"]))
                txn.put(key, row["tag"].encode(), db=tags)


def test_rejects_old_store_format_without_mutating_historical_bytes(tmp_path):
    path = tmp_path / "legacy.lmdb"
    _create_frozen_store(path)
    marker = (path / "FORMAT").read_bytes()
    pointer = (path / "CURRENT").read_bytes()
    with (
        LMDBReader(path) as reader,
        pytest.raises(ValueError, match="unsupported LMDB dataset format"),
    ):
        reader.read_as_primitive(("session", "count"))
    assert (path / "FORMAT").read_bytes() == marker
    assert (path / "CURRENT").read_bytes() == pointer
    assert not (path / ".coordination").exists()


def test_new_writer_retains_historical_key_body_bytes_with_new_integrity_envelope(tmp_path):
    path = tmp_path / "new.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_dict({"language": "日本語", "numpy_type": np.ndarray, "count": 2}, key="meta")
        writer.write_ndarray(
            np.arange(6, dtype=np.float32).reshape(2, 3), key=("features", "audio"), index=3
        )
        writer.write_primitive(-17, key=("session", "count"))
        writer.write_sequence(["x", "日本語"], key="labels")
    generation = path / "versions" / (path / "CURRENT").read_text().strip()
    with lmdb.open(str(generation), readonly=True, max_dbs=1) as env:
        tags = env.open_db(_METADATA_DB_NAME, create=False)
        with env.begin() as txn:
            for row in _fixture()["rows"]:
                key = bytes.fromhex(row["key"])
                assert txn.get(key) == bytes.fromhex(row["value"])
                metadata = _parse_metadata(txn.get(key, db=tags))
                assert metadata.writer_tag == row["tag"]
                _verify_checksum(key, metadata, txn.get(key))
                assert decode_key(key)[0] is not None
