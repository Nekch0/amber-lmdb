"""Test typed-batch transaction boundaries and opened-generation identities."""

import pickle
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock

import pytest

import amber_lmdb.codec as codec
from amber_lmdb import LMDBReader, LMDBSnapshot, LMDBWriter, register_bytes_codec


@pytest.mark.parametrize("method", ["read_as_dict_many", "read_as_primitive_many"])
@pytest.mark.parametrize(
    "keys,options,error",
    [
        ("abc", {}, TypeError),
        (["a"], {"max_items": True}, TypeError),
        (["a"], {"max_items": 0}, ValueError),
        (["a", "b"], {"max_items": 1}, ValueError),
        (["a"], {"indices": []}, ValueError),
        (["a"], {"indices": [-1]}, ValueError),
        ([("a", 1)], {}, TypeError),
    ],
)
def test_invalid_batch_does_not_open_store(tmp_path, method, keys, options, error):
    with LMDBReader(tmp_path / "missing") as reader:
        with pytest.raises(error):
            getattr(reader, method)(keys, **options)
        assert getattr(reader, method)([]) == []
        assert reader._env is None


@pytest.mark.parametrize("kind", ["dict", "primitive"])
def test_typed_batch_is_one_transaction_and_decodes_after_end(tmp_path, monkeypatch, kind):
    path = tmp_path / "data"
    values = [{"nested": [0]}, {"nested": [1]}] if kind == "dict" else [0, 1]
    with LMDBWriter(path, map_size=1024**2) as writer:
        for index, value in enumerate(values):
            getattr(writer, f"write_{kind}")(value, key=("s", "value"), index=index)
        writer.write_str("wrong tag", key="bad")
    with LMDBReader(path) as reader:
        environment = reader._get_env()
        active = []
        starts = []
        decoder = getattr(codec, f"bytes_to_{kind}")

        class Transaction:
            def __enter__(self):
                self.txn = environment.begin(write=False, buffers=True)
                active.append(True)
                return self.txn

            def __exit__(self, *exc):
                self.txn.abort()
                active.pop()

        def begin(**kwargs):
            starts.append(kwargs)
            return Transaction()

        def decode(*args, **kwargs):
            assert not active
            return decoder(*args, **kwargs)

        monkeypatch.setattr(reader, "_get_env", lambda: MagicMock(begin=begin))
        monkeypatch.setattr(codec, f"bytes_to_{kind}", decode)
        read = getattr(reader, f"read_as_{kind}_many")
        result = read([("s", "value")] * 3, indices=[1, 0, 1], max_items=3)
        assert result == [values[1], values[0], values[1]]
        assert starts == [{"write": False, "buffers": True}]
        if kind == "dict":
            result[0]["nested"].append(9)
            assert result[2] == values[1]
        with pytest.raises(ValueError):
            read(["bad"])
        with pytest.raises(KeyError):
            read([("s", "value"), "missing"], indices=[0, None])
        assert not active
    assert result[1] == values[0]


def test_dict_batch_strict_payload_and_custom_decoder(tmp_path):
    from frozendict import frozendict

    class Point:
        def __init__(self, value):
            self.value = value

    decoded = []

    def decode(value):
        decoded.append(value)
        return Point(value)

    register_bytes_codec("tests.typed-batch.point", Point, lambda point: point.value, decode)
    path = tmp_path / "data"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_dict(frozendict({"a": 1}), key="frozen")
        writer.write_dict({"point": Point(7)}, key="custom")
    with LMDBReader(path) as reader:
        with pytest.raises(ValueError, match="expected dict"):
            reader.read_as_dict_many(["frozen"])
        with pytest.raises(ValueError, match="allow_custom"):
            reader.read_as_dict_many(["custom"])
        assert not decoded
        assert reader.read_as_dict_many(["custom"], allow_custom=True)[0]["point"].value == 7
        with pytest.raises(TypeError):
            reader.read_as_dict_many([], allow_custom=1)
    assert decoded == [7]


def test_snapshot_tracks_opened_generation_refresh_and_pickle(tmp_path):
    path = tmp_path / "data"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_primitive(1, key="value")
    with LMDBReader(path) as reader:
        first = reader.get_snapshot()
        assert isinstance(first, LMDBSnapshot)
        assert first.root == str(path.resolve())
        assert first.last_txnid > 0
        with pytest.raises(FrozenInstanceError):
            first.generation = "changed"
        with LMDBWriter(path, overwrite=True, map_size=1024**2) as writer:
            writer.write_primitive(2, key="value")
        assert reader.get_snapshot() == first
        assert reader.read_as_primitive("value") == 1
        with pickle.loads(pickle.dumps(reader)) as restored:
            # A copy reads the generation the original opened, as DataLoader workers must.
            assert restored.get_snapshot().generation == first.generation
            assert restored.read_as_primitive("value") == 1
            restored.refresh()
            second = restored.get_snapshot()
            assert second.root == first.root
            assert second.generation != first.generation
            assert restored.read_as_primitive("value") == 2
        reader.refresh()
        assert reader.get_snapshot() == second
        assert reader.read_as_primitive("value") == 2
    assert LMDBSnapshot(first.root, first.generation, first.last_txnid) == first
