"""Writers record body locations; readers prefetch whole bodies from them."""

import pickle
from pathlib import Path

import numpy as np
import pytest

import amber_lmdb.lmdb as storage
from amber_lmdb import LMDBReader, LMDBWriter, _locations, get_loc, verify_store

LARGE = 40_000  # bytes per int8 body: well above MIN_BYTES


def _body(index: int, size: int = LARGE) -> np.ndarray:
    return np.full(size, index % 120, dtype=np.int8)


def _write(path, count=5, **writer_options):
    with LMDBWriter(path, map_size=64 * 1024**2, **writer_options) as writer:
        for index in range(count):
            writer.write_ndarray(_body(index), key=("s", "tensor", "a"), index=index)
            writer.write_ndarray(_body(index + 50), key=("s", "tensor", "b"), index=index)
            if index == 2:
                writer.commit()
        writer.write_ndarray(_body(7, 100), key=("s", "small"), index=0)
        writer.write_ndarray(_body(8), key="unindexed")


def _data_file(path):
    return path / "versions" / (path / "CURRENT").read_text().strip() / "data.mdb"


def _table(reader, key):
    environment = reader._get_env()
    return reader._location_table(environment, get_loc(key, 0)[:-9])


def test_recorded_locations_point_at_the_bodies(tmp_path):
    path = tmp_path / "store.lmdb"
    _write(path)
    data = _data_file(path).read_bytes()
    with LMDBReader(path) as reader:
        environment = reader._get_env()
        for name in ("a", "b"):
            table = _table(reader, ("s", "tensor", name))
            assert table is not None
            assert table[:, 0].tolist() == [0, 1, 2, 3, 4]
            with environment.begin(buffers=True) as transaction:
                for index, offset, length in table.tolist():
                    body = bytes(transaction.get(get_loc(("s", "tensor", name), index)))
                    assert data[offset : offset + length] == body
        # Small bodies and unindexed keys are not recorded.
        assert _table(reader, ("s", "small")) is None
        keys = [group.key for page in reader.iter_keys() for group in page]
    assert all(key[0] != "amber-lmdb" for key in keys if isinstance(key, tuple))
    assert verify_store(path).ok


def test_the_last_write_of_an_index_wins(tmp_path):
    path = tmp_path / "store.lmdb"
    with LMDBWriter(path, map_size=64 * 1024**2) as writer:
        writer.write_ndarray(_body(1), key=("k",), index=3)
        writer.commit()
        writer.write_ndarray(_body(2, LARGE * 2), key=("k",), index=3)
    data = _data_file(path).read_bytes()
    with LMDBReader(path) as reader:
        table = _table(reader, ("k",))
        assert table is not None and table.shape == (1, 3)
        _, offset, length = table[0].tolist()
        assert length > LARGE * 2
        expected = bytes(reader._get_env().begin(buffers=True).get(get_loc(("k",), 3)))
    assert data[offset : offset + length] == expected


def _record_advice(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _locations,
        "FADVISE",
        lambda fd, offset, length, advice: calls.append((offset, length, advice)),
    )
    return calls


def test_reads_ask_for_whole_bodies_in_pieces(tmp_path, monkeypatch):
    path = tmp_path / "store.lmdb"
    _write(path)
    monkeypatch.setattr(_locations, "PIECE", 16 * 1024)
    calls = _record_advice(monkeypatch)
    with LMDBReader(path) as reader:
        arrays = reader.read_as_ndarray_many([("s", "tensor", "a")] * 3, indices=[4, 0, 1])
        table = _table(reader, ("s", "tensor", "a"))
    assert [array[0] for array in arrays] == [4, 0, 1]
    assert calls and all(length <= 16 * 1024 for _, length, _ in calls)
    assert all(advice == _locations.WILLNEED for *_, advice in calls)

    def covered(position):
        return any(offset <= position < offset + length for offset, length, _ in calls)

    for index in (0, 1, 4):
        _, offset, length = table[index].tolist()
        assert covered(offset) and covered(offset + length - 1)

    calls.clear()
    with LMDBReader(path) as reader:
        assert reader.read_as_ndarray(("s", "tensor", "b"), index=2)[0] == 52
    assert calls
    calls.clear()
    with LMDBReader(path, prefetch=False) as reader:
        assert not reader.prefetch
        reader.read_as_ndarray(("s", "tensor", "b"), index=2)
    assert not calls


def test_neighboring_bodies_are_requested_together(monkeypatch):
    calls = _record_advice(monkeypatch)
    _locations.advise(9, [(300_000, 50_000), (100_000, 40_000), (141_000, 40_000)])
    starts = [offset for offset, _, _ in calls]
    assert starts[0] == 100_000
    assert sum(length for offset, length, _ in calls if offset < 300_000) == 81_000
    assert calls[-1][:2] == (300_000, 50_000)


def test_a_base_copy_drops_copied_tables(tmp_path):
    path = tmp_path / "store.lmdb"
    _write(path)
    with LMDBWriter(path, map_size=64 * 1024**2, base="current") as writer:
        writer.write_ndarray(_body(9), key=("s", "tensor", "c"), index=0)
    data = _data_file(path).read_bytes()
    with LMDBReader(path) as reader:
        assert _table(reader, ("s", "tensor", "a")) is None
        table = _table(reader, ("s", "tensor", "c"))
        assert table is not None
        _, offset, length = table[0].tolist()
        expected = bytes(
            reader._get_env().begin(buffers=True).get(get_loc(("s", "tensor", "c"), 0))
        )
        assert reader.read_as_ndarray(("s", "tensor", "a"), index=1)[0] == 1
    assert data[offset : offset + length] == expected
    assert verify_store(path).ok


def test_state_is_per_process_and_not_pickled(tmp_path):
    path = tmp_path / "store.lmdb"
    _write(path)
    reader = LMDBReader(path)
    reader.read_as_ndarray(("s", "tensor", "a"), index=0)
    copy = pickle.loads(pickle.dumps(reader))
    assert copy._data_descriptor is None and copy._location_tables == {}
    assert copy.prefetch
    assert copy.read_as_ndarray(("s", "tensor", "a"), index=3)[0] == 3
    copy.close()
    reader.close()
    assert reader._data_descriptor is None


def test_prefetch_must_be_bool(tmp_path):
    with pytest.raises(TypeError, match="prefetch"):
        LMDBReader(tmp_path / "store.lmdb", prefetch=1)


def test_reserved_component_matches_the_store_namespace():
    assert _locations.RESERVED_COMPONENT == storage._RESERVED_COMPONENT


def test_verify_store_prefetches_ahead_of_the_scan(tmp_path, monkeypatch):
    path = tmp_path / "store.lmdb"
    _write(path)
    calls = _record_advice(monkeypatch)
    report = verify_store(path)
    assert report.ok
    with LMDBReader(path) as reader:
        tables = [_table(reader, ("s", "tensor", name)) for name in ("a", "b")]

    def covered(position):
        return any(offset <= position < offset + length for offset, length, _ in calls)

    for table in tables:
        for _, offset, length in table.tolist():
            assert covered(offset) and covered(offset + length - 1)


def test_scan_prefetch_stops_at_the_budget(tmp_path, monkeypatch):
    path = tmp_path / "store.lmdb"
    _write(path)
    calls = _record_advice(monkeypatch)
    with LMDBReader(path) as reader:
        environment = reader._get_env()
        generation = str(Path(reader._env_key[1]))
        prefetcher = _locations.ScanPrefetcher(generation, budget=2 * LARGE + 1000)
        with environment.begin(buffers=True) as transaction:
            prefetcher.before(transaction, get_loc(("s", "tensor", "a"), 1))
            first = len(calls)
            # Indices 1 and 2 were requested together; index 2 needs nothing new.
            prefetcher.before(transaction, get_loc(("s", "tensor", "a"), 2))
            assert len(calls) == first
            prefetcher.before(transaction, get_loc(("s", "tensor", "a"), 3))
            assert len(calls) > first
            # A key without a table, and an unindexed key, request nothing.
            count = len(calls)
            prefetcher.before(transaction, get_loc(("s", "small"), 0))
            prefetcher.before(transaction, get_loc("unindexed"))
            assert len(calls) == count
        prefetcher.close()
        offset = int(_table(reader, ("s", "tensor", "a"))[3, 1])
    assert any(o <= offset < o + n for o, n, _ in calls[first:])
