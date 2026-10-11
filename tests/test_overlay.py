"""An overlay reads several stores as one keyspace and rejects repeated entries."""

import pickle
import threading

import numpy as np
import pytest

from amber_lmdb import (
    IntegrityError,
    KeyGroup,
    LMDBOverlayReader,
    LMDBWriter,
    OperationCancelled,
    OverlayConflictError,
    ReadLimitError,
    RecordTable,
    get_loc,
    open_environment,
)


def _write(path, build):
    with LMDBWriter(path, map_size=4 * 1024**2) as writer:
        build(writer)
        writer.commit()
    return path


@pytest.fixture
def layers(tmp_path):
    def base(writer):
        writer.write_str("base", key="meta")
        for index in range(3):
            writer.write_dict({"row": index}, key="rows", index=index)
        for index in range(4):
            writer.write_ndarray(np.full(4, index, np.float32), key="features", index=index)

    def extra(writer):
        writer.write_str("extra", key="extra_meta")
        for index in (3, 4):
            writer.write_dict({"row": index}, key="rows", index=index)
        for index in (4, 5):
            writer.write_ndarray(np.full(4, index, np.float32), key="features", index=index)
        for index in range(4):
            writer.write_ndarray(np.full(2, -index, np.int16), key=("pitch",), index=index)

    return _write(tmp_path / "base.lmdb", base), _write(tmp_path / "extra.lmdb", extra)


def test_reads_resolve_each_key_to_its_layer_in_request_order(layers):
    with LMDBOverlayReader(layers) as reader:
        values = reader.read_many(
            ["features", ("pitch",), "meta", "features", "extra_meta"],
            indices=[5, 1, None, 0, None],
        )
        assert values[0].tolist() == [5.0] * 4
        assert values[1].tolist() == [-1, -1] and values[1].dtype == np.int16
        assert values[2:3] == ["base"] and values[4] == "extra"
        assert values[3].tolist() == [0.0] * 4
        assert reader.read_as_ndarray("features", index=4).tolist() == [4.0] * 4
        assert reader.read_as_str("extra_meta") == "extra"
        assert reader.read_as_dict("rows", index=3) == {"row": 3}
        headers = reader.read_array_headers_many(["features", ("pitch",)], indices=[5, 2])
        assert [h.shape for h in headers] == [(4,), (2,)]
        assert reader.read_as_shape(("pitch",), index=0) == (2,)
        assert reader.read_as_dtype("features", index=1) == np.dtype(np.float32)
        with pytest.raises(KeyError):
            reader.read("missing")


def test_record_table_and_streams_span_layers(layers):
    with LMDBOverlayReader(layers) as reader:
        table = RecordTable(reader, "rows", lambda row: row["row"], count=5, batch_size=2)
        assert table.get_many([4, 0, 3, 1]) == [4, 0, 3, 1]
        assert list(table.iter_records()) == [0, 1, 2, 3, 4]
        keys, indices = ["features"] * 6, list(range(6))
        pages = list(reader.iter_read_pages(keys, indices=indices, max_bytes=200))
        assert len(pages) > 1
        assert [v[0] for page in pages for v in page] == [float(i) for i in range(6)]


def test_batch_byte_budget_covers_all_layers(layers):
    with LMDBOverlayReader(layers) as reader:
        single = len(reader.read_many(["features"], indices=[0])[0].tobytes())
        with pytest.raises(ReadLimitError):
            reader.read_many(["features"] * 6, indices=list(range(6)), max_bytes=4 * (single + 140))


def test_repeated_entries_raise_but_other_keys_still_read(layers, tmp_path):
    clash = _write(tmp_path / "clash.lmdb", lambda w: w.write_str("other", key="meta"))
    with LMDBOverlayReader([layers[0], clash]) as reader:
        assert reader.read_as_dict("rows", index=0) == {"row": 0}
        with pytest.raises(OverlayConflictError) as error:
            reader.read("meta")
        assert error.value.location == get_loc("meta")
        assert error.value.layers == (layers[0], clash)
        with pytest.raises(OverlayConflictError):
            reader.contains("meta")
        with pytest.raises(OverlayConflictError):
            reader.check_disjoint()


def test_check_disjoint_counts_entries_and_reports_progress(layers):
    reports = []
    with LMDBOverlayReader(layers) as reader:
        assert reader.check_disjoint(page_size=3, progress=reports.append) == 8 + 9
    assert reports[-1].finished and reports[-1].entries_done == 17
    assert reports[-1].operation == "check_disjoint"


def test_check_disjoint_can_be_cancelled(layers):
    cancel = threading.Event()
    cancel.set()
    with LMDBOverlayReader(layers) as reader, pytest.raises(OperationCancelled) as error:
        reader.check_disjoint(cancel=cancel)
    assert error.value.partial is None


def test_listing_merges_layers(layers):
    with LMDBOverlayReader(layers) as reader:
        assert reader.contains("features", 5) and not reader.contains("features", 6)
        assert reader.contains_many(["meta", "extra_meta", "rows"]) == [True, True, False]
        assert reader.count("features") == 6
        assert reader.count("rows") == 5
        assert [i for page in reader.iter_indices("features", page_size=4) for i in page] == list(
            range(6)
        )
        assert reader.index_range("features") == (0, 5)
        groups = [g for page in reader.iter_keys(page_size=2) for g in page]
        assert KeyGroup("features", False, 0, 5) in groups
        assert KeyGroup(("pitch",), False, 0, 3) in groups
        assert [g.key for g in groups] == sorted((g.key for g in groups), key=get_loc)
        assert len({get_loc(g.key) for g in groups}) == len(groups)


def test_interleaved_indices_are_allowed_but_repeated_indices_are_not(tmp_path):
    def evens(writer):
        for index in (0, 2, 4):
            writer.write_primitive(index, key="x", index=index)

    def odds(writer):
        for index in (1, 3):
            writer.write_primitive(index, key="x", index=index)

    def repeat(writer):
        writer.write_primitive(9, key="x", index=2)

    a, b, c = (
        _write(tmp_path / f"{n}.lmdb", f) for n, f in (("a", evens), ("b", odds), ("c", repeat))
    )
    with LMDBOverlayReader([a, b]) as reader:
        assert [g for page in reader.iter_keys() for g in page] == [KeyGroup("x", False, 0, 4)]
        assert reader.count("x") == 5
    with LMDBOverlayReader([a, c]) as reader:
        with pytest.raises(OverlayConflictError):
            reader.count("x")
        with pytest.raises(OverlayConflictError):
            list(reader.iter_keys())
        assert reader.read_as_primitive("x", index=4) == 4


def test_unindexed_entry_in_two_layers_is_a_listing_conflict(tmp_path):
    a = _write(tmp_path / "a.lmdb", lambda w: w.write_str("one", key="same"))
    b = _write(tmp_path / "b.lmdb", lambda w: w.write_str("two", key="same"))
    with LMDBOverlayReader([a, b]) as reader, pytest.raises(OverlayConflictError):
        list(reader.iter_keys())


def test_verify_checksums_option_reaches_every_layer(layers):
    with (
        open_environment(layers[1], readonly=False) as handle,
        handle.environment.begin(write=True) as txn,
    ):
        raw = txn.get(get_loc(("pitch",), 0))
        txn.put(get_loc(("pitch",), 0), raw[:-1] + bytes([raw[-1] ^ 1]))
    with LMDBOverlayReader(layers) as reader, pytest.raises(IntegrityError):
        reader.read_as_ndarray(("pitch",), index=0)
    with LMDBOverlayReader(layers, verify_checksums=False) as reader:
        assert reader.verify_checksums is False
        assert reader.read_as_ndarray(("pitch",), index=0)[-1] != 0


def test_snapshots_pickling_and_refresh(layers):
    with LMDBOverlayReader(layers) as reader:
        snapshots = reader.get_snapshot()
        assert len(snapshots) == 2 and snapshots[0].root != snapshots[1].root
        assert len(reader.get_lib_versions()) == 2
        clone = pickle.loads(pickle.dumps(reader))
        try:
            assert clone.read_as_str("extra_meta") == "extra"
            assert clone.paths == reader.paths
        finally:
            clone.close()
        reader.refresh()
        assert reader.read_as_str("meta") == "base"


@pytest.mark.parametrize(
    ("paths", "error"),
    [("single", TypeError), ([], ValueError)],
)
def test_constructor_rejects_invalid_layer_lists(paths, error):
    with pytest.raises(error):
        LMDBOverlayReader(paths)


def test_the_same_store_cannot_be_two_layers(layers):
    with pytest.raises(ValueError, match="distinct"):
        LMDBOverlayReader([layers[0], layers[0]])
