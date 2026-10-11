"""Chunked arrays: same values as plain arrays, row reads that touch only needed chunks."""

import pickle

import numpy as np
import pytest

import amber_lmdb.lmdb as lmdb_module
from amber_lmdb import (
    IntegrityError,
    LMDBOverlayReader,
    LMDBReader,
    LMDBWriter,
    OverlayConflictError,
    ReadLimitError,
    array_statistics,
    diff_stores,
    fingerprint_store,
    get_loc,
    inspect_store,
    verify_store,
)
from amber_lmdb.lmdb import _METADATA_DB_NAME, _chunk_location, open_environment

ARRAY = np.arange(1000 * 3, dtype=np.float32).reshape(1000, 3)
RANGES = [(0, 1), (0, 64), (63, 65), (100, 900), (990, 2000), (5, 5), (2000, 3000), (0, 1000)]


def _store(path, *, engine=1, checksum="blake2b-256", manifest=False, chunk_rows=64):
    with LMDBWriter(
        path, map_size=64 * 1024**2, lib_version=engine, checksum=checksum, manifest=manifest
    ) as writer:
        writer.write_ndarray(ARRAY, key="chunked", index=0, chunk_rows=chunk_rows)
        writer.write_ndarray(ARRAY, key="plain", index=0)
        writer.write_str("text", key="note")
    return path


def _tamper(path, change):
    with open_environment(path, readonly=False) as handle:
        tags = handle.environment.open_db(_METADATA_DB_NAME)
        with handle.environment.begin(write=True) as transaction:
            change(transaction, tags)


@pytest.mark.parametrize("engine", [0, 1])
@pytest.mark.parametrize("verify", [True, False])
def test_full_reads_match_plain_arrays(tmp_path, engine, verify):
    path = _store(tmp_path / "store.lmdb", engine=engine)
    with LMDBReader(path, verify_checksums=verify) as reader:
        for result in (
            reader.read_as_ndarray("chunked", 0),
            reader.read("chunked", 0),
            reader.read_many(["chunked", "plain"], indices=[0, 0])[0],
            next(reader.iter_read_pages(["chunked"], indices=[0]))[0],
        ):
            assert np.array_equal(result, ARRAY)
            assert result.dtype == ARRAY.dtype and not result.flags.writeable
        assert reader.read_as_ndarray("chunked", 0, dtype=np.float64).flags.writeable
        assert reader.read_as_shape("chunked", 0) == (1000, 3)
        assert reader.read_as_dtype("chunked", 0) == np.float32
        header = reader.read_array_headers_many(["chunked", "plain"], indices=[0, 0])
        assert header[0].chunk_rows == 64 and header[1].chunk_rows is None
        assert header[0].payload_bytes == header[1].payload_bytes == ARRAY.nbytes
        restored = pickle.loads(pickle.dumps(reader))
        assert np.array_equal(restored.read_as_ndarray("chunked", 0), ARRAY)
        restored.close()


@pytest.mark.parametrize(
    "array",
    [
        np.arange(10, dtype=">i2"),
        np.arange(24, dtype=np.float64).reshape(2, 3, 4),
        np.zeros((0, 4), np.float32),
        np.zeros((5, 0), np.int8),
        np.array([True, False, True]),
        np.array([(1, 2.0), (3, 4.0), (5, 6.0)], dtype=[("a", "<i4"), ("b", "<f8")]),
        np.asfortranarray(np.arange(12, dtype=np.int64).reshape(4, 3)),
    ],
)
@pytest.mark.parametrize("chunk_rows", [1, 2, 1000])
def test_shapes_dtypes_and_chunk_sizes(tmp_path, array, chunk_rows):
    path = tmp_path / "shapes.lmdb"
    with LMDBWriter(path, map_size=16 * 1024**2) as writer:
        writer.write_ndarray(array, key="value", chunk_rows=chunk_rows)
    with LMDBReader(path) as reader:
        restored = reader.read_as_ndarray("value")
        assert restored.dtype == array.dtype and np.array_equal(restored, array)
        rows = reader.read_array_rows("value", start=1, stop=3)
        assert rows.shape == array[1:3].shape and np.array_equal(rows, array[1:3])
    assert verify_store(path).ok


@pytest.mark.parametrize("verify", [True, False])
def test_row_reads_match_slicing(tmp_path, verify):
    path = _store(tmp_path / "store.lmdb")
    rng = np.random.default_rng(3)
    ranges = RANGES + [tuple(sorted(rng.integers(0, 1100, 2).tolist())) for _ in range(30)]
    with LMDBReader(path, verify_checksums=verify) as reader:
        for key in ("chunked", "plain"):
            for start, stop in ranges:
                rows = reader.read_array_rows(key, 0, start=start, stop=stop)
                assert np.array_equal(rows, ARRAY[start:stop]), (key, start, stop)
                assert not rows.flags.writeable
        assert reader.read_array_rows("chunked", 0, start=10, stop=5).shape == (0, 3)
        converted = reader.read_array_rows("chunked", 0, start=1, stop=3, dtype=np.float64)
        assert converted.dtype == np.float64 and converted.flags.writeable
        with pytest.raises(TypeError, match="casting='unsafe'"):
            reader.read_array_rows("chunked", 0, start=1, stop=3, dtype=np.int64)
        lossy = reader.read_array_rows(
            "chunked", 0, start=1, stop=3, dtype=np.int64, casting="unsafe"
        )
        assert lossy.dtype == np.int64
        many = reader.read_array_rows_many(
            ["plain", "chunked", "chunked"],
            indices=[0, 0, 0],
            starts=[5, 100, 100],
            stops=[10, 300, 300],
        )
        assert [item.shape for item in many] == [(5, 3), (200, 3), (200, 3)]
        assert np.array_equal(many[1], ARRAY[100:300])
        assert reader.read_array_rows_many([], starts=[], stops=[]) == []


def test_row_reads_touch_only_needed_chunks_and_own_their_memory(tmp_path, monkeypatch):
    path = _store(tmp_path / "store.lmdb")
    requested = []
    original = lmdb_module._chunk_location

    def recording(parent, number):
        requested.append(number)
        return original(parent, number)

    def owner(array):
        while isinstance(array, np.ndarray) and array.base is not None:
            array = array.base
        return array

    monkeypatch.setattr(lmdb_module, "_chunk_location", recording)
    for verify in (True, False):
        with LMDBReader(path, verify_checksums=verify) as reader:
            requested.clear()
            rows = reader.read_array_rows("chunked", 0, start=130, stop=260)
            assert requested == [2, 3, 4]
            assert owner(rows).nbytes == 3 * 64 * 12
            requested.clear()
            reader.read_array_rows("chunked", 0, start=128, stop=192)
            assert requested == [2]
            requested.clear()
            reader.read_array_rows("chunked", 0, start=500, stop=500)
            assert requested == []
            plain = reader.read_array_rows("plain", 0, start=0, stop=2)
            assert isinstance(owner(plain), bytes) and len(owner(plain)) == 2 * 12


def test_row_read_validation_and_budgets(tmp_path):
    path = _store(tmp_path / "store.lmdb")
    with LMDBWriter(tmp_path / "other.lmdb", map_size=16 * 1024**2) as writer:
        writer.write_ndarray(np.float32(1.0), key="scalar")
    with LMDBReader(path) as reader:
        with pytest.raises(ValueError, match="nonnegative"):
            reader.read_array_rows("chunked", 0, start=-1, stop=5)
        with pytest.raises(TypeError, match="integers"):
            reader.read_array_rows("chunked", 0, start=0, stop=5.0)
        with pytest.raises(ValueError, match="one value per key"):
            reader.read_array_rows_many(["chunked"], indices=[0], starts=[0, 1], stops=[1])
        with pytest.raises(ValueError, match="Invalid write/read method pair"):
            reader.read_array_rows("note", start=0, stop=1)
        with pytest.raises(KeyError):
            reader.read_array_rows("absent", start=0, stop=1)
        with pytest.raises(ReadLimitError):
            reader.read_array_rows_many(
                ["chunked"], indices=[0], starts=[0], stops=[1000], max_bytes=4096
            )
        assert reader.read_array_rows_many(
            ["chunked"], indices=[0], starts=[0], stops=[64], max_bytes=4096
        )[0].shape == (64, 3)
        with pytest.raises(ReadLimitError):
            reader.read_many(["chunked"], indices=[0], max_bytes=4096)
        _, metadata, value = reader._owned_entries([get_loc("chunked", 0)], "read", max_bytes=None)[
            0
        ]
        budget = len(value) + metadata.metadata_bytes  # The chunked array alone fits.
        pages = list(
            reader.iter_read_pages(["note", "chunked"], indices=[None, 0], max_bytes=budget)
        )
        assert [len(page) for page in pages] == [1, 1]
        assert np.array_equal(pages[1][0], ARRAY)
    with (
        LMDBReader(tmp_path / "other.lmdb") as reader,
        pytest.raises(ValueError, match="dimension"),
    ):
        reader.read_array_rows("scalar", start=0, stop=1)


def test_writer_validation_and_reserved_keys(tmp_path):
    with LMDBWriter(tmp_path / "w.lmdb", map_size=16 * 1024**2) as writer:
        for invalid in (0, -1, 1.5, True):
            with pytest.raises(ValueError, match="chunk_rows"):
                writer.write_ndarray(ARRAY, key="x", chunk_rows=invalid)
        with pytest.raises(ValueError, match="at least one dimension"):
            writer.write_ndarray(np.float32(1), key="x", chunk_rows=4)
        with pytest.raises(ValueError, match="reserved"):
            writer.write_ndarray(ARRAY, key=("amber-lmdb", "chunk", "00"), index=0, chunk_rows=4)
        with pytest.raises(ValueError, match="reserved"):
            writer.write_str("x", key=("amber-lmdb", "anything"))
        writer.write_ndarray(ARRAY, key="huge", chunk_rows=10**30)
    with LMDBReader(tmp_path / "w.lmdb") as reader:
        assert reader.read_array_headers_many(["huge"])[0].chunk_rows == 1000
        assert np.array_equal(reader.read_array_rows("huge", start=998, stop=1000), ARRAY[998:])
    assert verify_store(tmp_path / "w.lmdb").checked_entries == 2


@pytest.mark.parametrize("verify", [True, False])
def test_unchunked_row_reads_check_the_payload_size(tmp_path, verify):
    path = _store(tmp_path / "store.lmdb")
    location = get_loc("plain", 0)

    def lengthen(transaction, tags):
        body = transaction.get(location) + b"\x00" * 12
        transaction.put(location, body)
        transaction.put(
            location, lmdb_module._encode_metadata(location, "write_ndarray", body), db=tags
        )

    _tamper(path, lengthen)
    with (
        LMDBReader(path, verify_checksums=verify) as reader,
        pytest.raises(ValueError, match="payload size"),
    ):
        reader.read_array_rows("plain", 0, start=0, stop=1)


def test_replacing_and_deleting_chunked_arrays_removes_their_chunks(tmp_path):
    path = tmp_path / "replace.lmdb"
    with LMDBWriter(path, map_size=64 * 1024**2) as writer:
        writer.write_ndarray(ARRAY, key="a", chunk_rows=64)
        writer.write_ndarray(ARRAY[:10], key="a", chunk_rows=64)  # 16 chunks become 1
        writer.write_ndarray(ARRAY, key="b", chunk_rows=100)
        writer.write_ndarray(ARRAY, key="b")  # plain replaces chunked
        writer.write_ndarray(ARRAY, key="c", index=0, chunk_rows=10)
        writer.write_ndarray(ARRAY, key="c", index=1, chunk_rows=10)
        writer.write_ndarray(ARRAY, key="d", chunk_rows=10)
        assert writer.delete_key("c") == 2
        assert writer.delete("d")
    report = verify_store(path)
    assert report.ok and report.checked_entries == 3  # a header, its chunk, plain b
    with LMDBWriter(path, map_size=64 * 1024**2, base="current") as writer:
        writer.write_str("replaced", key="a")
    report = verify_store(path)
    assert report.ok and report.checked_entries == 2
    with LMDBReader(path) as reader:
        assert reader.read_as_str("a") == "replaced"
        assert np.array_equal(reader.read_as_ndarray("b"), ARRAY)


def test_derived_writer_keeps_and_replaces_chunked_arrays(tmp_path):
    path = _store(tmp_path / "store.lmdb")
    with LMDBWriter(path, map_size=64 * 1024**2, base="current") as writer:
        writer.write_ndarray(ARRAY * 2, key="chunked", index=0, chunk_rows=500)
        writer.write_ndarray(ARRAY, key="copy", index=0, chunk_rows=64)
    with LMDBReader(path) as reader:
        assert np.array_equal(reader.read_array_rows("chunked", 0, start=0, stop=3), ARRAY[:3] * 2)
        assert np.array_equal(reader.read_as_ndarray("copy", 0), ARRAY)
    report = verify_store(path)
    assert report.ok and report.checked_entries == 3 + 1 + 2 + 16


@pytest.mark.parametrize("checksum", ["blake2b-256", "sha256"])
def test_tampered_chunks_are_detected(tmp_path, checksum):
    path = _store(tmp_path / "store.lmdb", checksum=checksum)
    parent = get_loc("chunked", 0)
    chunk = _chunk_location(parent, 3)

    def flip(transaction, tags):
        body = bytearray(transaction.get(chunk))
        body[0] ^= 1
        transaction.put(chunk, bytes(body))

    _tamper(path, flip)
    with LMDBReader(path) as reader:
        with pytest.raises(IntegrityError, match="checksum mismatch"):
            reader.read_as_ndarray("chunked", 0)
        with pytest.raises(IntegrityError, match="checksum mismatch"):
            reader.read_array_rows("chunked", 0, start=200, stop=201)
        assert reader.read_array_rows("chunked", 0, start=0, stop=64).shape == (64, 3)
    with LMDBReader(path, verify_checksums=False) as reader:
        assert reader.read_as_ndarray("chunked", 0).shape == (1000, 3)  # same length, trusted
    report = verify_store(path)
    assert not report.ok and report.error_count == 1


@pytest.mark.parametrize("damage", ["missing", "envelope", "orphan", "foreign"])
def test_broken_chunk_links_fail_reads_and_verification(tmp_path, damage):
    path = _store(tmp_path / "store.lmdb")
    parent = get_loc("chunked", 0)
    chunk = _chunk_location(parent, 3)

    def change(transaction, tags):
        if damage == "missing":
            transaction.delete(chunk)
            transaction.delete(chunk, db=tags)
        elif damage == "envelope":
            # A valid envelope for different bytes: digests no longer match the header.
            other = _chunk_location(parent, 4)
            transaction.put(chunk, transaction.get(other))
            envelope = lmdb_module._encode_metadata(
                chunk, "write_array_chunk", transaction.get(other)
            )
            transaction.put(chunk, envelope, db=tags)
        elif damage == "orphan":
            stray = _chunk_location(parent, 99)
            body = transaction.get(chunk)
            transaction.put(stray, body)
            transaction.put(
                stray, lmdb_module._encode_metadata(stray, "write_array_chunk", body), db=tags
            )
        else:
            stray = _chunk_location(get_loc("plain", 0), 0)
            body = transaction.get(chunk)
            transaction.put(stray, body)
            transaction.put(
                stray, lmdb_module._encode_metadata(stray, "write_array_chunk", body), db=tags
            )

    _tamper(path, change)
    readable = damage in ("orphan", "foreign")
    for verify in (True, False):
        with LMDBReader(path, verify_checksums=verify) as reader:
            if readable:
                assert np.array_equal(reader.read_as_ndarray("chunked", 0), ARRAY)
            else:
                with pytest.raises(IntegrityError):
                    reader.read_as_ndarray("chunked", 0)
                assert reader.read_array_rows("chunked", 0, start=0, stop=10).shape == (10, 3)
    report = verify_store(path)
    assert not report.ok and report.error_count >= 1
    messages = " ".join(issue.message for issue in report.issues)
    assert ("not listed" in messages) if readable else ("chunk 3" in messages)


def test_chunks_are_internal_to_listings_fingerprints_and_diffs(tmp_path):
    path = _store(tmp_path / "store.lmdb", manifest=True)
    before = (path / "CURRENT").read_text().strip()
    with LMDBReader(path) as reader:
        assert [group.key for page in reader.iter_keys() for group in page] == [
            "note",
            "plain",
            "chunked",
        ]
        assert [
            group.key for page in reader.iter_keys(prefix=("amber-lmdb",)) for group in page
        ] == []
        manifest = reader.manifest()
    assert manifest is not None and manifest.entries == 3
    assert [group.key for group in manifest.groups] == ["note", "plain", "chunked"]
    assert manifest.groups[2].arrays[0].writer_tag == "write_ndarray_chunked"
    assert manifest.fingerprint == fingerprint_store(path).digest
    assert fingerprint_store(path).entries == 3
    changed = ARRAY.copy()
    changed[700, 1] += 1
    with LMDBWriter(path, map_size=64 * 1024**2, base="current") as writer:
        writer.write_ndarray(changed, key="chunked", index=0, chunk_rows=64)
    difference = diff_stores(path, path, before_generation=before)
    assert difference.changed == (("chunked", 0),) and difference.changed_count == 1
    assert difference.added_count == difference.removed_count == 0
    assert inspect_store(path, entry_limit=100).data_entries == 3 + 16
    statistics = array_statistics(path)
    assert {group.writer_tag for group in statistics.groups} == {
        "write_ndarray",
        "write_ndarray_chunked",
    }


def test_diff_compares_chunked_bodies_across_digest_algorithms(tmp_path):
    first = _store(tmp_path / "a.lmdb", checksum="sha256")
    second = _store(tmp_path / "b.lmdb", checksum="blake2b-256")
    difference = diff_stores(first, second)
    assert difference.undetermined_count == 3
    resolved = diff_stores(first, second, compare_bodies=True)
    assert resolved.identical
    with LMDBWriter(tmp_path / "c.lmdb", map_size=64 * 1024**2, checksum="sha256") as writer:
        changed = ARRAY.copy()
        changed[999, 2] = -1
        writer.write_ndarray(changed, key="chunked", index=0, chunk_rows=64)
        writer.write_ndarray(ARRAY, key="plain", index=0)
        writer.write_str("text", key="note")
    assert diff_stores(tmp_path / "c.lmdb", second, compare_bodies=True).changed == (
        ("chunked", 0),
    )


def test_overlay_reads_chunked_arrays_from_their_layer(tmp_path):
    base = _store(tmp_path / "base.lmdb")
    extra = tmp_path / "extra.lmdb"
    with LMDBWriter(extra, map_size=64 * 1024**2) as writer:
        writer.write_ndarray(ARRAY + 1, key="chunked", index=1, chunk_rows=64)
    with LMDBOverlayReader([base, extra]) as reader:
        assert reader.check_disjoint() == 4
        rows = reader.read_array_rows_many(
            ["chunked", "chunked", "plain"],
            indices=[1, 0, 0],
            starts=[10, 10, 10],
            stops=[20, 20, 20],
        )
        assert np.array_equal(rows[0], ARRAY[10:20] + 1)
        assert np.array_equal(rows[1], ARRAY[10:20])
        assert np.array_equal(reader.read_as_ndarray("chunked", 1), ARRAY + 1)
        with pytest.raises(ReadLimitError):
            reader.read_array_rows_many(
                ["chunked", "chunked"],
                indices=[1, 0],
                starts=[0, 0],
                stops=[500, 500],
                max_bytes=8000,
            )
    clash = tmp_path / "clash.lmdb"
    with LMDBWriter(clash, map_size=64 * 1024**2) as writer:
        writer.write_ndarray(ARRAY, key="chunked", index=0, chunk_rows=32)
    with LMDBOverlayReader([base, clash]) as reader, pytest.raises(OverlayConflictError):
        reader.check_disjoint()


def test_chunk_locations_are_reserved_keys_of_their_parent():
    parent = get_loc(("s01", "pitch"), 3)
    for number in (0, 1, 2**64 - 1):
        assert _chunk_location(parent, number) == get_loc(
            ("amber-lmdb", "chunk", parent.hex()), number
        )
        assert lmdb_module._chunk_parent(_chunk_location(parent, number)) == (parent, number)
    assert lmdb_module._chunk_parent(parent) is None
