"""Existence, counts and key listing read integrity metadata only, never bodies."""

import pytest

from amber_lmdb import (
    KeyGroup,
    LMDBReader,
    LMDBWriter,
    get_active_envs,
    get_loc,
    open_environment,
)
from amber_lmdb.collections import SourceChangedError
from amber_lmdb.lmdb import _METADATA_DB_NAME, TypedReader

LAST = 2**64 - 1


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "listing.lmdb"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_str("value", key="alpha")
        for index in (0, 1, 2, 5):
            writer.write_primitive(index, key="beta", index=index)
        for index in range(10):
            writer.write_primitive(index, key=("s1", "feat"), index=index)
        writer.write_str("plain", key=("s1", "feat2"))
        for index in (3, 4):
            writer.write_primitive(index, key=("s1", "feat2"), index=index)
        writer.write_str("root", key=("s1",))
        writer.write_primitive(0, key=("s2", "feat"), index=0)
        writer.write_primitive(1, key=("s2", "feat"), index=LAST)
        writer.write_primitive(7, key=("s1", "x", "y"), index=7)
        writer.write_primitive(1, key=("s1longer", "feat"), index=1)
    return path


def _flatten(pages):
    return [item for page in pages for item in page]


def test_contains_distinguishes_index_and_key_kind(store):
    with LMDBReader(store) as reader:
        assert reader.contains("alpha")
        assert not reader.contains("alpha", index=0)
        assert not reader.contains(("alpha",))
        assert reader.contains("beta", index=5)
        assert not reader.contains("beta", index=3)
        assert not reader.contains("beta")
        assert reader.contains(("s1", "feat2")) and reader.contains(("s1", "feat2"), 4)
        assert reader.contains_many(["beta", "beta", "missing"], indices=[0, 0, None]) == [
            True,
            True,
            False,
        ]


def test_empty_contains_many_opens_nothing(store):
    with LMDBReader(store) as reader:
        assert reader.contains_many([]) == []
        assert not get_active_envs()


def test_index_range_and_count(store):
    with LMDBReader(store) as reader:
        assert reader.index_range("beta") == (0, 5)
        assert reader.index_range(("s2", "feat")) == (0, LAST)
        assert reader.index_range("alpha") is None
        assert reader.index_range("missing") is None
        assert reader.count("beta") == 4
        assert reader.count(("s1", "feat")) == 10
        assert reader.count(("s2", "feat")) == 2
        assert reader.count("alpha") == 0
        assert reader.count("missing") == 0


def test_iter_indices_pages_and_start(store):
    with LMDBReader(store) as reader:
        assert list(reader.iter_indices("beta", page_size=3)) == [[0, 1, 2], [5]]
        assert list(reader.iter_indices("beta", start=2)) == [[2, 5]]
        assert list(reader.iter_indices("beta", start=6)) == []
        assert list(reader.iter_indices(("s2", "feat"), start=1)) == [[LAST]]
        assert _flatten(reader.iter_indices(("s1", "feat"), page_size=4)) == list(range(10))
        assert list(reader.iter_indices("missing")) == []


def test_iter_keys_lists_every_logical_key_once_in_stored_order(store):
    with LMDBReader(store) as reader:
        groups = _flatten(reader.iter_keys())
        assert _flatten(reader.iter_keys(page_size=2)) == groups
        assert all(len(page) <= 2 for page in reader.iter_keys(page_size=2))
    expected = [
        KeyGroup("alpha", True, None, None),
        KeyGroup("beta", False, 0, 5),
        KeyGroup(("s1",), True, None, None),
        KeyGroup(("s2", "feat"), False, 0, LAST),
        KeyGroup(("s1", "feat"), False, 0, 9),
        KeyGroup(("s1", "feat2"), True, 3, 4),
        KeyGroup(("s1longer", "feat"), False, 1, 1),
        KeyGroup(("s1", "x", "y"), False, 7, 7),
    ]
    # Stored order is the encoded key order: string keys, then tuple keys by component
    # count, each component by length before content.
    assert groups == sorted(expected, key=lambda group: get_loc(group.key))
    assert [group.key for group in groups][:3] == ["beta", "alpha", ("s1",)]


def test_iter_keys_prefix_selects_leading_tuple_components(store):
    with LMDBReader(store) as reader:
        keys = [group.key for group in _flatten(reader.iter_keys(prefix=("s1",)))]
        assert keys == [("s1",), ("s1", "feat"), ("s1", "feat2"), ("s1", "x", "y")]
        deep = [group.key for group in _flatten(reader.iter_keys(prefix=("s1", "x")))]
        assert deep == [("s1", "x", "y")]
        paged = [group.key for group in _flatten(reader.iter_keys(prefix=("s1",), page_size=1))]
        assert paged == keys
        assert _flatten(reader.iter_keys(prefix=("none",))) == []


def test_listing_never_reads_bodies(store, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("listing read a body")

    monkeypatch.setattr(TypedReader, "_get_bytes", fail)
    monkeypatch.setattr(LMDBReader, "_owned_entries", fail)
    monkeypatch.setattr(LMDBReader, "_get_checked_entry", fail)
    with LMDBReader(store) as reader:
        reader.contains("alpha")
        reader.count(("s1", "feat"))
        reader.index_range("beta")
        _flatten(reader.iter_keys())


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda r: list(r.iter_keys(prefix=["s1"])), TypeError),
        (lambda r: list(r.iter_keys(prefix=("s1", 2))), TypeError),
        (lambda r: list(r.iter_keys(page_size=0)), ValueError),
        (lambda r: list(r.iter_indices("beta", page_size=True)), ValueError),
        (lambda r: list(r.iter_indices("beta", start=-1)), ValueError),
        (lambda r: list(r.iter_indices("beta", start=2**64)), ValueError),
        (lambda r: r.contains_many("beta"), TypeError),
    ],
)
def test_listing_arguments_are_validated(store, call, error):
    with LMDBReader(store) as reader, pytest.raises(error):
        call(reader)


def test_generation_change_between_pages_is_detected(store):
    with LMDBReader(store) as reader:
        pages = reader.iter_keys(page_size=1)
        next(pages)
        with LMDBWriter(store, map_size=1024**2, overwrite=True) as writer:
            writer.write_str("new", key="alpha")
        reader.refresh()
        with pytest.raises(SourceChangedError):
            next(pages)


def test_malformed_metadata_key_is_reported(store):
    with open_environment(store, readonly=False) as handle:
        tags = handle.environment.open_db(_METADATA_DB_NAME)
        with handle.environment.begin(write=True) as txn:
            txn.put(b"not-a-typed-key", b"x", db=tags)
    with LMDBReader(store) as reader, pytest.raises(ValueError, match="typed key"):
        _flatten(reader.iter_keys())
