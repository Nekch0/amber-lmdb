"""Header-only statistics, bounded full validation and framework independence."""

import json
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import lmdb
import numpy as np
import pytest

import amber_lmdb.scanning as scanning
from amber_lmdb import (
    LMDBReader,
    LMDBWriter,
    SourceChangedError,
    array_statistics,
    codec,
    get_active_envs,
    get_loc,
    parse_array_header,
    register_bytes_codec,
    verify_store,
)
from amber_lmdb.cli import main
from amber_lmdb.integrity import _encode_metadata
from amber_lmdb.lmdb import _METADATA_DB_NAME


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "numbers.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_ndarray(np.zeros((2, 8), dtype=np.float32), key="a", index=0)
        writer.write_ndarray(np.ones((4, 3), dtype=np.float32), key="b", index=0)
        writer.write_ndarray(np.array(7, dtype=np.int16), key="scalar")
        writer.write_ndarray(np.empty((0, 5), dtype=np.float32), key="empty")
        writer.write_dict({"count": 4}, key="description")
    return path


def _mutate(path, update):
    generation = path / "versions" / (path / "CURRENT").read_text().strip()
    with lmdb.open(str(generation), max_dbs=1) as env:
        tags = env.open_db(_METADATA_DB_NAME)
        with env.begin(write=True) as txn:
            update(txn, tags)


def test_header_batch_preserves_order_duplicates_and_owned_data(store, monkeypatch):
    monkeypatch.setattr(codec, "bytes_to_ndarray", lambda *args: pytest.fail("body decode"))
    with LMDBReader(store) as reader:
        headers = reader.read_array_headers_many(["b", "a", "b"], indices=[0, 0, 0])
    assert [item.shape for item in headers] == [(4, 3), (2, 8), (4, 3)]
    assert [item.numel for item in headers] == [12, 16, 12]
    assert headers[0].payload_bytes == 48 and headers[0].dtype == "float32"
    assert not get_active_envs()


@pytest.mark.parametrize(
    "options",
    [{"max_items": 0}, {"max_items": 2**20 + 1}, {"max_header_bytes": 0}, {"indices": [0]}],
)
def test_header_batch_validates_before_open(tmp_path, options):
    reader = LMDBReader(tmp_path / "absent")
    with pytest.raises((ValueError, TypeError)):
        reader.read_array_headers_many([], **options)
    assert not get_active_envs()


def test_empty_header_batch_opens_nothing(tmp_path):
    assert LMDBReader(tmp_path / "absent").read_array_headers_many([]) == []
    assert not get_active_envs()


def test_header_batch_rejects_non_array_and_missing_keys(store):
    with LMDBReader(store) as reader:
        with pytest.raises(ValueError):
            reader.read_array_headers_many(["description"])
        with pytest.raises(KeyError):
            reader.read_array_headers_many(["missing"])


def test_tensor_header_needs_no_torch():
    raw = codec._pack({"kind": "torch", "shape": (2, 3), "dtype": "float32"}, b"x" * 24)
    header = parse_array_header(raw[: len(raw) - 24], stored_bytes=len(raw))
    assert header.dtype == "torch.float32" and header.numel == 6
    unknown = codec._pack({"kind": "torch", "shape": (1,), "dtype": "future_dtype"}, b"x")
    assert parse_array_header(unknown, stored_bytes=len(unknown)).itemsize is None
    with pytest.raises(ValueError, match="payload size"):
        parse_array_header(raw, stored_bytes=len(raw) - 1)


def test_statistics_separate_axis_bounds_from_real_smallest_sample(store, monkeypatch):
    monkeypatch.setattr(codec, "bytes_to_ndarray", lambda *args: pytest.fail("body decode"))
    stats = array_statistics(store, batch_size=1)
    assert stats.scanned_entries == 5 and stats.array_entries == 4
    group = next(group for group in stats.groups if group.rank == 2 and group.count == 3)
    assert group.min_shape == (0, 3) and group.max_shape == (4, 8)
    assert group.smallest_shape == (0, 5) and group.largest_shape == (2, 8)
    assert group.smallest_shape != group.min_shape
    assert group.payload_bytes == (16 + 12) * 4
    assert not get_active_envs()


def test_statistics_group_limit_and_invalid_options(store):
    with pytest.raises(ValueError, match="max_groups"):
        array_statistics(store, max_groups=1)
    for options in ({"batch_size": 0}, {"max_groups": 0}):
        with pytest.raises(ValueError):
            array_statistics(store, **options)
    assert not get_active_envs()


def test_scan_copy_budget_and_decoding_outside_transactions(store, monkeypatch):
    original_opened = scanning._opened
    original_pages = scanning._pages
    original_decode = LMDBReader._decode_entry
    active = [False]
    pages = []

    class TrackedEnvironment:
        def __init__(self, env):
            self.env = env

        def __getattr__(self, name):
            return getattr(self.env, name)

        @contextmanager
        def begin(self, **options):
            active[0] = True
            try:
                with self.env.begin(**options) as txn:
                    yield txn
            finally:
                active[0] = False

    @contextmanager
    def opened(path, **options):
        with original_opened(path, **options) as (env, tags, snapshot):
            yield TrackedEnvironment(env), tags, snapshot

    def checked_pages(*args, **options):
        for page in original_pages(*args, **options):
            assert not active[0]
            assert sum(len(row.payload) for row in page if row.payload is not None) <= 200
            assert len(page) <= 2
            pages.append(page)
            yield page

    def decode(*args, **options):
        assert not active[0]
        return original_decode(*args, **options)

    monkeypatch.setattr(scanning, "_opened", opened)
    monkeypatch.setattr(scanning, "_pages", checked_pages)
    monkeypatch.setattr(LMDBReader, "_decode_entry", staticmethod(decode))
    assert verify_store(store, batch_size=2, max_batch_bytes=200).ok
    assert len(pages) >= 3


def test_full_validation_missing_tensor_dependency_is_unverified(store, monkeypatch):
    def missing(*args):
        raise ImportError("tensor support is not installed")

    def add_tensor(txn, tags):
        location = get_loc("tensor")
        value = codec._pack({"kind": "torch", "shape": (2,), "dtype": "float32"}, b"x" * 8)
        txn.put(location, value)
        txn.put(location, _encode_metadata(location, "write_tensor", value), db=tags)

    _mutate(store, add_tensor)
    monkeypatch.setattr(codec, "bytes_to_tensor", missing)
    result = verify_store(store)
    assert not result.ok and not result.complete
    assert result.skipped_values == 1 and result.error_count == 0
    assert array_statistics(store).array_entries == 5


def test_full_validation_all_values_and_tags(store):
    report = verify_store(store, batch_size=1, max_batch_bytes=512)
    assert report.ok and report.complete
    assert report.checked_entries == report.tag_entries == report.decoded_values == 5
    assert report.error_count == report.skipped_values == 0
    assert report.issues == () and not get_active_envs()


def test_full_validation_catches_non_array_corruption_and_orphan_tags(store):
    def damage(txn, tags):
        txn.put(get_loc("description"), b"not a typed value")
        txn.put(get_loc("orphan"), b"write_dict", db=tags)

    _mutate(store, damage)
    report = verify_store(store, max_issues=1)
    assert not report.ok and report.complete
    assert report.error_count == 2 and len(report.issues) == 1
    assert report.tag_entries == 6


def test_oversized_values_are_unverified_not_success(store):
    report = verify_store(store, max_value_bytes=1, max_batch_bytes=512, max_issues=0)
    assert not report.complete and not report.ok
    assert report.skipped_values == 5 and report.decoded_values == 0
    assert report.error_count == 0 and report.issues == ()


def test_custom_decoders_require_opt_in_and_are_never_run_by_cli(tmp_path, capsys):
    @dataclass
    class Account:
        number: int

    calls = []

    def decode(value):
        calls.append(value)
        return Account(value)

    register_bytes_codec("scan.account.v1", Account, lambda value: value.number, decode)
    path = tmp_path / "custom"
    with LMDBWriter(path) as writer:
        writer.write_any(Account(7), key="account")
    report = verify_store(path)
    assert not report.ok and report.skipped_values == 1 and not calls
    assert main(["doctor", str(path), "--full"]) == 1
    assert json.loads(capsys.readouterr().out)["full_value_validation"] is False
    assert not calls
    assert verify_store(path, allow_custom=True).ok and calls == [7]


def test_corrupt_array_length_detected_by_header_and_full_scan(store):
    def damage(txn, tags):
        raw = bytes(txn.get(get_loc("a", 0)))
        txn.put(get_loc("a", 0), raw[:-1])

    _mutate(store, damage)
    with LMDBReader(store) as reader, pytest.raises(ValueError, match="integrity metadata"):
        reader.read_array_headers_many(["a"], indices=[0])
    assert verify_store(store).error_count == 1
    with pytest.raises(ValueError):
        array_statistics(store)


def test_paged_callbacks_run_after_transaction_and_pin_generation(store, monkeypatch):
    original = scanning._pages

    def replace_current(env, tags, snapshot, **options):
        pages = original(env, tags, snapshot, **options)
        first = next(pages)
        # A writer can publish while the scanner's short transaction is closed.
        with LMDBWriter(store, overwrite=True) as writer:
            writer.write_primitive(42, key="new")
        yield first
        yield from pages

    monkeypatch.setattr(scanning, "_pages", replace_current)
    report = verify_store(store, batch_size=1)
    assert report.ok and report.checked_entries == 5
    assert report.snapshot.generation != str(
        store / "versions" / (store / "CURRENT").read_text().strip()
    )


def test_changed_transaction_id_rejected(store, monkeypatch):
    original = scanning._opened

    @contextmanager
    def changed(path, **options):
        with original(path, **options) as (env, tags, snapshot):
            from dataclasses import replace

            yield env, tags, replace(snapshot, last_txnid=snapshot.last_txnid + 1)

    monkeypatch.setattr(scanning, "_opened", changed)
    with pytest.raises(SourceChangedError):
        verify_store(store, batch_size=1)
    with pytest.raises(SourceChangedError):
        array_statistics(store, batch_size=1)
    assert not get_active_envs()


def test_cli_full_stats_and_argument_errors(store, capsys):
    assert main(["doctor", str(store), "--full", "--batch-size", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["full_value_validation"]
    assert main(["stats", str(store)]) == 0
    assert json.loads(capsys.readouterr().out)["array_entries"] == 4
    with pytest.raises(SystemExit) as error:
        main(["doctor", str(store), "--full", "--after", "00"])
    with pytest.raises(SystemExit):
        main(["doctor", str(store), "--progress"])
    assert error.value.code == 2
    with pytest.raises(SystemExit):
        main(["stats", str(store), "--batch-size", "1025"])


def test_no_runtime_or_package_dependency_on_consumers_or_torch():
    script = r"""
import importlib.abc, sys, tempfile
from pathlib import Path
class BlockConsumers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mmfabric', 'dataset_cejc', 'torch'}:
            raise AssertionError('unexpected framework import: ' + fullname)
sys.meta_path.insert(0, BlockConsumers())
import numpy as np
import amber_lmdb as api
from amber_lmdb.cli import build_parser
assert 'mmfabric' not in build_parser().format_help()
with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / 'standalone'
    with api.LMDBWriter(path) as writer:
        writer.write_ndarray(np.arange(4), key='values')
    with api.LMDBReader(path) as reader:
        assert reader.read_array_headers_many(['values'])[0].numel == 4
    assert api.array_statistics(path).array_entries == 1
    assert api.verify_store(path).ok
"""
    subprocess.run([sys.executable, "-c", script], check=True)
    import tomllib

    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    requirements = project["project"]["dependencies"]
    for items in project["project"]["optional-dependencies"].values():
        requirements += items
    assert not any("mmfabric" in item or "cejc" in item for item in requirements)
