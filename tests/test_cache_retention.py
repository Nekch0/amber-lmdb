"""Cache inventory, retention, striped builder locks, safe cleanup and builder cancellation."""

import json
import os
import threading
import time

import pytest

from amber_lmdb import CacheEntry, DerivedIndexCache, OperationCancelled, index_lock
from amber_lmdb.cli import main
from amber_lmdb.codec import any_to_bytes, bytes_to_any


def _cache(directory, **options):
    return DerivedIndexCache(directory, encode=any_to_bytes, decode=bytes_to_any, **options)


def _age(path, seconds):
    stamp = time.time() - seconds
    os.utime(path, (stamp, stamp))


def _stripe(cache, path):
    digest = path.name[len(cache.namespace) + 1 if cache.namespace else 0 :][:64]
    locks = cache.directory / ".locks"
    locks.mkdir(parents=True, exist_ok=True)
    return locks / f"stripe-{int(digest[:8], 16) % cache._stripe_count(locks)}.lock"


def test_entries_list_only_this_cache_and_read_headers_only(tmp_path):
    cache = _cache(tmp_path, namespace="rows")
    other = _cache(tmp_path, namespace="other")
    for index in range(3):
        cache.publish({"source": index}, list(range(index)))
    other.publish({"source": 0}, "x")
    (tmp_path / "unrelated.txt").write_text("x")
    broken = cache.path_for({"source": 9})
    broken.write_bytes(b"not a cache file")

    def refuse(payload):
        raise AssertionError("entries must not decode payloads")

    cache.decode = refuse
    entries = list(cache.entries())
    assert len(entries) == 4 and all(isinstance(entry, CacheEntry) for entry in entries)
    valid = {entry.identity["source"]: entry for entry in entries if entry.valid}
    assert sorted(valid) == [0, 1, 2]
    assert [entry.path for entry in entries if not entry.valid] == [broken]
    assert all(entry.size_bytes == entry.path.stat().st_size for entry in entries)
    assert [entry.identity for entry in other.entries()] == [{"source": 0}]
    assert list(_cache(None).entries()) == []
    assert list(_cache(tmp_path / "missing").entries()) == []


def test_prune_selection_order_dry_run_and_report(tmp_path):
    cache = _cache(tmp_path)
    paths = {}
    for index in range(6):
        cache.publish({"source": index}, "x" * 100)
        paths[index] = cache.path_for({"source": index})
        _age(paths[index], 1000 - index * 100)  # source 0 is the oldest
    cache.path_for({"source": "bad"}).write_bytes(b"corrupt")
    report = cache.prune(
        remove=lambda entry: entry.identity["source"] == 5,
        older_than=850,
        max_entries=2,
    )
    assert report.dry_run and all(path.exists() for path in paths.values())
    assert report.reasons == {"invalid": 1, "predicate": 1, "expired": 2, "capacity": 1}
    assert set(report.removed) >= {paths[5], paths[0], paths[1], paths[2]}
    applied = cache.prune(
        remove=lambda entry: entry.identity["source"] == 5, older_than=850, max_entries=2
    )
    assert applied.dry_run and all(path.exists() for path in paths.values())
    applied = cache.prune(
        remove=lambda entry: entry.identity["source"] == 5,
        older_than=850,
        max_entries=2,
        dry_run=False,
    )
    assert sorted(path.name for path in tmp_path.glob("*.typed-index")) == sorted(
        [paths[3].name, paths[4].name]
    )
    assert applied.kept == 2 and applied.freed_bytes > 0 and not applied.in_use
    by_bytes = cache.prune(max_total_bytes=paths[4].stat().st_size, dry_run=False)
    assert by_bytes.removed == (paths[3],) and by_bytes.reasons == {"capacity": 1}
    assert cache.lookup({"source": 4}).hit


def test_striped_locks_stay_bounded_and_are_never_removed(tmp_path):
    cache = _cache(tmp_path, lock_stripes=4)
    for index in range(40):
        assert cache.get_or_build({"source": index}, lambda index=index: index) == index
    files = sorted(path.name for path in (tmp_path / ".locks").iterdir())
    stripes = [name for name in files if name != "STRIPES"]
    assert "STRIPES" in files and (tmp_path / ".locks" / "STRIPES").read_text() == "4\n"
    assert 1 <= len(stripes) <= 4 and all(name.startswith("stripe-") for name in stripes)
    assert not list(tmp_path.glob("*.lock"))
    cache.prune(max_entries=0, dry_run=False)
    assert not list(tmp_path.glob("*.typed-index"))
    assert sorted(path.name for path in (tmp_path / ".locks").iterdir()) == files


def test_stripe_count_is_recorded_by_the_first_builder_and_enforced(tmp_path):
    first = _cache(tmp_path, lock_stripes=8)
    first.publish({"source": 1}, "x")
    assert (tmp_path / ".locks" / "STRIPES").read_text() == "8\n"
    adopted = _cache(tmp_path)  # None adopts the recorded count
    adopted.publish({"source": 2}, "y")
    assert adopted._stripes == 8
    conflicting = _cache(tmp_path, lock_stripes=4)
    with pytest.raises(ValueError, match="8 lock stripes"):
        conflicting.publish({"source": 3}, "z")
    assert not conflicting.path_for({"source": 3}).exists()
    fresh = _cache(tmp_path / "fresh")
    fresh.publish({"source": 1}, "x")
    assert (tmp_path / "fresh" / ".locks" / "STRIPES").read_text() == "64\n"
    (tmp_path / "fresh" / ".locks" / "STRIPES").write_text("many\n")
    with pytest.raises(ValueError, match="stripe record"):
        _cache(tmp_path / "fresh").publish({"source": 2}, "y")


def test_prune_skips_entries_whose_stripe_is_held(tmp_path):
    cache = _cache(tmp_path, lock_stripes=1)
    cache.publish({"source": 1}, "x")
    path = cache.path_for({"source": 1})
    with index_lock(_stripe(cache, path)):
        report = cache.prune(max_entries=0, dry_run=False)
        assert report.in_use == (path,) and report.removed == () and path.exists()
        dry = cache.prune(max_entries=0)
        assert dry.in_use == (path,)
    assert cache.prune(max_entries=0, dry_run=False).removed == (path,)


@pytest.mark.skipif(os.name != "nt", reason="POSIX readers keep reading an unlinked file")
def test_windows_reader_holding_the_file_keeps_it(tmp_path):
    cache = _cache(tmp_path)
    cache.publish({"source": 1}, "x")
    path = cache.path_for({"source": 1})
    with path.open("rb"):
        assert cache.prune(max_entries=0, dry_run=False).in_use == (path,)
    assert not path.exists() or cache.prune(max_entries=0, dry_run=False).removed == (path,)


def test_abandoned_temporaries_are_removed_and_other_files_kept(tmp_path):
    cache = _cache(tmp_path, namespace="rows", suffix=".cejc.records", lock_stripes=1)
    cache.publish({"source": 1}, "x")
    entry = cache.path_for({"source": 1})
    old = tmp_path / f".{entry.stem}.abc_12x.tmp"
    recent = tmp_path / f".{entry.stem}.def34.tmp"
    other_lock = tmp_path / f"{entry.stem}.lock"
    foreign = tmp_path / ".something.else.tmp"
    for path in (old, recent, other_lock, foreign):
        path.write_bytes(b"x")
    _age(old, 7200)
    _age(foreign, 7200)
    report = cache.prune()
    assert report.temporaries == (old,) and old.exists()
    with index_lock(_stripe(cache, entry)):
        assert cache.prune(dry_run=False).temporaries == ()
    applied = cache.prune(dry_run=False)
    assert applied.temporaries == (old,) and not old.exists()
    assert other_lock.exists() and recent.exists() and foreign.exists() and entry.exists()
    assert cache.prune(temporary_grace=0, dry_run=False).temporaries == (recent,)


def test_touch_on_hit_refreshes_at_most_once_per_interval(tmp_path, monkeypatch):
    cache = _cache(tmp_path, touch_on_hit=True, touch_interval=600)
    cache.publish({"source": 1}, "x")
    path = cache.path_for({"source": 1})
    _age(path, 3600)
    before = path.stat().st_mtime_ns
    assert cache.lookup({"source": 1}).hit
    refreshed = path.stat().st_mtime_ns
    assert refreshed > before
    _age(path, 60)
    within = path.stat().st_mtime_ns
    assert cache.lookup({"source": 1}).hit and path.stat().st_mtime_ns == within

    def read_only(*args, **kwargs):
        raise PermissionError("read-only")

    _age(path, 3600)
    monkeypatch.setattr(os, "utime", read_only)
    assert cache.lookup({"source": 1}).hit
    untouched = _cache(tmp_path)
    monkeypatch.undo()
    _age(path, 3600)
    stamp = path.stat().st_mtime_ns
    assert untouched.lookup({"source": 1}).hit and path.stat().st_mtime_ns == stamp


def test_publish_takes_the_stripe_lock(tmp_path):
    cache = _cache(tmp_path, lock_stripes=1, lock_timeout=0)
    target = cache.path_for({"source": 1})
    with index_lock(_stripe(cache, target)), pytest.raises(TimeoutError):
        cache.publish({"source": 1}, "x")
    assert not target.exists()


def test_publish_reenters_a_stripe_held_by_the_same_thread(tmp_path):
    cache = _cache(tmp_path, lock_stripes=1, lock_timeout=5)
    published = []
    original = DerivedIndexCache.publish

    def recording(self, identity, value):
        published.append(identity)
        original(self, identity, value)

    def build():
        cache.publish({"source": "side"}, "also published")  # same stripe, same thread
        return "main"

    DerivedIndexCache.publish = recording
    try:
        assert cache.get_or_build({"source": "main"}, build) == "main"
    finally:
        DerivedIndexCache.publish = original
    assert published == [{"source": "side"}, {"source": "main"}]  # get_or_build uses publish
    assert cache.lookup({"source": "side"}).value == "also published"


def test_builders_of_different_identities_on_one_stripe_serialize(tmp_path):
    cache = _cache(tmp_path, lock_stripes=1)
    active = []
    overlap = []

    def build(value):
        active.append(value)
        overlap.append(len(active))
        time.sleep(0.2)
        active.remove(value)
        return value

    threads = [
        threading.Thread(target=cache.get_or_build, args=({"source": n}, lambda n=n: build(n)))
        for n in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert overlap == [1, 1]
    assert cache.lookup({"source": 0}).hit and cache.lookup({"source": 1}).hit


def test_cancellation_before_during_and_while_waiting(tmp_path):
    cache = _cache(tmp_path, lock_stripes=1)
    event = threading.Event()
    event.set()
    with pytest.raises(OperationCancelled):
        cache.get_or_build({"source": 1}, lambda: pytest.fail("must not build"), cancel=event)
    event.clear()

    def build_then_cancel():
        event.set()
        return "late"

    with pytest.raises(OperationCancelled):
        cache.get_or_build({"source": 1}, build_then_cancel, cancel=event)
    assert not cache.lookup({"source": 1}).hit
    event.clear()
    errors = []

    def waiting():
        try:
            cache.get_or_build({"source": 2}, lambda: "built", cancel=event)
        except OperationCancelled as error:
            errors.append(error)

    with index_lock(_stripe(cache, cache.path_for({"source": 2}))):
        thread = threading.Thread(target=waiting)
        thread.start()
        time.sleep(0.3)
        event.set()
        thread.join(10)
    assert len(errors) == 1 and not cache.lookup({"source": 2}).hit
    with pytest.raises(TypeError, match="is_set"):
        cache.get_or_build({"source": 3}, lambda: 1, cancel=object())
    event.clear()
    assert cache.get_or_build({"source": 3}, lambda: 3, cancel=event) == 3


def test_index_lock_cancel(tmp_path):
    event = threading.Event()
    event.set()
    with pytest.raises(OperationCancelled), index_lock(tmp_path / "x.lock", cancel=event):
        pass


def test_option_validation(tmp_path):
    for stripes in (0, 4097, True, 1.0):
        with pytest.raises(ValueError, match="lock_stripes"):
            _cache(tmp_path, lock_stripes=stripes)
    with pytest.raises(TypeError, match="touch_on_hit"):
        _cache(tmp_path, touch_on_hit=1)
    for interval in (-1, float("nan"), None):
        with pytest.raises(ValueError, match="touch_interval"):
            _cache(tmp_path, touch_interval=interval)
    cache = _cache(tmp_path)
    for options in (
        {"older_than": -1},
        {"max_total_bytes": -1},
        {"max_entries": 1.5},
        {"temporary_grace": None},
    ):
        with pytest.raises(ValueError):
            cache.prune(**options)
    with pytest.raises(TypeError):
        cache.prune(remove=1)
    with pytest.raises(TypeError):
        cache.prune(dry_run=0)


def test_cli_cache_list_and_prune(tmp_path, capsys):
    cache = _cache(tmp_path, namespace="rows")
    for index in range(3):
        cache.publish({"source": index}, "x")
    _age(cache.path_for({"source": 0}), 7200)
    assert main(["cache-list", str(tmp_path), "--namespace", "rows"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert sorted(item["identity"]["source"] for item in listed) == [0, 1, 2]
    assert main(["cache-prune", str(tmp_path), "--namespace", "rows", "--older-than", "1h"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] and len(report["namespaces"]["rows"]["removed"]) == 1
    assert cache.path_for({"source": 0}).exists()
    assert (
        main(["cache-prune", str(tmp_path), "--namespace", "rows", "--max-entries", "1", "--apply"])
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert not report["dry_run"] and report["namespaces"]["rows"]["reasons"] == {"capacity": 2}
    assert len(list(cache.entries())) == 1
    for bad in (["--older-than", "-1"], ["--max-total-bytes", "lots"], ["--older-than", "3x"]):
        with pytest.raises(SystemExit):
            main(["cache-prune", str(tmp_path), *bad])
    assert main(["cache-prune", str(tmp_path), "--max-total-bytes", "1K"]) == 0
    assert set(json.loads(capsys.readouterr().out)["namespaces"]) == {"rows"}
    other = _cache(tmp_path, namespace="cols")
    other.publish({"source": 9}, "y")
    assert main(["cache-list", str(tmp_path)]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert sorted(item["namespace"] for item in listed) == ["cols", "rows"]
    assert main(["cache-list", str(tmp_path / "absent")]) == 3
    error = json.loads(capsys.readouterr().out)["error"]
    assert error["type"] == "FileNotFoundError"
    assert not (tmp_path / "absent").exists()
    capsys.readouterr()
