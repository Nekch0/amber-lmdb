"""Test benchmark contracts and correctness, without performance thresholds."""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from benchmark_storage import Config, main, run, summarize


@pytest.mark.parametrize(
    "options",
    [
        {"rows": 0},
        {"array_items": 0},
        {"array_bytes": 3},
        {"array_bytes": 256 * 1024**2},
        {"batch_sizes": (1025,)},
        {"workers": 0},
        {"repeats": 0},
        {"trials": 0},
        {"engines": (2,)},
        {"extended": 1},
        {"checksum": "md5"},
        {"extended": True, "unique_items": 0},
        {"extended": True, "unique_items": 4096, "array_bytes": 1024},
        {"extended": True, "large_items": 0},
        {"extended": True, "large_bytes": 4100},
        {"extended": True, "large_items": 8, "large_bytes": 4096},
        {"extended": True, "large_items": 64, "large_bytes": 4 * 1024**2},
    ],
)
def test_bad_config_is_rejected(options):
    with pytest.raises(ValueError):
        Config(**options).validate()


def test_synthetic_trials_verify_data_and_use_fresh_processes(tmp_path):
    config = Config(
        rows=12,
        array_items=2,
        array_bytes=32,
        batch_sizes=(4, 16),
        repeats=1,
        trials=2,
        engines=(1,),
    )
    before = random.getstate()
    report = run(config, temporary_directory=tmp_path)
    assert random.getstate() == before
    assert not list(tmp_path.iterdir())
    assert len({worker["pid"] for worker in report["workers"]}) == 2
    assert all(worker["verified"] for worker in report["workers"])
    assert report["conditions"]["page_cache_cleared"] is False
    assert report["conditions"]["timings_include_profiling_or_tracemalloc"] is False
    assert len(summarize(report)["1"]["records.typed/4"]["trial_median_wall_ns"]) == 2
    assert report["workers"][0]["expected_explicit_transactions"]["records.typed/4"] == 6


def test_extended_cases_verify_parts_streams_and_lifecycles(tmp_path):
    config = Config(
        rows=12,
        array_items=2,
        array_bytes=64,
        batch_sizes=(4, 16),
        repeats=1,
        trials=1,
        engines=(1,),
        extended=True,
        unique_items=4,
        large_items=2,
        large_bytes=3 * 4096,
    )
    report = run(config, temporary_directory=tmp_path)
    assert not list(tmp_path.iterdir())
    (worker,) = report["workers"]
    assert worker["verified"]
    assert worker["request_counts"] == {"records": 24, "arrays": 4, "shapes": 8, "large": 4}
    names = set(worker["samples"])
    for family in ("records", "arrays", "shapes", "large"):
        for part in ("lookup", "copy", "checksum", "decode"):
            assert f"{family}.parts.{part}/16" in names
        assert f"{family}.stream" in names
        assert f"{family}.trusted/16" in names
    assert {"shapes.batch/4", "shapes.headers/16", "large.batch/16", "large.headers/16"} <= names
    assert {"records.dict_stream", "lifecycle.reader_shared", "lifecycle.reader_first"} <= names
    transactions = worker["expected_explicit_transactions"]
    assert transactions["shapes.batch/4"] == 2
    assert transactions["large.parts.copy/16"] == 1
    assert transactions["large.parts.checksum/16"] == 0
    assert transactions["records.trusted/16"] == 2
    assert report["conditions"]["first_open_lifecycle_timed_after_other_cases"] is True


def test_default_config_keeps_the_original_case_set(tmp_path):
    config = Config(
        rows=4, array_items=2, array_bytes=32, batch_sizes=(4,), repeats=1, trials=1, engines=(1,)
    )
    (worker,) = run(config, temporary_directory=tmp_path)["workers"]
    assert set(worker["samples"]) == {
        "records.codec",
        "records.single",
        "records.typed_single",
        "numbers.single",
        "arrays.single",
        "records.generic/4",
        "records.typed/4",
        "records.table/4",
        "records.native/4",
        "numbers.batch/4",
        "arrays.batch/4",
        "arrays.headers/4",
        "scan.statistics",
        "scan.values",
        "scan.physical",
    }


def test_cli_never_overwrites_an_existing_report(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["--output", str(output)])
    assert output.read_text(encoding="utf-8") == "preserve"
