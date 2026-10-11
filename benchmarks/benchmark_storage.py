"""Measure synthetic owned-value reads in fresh processes; never accept existing stores."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import pstats
import random
import statistics
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np

from amber_lmdb import (
    DEFAULT_MAX_BYTES,
    LMDBReader,
    LMDBWriter,
    RecordTable,
    array_statistics,
    codec,
    get_backend_info,
    get_loc,
    open_environment,
    verify_physical,
    verify_store,
)
from amber_lmdb.integrity import _parse_metadata, _verify_checksum

_PAYLOAD_LIMIT = 128 * 1024**2
_LARGE_ROW_BYTES = 1024 * 4  # One float32 row of a (rows, 1024) feature matrix.


@dataclass(frozen=True)
class Config:
    """Resource-bounded fixture and sampling settings, independent of consumers."""

    rows: int = 1024
    array_items: int = 256
    array_bytes: int = 16384
    batch_sizes: tuple[int, ...] = (32, 256, 1024)
    repeats: int = 5
    trials: int = 3
    workers: int = 1
    engines: tuple[Literal[0, 1], ...] = (0, 1)
    seed: int = 42
    extended: bool = False
    unique_items: int = 256
    large_items: int = 32
    large_bytes: int = 2 * 1024**2
    checksum: Literal["blake2b-256", "sha256"] = "blake2b-256"

    def validate(self) -> None:
        """Reject invalid or impractically large default synthetic workloads."""
        if not 1 <= self.rows <= 100000 or not 1 <= self.array_items <= 4096:
            raise ValueError("rows must be 1..100000 and array-items 1..4096")
        if self.array_bytes < 4 or self.array_bytes % 4:
            raise ValueError("array-bytes must be a positive multiple of float32 itemsize")
        if self.array_bytes * self.array_items > _PAYLOAD_LIMIT:
            raise ValueError("array fixture exceeds the 128 MiB payload limit")
        if not self.batch_sizes or any(not 1 <= size <= 1024 for size in self.batch_sizes):
            raise ValueError("batch sizes must be within 1..1024")
        if not 1 <= self.repeats <= 50 or not 1 <= self.trials <= 20:
            raise ValueError("repeats must be 1..50 and trials 1..20")
        if (
            not 1 <= self.workers <= 8
            or not self.engines
            or any(e not in (0, 1) for e in self.engines)
        ):
            raise ValueError("workers must be 1..8 and engines must be 0 or 1")
        if self.checksum not in ("blake2b-256", "sha256"):
            raise ValueError("checksum must be blake2b-256 or sha256")
        if type(self.extended) is not bool:
            raise ValueError("extended must be bool")
        if self.extended:
            self._validate_extended()

    def _validate_extended(self) -> None:
        if not 1 <= self.unique_items <= 4096 or _unique_length(self, 0) < 1:
            raise ValueError("unique-items must be 1..4096 and leave every length positive")
        if sum(_unique_length(self, i) for i in range(self.unique_items)) * 4 > _PAYLOAD_LIMIT:
            raise ValueError("unique-shape fixture exceeds the 128 MiB payload limit")
        if not 1 <= self.large_items <= 256 or self.large_bytes % _LARGE_ROW_BYTES:
            raise ValueError("large-items must be 1..256 and large-bytes a multiple of 4096")
        if _large_rows(self, 0) < 1:
            raise ValueError("large-bytes must leave every large array with at least one row")
        if (
            sum(_large_rows(self, i) for i in range(self.large_items)) * _LARGE_ROW_BYTES
            > _PAYLOAD_LIMIT
        ):
            raise ValueError("large-array fixture exceeds the 128 MiB payload limit")


def _unique_length(config: Config, index: int) -> int:
    """Distinct 1-D lengths whose mean matches ``array_bytes``, isolating shape reuse."""
    return config.array_bytes // 4 - config.unique_items // 2 + index


def _large_rows(config: Config, index: int) -> int:
    """Distinct row counts of (rows, 1024) float32 matrices averaging ``large_bytes``."""
    return config.large_bytes // _LARGE_ROW_BYTES - config.large_items // 2 + index


def _unique_array(config: Config, index: int) -> np.ndarray:
    return np.arange(_unique_length(config, index), dtype=np.float32) + index


def _large_array(config: Config, index: int) -> np.ndarray:
    rows = _large_rows(config, index)
    return np.arange(rows * 1024, dtype=np.float32).reshape(rows, 1024) + index


def _record(index: int) -> dict[str, Any]:
    return {
        "id": index,
        "name": f"sample-{index}",
        "text": "synthetic metadata " * 16,
        "labels": ["alpha", "beta", "gamma"],
        "bounds": (index / 10.0, index / 10.0 + 1.0),
        "attributes": {"enabled": True, "level": index % 7},
    }


def _batches(indices: Sequence[int], size: int):
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def _build(path: Path, config: Config, engine: Literal[0, 1]) -> dict[str, Any]:
    started, cpu = time.perf_counter_ns(), time.process_time_ns()
    # A small explicit map avoids huge logical sparse files on Windows/LMDB 0.9.
    capacity = 32 * 1024**2 + config.rows * 2048 + config.array_items * (config.array_bytes + 8192)
    if config.extended:
        capacity += sum(
            _unique_length(config, i) * 4 + 8192 for i in range(config.unique_items)
        ) + sum(_large_rows(config, i) * _LARGE_ROW_BYTES + 8192 for i in range(config.large_items))
    with LMDBWriter(
        path, lib_version=engine, map_size=capacity, checksum=config.checksum
    ) as writer:
        for index in range(config.rows):
            writer.write_dict(_record(index), key="records", index=index)
            writer.write_primitive(index, key="numbers", index=index)
        value = np.arange(config.array_bytes // 4, dtype=np.float32)
        for index in range(config.array_items):
            writer.write_ndarray(value + index, key="arrays", index=index)
        if config.extended:
            for index in range(config.unique_items):
                writer.write_ndarray(_unique_array(config, index), key="shapes", index=index)
            for index in range(config.large_items):
                writer.write_ndarray(_large_array(config, index), key="large", index=index)
        writer.commit()
    elapsed = time.perf_counter_ns() - started
    data = path / "versions" / (path / "CURRENT").read_text(encoding="ascii").strip() / "data.mdb"
    return {
        "wall_ns": elapsed,
        "cpu_ns": time.process_time_ns() - cpu,
        "logical_file_bytes": data.stat().st_size,
        "engine_selector": engine,
    }


def _profile(function: Callable[[], Any]) -> dict[str, Any]:
    profiler = cProfile.Profile()
    profiler.runcall(function)
    stats = pstats.Stats(profiler)
    rows = [
        {
            "file": key[0],
            "line": key[1],
            "function": key[2],
            "primitive_calls": values[0],
            "calls": values[1],
            "self_seconds": values[2],
            "cumulative_seconds": values[3],
        }
        for key, values in stats.stats.items()
    ]
    return {
        "total_seconds": stats.total_tt,
        "by_self_time": sorted(rows, key=lambda row: row["self_seconds"], reverse=True)[:20],
    }


_LIFECYCLE_CYCLES = 16


def _expected_arrays(family: str, requests: dict[str, list[int]], config: Config) -> list[Any]:
    if family == "arrays":
        base = np.arange(config.array_bytes // 4, dtype=np.float32)
        return [base + index for index in requests["arrays"]]
    if family == "shapes":
        return [_unique_array(config, index) for index in requests["shapes"]]
    return [_large_array(config, index) for index in requests["large"]]


def _check_values(family: str, result: Any, requests: dict[str, list[int]], config: Config) -> None:
    if family == "records":
        assert result == [_record(i) for i in requests["records"]]
        return
    expected = _expected_arrays(family, requests, config)
    assert len(result) == len(expected)
    assert all(
        value.dtype == wanted.dtype
        and value.shape == wanted.shape
        and np.array_equal(value, wanted)
        for value, wanted in zip(result, expected, strict=True)
    )


def _check(name: str, result: Any, requests: dict[str, list[int]], config: Config) -> None:
    """Check values/order/duplicates outside timing, including every array's numeric body."""
    indices, arrays = requests["records"], requests["arrays"]
    family, _, case = name.partition(".")
    if case.startswith(("parts.lookup", "parts.checksum")):
        assert result == len(requests[family])
    elif case.startswith("parts.copy"):
        assert len(result) == len(requests[family])
        assert all(metadata and body for _, metadata, body in result)
    elif case.startswith(("parts.decode", "trusted")) or case in ("stream", "dict_stream"):
        _check_values(family, result, requests, config)
    elif family == "lifecycle":
        assert result == [0] * _LIFECYCLE_CYCLES
    elif family in ("shapes", "large") and case.startswith("headers"):
        expected = _expected_arrays(family, requests, config)
        assert [header.shape for header in result] == [value.shape for value in expected]
    elif family in ("shapes", "large"):
        _check_values(family, result, requests, config)
    elif name.startswith("records.native"):
        assert [codec.bytes_to_dict(value) for value in result] == [_record(i) for i in indices]
    elif name.startswith("records"):
        assert result == [_record(i) for i in indices]
    elif name.startswith("numbers"):
        assert result == indices
    elif name.startswith("arrays.headers"):
        assert len(result) == len(arrays)
        assert all(header.shape == (config.array_bytes // 4,) for header in result)
    elif name.startswith("arrays"):
        assert len(result) == len(arrays)
        expected = np.arange(config.array_bytes // 4, dtype=np.float32)
        assert all(
            np.array_equal(value, expected + index) and value.dtype == expected.dtype
            for value, index in zip(result, arrays, strict=True)
        )
    elif name == "scan.statistics":
        stored = config.array_items
        if config.extended:
            stored += config.unique_items + config.large_items
        assert sum(group.count for group in result.groups) == stored
    elif name.startswith("scan"):
        assert result.ok


def _parts(
    reader: LMDBReader, family: str, key: str, request: list[int], size: int
) -> dict[str, Callable[[], Any]]:
    """Isolate native lookup, owned copy, checksum and decode on the same batches.

    Locations are precomputed, and the hash/decode inputs are copied once outside
    timing. Private reader/integrity helpers keep each part on the code path used by
    ``read_many``. A public read minus copy, hash and decode leaves location
    construction, metadata parsing, tag/budget checks and bookkeeping.
    """
    environment = reader._get_env()
    metadata_db = reader._metadata_db
    batches = [[get_loc(key, index) for index in batch] for batch in _batches(request, size)]

    def lookup() -> int:
        count = 0
        for locations in batches:
            with environment.begin(buffers=True) as transaction:
                for location in locations:
                    transaction.get(location, db=metadata_db)
                    transaction.get(location)
                    count += 1
        return count

    def copy() -> list[tuple[bytes, bytes, bytes]]:
        owned = []
        for locations in batches:
            with environment.begin(buffers=True) as transaction:
                views = [
                    (location, transaction.get(location, db=metadata_db), transaction.get(location))
                    for location in locations
                ]
                owned.extend((location, bytes(meta), bytes(body)) for location, meta, body in views)
        return owned

    copied = [(location, _parse_metadata(meta), body) for location, meta, body in copy()]

    def checksum() -> int:
        for location, metadata, body in copied:
            _verify_checksum(location, metadata, body)
        return len(copied)

    def decode() -> list[Any]:
        return [
            LMDBReader._decode_entry(metadata.writer_tag, body, allow_custom=False)
            for _, metadata, body in copied
        ]

    return {
        f"{family}.parts.lookup/{size}": lookup,
        f"{family}.parts.copy/{size}": copy,
        f"{family}.parts.checksum/{size}": checksum,
        f"{family}.parts.decode/{size}": decode,
    }


def _lifecycle(path: str) -> list[Any]:
    """Open, read one value and close a reader repeatedly; leases follow reader lifetime."""
    values = []
    for _ in range(_LIFECYCLE_CYCLES):
        with LMDBReader(path) as reader:
            values.append(reader.read_as_primitive("numbers", 0))
    return values


def _time_cases(
    cases: dict[str, Callable[[], Any]], config: Config, trial: int, salt: int = 0
) -> dict[str, list[dict[str, int]]]:
    samples: dict[str, list[dict[str, int]]] = {name: [] for name in cases}
    order = list(cases)
    for repeat in range(config.repeats):
        random.Random(config.seed + salt + trial * config.repeats + repeat).shuffle(order)
        for name in order:
            start, cpu = time.perf_counter_ns(), time.process_time_ns()
            result = cases[name]()
            cpu_ns = time.process_time_ns() - cpu
            wall_ns = time.perf_counter_ns() - start
            samples[name].append({"wall_ns": wall_ns, "cpu_ns": cpu_ns})
            del result  # Destruction of the returned collection is outside the call sample.
    return samples


def _allocation_peaks(cases: dict[str, Callable[[], Any]]) -> dict[str, int]:
    allocations = {}
    for name, call in cases.items():
        tracemalloc.start()
        result = call()
        allocations[name] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        del result
    return allocations


def _worker(path: str, config: Config, trial: int) -> dict[str, Any]:
    """One fresh-process trial; imports/reader setup are outside measured call samples."""
    setup = time.perf_counter_ns()
    indices = list(range(config.rows))
    random.Random(config.seed).shuffle(indices)
    indices.extend(indices[: min(64, config.rows)])
    arrays = list(range(config.array_items))
    arrays.extend(arrays[: min(16, config.array_items)])
    shapes = list(range(config.unique_items))
    shapes.extend(shapes[: min(16, config.unique_items)])
    large = list(range(config.large_items))
    large.extend(large[: min(4, config.large_items)])
    requests = {"records": indices, "arrays": arrays}
    if config.extended:
        requests.update({"shapes": shapes, "large": large})
    array_copy_budget = max(DEFAULT_MAX_BYTES, len(arrays) * (config.array_bytes + 64 * 1024))
    largest_unique = _unique_length(config, config.unique_items - 1) * 4
    largest_large = _large_rows(config, config.large_items - 1) * _LARGE_ROW_BYTES
    shape_copy_budget = max(DEFAULT_MAX_BYTES, len(shapes) * (largest_unique + 64 * 1024))
    large_copy_budget = max(DEFAULT_MAX_BYTES, len(large) * (largest_large + 64 * 1024))
    largest_value = config.array_bytes
    if config.extended:
        largest_value = max(largest_value, largest_unique, largest_large)
    verify_value_budget = max(DEFAULT_MAX_BYTES, largest_value + 64 * 1024)
    verify_batch_budget = max(array_copy_budget, verify_value_budget)
    cases: dict[str, Callable[[], Any]] = {}
    transactions: dict[str, int | None] = {}
    with (
        LMDBReader(path) as reader,
        LMDBReader(path, verify_checksums=False) as trusted,
        open_environment(path) as handle,
    ):
        snapshot = reader.get_snapshot()
        environment = handle.environment
        locations = [get_loc("records", index) for index in indices]
        with environment.begin() as transaction:
            raw = [transaction.get(location) for location in locations]
        assert all(value is not None for value in raw)
        cases["records.codec"] = lambda: [codec.bytes_to_dict(value) for value in raw]
        cases["records.single"] = lambda: [reader.read("records", i) for i in indices]
        cases["records.typed_single"] = lambda: [reader.read_as_dict("records", i) for i in indices]
        cases["numbers.single"] = lambda: [reader.read_as_primitive("numbers", i) for i in indices]
        cases["arrays.single"] = lambda: [reader.read_as_ndarray("arrays", i) for i in arrays]
        transactions.update(
            {
                "records.codec": 0,
                "records.single": len(indices),
                "records.typed_single": len(indices),
                "numbers.single": len(indices),
                "arrays.single": len(arrays),
            }
        )
        for size in config.batch_sizes:

            def generic(size=size):
                return [
                    value
                    for batch in _batches(indices, size)
                    for value in reader.read_many(["records"] * len(batch), indices=batch)
                ]

            def typed(size=size):
                return [
                    value
                    for batch in _batches(indices, size)
                    for value in reader.read_as_dict_many(["records"] * len(batch), indices=batch)
                ]

            def primitive(size=size):
                return [
                    value
                    for batch in _batches(indices, size)
                    for value in reader.read_as_primitive_many(
                        ["numbers"] * len(batch), indices=batch
                    )
                ]

            def array_values(size=size, key="arrays", request=arrays, budget=array_copy_budget):
                return [
                    value
                    for batch in _batches(request, size)
                    for value in reader.read_many(
                        [key] * len(batch),
                        indices=batch,
                        max_bytes=budget,
                    )
                ]

            def headers(size=size, key="arrays", request=arrays):
                return [
                    value
                    for batch in _batches(request, size)
                    for value in reader.read_array_headers_many([key] * len(batch), indices=batch)
                ]

            def native(size=size):
                result = []
                for batch in _batches(locations, size):
                    with environment.begin(buffers=False) as transaction:
                        result.extend(transaction.get(location) for location in batch)
                return result

            table = RecordTable(
                reader, "records", lambda row: row, count=config.rows, batch_size=size
            )
            variants = {
                f"records.generic/{size}": generic,
                f"records.typed/{size}": typed,
                f"records.table/{size}": lambda table=table: table.get_many(indices),
                f"records.native/{size}": native,
                f"numbers.batch/{size}": primitive,
                f"arrays.batch/{size}": array_values,
                f"arrays.headers/{size}": headers,
            }
            if config.extended:
                variants[f"shapes.batch/{size}"] = partial(
                    array_values, key="shapes", request=shapes, budget=shape_copy_budget
                )
                variants[f"shapes.headers/{size}"] = partial(headers, key="shapes", request=shapes)
            cases.update(variants)
            transactions.update(
                {
                    name: math.ceil(len(requests[name.split(".")[0]]) / size)
                    if not name.startswith("numbers")
                    else math.ceil(len(indices) / size)
                    for name in variants
                }
            )
        cases["scan.statistics"] = lambda: array_statistics(path)
        cases["scan.values"] = lambda: verify_store(
            path,
            max_value_bytes=verify_value_budget,
            max_batch_bytes=verify_batch_budget,
        )
        cases["scan.physical"] = lambda: verify_physical(path)
        transactions.update({"scan.statistics": None, "scan.values": None, "scan.physical": 0})
        if config.extended:
            largest = config.batch_sizes[-1]
            cases[f"large.batch/{largest}"] = lambda: [
                value
                for batch in _batches(large, largest)
                for value in reader.read_many(
                    ["large"] * len(batch), indices=batch, max_bytes=large_copy_budget
                )
            ]
            cases[f"large.headers/{largest}"] = lambda: [
                value
                for batch in _batches(large, largest)
                for value in reader.read_array_headers_many(["large"] * len(batch), indices=batch)
            ]
            transactions.update(
                {
                    f"large.batch/{largest}": math.ceil(len(large) / largest),
                    f"large.headers/{largest}": math.ceil(len(large) / largest),
                }
            )
            budgets = {
                "records": DEFAULT_MAX_BYTES,
                "arrays": array_copy_budget,
                "shapes": shape_copy_budget,
                "large": large_copy_budget,
            }
            for family, request in requests.items():

                def trusted_batch(key=family, request=request, budget=budgets[family]):
                    return [
                        value
                        for batch in _batches(request, largest)
                        for value in trusted.read_many(
                            [key] * len(batch), indices=batch, max_bytes=budget
                        )
                    ]

                cases[f"{family}.trusted/{largest}"] = trusted_batch
                transactions[f"{family}.trusted/{largest}"] = math.ceil(len(request) / largest)
                parts = _parts(reader, family, family, request, largest)
                cases.update(parts)
                transactions.update(
                    {
                        name: math.ceil(len(request) / largest)
                        if ".parts.lookup/" in name or ".parts.copy/" in name
                        else 0
                        for name in parts
                    }
                )

                def stream(key=family, request=request):
                    return [
                        value
                        for page in reader.iter_read_pages([key] * len(request), indices=request)
                        for value in page
                    ]

                cases[f"{family}.stream"] = stream
                transactions[f"{family}.stream"] = None
            cases["records.dict_stream"] = lambda: [
                value
                for page in reader.iter_dict_pages(["records"] * len(indices), indices=indices)
                for value in page
            ]
            transactions["records.dict_stream"] = None
            cases["lifecycle.reader_shared"] = partial(_lifecycle, path)
            transactions["lifecycle.reader_shared"] = _LIFECYCLE_CYCLES
        setup_ns = time.perf_counter_ns() - setup
        for name, call in cases.items():
            _check(name, call(), requests, config)
        samples = _time_cases(cases, config, trial)
        print(
            f"trial={trial} pid={os.getpid()} engine={reader.get_lib_version()[0]} timing complete",
            file=sys.stderr,
            flush=True,
        )
        allocations = _allocation_peaks(cases)
        profile = _profile(cases[f"records.typed/{config.batch_sizes[-1]}"])
        engine_version = reader.get_lib_version()
    if config.extended:
        # No other reference holds this generation now: every cycle acquires the root gate,
        # the shared generation lease and a native environment, then releases them.
        first = {"lifecycle.reader_first": partial(_lifecycle, path)}
        _check("lifecycle.reader_first", first["lifecycle.reader_first"](), requests, config)
        samples.update(_time_cases(first, config, trial, salt=1))
        allocations.update(_allocation_peaks(first))
        transactions["lifecycle.reader_first"] = _LIFECYCLE_CYCLES
    print(f"trial={trial} pid={os.getpid()} diagnostics complete", file=sys.stderr, flush=True)
    copy_limits = {
        "records_and_headers": DEFAULT_MAX_BYTES,
        "arrays_and_full_scan": array_copy_budget,
        "full_scan_value": verify_value_budget,
        "full_scan_batch": verify_batch_budget,
    }
    if config.extended:
        copy_limits.update(
            {"shapes": shape_copy_budget, "large": large_copy_budget, "streams": DEFAULT_MAX_BYTES}
        )
    return {
        "pid": os.getpid(),
        "trial": trial,
        "engine_version": engine_version,
        "snapshot": asdict(snapshot),
        "setup_wall_ns": setup_ns,
        "request_counts": {name: len(request) for name, request in requests.items()},
        "copy_limits": copy_limits,
        "samples": samples,
        "separate_python_allocation_peak_bytes": allocations,
        "expected_explicit_transactions": transactions,
        "separate_cprofile": profile,
        "verified": True,
    }


def _provenance() -> dict[str, Any]:
    source = Path(__file__).resolve().parents[1] / "src/amber_lmdb"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("amber-lmdb", "lmdb", "numpy", "msgpack", "frozendict")
        },
        "backend": asdict(get_backend_info()),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.glob("*.py"))
        },
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def run(config: Config, *, temporary_directory: Path | None = None) -> dict[str, Any]:
    """Build only private synthetic stores and return samples without writing report files."""
    config.validate()
    started = time.perf_counter_ns()
    with tempfile.TemporaryDirectory(
        prefix="amber-lmdb-benchmark-", dir=temporary_directory
    ) as tmp:
        paths = {engine: Path(tmp) / f"engine-{engine}.lmdb" for engine in config.engines}
        builds = {str(engine): _build(path, config, engine) for engine, path in paths.items()}
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=config.workers, mp_context=context, max_tasks_per_child=1
        ) as pool:
            futures = [
                pool.submit(_worker, str(path), config, trial)
                for engine, path in paths.items()
                for trial in range(config.trials)
            ]
            workers = [future.result() for future in futures]
    return {
        "format": "amber-lmdb-benchmark-v1",
        "config": asdict(config),
        "provenance": _provenance(),
        "builds": builds,
        "workers": workers,
        "whole_run_wall_ns": time.perf_counter_ns() - started,
        "conditions": {
            "page_cache_cleared": False,
            "synthetic": True,
            "fresh_process_per_trial": True,
            "native_baseline_checks_tags": False,
            "timings_include_profiling_or_tracemalloc": False,
            "timings_include_imports": False,
            "full_output_deletion_timed": False,
            "parts_use_private_reader_and_integrity_helpers": config.extended,
            "first_open_lifecycle_timed_after_other_cases": config.extended,
        },
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize independent trial medians, retaining all raw samples in the main report."""
    summary = {}
    for engine in report["config"]["engines"]:
        workers = [w for w in report["workers"] if w["engine_version"][0] == engine]
        summary[str(engine)] = {
            name: {
                "trial_median_wall_ns": [
                    statistics.median(sample["wall_ns"] for sample in w["samples"][name])
                    for w in workers
                ],
                "median_wall_ns": statistics.median(
                    statistics.median(sample["wall_ns"] for sample in w["samples"][name])
                    for w in workers
                ),
                "trial_median_cpu_ns": [
                    statistics.median(sample["cpu_ns"] for sample in w["samples"][name])
                    for w in workers
                ],
            }
            for name in workers[0]["samples"]
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run a bounded local benchmark, writing a new caller-selected JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporary-directory", type=Path)
    for name in (
        "rows",
        "array-items",
        "array-bytes",
        "repeats",
        "trials",
        "workers",
        "unique-items",
        "large-items",
        "large-bytes",
    ):
        parser.add_argument(
            f"--{name}", type=int, default=getattr(Config(), name.replace("-", "_"))
        )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(Config().batch_sizes))
    parser.add_argument("--engines", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    parser.add_argument("--checksum", choices=("blake2b-256", "sha256"), default="blake2b-256")
    parser.add_argument(
        "--extended",
        action="store_true",
        help="add unique-shape and large arrays, read-path parts, streams and reader lifecycles",
    )
    args = parser.parse_args(argv)
    config = Config(
        args.rows,
        args.array_items,
        args.array_bytes,
        tuple(args.batch_sizes),
        args.repeats,
        args.trials,
        args.workers,
        tuple(args.engines),
        extended=args.extended,
        unique_items=args.unique_items,
        large_items=args.large_items,
        large_bytes=args.large_bytes,
        checksum=args.checksum,
    )
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists():
        parser.error("output already exists; choose a new report path")
    report = run(config, temporary_directory=args.temporary_directory)
    report["summary"] = summarize(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(f"Verified {len(report['workers'])} fresh-process trials; report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
