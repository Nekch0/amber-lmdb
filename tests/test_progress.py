"""Long scans report progress between pages and stop cleanly when cancelled."""

import json
import threading
from itertools import pairwise

import numpy as np
import pytest

import amber_lmdb.progress as progress_module
from amber_lmdb import (
    LMDBWriter,
    OperationCancelled,
    ValidationReport,
    array_statistics,
    fingerprint_store,
    verify_store,
)
from amber_lmdb.cli import main

ENTRIES = 12


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "store.lmdb"
    with LMDBWriter(path, map_size=1024**2) as writer:
        for index in range(ENTRIES):
            writer.write_ndarray(np.full(3, index, np.float32), key="values", index=index)
    return path


@pytest.fixture
def every_page(monkeypatch):
    monkeypatch.setattr(progress_module, "_INTERVAL", 0)


@pytest.mark.parametrize(
    "run",
    [
        lambda p, **kw: verify_store(p, batch_size=4, **kw),
        lambda p, **kw: array_statistics(p, batch_size=4, **kw),
        lambda p, **kw: fingerprint_store(p, page_size=4, **kw),
    ],
)
def test_reports_arrive_after_pages_and_once_at_the_end(store, every_page, run):
    reports = []
    run(store, progress=reports.append)
    assert [r.entries_done for r in reports] == [4, 8, 12, 12]
    assert [r.finished for r in reports] == [False, False, False, True]
    assert reports[-1].entries_total in (ENTRIES, ENTRIES + 1)  # includes the tag descriptor
    assert reports[0].bytes_done > 0
    assert all(b.elapsed_seconds >= a.elapsed_seconds for a, b in pairwise(reports))


def test_cancelled_verification_carries_an_incomplete_partial_report(store, every_page):
    cancel = threading.Event()
    with pytest.raises(OperationCancelled) as error:
        verify_store(store, batch_size=4, progress=lambda report: cancel.set(), cancel=cancel)
    partial = error.value.partial
    assert isinstance(partial, ValidationReport)
    assert partial.cancelled and partial.checked_entries == 4
    assert partial.error_count == 0 and not partial.complete and not partial.ok


def test_cancel_set_before_start_stops_before_any_page(store):
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(OperationCancelled) as error:
        verify_store(store, cancel=cancel)
    assert error.value.partial.checked_entries == 0
    with pytest.raises(OperationCancelled) as error:
        array_statistics(store, cancel=cancel)
    assert error.value.partial is None


def test_callback_exceptions_abort_the_operation(store, every_page):
    def fail(report):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        verify_store(store, batch_size=4, progress=fail)


def test_reports_are_throttled(store):
    reports = []
    verify_store(store, batch_size=1, progress=reports.append)
    # The default interval suppresses most page reports on a fast scan.
    assert reports[-1].finished and len(reports) < ENTRIES


@pytest.mark.parametrize(
    "options",
    [{"progress": "print"}, {"cancel": object()}],
)
def test_invalid_hooks_are_rejected(store, options):
    with pytest.raises(TypeError):
        verify_store(store, **options)


def test_cli_progress_goes_to_stderr_and_json_stays_on_stdout(store, capsys):
    assert main(["doctor", str(store), "--full", "--progress"]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out)["ok"] is True
    assert "verify_store:" in output.err
    assert main(["stats", str(store), "--progress"]) == 0
    assert "array_statistics:" in capsys.readouterr().err


def test_cli_interrupt_returns_130(store, monkeypatch, capsys):
    import amber_lmdb.cli as cli

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "verify_store", interrupt)
    assert main(["doctor", str(store), "--full"]) == 130
    assert "interrupted" in capsys.readouterr().err
