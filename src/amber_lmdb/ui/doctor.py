"""Doctor reports: sections of OK/WARN/FAIL/SKIP checks and the shared store checks."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from ..inspection import inspect_store
from ..lmdb import LMDBReader
from ._common import ACCENT, STATUS_STYLE, Status, cap_list, short_version
from .messages import ENGLISH, Messages
from .render import print_section, status_icons, status_text


@dataclass
class Check:
    """One reported result."""

    name: str
    status: Status
    detail: str = ""


@dataclass
class Section:
    """A titled group of checks."""

    title: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: Status, detail: str = "") -> Check:
        check = Check(name=name, status=status, detail=detail)
        self.checks.append(check)
        return check


def doctor_progress(console: Console, *, accent: str = ACCENT) -> Progress:
    """Transient progress bar for long checks; disabled off a terminal."""
    return Progress(
        SpinnerColumn("dots", style=accent),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=None, style="dim", complete_style=accent, finished_style="green"),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
        disable=not console.is_terminal,
    )


def tally(sections: Iterable[Section]) -> dict[str, int]:
    """Number of checks per status."""
    counts = {"OK": 0, "WARN": 0, "FAIL": 0, "SKIP": 0}
    for section in sections:
        for check in section.checks:
            counts[check.status] += 1
    return counts


def report_document(sections: Sequence[Section], *, elapsed: float | None = None) -> dict[str, Any]:
    """JSON-ready report: ``ok``, per-status ``counts``, ``sections`` and ``elapsed_seconds``."""
    counts = tally(sections)
    return {
        "ok": counts["FAIL"] == 0,
        "counts": {status.lower(): count for status, count in counts.items()},
        "elapsed_seconds": None if elapsed is None else round(elapsed, 3),
        "sections": [
            {"title": section.title, "checks": [asdict(check) for check in section.checks]}
            for section in sections
        ],
    }


def render_report(
    console: Console,
    sections: Sequence[Section],
    *,
    messages: Messages = ENGLISH,
    accent: str = ACCENT,
    elapsed: float | None = None,
) -> bool:
    """Print every section and a one-line verdict; return whether nothing failed.

    Each check is a colored glyph, its name (colored unless OK) and a dim detail.
    """
    icons = status_icons(console)
    console.print()
    for section in sections:
        grid = Table.grid(padding=(0, 2, 0, 0))
        grid.add_column(no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(overflow="fold")
        for check in section.checks:
            name_style = "" if check.status == "OK" else STATUS_STYLE[check.status]
            grid.add_row(
                status_text(console, check.status),
                Text(check.name, style=name_style),
                Text(check.detail, style="dim"),
            )
        print_section(console, section.title, grid, accent=accent)
    counts = tally(sections)
    failures = counts["FAIL"]
    if failures:
        headline = Text(
            f"{icons['FAIL']} {messages.failures_found.format(count=failures)}", style="bold red"
        )
    else:
        headline = Text(f"{icons['OK']} {messages.all_passed}", style="bold green")
    detail = messages.tally.format(
        ok=counts["OK"], warn=counts["WARN"], skip=counts["SKIP"], seconds=elapsed or 0.0
    )
    if elapsed is None:
        detail = detail.rsplit(" · ", 1)[0]
    headline.append(f"  {detail}", style="dim not bold")
    console.print(headline)
    console.print()
    return not failures


def check_store_layout(store: Path, section: Section, *, messages: Messages = ENGLISH) -> None:
    """Report whether one store opens and how many entries its current generation has."""
    try:
        info = inspect_store(store)
    except Exception as exc:
        prefix = (
            messages.format_mismatch
            if "unsupported LMDB dataset format" in str(exc)
            else messages.store_open_failed
        )
        section.add(store.name, "FAIL", f"{prefix}: {type(exc).__name__}: {exc}")
        return
    version = short_version(Path(info.snapshot.generation).name)
    section.add(
        store.name,
        "WARN" if info.entries == 0 else "OK",
        f"{info.entries:,} entries · {version}",
    )


def check_store_meta(
    store: Path,
    section: Section,
    *,
    required_keys: Iterable[str] = (),
    describe: Mapping[str, str] | None = None,
    metadata_key: str = "meta",
) -> dict[str, Any] | None:
    """Read the metadata dictionary and require keys; return it when the check passes.

    ``describe`` maps detail labels to metadata keys for the OK line, e.g.
    ``{"record": "record_type"}`` prints ``record=<meta["record_type"]>``.
    """
    name = f"{store.name} {metadata_key}"
    try:
        with LMDBReader(store) as reader:
            meta = cast("dict[str, Any]", reader.read_as_dict(metadata_key))
    except Exception as exc:
        section.add(name, "FAIL", f"{type(exc).__name__}: {exc}")
        return None
    missing = [key for key in required_keys if key not in meta]
    if missing:
        section.add(name, "FAIL", f"missing keys: {cap_list(missing)}")
        return None
    detail = ", ".join(f"{label}={meta.get(key)}" for label, key in (describe or {}).items())
    section.add(name, "OK", detail)
    return meta


__all__ = [
    "Check",
    "Section",
    "Status",
    "check_store_layout",
    "check_store_meta",
    "doctor_progress",
    "render_report",
    "report_document",
    "tally",
]
