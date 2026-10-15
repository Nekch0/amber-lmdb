"""Rich renderings of store summaries: detail grids, feature tables and listings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ._common import ACCENT, ZEBRA_ROW_STYLES, flatten_tree, format_size, short_version
from .messages import ENGLISH, Messages
from .render import code_syntax, supports_unicode
from .stores import FEATURE_COLUMNS, StoreProfile, StoreSummary, feature_rows, has_features


def _zebra(index: int, enabled: bool) -> str:
    return ZEBRA_ROW_STYLES[index % 2] if enabled else ""


def store_grid(
    summary: StoreSummary,
    profile: StoreProfile,
    *,
    messages: Messages = ENGLISH,
    accent: str = ACCENT,
    expand: bool = True,
    with_contents: bool = True,
    zebra: bool = False,
) -> Table:
    """Group, key and value columns: the file, the profile summary and the contents tree."""
    grid = Table.grid(expand=expand, padding=(0, 2, 0, 0))
    grid.add_column(no_wrap=True, style=f"bold {accent}")
    grid.add_column(no_wrap=True, style="dim")
    grid.add_column(ratio=1 if expand else None, overflow="fold")
    # Blank lines between groups do not count for the zebra parity.
    row_index = 0

    def row(group: str, key: str, value: Any) -> None:
        nonlocal row_index
        if group and grid.row_count:
            grid.add_row("", "", "")
        grid.add_row(Text(group), Text(key), Text(str(value)), style=_zebra(row_index, zebra))
        row_index += 1

    row("file", "path", summary.path)
    row("", "version", f"{summary.version}  ({summary.n_versions} versions)")
    row("", "entries", "-" if summary.entries is None else summary.entries)
    row("", "file size", format_size(summary.size_bytes))
    if summary.used_bytes is not None:
        row("", "used", format_size(summary.used_bytes))
    row("", "modified", summary.modified_text)

    if summary.meta is None:
        grid.add_row("", "", "")
        reason = f" ({summary.meta_error})" if summary.meta_error else ""
        grid.add_row(
            Text("summary", style="bold yellow"),
            Text(""),
            Text(messages.meta_unreadable.format(reason=reason), style="yellow"),
        )
        return grid

    for group, key, value in profile.summary(summary):
        row(group, key, value)

    if with_contents and profile.contents is not None:
        grid.add_row("", "", "")
        grid.add_row(Text("contents"), Text("store", style="dim"), Text())
        for line, note in flatten_tree(profile.contents(summary)):
            grid.add_row("", Text(line), Text(note, style="dim"), style=_zebra(row_index, zebra))
            row_index += 1
    return grid


def contents_grid(summary: StoreSummary, profile: StoreProfile) -> Table | None:
    """The contents tree alone, as tree line and note columns."""
    if profile.contents is None:
        return None
    grid = Table.grid(expand=True, padding=(0, 2, 0, 0))
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1, overflow="fold")
    grid.row_styles = ZEBRA_ROW_STYLES
    grid.add_row(Text("store", style="dim"), Text())
    for line, note in flatten_tree(profile.contents(summary)):
        grid.add_row(Text(line), Text(note, style="dim"))
    return grid


def feature_table(summary: StoreSummary, *, safe_box: bool) -> Table:
    """Bordered feature table; SHAPE and ENTRIES only when the meta records them."""
    table = Table(
        box=box.ASCII2 if safe_box else box.SIMPLE_HEAD,
        border_style="bright_black",
        header_style="bold",
    )
    table.add_column("FEATURE", no_wrap=True)
    table.add_column("TYPE", no_wrap=True)
    table.add_column("DTYPE", style="dim", no_wrap=True)
    table.add_column("KIND", style="dim", no_wrap=True)
    columns = 4
    if summary.shapes is not None:
        table.add_column("SHAPE", no_wrap=True)
        columns = 5
    if summary.counts is not None:
        table.add_column("ENTRIES", justify="right", no_wrap=True)
    for row in feature_rows(summary):
        cells = list(row[:columns])
        if summary.counts is not None:
            cells.append(row[5])
        table.add_row(*cells)
    return table


def feature_grid(summary: StoreSummary, *, messages: Messages = ENGLISH) -> Table:
    """Borderless six-column feature grid for the explorer."""
    grid = Table.grid(expand=True, padding=(0, 2, 0, 0))
    grid.add_column(ratio=1, overflow="fold")
    for _ in range(5):
        grid.add_column(no_wrap=True)
    grid.row_styles = ZEBRA_ROW_STYLES
    grid.add_row(*(Text(head, style="bold dim") for head in FEATURE_COLUMNS))
    rows = feature_rows(summary)
    if not rows:
        grid.add_row(Text(messages.features_unreadable, style="dim"))
    for row in rows:
        grid.add_row(*row)
    return grid


def store_table(
    summaries: Sequence[StoreSummary], *, safe_box: bool, messages: Messages = ENGLISH
) -> Table:
    """NAME, TYPE, HASH, VERSION, ENTRIES, USED and MODIFIED, with a dim total below."""
    total = sum(summary.listed_size for summary in summaries)
    table = Table(
        box=box.ASCII2 if safe_box else box.SIMPLE_HEAD,
        border_style="bright_black",
        header_style="bold dim",
        show_edge=False,
        pad_edge=False,
        caption=messages.store_total.format(count=len(summaries), size=format_size(total)),
        caption_justify="left",
        caption_style="dim",
    )
    table.add_column("NAME", no_wrap=True)
    table.add_column("TYPE", no_wrap=True)
    table.add_column("HASH", style="dim", no_wrap=True)
    table.add_column("VERSION", style="dim", no_wrap=True)
    table.add_column("ENTRIES", justify="right")
    table.add_column("USED", justify="right")
    table.add_column("MODIFIED", style="dim", no_wrap=True)
    for summary in summaries:
        table.add_row(
            summary.name,
            summary.typename,
            summary.digest,
            short_version(summary.version, ellipsis="..." if safe_box else "…"),
            "-" if summary.entries is None else f"{summary.entries:,}",
            format_size(summary.listed_size),
            summary.modified_text,
        )
    return table


def usage_syntax(summary: StoreSummary, profile: StoreProfile) -> Syntax | None:
    code = profile.usage_code(summary)
    return None if code is None else code_syntax(code)


def feature_section(
    summary: StoreSummary, profile: StoreProfile, console: Console, *, messages: Messages = ENGLISH
) -> Group:
    """Features table and usage snippet printed below a store grid."""
    parts: list[RenderableType] = []
    if has_features(summary):
        parts += [
            Text(messages.features_title, style="bold"),
            feature_table(summary, safe_box=not supports_unicode(console)),
        ]
    usage = usage_syntax(summary, profile)
    if usage is not None:
        if parts:
            parts.append(Text(""))
        parts += [Text(messages.usage_title, style="bold"), usage]
    return Group(*parts)


__all__ = [
    "contents_grid",
    "feature_grid",
    "feature_section",
    "feature_table",
    "short_version",
    "store_grid",
    "store_table",
    "usage_syntax",
]
