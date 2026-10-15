"""Rich rendering helpers shared by the command-line views and the explorers."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from typing import Any

from rich.console import Console, RenderableType
from rich.padding import Padding
from rich.style import Style
from rich.syntax import PygmentsSyntaxTheme, Syntax
from rich.table import Table
from rich.text import Text

from ._common import (
    ACCENT,
    STATUS_STYLE,
    ZEBRA_ROW_STYLES,
    Status,
    TreeNode,
    cap_list,
    flatten_tree,
    format_size,
    is_interactive,
)

# Narrow glyphs only: emoji-like symbols such as U+26A0 are two cells wide in many
# East Asian terminals and break the alignment of the report columns.
_STATUS_ICON_UNICODE = {"OK": "✓", "WARN": "!", "FAIL": "✗", "SKIP": "-"}
_STATUS_ICON_ASCII = {"OK": "+", "WARN": "!", "FAIL": "x", "SKIP": "-"}

_SYNTAX_THEME = "one-dark"


def new_console(*, stderr: bool = False) -> Console:
    """Console of the command-line views; fixed width off a terminal keeps wrapping stable."""
    stream = sys.stderr if stderr else sys.stdout
    return Console(stderr=stderr, width=None if stream.isatty() else 200)


def supports_unicode(console: Console) -> bool:
    """Whether the console can print Unicode box and status glyphs."""
    encoding = (getattr(console.file, "encoding", "") or "").lower()
    return not console.options.ascii_only and "utf" in encoding


def status_icons(console: Console) -> dict[str, str]:
    """Status glyphs for this console, ASCII when Unicode is unavailable."""
    return _STATUS_ICON_UNICODE if supports_unicode(console) else _STATUS_ICON_ASCII


def status_text(console: Console, status: Status) -> Text:
    """Colored status glyph, e.g. a green ``✓``."""
    return Text(status_icons(console)[status], style=STATUS_STYLE[status])


def print_section(
    console: Console,
    title: str,
    body: RenderableType,
    *,
    note: str | None = None,
    accent: str = ACCENT,
) -> None:
    """Print a bold accent-colored title with an optional dim note, the body indented by
    two cells, and a blank line."""
    heading = Text(title, style=f"bold {accent}")
    if note:
        heading.append(f"  {note}", style="dim")
    console.print(heading)
    console.print(Padding(body, (0, 0, 0, 2), expand=False))
    console.print()


def print_error(
    console: Console,
    message: str,
    hint: str | None = None,
    *,
    prefix: str = "error",
    hint_prefix: str = "hint",
) -> None:
    """``error: message`` in red, then an optional dim ``hint: ...`` line."""
    console.print(Text.assemble((f"{prefix}: ", "bold red"), message))
    if hint:
        console.print(Text.assemble((f"{hint_prefix}: ", "bold"), (hint, "dim")))


def label_grid() -> Table:
    """Borderless two-column grid of labels and details."""
    grid = Table.grid(padding=(0, 2, 0, 0))
    grid.add_column(no_wrap=True)
    grid.add_column(overflow="fold")
    return grid


def key_value_grid(rows: Iterable[tuple[str, Any]], *, zebra: bool = True) -> Table:
    """Full-width grid of dim keys and folded values."""
    grid = Table.grid(expand=True, padding=(0, 2, 0, 0))
    grid.add_column(no_wrap=True, style="dim")
    grid.add_column(ratio=1, overflow="fold")
    if zebra:
        grid.row_styles = ZEBRA_ROW_STYLES
    for key, value in rows:
        grid.add_row(Text(key), value if isinstance(value, Text) else Text(str(value)))
    return grid


def _syntax_theme() -> PygmentsSyntaxTheme:
    theme = PygmentsSyntaxTheme(_SYNTAX_THEME)
    # Drop the theme background so the terminal background shows through. Token
    # styles read _background_color and the canvas reads _background_style.
    theme._background_color = None  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]
    theme._background_style = Style.null()  # pyright: ignore[reportPrivateUsage]
    return theme


def code_syntax(code: str, lexer: str = "python") -> Syntax:
    """Line-numbered code with a transparent background."""
    return Syntax(code, lexer, theme=_syntax_theme(), word_wrap=True, line_numbers=True)


__all__ = [
    "ACCENT",
    "STATUS_STYLE",
    "ZEBRA_ROW_STYLES",
    "Status",
    "TreeNode",
    "cap_list",
    "code_syntax",
    "flatten_tree",
    "format_size",
    "is_interactive",
    "key_value_grid",
    "label_grid",
    "new_console",
    "print_error",
    "print_section",
    "status_icons",
    "status_text",
    "supports_unicode",
]
