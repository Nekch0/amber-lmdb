"""Values and helpers of the views that do not need Rich, so importing them stays cheap."""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence
from typing import Literal

ACCENT = "#d77757"
"""Default accent color of section markers, spinners and explorer themes."""

Status = Literal["OK", "WARN", "FAIL", "SKIP"]

STATUS_STYLE: dict[str, str] = {
    "OK": "green",
    "WARN": "yellow",
    "FAIL": "bold red",
    "SKIP": "dim",
}

# Zebra rows dim the text instead of painting a background (grey70 is close to
# textual's $foreground-darken-2).
ZEBRA_ROW_STYLES = ("", "grey70")

TreeNode = tuple[str, str, Sequence["TreeNode"]]
"""``(name, note, children)`` node of a contents tree."""


def is_interactive() -> bool:
    """Whether both stdin and stdout are terminals, so an explorer can run."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def format_size(size: int) -> str:
    """Byte count with a binary unit, e.g. ``1.5 MiB``."""
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024:
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def cap_list(values: Iterable[str], limit: int = 8) -> str:
    """Comma-separated values, the excess summarized as ``+N more``."""
    items = list(values)
    shown = ", ".join(items[:limit])
    if len(items) > limit:
        shown += f" +{len(items) - limit} more"
    return shown


def flatten_tree(nodes: Sequence[TreeNode], prefix: str = "") -> list[tuple[str, str]]:
    """Expand nested nodes into ``(tree line, note)`` rows with box-drawing branches."""
    rows: list[tuple[str, str]] = []
    for index, (name, note, children) in enumerate(nodes):
        last = index == len(nodes) - 1
        rows.append((f"{prefix}{'└── ' if last else '├── '}{name}", note))
        rows.extend(flatten_tree(children, prefix + ("    " if last else "│   ")))
    return rows


def short_version(version: str, width: int = 12, *, ellipsis: str = "…") -> str:
    """Generation name cut to ``width`` characters with an ellipsis (``v-1a2b3c4d5e…``)."""
    if len(version) <= width:
        return version
    return version[: width - len(ellipsis)] + ellipsis


__all__ = [
    "ACCENT",
    "STATUS_STYLE",
    "ZEBRA_ROW_STYLES",
    "Status",
    "TreeNode",
    "cap_list",
    "flatten_tree",
    "format_size",
    "is_interactive",
    "short_version",
]
