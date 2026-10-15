"""Models of the dataset browser, importable without loading textual.

:class:`Listing` describes one list screen and :class:`Detail` one detail screen;
:class:`amber_lmdb.ui.tui.BrowserApp` displays them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from rich.console import RenderableType

Severity = Literal["information", "warning", "error"]


@dataclass(frozen=True)
class Detail:
    """Content of a detail screen, prepared off the interface thread.

    ``info`` is a renderable or ``(key, value)`` rows. ``columns`` and ``rows`` fill an
    optional table whose cells copy on double click. ``notices`` are shown as
    notifications when the screen opens.
    """

    title: str
    info: RenderableType | Sequence[tuple[str, Any]]
    table_title: str = "Items"
    columns: Sequence[str] = ()
    rows: Sequence[Sequence[str]] = ()
    notices: Sequence[tuple[str, Severity]] = ()


@dataclass(frozen=True)
class Listing[T]:
    """One list screen.

    ``items`` is a sequence or a function returning one. ``row`` gives the cells of an
    item. ``key`` identifies an item (default: its index). ``filter_text`` gives the text
    the filter matches as a case-insensitive substring; without it there is no filter.
    ``open`` runs in a worker thread and returns the next screen; without it rows do not
    open. ``open_error`` replaces the generic failure notification.
    """

    title: str
    columns: Sequence[str]
    items: Sequence[T] | Callable[[], Sequence[T]]
    row: Callable[[T], Sequence[str]]
    open: Callable[[T], Listing[Any] | Detail] | None = None
    key: Callable[[T], str] | None = None
    filter_text: Callable[[T], str] | None = None
    filter_placeholder: str | None = None
    open_error: str | None = None

    def resolve_items(self) -> Sequence[T]:
        return self.items() if callable(self.items) else self.items


__all__ = ["Detail", "Listing", "Severity"]
