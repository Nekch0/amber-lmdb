"""Hierarchical browser: filterable lists that open nested lists or detail screens.

A :class:`Listing` describes one list screen. Selecting a row calls ``open`` in a
worker thread, so slow parsing never blocks the interface; ``open`` returns either
another :class:`Listing` or a :class:`Detail` (an information panel and an optional
table).

>>> from amber_lmdb.ui.tui import BrowserApp, Detail, Listing
>>> records = Listing(
...     title="Records",
...     columns=("id", "text"),
...     items=[("a", "hello"), ("b", "world")],
...     row=lambda item: item,
...     key=lambda item: item[0],
...     open=lambda item: Detail(title="Record", info=[("id", item[0]), ("text", item[1])]),
... )
>>> app = BrowserApp(records, title="Example")  # app.run() starts it
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Header, Input, Static

from ..browse import Detail, Listing, Severity
from ..messages import ENGLISH, Messages
from ..render import ACCENT, key_value_grid
from .widgets import (
    PANEL_CSS,
    DragCopyApp,
    HintBar,
    LeanDataTable,
    SelectableStatic,
    SyncSelectScreen,
    messages_of,
    titled_panel,
)


class DetailScreen(SyncSelectScreen[None]):
    """Information panel and an optional table."""

    BINDINGS: ClassVar = [Binding("escape", "back", "back")]

    DEFAULT_CSS: ClassVar[str] = (
        PANEL_CSS
        + """
    DetailScreen #detail-info {
        height: auto;
    }
    """
    )

    def __init__(self, detail: Detail) -> None:
        super().__init__()
        self.detail = detail
        self.add_class("opaque")

    def compose(self) -> ComposeResult:
        messages = messages_of(self.app)
        yield Header()
        yield titled_panel(SelectableStatic(id="detail-info"), self.detail.title)
        if self.detail.columns:
            yield titled_panel(
                LeanDataTable(id="detail-table", cell_copy=True), self.detail.table_title
            )
        yield HintBar(
            ("↑↓", messages.hint_select), ("Esc", messages.hint_back), ("q", messages.hint_quit)
        )

    def on_mount(self) -> None:
        info = self.detail.info
        if isinstance(info, Sequence) and not isinstance(info, str):
            info = key_value_grid(info)
        self.query_one("#detail-info", Static).update(info)
        if self.detail.columns:
            table = self.query_one("#detail-table", DataTable)
            table.cursor_type = "row"
            table.add_columns(*self.detail.columns)
            for row in self.detail.rows:
                table.add_row(*row)
        for message, severity in self.detail.notices:
            self.app.notify(message, severity=severity)

    def action_back(self) -> None:
        self.app.pop_screen()


class ListingScreen(SyncSelectScreen[None]):
    """A filterable list; Escape quits at the root and goes back elsewhere."""

    BINDINGS: ClassVar = [Binding("escape", "leave", "back")]

    DEFAULT_CSS: ClassVar[str] = PANEL_CSS

    def __init__(self, listing: Listing[Any], *, root: bool = False) -> None:
        super().__init__()
        self.listing = listing
        self._root = root
        self._items: Sequence[Any] = ()
        self._loading = False
        if not root:
            self.add_class("opaque")

    def compose(self) -> ComposeResult:
        messages = messages_of(self.app)
        listing = self.listing
        yield Header()
        if listing.filter_text is not None:
            yield Input(
                placeholder=listing.filter_placeholder or messages.filter_placeholder,
                id="filter",
            )
        yield titled_panel(
            LeanDataTable(id="items", double_click_select=listing.open is not None),
            listing.title,
        )
        move = "Tab/↑↓" if listing.filter_text is not None else "↑↓"
        if self._root:
            yield HintBar((move, messages.hint_move), ("Esc", messages.hint_quit))
        else:
            yield HintBar(
                (move, messages.hint_move), ("Esc", messages.hint_back), ("q", messages.hint_quit)
            )

    def on_mount(self) -> None:
        table = self.query_one("#items", DataTable)
        table.cursor_type = "row"
        table.add_columns(*self.listing.columns)
        self._items = self.listing.resolve_items()
        self.refill("")
        if self.listing.filter_text is not None:
            self.query_one("#filter", Input).focus()

    def _key(self, index: int, item: Any) -> str:
        return str(index) if self.listing.key is None else self.listing.key(item)

    def refill(self, needle: str) -> None:
        table = self.query_one("#items", DataTable)
        table.clear()
        needle = needle.strip().lower()
        filter_text = self.listing.filter_text
        for index, item in enumerate(self._items):
            if needle and filter_text is not None and needle not in filter_text(item).lower():
                continue
            table.add_row(*self.listing.row(item), key=f"{index}\x00{self._key(index, item)}")

    @on(Input.Changed, "#filter")
    def _filter_changed(self, event: Input.Changed) -> None:
        self.refill(event.value)

    @on(DataTable.RowSelected, "#items")
    def _open_item(self, event: DataTable.RowSelected) -> None:
        if self.listing.open is None or self._loading:
            return
        self._loading = True
        # Opening may parse files; show a loading overlay over the list meanwhile.
        self.query_one("#items", DataTable).loading = True
        index = int(str(event.row_key.value).partition("\x00")[0])
        self._load(self._items[index])

    @work(thread=True)
    def _load(self, item: Any) -> None:
        opener = self.listing.open
        assert opener is not None
        try:
            target: Listing[Any] | Detail | None = opener(item)
        except Exception:
            target = None

        def push() -> None:
            self._loading = False
            self.query_one("#items", DataTable).loading = False
            if target is None:
                message = self.listing.open_error or messages_of(self.app).item_open_error
                self.notify(message, severity="error")
            elif isinstance(target, Detail):
                self.app.push_screen(DetailScreen(target))
            else:
                self.app.push_screen(ListingScreen(target))

        try:
            self.app.call_from_thread(push)
        except RuntimeError:
            self._loading = False

    def action_leave(self) -> None:
        if self._root:
            self.app.exit()
        else:
            self.app.pop_screen()


class BrowserApp(DragCopyApp):
    """Explorer starting at a root :class:`Listing`."""

    def __init__(
        self,
        root: Listing[Any],
        *,
        title: str,
        theme_name: str = "amber",
        accent: str = ACCENT,
        messages: Messages = ENGLISH,
    ) -> None:
        super().__init__(title=title, theme_name=theme_name, accent=accent, messages=messages)
        self.root_listing = root

    def on_mount(self) -> None:
        self.push_screen(ListingScreen(self.root_listing, root=True))


__all__ = ["BrowserApp", "Detail", "DetailScreen", "Listing", "ListingScreen", "Severity"]
