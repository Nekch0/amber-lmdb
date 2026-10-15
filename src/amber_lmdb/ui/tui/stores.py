"""Store explorer: a store list, store details with features and usage, and deletion."""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import shutil
from pathlib import Path
from typing import ClassVar

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Header, Static

from ..messages import ENGLISH, Messages
from ..render import ACCENT, code_syntax, format_size
from ..stores import (
    StoreProfile,
    StoreSummary,
    describe_store,
    describe_stores,
    find_stores,
    has_features,
)
from ..views import contents_grid, feature_grid, store_grid
from .widgets import (
    PANEL_CSS,
    ConfirmDelete,
    DragCopyApp,
    FastVerticalScroll,
    HintBar,
    LeanDataTable,
    SelectableStatic,
    SyncSelectScreen,
    code_gutter_width,
    copy_text,
    messages_of,
    titled_panel,
)

# Generation names (v-<32 hex>) are long; show their start only.
VERSION_COLUMN_WIDTH = 18


class StoreScreen(SyncSelectScreen[str | None]):
    """Details of one store; dismisses with ``"deleted"`` after a deletion."""

    BINDINGS: ClassVar = [
        Binding("escape", "back", "back"),
        Binding("c", "copy_path", "copy path"),
        Binding("d", "delete", "delete"),
    ]

    DEFAULT_CSS: ClassVar[str] = (
        PANEL_CSS
        + """
    StoreScreen SelectableStatic {
        height: auto;
    }
    """
    )

    def __init__(self, summary: StoreSummary, profile: StoreProfile) -> None:
        super().__init__()
        self._summary = summary
        self._profile = profile
        self.add_class("opaque")

    def compose(self) -> ComposeResult:
        messages = messages_of(self.app)
        summary, profile = self._summary, self._profile
        yield Header()
        with FastVerticalScroll(id="store-scroll"):
            yield titled_panel(SelectableStatic(id="store-info"), "Stored Information")
            if profile.contents is not None:
                yield titled_panel(SelectableStatic(id="store-contents"), "Content Tree")
            if has_features(summary) or summary.meta is None:
                yield titled_panel(SelectableStatic(id="store-features"), messages.features_title)
            if profile.source_file is not None:
                yield titled_panel(SelectableStatic(id="store-code"), "Source")
            if profile.usage is not None:
                yield titled_panel(SelectableStatic(id="store-usage"), messages.usage_title)
        yield HintBar(
            ("c", messages.hint_copy_path),
            ("d", messages.hint_delete),
            ("Esc", messages.hint_back),
            ("q", messages.hint_quit),
        )

    def on_mount(self) -> None:
        messages = messages_of(self.app)
        accent = getattr(self.app, "accent", ACCENT)
        summary, profile = self._summary, self._profile
        self.query_one("#store-info", Static).update(
            store_grid(
                summary, profile, messages=messages, accent=accent, with_contents=False, zebra=True
            )
        )
        if (contents := contents_grid(summary, profile)) is not None:
            self.query_one("#store-contents", Static).update(contents)
        if self.query("#store-features"):
            self.query_one("#store-features", Static).update(
                feature_grid(summary, messages=messages)
            )
        if profile.source_file is not None:
            widget = self.query_one("#store-code", SelectableStatic)
            source = profile.source_code(summary)
            if source is not None:
                widget.update(code_syntax(source))
                widget.selection_gutter = code_gutter_width(source)
            else:
                widget.update(
                    Text(messages.source_missing.format(file=profile.source_file), style="dim")
                )
        if profile.usage is not None:
            widget = self.query_one("#store-usage", SelectableStatic)
            code = profile.usage_code(summary)
            if code is not None:
                widget.update(code_syntax(code))
                widget.selection_gutter = code_gutter_width(code)
            else:
                widget.display = False

    def action_back(self) -> None:
        self.dismiss(None)

    def action_copy_path(self) -> None:
        path = str(self._summary.path)
        self.app.copy_to_clipboard(path)
        try:
            import pyperclip

            pyperclip.copy(path)
        except Exception:
            pass
        self.app.notify(messages_of(self.app).path_copied)

    def action_delete(self) -> None:
        path = self._summary.path
        messages = messages_of(self.app)

        def after_final(confirmed: bool | None) -> None:
            if not confirmed:
                return
            try:
                shutil.rmtree(path)
            except Exception as exc:
                self.app.notify(messages.delete_failed.format(error=exc), severity="error")
                return
            self.app.notify(messages.deleted.format(name=path.name))
            self.dismiss("deleted")

        def after_first(confirmed: bool | None) -> None:
            if confirmed:
                self.app.push_screen(ConfirmDelete(path, final=True), after_final)

        self.app.push_screen(ConfirmDelete(path), after_first)


class StoreListScreen(SyncSelectScreen[None]):
    """Stores of one directory; opening a store reads its metadata in a worker."""

    BINDINGS: ClassVar = [
        Binding("escape", "quit_app", "quit"),
        Binding("c", "copy_path", "copy path"),
    ]

    DEFAULT_CSS: ClassVar[str] = (
        PANEL_CSS
        + """
    StoreListScreen #stores {
        border-subtitle-align: right;
        border-subtitle-color: $text-muted;
    }
    """
    )

    def __init__(self, store_dir: Path, profile: StoreProfile) -> None:
        super().__init__()
        self._store_dir = store_dir
        self._profile = profile
        self._loading = False

    def compose(self) -> ComposeResult:
        messages = messages_of(self.app)
        yield Header()
        panel = titled_panel(
            LeanDataTable(id="stores", double_click_select=True), self._profile.title
        )
        panel.border_subtitle = str(self._store_dir)
        yield panel
        yield HintBar(
            ("↑↓", messages.hint_move),
            ("c", messages.hint_copy_path),
            ("Esc", messages.hint_quit),
        )

    def on_mount(self) -> None:
        table = self.query_one("#stores", DataTable)
        table.cursor_type = "row"
        table.add_column("NAME")
        table.add_column("TYPE")
        table.add_column("HASH")
        table.add_column("VERSION", width=VERSION_COLUMN_WIDTH)
        table.add_column("ENTRIES")
        table.add_column("USED")
        table.add_column("MODIFIED")
        self.refill()

    def refill(self) -> None:
        table = self.query_one("#stores", DataTable)
        table.clear()
        for summary in describe_stores(find_stores(self._store_dir)):
            table.add_row(
                summary.name,
                summary.typename,
                summary.digest,
                summary.version,
                "-" if summary.entries is None else str(summary.entries),
                format_size(summary.listed_size),
                summary.modified_text,
                key=summary.name,
            )

    def action_copy_path(self) -> None:
        table = self.query_one("#stores", DataTable)
        rows = table.ordered_rows
        if 0 <= table.cursor_row < len(rows):
            copy_text(self, str(self._store_dir / str(rows[table.cursor_row].key.value)))

    @on(DataTable.RowSelected, "#stores")
    def _open_store(self, event: DataTable.RowSelected) -> None:
        if self._loading:
            return
        self._loading = True
        self.query_one("#stores", DataTable).loading = True
        self._load_store(self._store_dir / str(event.row_key.value))

    @work(thread=True)
    def _load_store(self, store: Path) -> None:
        messages = messages_of(self.app)
        try:
            summary: StoreSummary | None = describe_store(
                store, with_entries=True, with_meta=True, messages=messages
            )
        except Exception:
            summary = None

        def after(result: str | None) -> None:
            if result == "deleted":
                self.refill()

        def push() -> None:
            self._loading = False
            self.query_one("#stores", DataTable).loading = False
            if summary is None:
                self.notify(messages.store_open_error, severity="error")
                return
            self.app.push_screen(StoreScreen(summary, self._profile), after)

        try:
            self.app.call_from_thread(push)
        except RuntimeError:
            self._loading = False

    def action_quit_app(self) -> None:
        self.app.exit()


class StoreExplorerApp(DragCopyApp):
    """Explorer of the stores in one directory."""

    def __init__(
        self,
        store_dir: Path,
        *,
        profile: StoreProfile | None = None,
        title: str = "amber-lmdb Store Explorer",
        theme_name: str = "amber",
        accent: str = ACCENT,
        messages: Messages = ENGLISH,
    ) -> None:
        super().__init__(title=title, theme_name=theme_name, accent=accent, messages=messages)
        self._store_dir = store_dir
        self._profile = profile or StoreProfile()

    def on_mount(self) -> None:
        self.push_screen(StoreListScreen(self._store_dir, self._profile))


__all__ = ["StoreExplorerApp", "StoreListScreen", "StoreScreen"]
