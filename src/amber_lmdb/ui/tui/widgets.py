"""Textual widgets shared by the explorers: fast scrolling, drag selection and copying."""

# textual's annotations are incomplete; silence the Unknown-derived strict diagnostics.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from contextlib import suppress
from pathlib import Path
from typing import Any, ClassVar

from rich.color import Color
from rich.segment import Segment
from rich.style import Style
from rich.syntax import NUMBERS_COLUMN_DEFAULT_PADDING
from rich.text import Text
from textual import errors, events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.coordinate import Coordinate
from textual.geometry import Offset, Region
from textual.reactive import Reactive
from textual.screen import ModalScreen, Screen, ScreenResultType
from textual.scrollbar import ScrollBar, ScrollBarRender
from textual.selection import Selection, SelectState
from textual.strip import Strip
from textual.theme import Theme
from textual.visual import Visual
from textual.widget import Widget
from textual.widgets import Button, DataTable, Label, Static

from ..messages import ENGLISH, Messages
from ..render import ACCENT

PANEL_CSS = """
Header {
    color: $text-muted;
    text-style: none;
}
.panel {
    border: round $surface-lighten-1;
    border-title-color: $text-muted;
    padding: 1 2;
}
.panel:focus-within {
    border: round $primary;
}
.panel .datatable--header {
    text-style: bold;
}
/* Type plus class beats the default :dark row stripes (specificity 21). */
DataTable.panel > .datatable--even-row {
    background: transparent;
    color: $foreground-darken-2;
}
"""


def titled_panel(child: Widget, title: str) -> Widget:
    """Give a widget a titled rounded border directly, without a wrapper container."""
    child.add_class("panel")
    child.border_title = title
    return child


def code_gutter_width(code: str) -> int:
    """Cell width of the line-number gutter ``code_syntax`` draws before each line."""
    return len(str(1 + code.count("\n"))) + NUMBERS_COLUMN_DEFAULT_PADDING + 1


def messages_of(app: App[Any]) -> Messages:
    """Messages of an explorer app, English for other apps."""
    return getattr(app, "messages", ENGLISH)


def copy_text(screen: Screen[Any], text: str) -> None:
    """Copy text to the clipboard (terminal and system) and notify."""
    screen.app.copy_to_clipboard(text)
    with suppress(Exception):
        import pyperclip

        pyperclip.copy(text)
    screen.app.notify(messages_of(screen.app).copied.format(text=text))


class DragCopyApp(App[None]):
    """App base that copies drag selections and paints a transparent background."""

    # Selection colors must be set at the app level: screen DEFAULT_CSS is scoped and
    # loses on specificity. Screens are transparent so the terminal background shows;
    # only functional layers (selection, modal dim) stay painted.
    CSS: ClassVar[str] = """
    Screen > .screen--selection {
        background: $primary;
        color: $background;
    }
    Screen {
        background: transparent;
        scrollbar-background: transparent;
        scrollbar-background-hover: transparent;
        scrollbar-background-active: transparent;
        scrollbar-corner-color: transparent;
    }
    /* Stacked detail screens hide the list below them completely. */
    Screen.opaque {
        background: $background;
    }
    /* Modals keep a translucent dim for legibility; same specificity, so it comes last. */
    ModalScreen {
        background: $background 60%;
    }
    ScrollableContainer {
        scrollbar-background: transparent;
        scrollbar-background-hover: transparent;
        scrollbar-background-active: transparent;
        scrollbar-corner-color: transparent;
    }
    DataTable {
        background: transparent;
    }
    Input {
        background: transparent;
    }
    Header {
        background: transparent;
    }
    HeaderIcon {
        background: transparent;
    }
    HeaderTitle {
        background: transparent;
    }
    HeaderClock {
        background: transparent;
    }
    HeaderClockSpace {
        background: transparent;
    }
    """

    BINDINGS: ClassVar = [Binding("q", "quit", "quit")]

    _mouse_down_cell: Offset | None = None

    def __init__(
        self,
        *,
        title: str,
        theme_name: str = "amber",
        accent: str = ACCENT,
        messages: Messages = ENGLISH,
    ) -> None:
        super().__init__()
        self.messages = messages
        self.accent = accent
        self._initial_title = title
        self.register_theme(Theme(name=theme_name, primary=accent, accent=accent, dark=True))
        self.theme = theme_name

    def on_mount(self) -> None:
        self.title = self._initial_title

    async def on_event(self, event: events.Event) -> None:
        # Stale MouseMove events queued during fast motion are superseded by the next
        # one; dropping the backlog keeps hit-testing and hover updates from lagging.
        if isinstance(event, events.MouseMove) and any(
            isinstance(pending, events.MouseMove)
            for pending in self._message_queue.values  # pyright: ignore[reportPrivateUsage]
        ):
            return
        await super().on_event(event)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        self._mouse_down_cell = event.screen_offset

    def on_mouse_up(self, event: events.MouseUp) -> None:
        down_offset = self._mouse_down_cell
        self._mouse_down_cell = None
        # Copy only after a drag, not after clicks or double-click select-all.
        if down_offset is not None and down_offset != event.screen_offset:
            text = self.screen.get_selected_text()
            if text:
                copy_text(self.screen, text)
            self.screen.clear_selection()


def _highlight_selection(
    widget: Widget,
    selection: Selection | None,
    strip: Strip,
    content_y: int,
    gutter: int = 0,
) -> Strip:
    """Paint the selected span of line ``content_y`` with the screen selection style.

    ``gutter`` cells at the line start are not selectable; selection coordinates
    exclude them, so they are added back when converting to cell positions.
    """
    if selection is None or (span := selection.get_span(content_y)) is None:
        return strip
    start, end = span
    text_len = len(strip.text) - gutter
    end = text_len if end == -1 else min(end, text_len)
    if start >= end:
        return strip
    x1 = strip.index_to_cell_position(start + gutter)
    x2 = strip.index_to_cell_position(end + gutter)
    selection_style = widget.screen.get_component_rich_style("screen--selection")
    middle = strip.crop(x1, x2)
    selected = Strip(Segment.apply_style(middle, post_style=selection_style), middle.cell_length)
    return strip.crop(0, x1) + selected + strip.crop(x2, strip.cell_length)


def _repaint_rows_later(screen: Screen[Any], widget: Widget) -> None:
    """Let the next update tick draw the regions queued in the compositor.

    Repainting synchronously on every mouse move backs up during fast drags, so the
    drawing is left to the update timer; any dirty widget makes it refresh the
    compositor, which draws only the queued regions.
    """
    screen._dirty_widgets.add(widget)  # pyright: ignore[reportPrivateUsage]
    screen._update_timer.resume()  # pyright: ignore[reportPrivateUsage]


class SyncSelectScreen(Screen[ScreenResultType]):
    """Screen that applies selection changes to widgets synchronously.

    The base watcher is async and runs one message late, so drag highlighting lags.
    """

    # The base reactive repaints the whole screen on every mouse move while dragging;
    # widgets repaint only their changed lines in selection_updated instead.
    _select_state: Reactive[SelectState | None] = Reactive(None, repaint=False)

    def _watch_selections(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        old_selections: dict[Widget, Selection],
        selections: dict[Widget, Selection],
    ) -> None:
        for widget in old_selections.keys() | selections.keys():
            widget.selection_updated(selections.get(widget))

    def get_style_at(self, x: int, y: int) -> Style:
        # Event forwarding reads the style under the pointer on every mouse event.
        # DataTable lines are expensive to render, so LeanDataTable computes the cell
        # metadata from coordinates. Other widgets read cached lines, and scroll bar
        # actions need the real style metadata.
        try:
            widget, region = self.get_widget_at(x, y)
        except errors.NoWidget:
            return Style.null()
        if isinstance(widget, LeanDataTable) and widget in self._compositor.visible_widgets:  # pyright: ignore[reportPrivateUsage]
            meta_style = widget._meta_style_at(x - region.x, y - region.y)  # pyright: ignore[reportPrivateUsage]
            if meta_style is not None:
                return meta_style
        return super().get_style_at(x, y)


class SelectableStatic(Static):
    """Static whose rich renderables (tables, syntax) support drag selection.

    Rich visuals emit no ``offset`` metadata for selection, so ``render_line`` adds it
    and ``get_selection`` extracts text from the rendered lines.
    ``selection_gutter`` cells at the line start are not selectable (line numbers).
    """

    selection_gutter: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Offset metadata changes the hover link id per cell; with auto links every
        # mouse move would repaint the whole content.
        self.auto_links = False

    _offset_styles: ClassVar[dict[tuple[int, int], Style]] = {}

    def _apply_offsets(self, strip: Strip, y: int) -> Strip:
        """``strip.apply_offsets(0, y)`` reusing metadata styles across widgets and frames."""
        offset_styles = self._offset_styles
        segments = []
        append = segments.append
        x = 0
        for seg in strip:
            key = (x, y)
            offset_style = offset_styles.get(key)
            if offset_style is None:
                offset_style = offset_styles[key] = Style.from_meta({"offset": key})
            append(Segment(seg.text, seg.style + offset_style if seg.style else offset_style))
            x += len(seg.text)
        new_strip = Strip(segments, strip.cell_length)
        new_strip._render_cache = strip._render_cache  # pyright: ignore[reportPrivateUsage]
        return new_strip

    def render_line(self, y: int) -> Strip:
        strip = super().render_line(y)
        gutter = self.selection_gutter
        # Offsets mark the first character of each segment, so they are added after
        # highlighting splits the segments; otherwise split parts inherit the offset.
        strip = _highlight_selection(self, self._last_selection, strip, y, gutter)
        if gutter:
            # Every gutter cell maps to the line start: dragging over the gutter
            # selects from the line start without the line number.
            offset_style = self._offset_styles.get((0, y))
            if offset_style is None:
                offset_style = self._offset_styles[(0, y)] = Style.from_meta({"offset": (0, y)})
            gutter_segments = [
                Segment(char, seg.style + offset_style if seg.style else offset_style)
                for seg in strip.crop(0, gutter)
                for char in seg.text
            ]
            return Strip(gutter_segments) + self._apply_offsets(
                strip.crop(gutter, strip.cell_length), y
            )
        return self._apply_offsets(strip, y)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        visual = self._render()
        if isinstance(visual, (Text, Content)):
            return super().get_selection(selection)
        strips = Visual.to_strips(
            self, visual, self.size.width, None, self.visual_style, apply_selection=False
        )
        gutter = self.selection_gutter
        lines = [strip.text[gutter:].rstrip() for strip in strips]
        return selection.extract("\n".join(lines)), "\n"

    _last_selection: Selection | None = None

    def selection_updated(self, selection: Selection | None) -> None:
        # render_line applies the selection style, so the content need not re-render:
        # only the line cache of rows covered by the old or new selection is dirtied and
        # their screen regions are queued for the next update tick.
        if not self.is_attached:
            return
        old, self._last_selection = self._last_selection, selection
        region = self.region
        if not region:
            return
        # Selection rows count content lines; cache rows include the border and padding.
        top = self.gutter.top
        height = max(0, region.height - self.gutter.height)
        rows: set[int] = set()
        for sel in (old, selection):
            if sel is None:
                continue
            if sel.start is None and sel.end is None:
                rows.update(range(height))
                continue
            lo = sel.start.y if sel.start is not None else 0
            hi = sel.end.y if sel.end is not None else height - 1
            lo, hi = sorted((lo, hi))
            rows.update(range(max(lo, 0), min(hi, height - 1) + 1))
        if not rows:
            return
        # Dirty only the line cache; _set_dirty would re-render the whole content.
        self._styles_cache.set_dirty(  # pyright: ignore[reportPrivateUsage]
            *(Region(0, y + top, region.width, 1) for y in rows)
        )
        screen = self.screen
        compositor = screen._compositor  # pyright: ignore[reportPrivateUsage]
        # Partial updates do not clip to the screen; rows scrolled off screen would
        # index past the compositor chops.
        bounds = compositor.size.region
        dirty = [
            clip
            for y in rows
            if (clip := Region(region.x, region.y + top + y, region.width, 1)).intersection(bounds)
        ]
        if not dirty:
            return
        if screen.app._batch_count:  # pyright: ignore[reportPrivateUsage]
            # Batched updates drop _display, so take the normal path.
            screen.refresh(*dirty)
            return
        compositor._dirty_regions.update(dirty)  # pyright: ignore[reportPrivateUsage]
        _repaint_rows_later(screen, self)


class LeanScrollBar(ScrollBar):
    """Scroll bar that returns cached strips directly.

    The standard bar re-renders every line through rich on each position change,
    which dominates scrolling cost; the bar is one segment per line, so strips are
    built from ``ScrollBarRender.render_bar`` and cached.
    """

    _strips: list[Strip] | None = None
    _strips_key: tuple[Any, ...] | None = None

    def _bar_strips(self) -> list[Strip]:
        renderable = self.render()
        assert isinstance(renderable, ScrollBarRender)
        width, height = self.size
        key = (
            renderable.virtual_size,
            renderable.window_size,
            renderable.position,
            renderable.style,
            width,
            height,
        )
        if self._strips is not None and self._strips_key == key:
            return self._strips
        style = self.app.console.get_style(renderable.style)
        segments = ScrollBarRender.render_bar(
            size=height,
            virtual_size=renderable.virtual_size,
            window_size=renderable.window_size,
            position=renderable.position,
            thickness=width,
            vertical=True,
            back_color=style.bgcolor or Color.parse("#555555"),
            bar_color=style.color or Color.parse("bright_magenta"),
        )
        self._strips = [Strip([segment]) for segment in segments.segments]
        self._strips_key = key
        return self._strips

    def render_lines(self, crop: Region) -> list[Strip]:
        if not self.vertical:
            return super().render_lines(crop)
        lines = self._bar_strips()[crop.y : crop.bottom]
        if crop.column_span != (0, self.size.width):
            x1, x2 = crop.column_span
            lines = [strip.crop(x1, x2) for strip in lines]
        return lines

    def render_line(self, y: int) -> Strip:
        # Hit-testing reads the @mouse.down metadata (grab, page) through render_line.
        strips = self._bar_strips()
        if 0 <= y < len(strips):
            return strips[y]
        return Strip.blank(self.size.width, self.visual_style.rich_style)


def _lean_vertical_scrollbar(widget: Widget) -> ScrollBar:
    """Create and register a LeanScrollBar the same way ``Widget.vertical_scrollbar`` does."""
    if widget._vertical_scrollbar is not None:  # pyright: ignore[reportPrivateUsage]
        return widget._vertical_scrollbar  # pyright: ignore[reportPrivateUsage]
    scroll_bar = LeanScrollBar(
        vertical=True, name="vertical", thickness=widget.scrollbar_size_vertical
    )
    widget._vertical_scrollbar = scroll_bar  # pyright: ignore[reportPrivateUsage]
    scroll_bar.display = False
    widget.app._start_widget(widget, scroll_bar)  # pyright: ignore[reportPrivateUsage]
    return scroll_bar


class FastVerticalScroll(VerticalScroll):
    """VerticalScroll with a LeanScrollBar."""

    @property
    def vertical_scrollbar(self) -> ScrollBar:
        return _lean_vertical_scrollbar(self)


class LeanDataTable(DataTable[Any]):
    """DataTable with a LeanScrollBar, cheap hover handling and optional cell copy.

    ``cell_copy`` copies a cell on double click; tables that open rows use
    ``double_click_select`` instead, so a single click only moves the cursor.
    """

    def __init__(
        self,
        *args: Any,
        cell_copy: bool = False,
        double_click_select: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("zebra_stripes", True)
        super().__init__(*args, **kwargs)
        self._cell_copy = cell_copy
        self._double_click_select = double_click_select
        # Per-cell row/column metadata changes the link id; with auto links crossing a
        # cell would repaint the whole table. Cells hold no links.
        self.auto_links = False

    def _meta_style_at(self, x: int, y: int) -> Style | None:
        """Compute the ``{"row", "column"}`` metadata of a widget coordinate.

        Replaces the rendering hit-test with the same rules: the header row is -1,
        the label column is -1, space after the last column is out of bounds, and the
        gutter has no metadata. Tables with fixed rows or columns return None so the
        caller falls back to rendering.
        """
        if self.fixed_rows or self.fixed_columns:
            return None
        gutter = self.gutter
        if (
            x < gutter.left
            or y < gutter.top
            or x >= gutter.left + self.size.width
            or y >= gutter.top + self.size.height
        ):
            return Style()
        content_x = x - gutter.left + self.scroll_offset.x
        content_y = y - gutter.top + self.scroll_offset.y
        try:
            row_key, _ = self._get_offsets(content_y)  # pyright: ignore[reportPrivateUsage]
        except (LookupError, IndexError):
            return Style()
        if row_key is self._header_row_key:  # pyright: ignore[reportPrivateUsage]
            row_index = -1
        else:
            located = self._row_locations.get(row_key)  # pyright: ignore[reportPrivateUsage]
            row_index = located if located is not None else -1
        cells_x = content_x - self._row_label_column_width  # pyright: ignore[reportPrivateUsage]
        column_index = -1
        width = 0
        for index, column in enumerate(self.ordered_columns):
            width += column.get_render_width(self)
            if cells_x < width:
                column_index = index
                break
        meta: dict[str, Any]
        if cells_x < 0:
            meta = {"row": row_index, "column": -1}
        elif cells_x >= width:
            meta = {"row": row_index, "column": 0, "out_of_bounds": True}
        else:
            meta = {"row": row_index, "column": column_index}
        return Style.from_meta(meta)

    def _set_hover_cursor(self, active: bool) -> None:
        # The base repaints the hover row even when nothing changes.
        if self._show_hover_cursor == active:  # pyright: ignore[reportPrivateUsage]
            return
        super()._set_hover_cursor(active)

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        if self.cursor_type == "row":
            meta = event.style.meta
            if not meta:
                self._set_hover_cursor(False)
            else:
                self._set_hover_cursor(True)
                if self.show_cursor:
                    row, column = meta.get("row"), meta.get("column")
                    if isinstance(row, int) and isinstance(column, int):
                        new_coordinate = Coordinate(row, column)
                        if new_coordinate.row != self.hover_coordinate.row:
                            self.hover_coordinate = new_coordinate
                        elif new_coordinate.column != self.hover_coordinate.column:
                            # The row highlight spans the row: keep the coordinate
                            # current without repainting.
                            self.set_reactive(DataTable.hover_coordinate, new_coordinate)
        else:
            super()._on_mouse_move(event)
        # The message pump calls _on_mouse_move of every class in the MRO; stop the
        # base handler from firing the always-update watcher again.
        event.prevent_default()

    def watch_hover_coordinate(self, old: Coordinate, value: Coordinate) -> None:
        # Repaint in the unit of the cursor, or stale cells keep a partial highlight.
        if self.cursor_type == "row":
            self.refresh_row(old.row)
            self.refresh_row(value.row)
        elif self.cursor_type == "column":
            self.refresh_column(old.column)
            self.refresh_column(value.column)
        else:
            super().watch_hover_coordinate(old, value)

    async def _on_click(self, event: events.Click) -> None:
        meta = event.style.meta or {}
        row, column = meta.get("row"), meta.get("column")
        if (
            self._double_click_select
            and isinstance(row, int)
            and isinstance(column, int)
            and row >= 0
            and column >= 0
            and self.show_cursor
            and self.cursor_type != "none"
            and not (self.cursor_type != "row" and meta.get("out_of_bounds", False))
        ):
            self._set_hover_cursor(True)
            new_coordinate = Coordinate(row, column)
            if event.chain >= 2:
                self.cursor_coordinate = new_coordinate
                self._post_selected_message()
            elif new_coordinate != self.cursor_coordinate:
                self.cursor_coordinate = new_coordinate
            self._scroll_cursor_into_view(animate=True)
            event.stop()
        else:
            await super()._on_click(event)
        # The base _on_click would run again through the MRO and, with the cursor
        # already on the cell, post a second selection.
        event.prevent_default()
        if event.chain != 2 or not self._cell_copy:
            return
        if (
            not isinstance(row, int)
            or not isinstance(column, int)
            or row < 0
            or column < 0
            or row >= len(self.ordered_rows)
            or column >= len(self.ordered_columns)
        ):
            return
        text = str(self.get_cell_at(Coordinate(row, column)))
        if text:
            copy_text(self.screen, text)

    @property
    def vertical_scrollbar(self) -> ScrollBar:
        return _lean_vertical_scrollbar(self)


class HintBar(Static):
    """Key hints docked at the bottom of a screen."""

    DEFAULT_CSS: ClassVar[str] = """
    HintBar {
        dock: bottom;
        height: 1;
        padding: 0 1;
        background: transparent;
        color: $text-muted;
    }
    """

    def __init__(self, *hints: tuple[str, str]) -> None:
        text = "    ".join(f"[bold reverse] {key} [/] {desc}" for key, desc in hints)
        super().__init__(text)


class ConfirmDelete(ModalScreen[bool]):
    """Deletion confirmation; the initial focus is on Cancel."""

    BINDINGS: ClassVar = [
        Binding("y", "confirm", "confirm"),
        Binding("n", "cancel", "cancel"),
        Binding("escape", "cancel", "cancel"),
    ]

    DEFAULT_CSS: ClassVar[str] = """
    ConfirmDelete {
        align: center middle;
    }
    ConfirmDelete #confirm-dialog {
        width: auto;
        height: auto;
        max-width: 80%;
        padding: 1 2;
        border: round $error;
        background: $surface;
    }
    ConfirmDelete #confirm-dialog Horizontal {
        height: auto;
        padding-top: 1;
    }
    ConfirmDelete #confirm-dialog Button {
        margin-right: 2;
    }
    """

    def __init__(self, path: Path, *, final: bool = False) -> None:
        super().__init__()
        self._path = path
        self._final = final

    def compose(self) -> ComposeResult:
        messages = messages_of(self.app)
        template = messages.confirm_delete_final if self._final else messages.confirm_delete
        with Vertical(id="confirm-dialog") as dialog:
            dialog.border_title = "Delete"
            yield Label(template.format(path=self._path))
            with Horizontal():
                yield Button("Delete (y)", id="delete", variant="error")
                yield Button("Cancel (n)", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#cancel", Button).focus()

    @on(Button.Pressed, "#delete")
    def action_confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(False)


__all__ = [
    "PANEL_CSS",
    "ConfirmDelete",
    "DragCopyApp",
    "FastVerticalScroll",
    "HintBar",
    "LeanDataTable",
    "LeanScrollBar",
    "SelectableStatic",
    "SyncSelectScreen",
    "code_gutter_width",
    "copy_text",
    "messages_of",
    "titled_panel",
]
