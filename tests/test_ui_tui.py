"""Textual explorers: the store explorer, the hierarchical browser and shared widgets."""

from __future__ import annotations

import asyncio
import io
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rich.console import Console
from textual._compositor import CompositorUpdate
from textual.app import App, ComposeResult
from textual.coordinate import Coordinate
from textual.events import MouseMove
from textual.geometry import Offset, Region
from textual.pilot import Pilot
from textual.screen import Screen
from textual.selection import Selection
from textual.widgets import DataTable, Header, Input, Label, Static

import amber_lmdb.ui.tui.stores as stores_module
from amber_lmdb import LMDBWriter
from amber_lmdb.ui import ENGLISH, StoreProfile, feature_nodes
from amber_lmdb.ui.tui import (
    BrowserApp,
    ConfirmDelete,
    Detail,
    DetailScreen,
    DragCopyApp,
    HintBar,
    LeanDataTable,
    Listing,
    ListingScreen,
    StoreExplorerApp,
    StoreListScreen,
    StoreScreen,
)


def make_store(store_dir: Path, name: str = "testfeat_a1b2c3d4.lmdb") -> Path:
    """Store directory with CURRENT and an empty generation: listed, but unreadable."""
    store = store_dir / name
    version = f"v-{uuid.uuid4().hex}"
    (store / "versions" / version).mkdir(parents=True)
    (store / "CURRENT").write_text(version + "\n", encoding="ascii")
    return store


def make_real_store(store_dir: Path, name: str = "testfeat_a1b2c3d4.lmdb") -> Path:
    """Store with a feature-convention meta and one array."""
    store = store_dir / name
    meta = {
        "schema_version": 1,
        "complete": True,
        "record_type": "ToyRecord",
        "n_utterances": 1,
        "names": ("logmel", "label"),
        "type_map": {"logmel": np.ndarray, "label": int},
        "dtype_map": {"logmel": "<f4"},
        "shape_map": {"logmel": ((2, 4), (2, 4))},
        "count_map": {"logmel": 1, "label": 1},
        "python_version": "3.12.1",
    }
    with LMDBWriter(store) as writer:
        writer.write_dict(meta, key="meta")
        writer.write_ndarray(np.zeros((2, 4), dtype=np.float32), key=("ndarray", "logmel"), index=0)
        writer.commit()
    return store


def render_static(static: Static) -> str:
    """Plain text of a Static's renderable."""
    out = io.StringIO()
    Console(file=out, width=160, force_terminal=False).print(static.content)
    return out.getvalue()


async def wait_screen(
    pilot: Pilot[Any], app: App[Any], screen_type: type[Screen[Any]]
) -> Screen[Any]:
    """Advance events until a screen of this type is active."""
    for _ in range(50):
        await pilot.pause()
        if isinstance(app.screen, screen_type):
            return app.screen
    raise AssertionError(f"{screen_type.__name__} did not appear: {app.screen!r}")


async def wait_region_stable(pilot: Pilot[Any], widget: DataTable[Any]) -> None:
    """Wait until the widget region is unchanged for five ticks.

    Screens with auto-height sibling panels keep relaying out after a push, which
    would shift computed click coordinates.
    """
    stable = 0
    region = None
    for _ in range(120):
        await pilot.pause()
        if widget.region == region:
            stable += 1
            if stable >= 5:
                return
        else:
            stable = 0
            region = widget.region


async def open_first_store(pilot: Pilot[Any], app: App[Any]) -> StoreScreen:
    screen = await wait_screen(pilot, app, StoreListScreen)
    screen.query_one("#stores", DataTable).focus()
    await pilot.pause()
    await pilot.press("enter")
    store_screen = await wait_screen(pilot, app, StoreScreen)
    assert isinstance(store_screen, StoreScreen)
    return store_screen


# Hierarchical browser ------------------------------------------------------------

RECORDS = [("f01", "ang", "angry text"), ("f01", "joy", "joyful text"), ("m02", "sad", "sad text")]


def record_detail(record: tuple[str, str, str]) -> Detail:
    return Detail(
        title="Record",
        info=[("speaker", record[0]), ("emotion", record[1])],
        table_title="Utterances",
        columns=("id", "text"),
        rows=[("1", record[2])],
        notices=[("annotation missing", "warning")],
    )


def records_listing(**changes: Any) -> Listing[Any]:
    options: dict[str, Any] = {
        "title": "Records",
        "columns": ("speaker", "emotion", "text"),
        "items": RECORDS,
        "row": lambda record: record,
        "filter_text": lambda record: f"{record[0]} {record[1]}",
        "open": record_detail,
    }
    options.update(changes)
    return Listing(**options)


class TestBrowser:
    def test_root_list_populates(self) -> None:
        async def scenario() -> tuple[int, bool]:
            app = BrowserApp(records_listing(), title="Toy")
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, ListingScreen)
                return screen.query_one("#items", DataTable).row_count, app.title == "Toy"

        assert asyncio.run(scenario()) == (3, True)

    def test_items_may_be_a_function(self) -> None:
        async def scenario() -> int:
            app = BrowserApp(records_listing(items=lambda: RECORDS[:2]), title="Toy")
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, ListingScreen)
                return screen.query_one("#items", DataTable).row_count

        assert asyncio.run(scenario()) == 2

    def test_filter_narrows_rows(self) -> None:
        async def scenario() -> tuple[int, int]:
            app = BrowserApp(records_listing(), title="Toy")
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, ListingScreen)
                table = screen.query_one("#items", DataTable)
                before = table.row_count
                screen.query_one("#filter", Input).value = "F01 ANG"
                await pilot.pause()
                await pilot.pause()
                return before, table.row_count

        assert asyncio.run(scenario()) == (3, 1)

    def test_listing_without_filter_has_no_input(self) -> None:
        async def scenario() -> int:
            app = BrowserApp(records_listing(filter_text=None), title="Toy")
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, ListingScreen)
                return len(screen.query(Input))

        assert asyncio.run(scenario()) == 0

    def test_open_shows_detail_with_table_and_notices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        notified: list[tuple[str, str]] = []

        async def scenario() -> tuple[str, int, float, bool]:
            app = BrowserApp(records_listing(), title="Toy")
            monkeypatch.setattr(
                app,
                "notify",
                lambda message, severity="information", **_: notified.append((message, severity)),
            )
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, ListingScreen)
                screen.query_one("#items", DataTable).focus()
                await pilot.pause()
                await pilot.press("enter")
                detail = await wait_screen(pilot, app, DetailScreen)
                await pilot.pause()
                info = render_static(detail.query_one("#detail-info", Static))
                rows = detail.query_one("#detail-table", DataTable).row_count
                alpha = detail.styles.background.a
                await pilot.press("escape")
                await pilot.pause()
                await pilot.pause()
                return info, rows, alpha, isinstance(app.screen, ListingScreen)

        info, rows, alpha, back = asyncio.run(scenario())
        assert "speaker" in info and "f01" in info
        assert rows == 1
        assert alpha == 1.0
        assert back
        assert ("annotation missing", "warning") in notified

    def test_detail_without_columns_has_no_table(self) -> None:
        async def scenario() -> int:
            app = BrowserApp(
                records_listing(open=lambda r: Detail(title="Record", info=[("id", r[0])])),
                title="Toy",
            )
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, ListingScreen)
                screen.query_one("#items", DataTable).focus()
                await pilot.pause()
                await pilot.press("enter")
                detail = await wait_screen(pilot, app, DetailScreen)
                return len(detail.query(DataTable))

        assert asyncio.run(scenario()) == 0

    def test_nested_listing_and_escape_goes_back(self) -> None:
        sessions = Listing(
            title="Sessions",
            columns=("session",),
            items=["s1", "s2"],
            row=lambda session: (session,),
            open=lambda session: records_listing(title=f"Records of {session}"),
        )

        async def scenario() -> tuple[str, float, str, bool]:
            app = BrowserApp(sessions, title="Toy")
            async with app.run_test() as pilot:
                root = await wait_screen(pilot, app, ListingScreen)
                root_alpha = root.styles.background.a
                root.query_one("#items", DataTable).focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(50):
                    await pilot.pause()
                    if app.screen is not root and isinstance(app.screen, ListingScreen):
                        break
                nested = app.screen
                assert isinstance(nested, ListingScreen)
                title = nested.listing.title
                nested_alpha = nested.styles.background.a
                hint = str(nested.query_one(HintBar).render())
                await pilot.press("escape")
                await pilot.pause()
                await pilot.pause()
                return title, nested_alpha - root_alpha, hint, app.screen is root

        title, alpha_gap, hint, back = asyncio.run(scenario())
        assert title == "Records of s1"
        assert alpha_gap == 1.0
        assert "back" in hint
        assert back

    def test_escape_at_root_quits(self) -> None:
        async def scenario() -> int | None:
            app = BrowserApp(records_listing(filter_text=None), title="Toy")
            async with app.run_test() as pilot:
                await wait_screen(pilot, app, ListingScreen)
                await pilot.press("escape")
                await pilot.pause()
            return app.return_code

        assert asyncio.run(scenario()) == 0

    def test_rows_without_open_do_not_navigate(self) -> None:
        async def scenario() -> bool:
            app = BrowserApp(records_listing(open=None, filter_text=None), title="Toy")
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, ListingScreen)
                screen.query_one("#items", DataTable).focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(10):
                    await pilot.pause()
                return app.screen is screen

        assert asyncio.run(scenario())

    def test_open_failure_notifies_open_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        notified: list[tuple[str, str]] = []

        def fail(_record: Any) -> Detail:
            raise OSError("unreadable")

        async def scenario() -> tuple[bool, bool]:
            app = BrowserApp(
                records_listing(open=fail, open_error="cannot open record"), title="Toy"
            )
            monkeypatch.setattr(
                app,
                "notify",
                lambda message, severity="information", **_: notified.append((message, severity)),
            )
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, ListingScreen)
                table = screen.query_one("#items", DataTable)
                table.focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(50):
                    await pilot.pause()
                    if notified:
                        break
                return app.screen is screen, table.loading

        stays, loading = asyncio.run(scenario())
        assert stays and not loading
        assert notified == [("cannot open record", "error")]

    def test_loading_overlay_while_opening(self) -> None:
        release = threading.Event()

        def slow(record: tuple[str, str, str]) -> Detail:
            assert release.wait(timeout=10)
            return record_detail(record)

        async def scenario() -> tuple[bool, bool]:
            app = BrowserApp(records_listing(open=slow), title="Toy")
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, ListingScreen)
                table = screen.query_one("#items", DataTable)
                table.focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(50):
                    await pilot.pause()
                    if table.loading:
                        break
                during = table.loading
                release.set()
                await wait_screen(pilot, app, DetailScreen)
                return during, table.loading

        assert asyncio.run(scenario()) == (True, False)

    def test_custom_hints(self) -> None:
        messages = replace(ENGLISH, hint_move="go", hint_quit="leave")

        async def scenario() -> str:
            app = BrowserApp(records_listing(), title="Toy", messages=messages)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, ListingScreen)
                return str(screen.query_one(HintBar).render())

        hint = asyncio.run(scenario())
        assert "go" in hint and "leave" in hint


# Store explorer --------------------------------------------------------------------


class TestStoreExplorer:
    def test_store_list_populates(self, tmp_path: Path) -> None:
        make_store(tmp_path)

        async def scenario() -> tuple[int, int]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                table = screen.query_one("#stores", DataTable)
                version = next(c for c in table.columns.values() if c.label.plain == "VERSION")
                return table.row_count, version.width

        # The VERSION column is capped instead of showing all of v-<32 hex>.
        assert asyncio.run(scenario()) == (1, 18)

    def test_backgrounds_are_transparent(self, tmp_path: Path) -> None:
        """Screens, tables, headers, hint bars and scroll bars paint no background.

        Modals keep a translucent dim for legibility over the content behind them.
        """
        make_store(tmp_path)

        async def scenario() -> tuple[float, ...]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                table = screen.query_one("#stores", DataTable)
                alphas = (
                    screen.styles.background.a,
                    table.styles.background.a,
                    table.styles.scrollbar_background.a,
                    screen.query_one(Header).styles.background.a,
                    screen.query_one(HintBar).styles.background.a,
                )
                app.push_screen(ConfirmDelete(Path("x.lmdb")))
                modal = await wait_screen(pilot, app, ConfirmDelete)
                return (*alphas, modal.styles.background.a)

        *alphas, modal_alpha = asyncio.run(scenario())
        assert alphas == [0.0] * 5
        assert modal_alpha == 0.6

    def test_pushed_screens_hide_list(self, tmp_path: Path) -> None:
        make_store(tmp_path)

        async def scenario() -> tuple[float, float]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                list_alpha = screen.styles.background.a
                detail = await open_first_store(pilot, app)
                return list_alpha, detail.styles.background.a

        assert asyncio.run(scenario()) == (0.0, 1.0)

    def test_open_store_shows_detail(self, tmp_path: Path) -> None:
        make_store(tmp_path)

        async def scenario() -> tuple[str, str]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                store_screen = await open_first_store(pilot, app)
                info = render_static(store_screen.query_one("#store-info", Static))
                usage = render_static(store_screen.query_one("#store-usage", Static))
                return info, usage

        info, usage = asyncio.run(scenario())
        assert "testfeat_a1b2c3d4.lmdb" in info
        assert "from amber_lmdb import LMDBReader" in usage

    def test_store_detail_shows_meta_and_features(self, tmp_path: Path) -> None:
        make_real_store(tmp_path)

        async def scenario() -> tuple[str, str]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                store_screen = await open_first_store(pilot, app)
                info = render_static(store_screen.query_one("#store-info", Static))
                features = render_static(store_screen.query_one("#store-features", Static))
                return info, features

        info, features = asyncio.run(scenario())
        assert "record_type" in info and "ToyRecord" in info
        assert "FEATURE" in features and "SHAPE" in features and "ENTRIES" in features
        assert "logmel" in features and "ndarray" in features and "<f4" in features
        assert "label" in features and "primitive" in features
        assert "(2, 4)" in features

    def test_store_detail_grids_use_zebra(self, tmp_path: Path) -> None:
        make_real_store(tmp_path)

        async def scenario() -> tuple[Any, Any]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                store_screen = await open_first_store(pilot, app)
                return (
                    store_screen.query_one("#store-info", Static).content,
                    store_screen.query_one("#store-features", Static).content,
                )

        info_grid, features_grid = asyncio.run(scenario())
        # Data rows alternate a dimmed text color, starting plain.
        assert not info_grid.rows[0].style
        assert info_grid.rows[1].style == "grey70"
        assert list(features_grid.row_styles)

    def test_store_list_uses_zebra_stripes(self, tmp_path: Path) -> None:
        make_store(tmp_path)

        async def scenario() -> tuple[bool, Any]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                table = screen.query_one("#stores", DataTable)
                return table.zebra_stripes, table.get_component_styles("datatable--even-row")

        zebra, styles = asyncio.run(scenario())
        assert zebra
        # Even rows dim the text instead of painting a background.
        assert not styles.has_rule("background") or styles.background.a == 0
        assert styles.color.hex == "#C2C2C2"

    def test_store_reopen_shows_features(self, tmp_path: Path) -> None:
        """Reopening a store still reads its meta (no environment left open)."""
        make_real_store(tmp_path)

        async def scenario() -> tuple[str, str]:
            app = StoreExplorerApp(tmp_path)
            info = features = ""
            async with app.run_test() as pilot:
                for _ in range(2):
                    store_screen = await open_first_store(pilot, app)
                    info = render_static(store_screen.query_one("#store-info", Static))
                    features = render_static(store_screen.query_one("#store-features", Static))
                    await pilot.press("escape")
                return info, features

        info, features = asyncio.run(scenario())
        assert "Could not read the meta key" not in info
        assert "(2, 4)" in features

    def test_open_store_shows_loading(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_real_store(tmp_path)
        release = threading.Event()
        real_describe = stores_module.describe_store

        def slow_describe(*args: Any, **kwargs: Any) -> Any:
            # Delay only the detail read, which includes the meta.
            if kwargs.get("with_meta"):
                assert release.wait(timeout=10)
            return real_describe(*args, **kwargs)

        monkeypatch.setattr(stores_module, "describe_store", slow_describe)

        async def scenario() -> tuple[bool, bool]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                table = screen.query_one("#stores", DataTable)
                table.focus()
                await pilot.pause()
                await pilot.press("enter")
                for _ in range(50):
                    await pilot.pause()
                    if table.loading:
                        break
                during = table.loading
                release.set()
                await wait_screen(pilot, app, StoreScreen)
                await pilot.press("escape")
                list_screen = await wait_screen(pilot, app, StoreListScreen)
                return during, list_screen.query_one("#stores", DataTable).loading

        assert asyncio.run(scenario()) == (True, False)

    def test_store_detail_shows_meta_error_reason(self, tmp_path: Path) -> None:
        make_store(tmp_path)  # no generation LMDB, so the meta cannot be read

        async def scenario() -> tuple[str, str | None]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                store_screen = await open_first_store(pilot, app)
                info = render_static(store_screen.query_one("#store-info", Static))
                return info, store_screen._summary.meta_error  # pyright: ignore[reportPrivateUsage]

        info, meta_error = asyncio.run(scenario())
        assert "Could not read the meta key" in info
        assert meta_error is not None
        # The rendering wraps, so compare without whitespace.
        assert meta_error.replace(" ", "") in "".join(info.split())

    def test_copy_path_copies_store_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = make_store(tmp_path)
        copied: list[str] = []
        monkeypatch.setattr("pyperclip.copy", copied.append)

        async def scenario() -> None:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                await open_first_store(pilot, app)
                await pilot.press("c")
                await pilot.pause()

        asyncio.run(scenario())
        assert copied == [str(store)]

    def test_copy_path_on_list_copies_selected_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = make_store(tmp_path)
        copied: list[str] = []
        monkeypatch.setattr("pyperclip.copy", copied.append)

        async def scenario() -> None:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                screen.query_one("#stores", DataTable).focus()
                await pilot.pause()
                await pilot.press("c")
                await pilot.pause()

        asyncio.run(scenario())
        assert copied == [str(store)]

    def test_delete_requires_confirmation(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)

        async def scenario() -> tuple[bool, int]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                await open_first_store(pilot, app)
                await pilot.press("d")
                await wait_screen(pilot, app, ConfirmDelete)
                await pilot.press("n")
                await pilot.pause()
                await pilot.pause()
                cancelled = store.exists()
                await pilot.press("d")
                first = await wait_screen(pilot, app, ConfirmDelete)
                await pilot.press("y")
                for _ in range(50):
                    await pilot.pause()
                    if isinstance(app.screen, ConfirmDelete) and app.screen is not first:
                        break
                await pilot.press("y")
                await wait_screen(pilot, app, StoreListScreen)
                return cancelled, app.screen.query_one("#stores", DataTable).row_count

        cancelled, remaining = asyncio.run(scenario())
        assert cancelled
        assert not store.exists()
        assert remaining == 0

    def test_delete_modal_focuses_cancel(self, tmp_path: Path) -> None:
        make_store(tmp_path)

        async def scenario() -> str | None:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                await open_first_store(pilot, app)
                await pilot.press("d")
                modal = await wait_screen(pilot, app, ConfirmDelete)
                await pilot.pause()
                return getattr(modal.app.focused, "id", None)

        assert asyncio.run(scenario()) == "cancel"

    def test_delete_cancelled_at_final_confirmation(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)

        async def scenario() -> bool:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                await open_first_store(pilot, app)
                await pilot.press("d")
                first = await wait_screen(pilot, app, ConfirmDelete)
                await pilot.press("y")
                for _ in range(50):
                    await pilot.pause()
                    if isinstance(app.screen, ConfirmDelete) and app.screen is not first:
                        break
                await pilot.press("n")
                await pilot.pause()
                await pilot.pause()
                return store.exists()

        assert asyncio.run(scenario())

    def test_hint_bar_visible_on_list_and_detail(self, tmp_path: Path) -> None:
        make_store(tmp_path)

        async def scenario() -> tuple[str, str]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                list_hint = str(screen.query_one(HintBar).render())
                store_screen = await open_first_store(pilot, app)
                return list_hint, str(store_screen.query_one(HintBar).render())

        list_hint, detail_hint = asyncio.run(scenario())
        assert "move" in list_hint
        assert "copy path" in detail_hint

    def test_custom_messages_reach_hints_and_dialog(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        messages = replace(
            ENGLISH, hint_copy_path="yank", hint_delete="remove", confirm_delete="Remove? {path}"
        )

        async def scenario() -> tuple[str, str]:
            app = StoreExplorerApp(tmp_path, messages=messages)
            async with app.run_test() as pilot:
                store_screen = await open_first_store(pilot, app)
                hint = str(store_screen.query_one(HintBar).render())
                await pilot.press("d")
                modal = await wait_screen(pilot, app, ConfirmDelete)
                await pilot.pause()
                return hint, str(modal.query_one(Label).render())

        hint, dialog = asyncio.run(scenario())
        assert "yank" in hint and "remove" in hint
        assert "Remove?" in dialog and store.name in dialog

    def test_mouse_up_copies_drag_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_store(tmp_path)
        copied: list[str] = []
        monkeypatch.setattr("pyperclip.copy", copied.append)
        selected = ["clicked"]
        monkeypatch.setattr(Screen, "get_selected_text", lambda self: selected[0])

        async def scenario() -> None:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                await wait_screen(pilot, app, StoreListScreen)
                table = app.screen.query_one("#stores", DataTable)
                # A click at one position (double clicks too) does not copy.
                await pilot.mouse_down(table, offset=(2, 1))
                await pilot.mouse_up(table, offset=(2, 1))
                await pilot.pause()
                # A drag to another position copies.
                selected[0] = "dragged text"
                await pilot.mouse_down(table, offset=(2, 1))
                await pilot.mouse_up(table, offset=(8, 1))
                await pilot.pause()

        asyncio.run(scenario())
        assert copied == ["dragged text"]

    def test_double_click_does_not_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Double clicking a panel (select-all) does not copy."""
        make_real_store(tmp_path)
        copied: list[str] = []

        async def scenario() -> str | None:
            app = StoreExplorerApp(tmp_path)
            monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
            async with app.run_test(size=(100, 70)) as pilot:
                store_screen = await open_first_store(pilot, app)
                for _ in range(50):
                    await pilot.pause()
                widget = store_screen.query_one("#store-features", Static)
                await pilot.double_click(widget, offset=(3, 1))
                await pilot.pause()
                return app.screen.get_selected_text()

        selected = asyncio.run(scenario())
        assert not copied
        assert selected is not None  # the select-all highlight itself stays

    def test_double_click_cell_on_navigating_table_does_not_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_store(tmp_path)
        copied: list[str] = []

        async def scenario() -> None:
            app = StoreExplorerApp(tmp_path)
            monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                table = screen.query_one("#stores", DataTable)
                cell = table._get_cell_region(Coordinate(0, 0))  # pyright: ignore[reportPrivateUsage]
                await pilot.double_click(
                    table, offset=(cell.x + table.gutter.left, cell.y + table.gutter.top)
                )
                await pilot.pause()

        asyncio.run(scenario())
        assert not copied

    def test_table_hover_tracks_without_full_repaint(self, tmp_path: Path) -> None:
        """Crossing cells does not trigger a full repaint through highlight_link_id."""
        make_store(tmp_path)

        async def scenario() -> tuple[str, Coordinate]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                table = screen.query_one("#stores", DataTable)
                gutter = table.gutter
                cell_a = table._get_cell_region(Coordinate(0, 0))  # pyright: ignore[reportPrivateUsage]
                cell_b = table._get_cell_region(Coordinate(0, 1))  # pyright: ignore[reportPrivateUsage]
                await pilot.hover(table, offset=(cell_a.x + gutter.left + 1, cell_a.y + gutter.top))
                await pilot.pause()
                await pilot.hover(table, offset=(cell_b.x + gutter.left + 1, cell_b.y + gutter.top))
                await pilot.pause()
                return table.highlight_link_id, table.hover_coordinate

        assert asyncio.run(scenario()) == ("", Coordinate(0, 1))

    def test_row_hover_repaints_by_row(self, tmp_path: Path) -> None:
        make_store(tmp_path)

        async def scenario() -> tuple[list[int], list[Coordinate]]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                table = screen.query_one("#stores", DataTable)
                rows: list[int] = []
                cells: list[Coordinate] = []
                orig_row = table.refresh_row
                orig_cell = table.refresh_coordinate
                table.refresh_row = lambda i: rows.append(i) or orig_row(i)  # type: ignore[method-assign]
                table.refresh_coordinate = lambda c: cells.append(c) or orig_cell(c)  # type: ignore[method-assign]
                cell = table._get_cell_region(Coordinate(0, 1))  # pyright: ignore[reportPrivateUsage]
                await pilot.hover(
                    table, offset=(cell.x + table.gutter.left + 1, cell.y + table.gutter.top)
                )
                await pilot.pause()
                return rows, cells

        rows, cells = asyncio.run(scenario())
        assert 0 in rows
        assert not cells

    def test_meta_style_at_matches_rendered_segments(self, tmp_path: Path) -> None:
        """The fast hit-test metadata equals the rendered segment metadata everywhere.

        Covers the table fast path and the fallback for other widgets (scroll bar
        @mouse.down actions).
        """
        make_store(tmp_path)

        async def scenario() -> list[tuple[int, int, dict[Any, Any], dict[Any, Any]]]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                region = screen.region
                mismatches = []
                for y in range(region.y, region.bottom):
                    for x in range(region.x, region.right):
                        real = screen._compositor.get_style_at(x, y).meta or {}  # pyright: ignore[reportPrivateUsage]
                        fast = screen.get_style_at(x, y).meta or {}
                        if dict(real) != dict(fast):
                            mismatches.append((x, y, dict(real), dict(fast)))
                return mismatches

        assert asyncio.run(scenario()) == []

    def test_mouse_move_coalesces_queued_events(self, tmp_path: Path) -> None:
        make_store(tmp_path, "feat_a.lmdb")
        make_store(tmp_path, "feat_b.lmdb")

        async def scenario() -> tuple[int, Coordinate]:
            app = StoreExplorerApp(tmp_path)
            handled = 0
            orig = Screen._handle_mouse_move  # pyright: ignore[reportPrivateUsage]

            def counting(self: Screen[Any], event: MouseMove) -> None:
                nonlocal handled
                handled += 1
                orig(self, event)

            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                table = screen.query_one("#stores", DataTable)
                cell = table._get_cell_region(Coordinate(0, 0))  # pyright: ignore[reportPrivateUsage]
                base_x = table.region.x + table.gutter.left + cell.x + 2
                base_y = table.region.y + table.gutter.top + cell.y
                Screen._handle_mouse_move = counting  # pyright: ignore[reportPrivateUsage]
                try:
                    for i in range(20):
                        y = base_y + i % 2
                        app.post_message(
                            MouseMove(
                                table,
                                base_x,
                                y,
                                0,
                                0,
                                0,
                                False,
                                False,
                                False,
                                screen_x=base_x,
                                screen_y=y,
                            )
                        )
                    await pilot.pause()
                    await pilot.pause()
                finally:
                    Screen._handle_mouse_move = orig  # pyright: ignore[reportPrivateUsage]
                return handled, table.hover_coordinate

        handled, hover = asyncio.run(scenario())
        assert handled < 20
        assert hover.row == 1

    def test_row_hover_skips_horizontal_moves(self, tmp_path: Path) -> None:
        make_store(tmp_path, "feat_a.lmdb")
        make_store(tmp_path, "feat_b.lmdb")

        async def scenario() -> tuple[int, int, Coordinate]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                table = screen.query_one("#stores", DataTable)
                cell_a = table._get_cell_region(Coordinate(0, 0))  # pyright: ignore[reportPrivateUsage]
                cell_b = table._get_cell_region(Coordinate(1, 0))  # pyright: ignore[reportPrivateUsage]
                left, top = table.gutter.left, table.gutter.top
                await pilot.hover(table, offset=(cell_a.x + left + 1, cell_a.y + top))
                await pilot.pause()
                calls = 0
                orig = table.refresh_row

                def counting(row: int) -> None:
                    nonlocal calls
                    calls += 1
                    orig(row)

                table.refresh_row = counting  # type: ignore[method-assign]
                try:
                    await pilot.hover(table, offset=(cell_a.x + left + 3, cell_a.y + top))
                    await pilot.pause()
                    same_row = calls
                    await pilot.hover(table, offset=(cell_b.x + left + 1, cell_b.y + top))
                    await pilot.pause()
                    return same_row, calls - same_row, table.hover_coordinate
                finally:
                    table.refresh_row = orig  # type: ignore[method-assign]

        same_row, other_row, hover = asyncio.run(scenario())
        assert same_row == 0
        assert other_row > 0
        assert hover.row == 1

    def test_single_click_selects_double_click_navigates(self, tmp_path: Path) -> None:
        make_store(tmp_path)

        async def scenario() -> tuple[bool, Coordinate, bool]:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                table = screen.query_one("#stores", DataTable)
                cell = table._get_cell_region(Coordinate(0, 1))  # pyright: ignore[reportPrivateUsage]
                offset = (cell.x + table.gutter.left + 1, cell.y + table.gutter.top)
                await pilot.click(table, offset=offset)
                await pilot.pause()
                stays = isinstance(app.screen, StoreListScreen)
                cursor = table.cursor_coordinate
                await pilot.double_click(table, offset=offset)
                detail = await wait_screen(pilot, app, StoreScreen)
                return stays, cursor, isinstance(detail, StoreScreen)

        assert asyncio.run(scenario()) == (True, Coordinate(0, 1), True)

    def test_double_click_cell_copies_value_on_linkless_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class GridScreen(Screen[None]):
            def compose(self) -> ComposeResult:
                yield LeanDataTable(id="grid", cell_copy=True)

            def on_mount(self) -> None:
                table = self.query_one("#grid", DataTable)
                table.add_columns("name")
                table.add_row("alpha", key="alpha")

        class GridApp(DragCopyApp):
            def on_mount(self) -> None:
                self.push_screen(GridScreen())

        copied: list[str] = []

        async def scenario() -> None:
            app = GridApp(title="grid")
            monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, GridScreen)
                table = screen.query_one("#grid", DataTable)
                await wait_region_stable(pilot, table)
                cell = table._get_cell_region(Coordinate(0, 0))  # pyright: ignore[reportPrivateUsage]
                await pilot.double_click(
                    table, offset=(cell.x + table.gutter.left, cell.y + table.gutter.top)
                )
                await pilot.pause()

        asyncio.run(scenario())
        assert copied == ["alpha"]

    def test_store_detail_separates_content_tree(self, tmp_path: Path) -> None:
        make_real_store(tmp_path)
        profile = StoreProfile(
            contents=lambda summary: [
                ("meta", "store metadata", ()),
                ("utterance[i]", "entry", feature_nodes(summary.meta or {})),
            ]
        )

        async def scenario() -> tuple[str, str]:
            app = StoreExplorerApp(tmp_path, profile=profile)
            async with app.run_test() as pilot:
                store_screen = await open_first_store(pilot, app)
                await pilot.pause()
                info = render_static(store_screen.query_one("#store-info", Static))
                tree = render_static(store_screen.query_one("#store-contents", Static))
                return info, tree

        info, tree = asyncio.run(scenario())
        assert "environment" in info and "├" not in info
        assert "store" in tree and "├── meta" in tree and "logmel" in tree

    def test_default_profile_has_no_content_tree(self, tmp_path: Path) -> None:
        make_real_store(tmp_path)

        async def scenario() -> int:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                store_screen = await open_first_store(pilot, app)
                return len(store_screen.query("#store-contents"))

        assert asyncio.run(scenario()) == 0

    def test_store_detail_shows_source(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        (store / "code.log").write_text("def extract():\n    return 42\n", encoding="utf-8")

        async def scenario() -> str:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                store_screen = await open_first_store(pilot, app)
                await pilot.pause()
                return render_static(store_screen.query_one("#store-code", Static))

        source = asyncio.run(scenario())
        assert "def extract():" in source
        assert "return 42" in source

    def test_store_detail_source_missing(self, tmp_path: Path) -> None:
        make_store(tmp_path)

        async def scenario() -> str:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test() as pilot:
                store_screen = await open_first_store(pilot, app)
                await pilot.pause()
                return render_static(store_screen.query_one("#store-code", Static))

        assert "code.log" in asyncio.run(scenario())

    def test_drag_on_table_does_not_select(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_store(tmp_path)
        copied: list[str] = []

        async def scenario() -> str | None:
            app = StoreExplorerApp(tmp_path)
            monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
            async with app.run_test() as pilot:
                screen = await wait_screen(pilot, app, StoreListScreen)
                table = screen.query_one("#stores", DataTable)
                await pilot.mouse_down(table, offset=(0, 1))
                await pilot.pause()
                await pilot.mouse_up(table, offset=(24, 1))
                await pilot.pause()
                return app.screen.get_selected_text()

        selected = asyncio.run(scenario())
        assert not copied
        assert not selected

    def test_drag_select_on_static_panel_copies_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Drag selection copies from rich renderable panels (source, usage)."""
        store = make_store(tmp_path)
        (store / "code.log").write_text("def extract():\n    return 42\n", encoding="utf-8")
        copied: list[str] = []

        async def scenario() -> None:
            app = StoreExplorerApp(tmp_path)
            monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
            async with app.run_test(size=(110, 90)) as pilot:
                store_screen = await open_first_store(pilot, app)
                for _ in range(40):
                    await pilot.pause()
                source = store_screen.query_one("#store-code", Static)
                # Offsets include the border (1) and padding (left 2, top 1). Dragging
                # from the line-number gutter selects without the line number.
                await pilot.mouse_down(source, offset=(4, 2))
                await pilot.mouse_up(source, offset=(21, 2))
                await pilot.pause()
                # The selection disappears right after copying.
                assert not app.screen.selections
                usage = store_screen.query_one("#store-usage", Static)
                await pilot.mouse_down(usage, offset=(7, 2))
                await pilot.mouse_up(usage, offset=(21, 2))
                await pilot.pause()
                assert not app.screen.selections

        asyncio.run(scenario())
        assert "def extract" in copied[0]
        assert copied[1].startswith("from amber")
        # Line numbers are never part of the copy.
        assert not copied[0].lstrip().startswith(("1", "2"))
        assert not copied[1].lstrip().startswith(("1", "2"))

    def test_drag_select_incremental_moves_match_copy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dragging across highlighted cells keeps the selection equal to the copy."""
        store = make_store(tmp_path)
        (store / "code.log").write_text("def extract():\n    return 42\n", encoding="utf-8")
        copied: list[str] = []

        async def scenario() -> None:
            app = StoreExplorerApp(tmp_path)
            monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
            async with app.run_test(size=(110, 90)) as pilot:
                store_screen = await open_first_store(pilot, app)
                for _ in range(40):
                    await pilot.pause()
                source = store_screen.query_one("#store-code", Static)
                # Absolute coordinates of the content area (border 1 + padding 2, 1).
                rx, ry = source.region.x + 3, source.region.y + 2
                screen = app.screen
                # Move one cell at a time; each move crosses the previous highlight, so
                # wrong offset metadata would stall the end point.
                await pilot.mouse_down(source, offset=(8, 2))
                for x in range(6, 15):
                    screen._forward_event(  # pyright: ignore[reportPrivateUsage]
                        MouseMove(
                            source,
                            x=rx + x,
                            y=ry,
                            delta_x=0,
                            delta_y=0,
                            button=1,
                            shift=False,
                            meta=False,
                            ctrl=False,
                            screen_x=rx + x,
                            screen_y=ry,
                        )
                    )
                    await pilot.pause()
                selection_style = screen.get_component_rich_style("screen--selection")
                strip = source.render_line(0)
                highlighted = "".join(
                    seg.text
                    for seg in strip
                    if seg.style and seg.style.bgcolor == selection_style.bgcolor
                )
                assert highlighted == "ef extract"
                await pilot.mouse_up(source, offset=(17, 2))
                await pilot.pause()

        asyncio.run(scenario())
        # The copy equals the highlight (a one-character copy was a regression).
        assert copied == ["ef extract"]

    def test_drag_copy_falls_back_to_osc52(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_store(tmp_path)

        def fail(_text: str) -> None:
            raise RuntimeError("xclip not found")

        monkeypatch.setattr("pyperclip.copy", fail)
        monkeypatch.setattr(Screen, "get_selected_text", lambda self: "dragged text")
        osc52: list[str] = []

        async def scenario() -> None:
            app = StoreExplorerApp(tmp_path)
            monkeypatch.setattr(app, "copy_to_clipboard", osc52.append)
            async with app.run_test() as pilot:
                await wait_screen(pilot, app, StoreListScreen)
                table = app.screen.query_one("#stores", DataTable)
                await pilot.mouse_down(table, offset=(2, 1))
                await pilot.mouse_up(table, offset=(8, 1))
                await pilot.pause()

        asyncio.run(scenario())
        assert osc52 == ["dragged text"]

    def test_selection_repaint_clips_offscreen_rows(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        (store / "code.log").write_text(
            "".join(f"line_{i:03d} = {i}\n" for i in range(80)), encoding="utf-8"
        )

        async def scenario() -> None:
            app = StoreExplorerApp(tmp_path)
            # The source panel starts on screen and extends past the bottom.
            async with app.run_test(size=(80, 50)) as pilot:
                store_screen = await open_first_store(pilot, app)
                for _ in range(40):
                    await pilot.pause()
                source = store_screen.query_one("#store-code", Static)
                compositor = app.screen._compositor  # pyright: ignore[reportPrivateUsage]
                assert source.region.y < compositor.size.height
                assert source.region.y + source.region.height > compositor.size.height
                compositor._dirty_regions.clear()  # pyright: ignore[reportPrivateUsage]
                # A select-all queues regions of off-screen rows, clipped to the screen.
                app.screen.selections = {source: Selection(None, None)}
                assert compositor._dirty_regions  # pyright: ignore[reportPrivateUsage]
                assert all(
                    region.intersection(compositor.size.region) == region
                    for region in compositor._dirty_regions  # pyright: ignore[reportPrivateUsage]
                )
                # Rendering the partial update never indexes past the chops.
                update = compositor.render_update()
                if isinstance(update, CompositorUpdate):
                    update.render_segments(app.console)

        asyncio.run(scenario())

    def test_selection_repaint_marks_content_rows(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)
        (store / "code.log").write_text("def extract():\n    return 42\n", encoding="utf-8")

        async def scenario() -> None:
            app = StoreExplorerApp(tmp_path)
            async with app.run_test(size=(110, 90)) as pilot:
                store_screen = await open_first_store(pilot, app)
                for _ in range(40):
                    await pilot.pause()
                source = store_screen.query_one("#store-code", Static)
                compositor = app.screen._compositor  # pyright: ignore[reportPrivateUsage]
                compositor._dirty_regions.clear()  # pyright: ignore[reportPrivateUsage]
                source._styles_cache._dirty_lines.clear()  # pyright: ignore[reportPrivateUsage]
                top = source.gutter.top
                # Content rows 0-1 dirty widget rows top..top+1 (border and padding).
                source.selection_updated(Selection(Offset(0, 0), Offset(5, 1)))
                assert {top, top + 1} <= source._styles_cache._dirty_lines  # pyright: ignore[reportPrivateUsage]
                region = source.region
                for y in range(2):
                    assert (
                        Region(region.x, region.y + top + y, region.width, 1)
                        in compositor._dirty_regions  # pyright: ignore[reportPrivateUsage]
                    )
                # The border row is not dirtied.
                assert (
                    Region(region.x, region.y, region.width, 1) not in compositor._dirty_regions  # pyright: ignore[reportPrivateUsage]
                )

        asyncio.run(scenario())
