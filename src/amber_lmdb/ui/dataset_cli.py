"""Ready-made command line for a dataset package that writes feature stores.

A dataset package describes itself with one :class:`DatasetCLI` and exposes its
``main``. The commands are:

``doctor [--all] [--root R] [--resources R] [--json]``
    Check the dataset root, the resources root, the dataset index (``inspect_dataset``)
    and every feature store; ``--all`` adds ``deep_check_dataset``, the metadata of
    every store and ``check_store``. Exit code 1 when a check fails.
``dataset [--root R] [--resources R]``
    Browse the dataset (``load_dataset`` then ``browse``) in an interactive explorer.
``paths [--root R] [--resources R] [--json]``
    Show how the dataset and resources roots resolve.
``features [NAME] [--list] [--detail] [--output-dir D] [--no-entries] [--json]``
    Open the store explorer on a terminal, or print the store table or details.

``--version`` prints the package and amber-lmdb versions. Errors go to stderr as
``error: ...`` with a hint; Ctrl+C exits with 130. Formatted output follows the
``NO_COLOR`` and ``FORCE_COLOR`` conventions; ``--json`` output is stable for scripts.

Rich and Textual load only when a command prints or opens something, so parsing
arguments and ``--version`` stay fast. The resources root holds ``feature_root`` and
``store_directory`` (both relative).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._common import ACCENT, cap_list, is_interactive
from .messages import ENGLISH, Messages
from .stores import StoreProfile, StoreSummary, describe_store, describe_stores, find_stores

if TYPE_CHECKING:
    from rich.console import Console
    from rich.progress import Progress

    from .browse import Listing
    from .doctor import Section

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130


def _json_dump(document: Any) -> None:
    from ..cli import _json_value  # pyright: ignore[reportPrivateUsage]

    print(json.dumps(_json_value(document), ensure_ascii=False, indent=2, allow_nan=False))


def _summary_document(summary: StoreSummary, *, detail: bool) -> dict[str, Any]:
    from .stores import feature_rows

    document: dict[str, Any] = {
        "name": summary.name,
        "path": str(summary.path),
        "type": summary.typename,
        "hash": None if summary.digest == "-" else summary.digest,
        "version": None if summary.version == "-" else summary.version,
        "versions": summary.n_versions,
        "entries": summary.entries,
        "used_bytes": summary.used_bytes,
        "size_bytes": summary.size_bytes,
        "modified": datetime.fromtimestamp(summary.modified).astimezone().isoformat(),
    }
    if detail:
        columns = ("name", "type", "dtype", "kind", "shape", "entries")
        document["meta"] = summary.meta
        document["meta_error"] = summary.meta_error
        document["features"] = [
            dict(zip(columns, row, strict=True)) for row in feature_rows(summary)
        ]
    return document


def _wants_help(argv: Sequence[str]) -> bool:
    return any(argument in ("-h", "--help") for argument in argv)


def _package_version(distribution: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


@dataclass(frozen=True)
class DatasetCLI:
    """Configuration and hooks of a dataset command line.

    Required: ``name`` (display name), ``prog``, the two environment variables and
    ``resolve_root``, which maps ``--root`` (or None for the environment) to the
    dataset root and raises with a readable message when it cannot.

    Hooks receive the object ``load_dataset(root, resources)`` returns:
    ``browse(dataset)`` builds the explorer, ``inspect_dataset(dataset, section)`` adds
    doctor checks, ``deep_check_dataset(dataset, section, progress)`` adds ``--all``
    checks and ``check_store(store, meta, dataset_or_none, section)`` compares a store
    with the dataset. ``extend_parser(subparsers)`` adds commands; give each a
    ``func`` default that takes the parsed arguments and returns an exit code.
    ``distribution`` names the package whose version ``--version`` prints (default
    ``prog``).
    """

    name: str
    prog: str
    dataset_env: str
    resources_env: str
    resolve_root: Callable[[str | None], Path]
    required_entries: Sequence[str] = ()
    feature_root: Sequence[str] = ()
    store_directory: Sequence[str] = ()
    profile: StoreProfile = field(default_factory=StoreProfile)
    load_dataset: Callable[[Path | str | None, str | None], Any] | None = None
    browse: Callable[[Any], Listing[Any]] | None = None
    inspect_dataset: Callable[[Any, Section], None] | None = None
    deep_check_dataset: Callable[[Any, Section, Progress], None] | None = None
    check_store: Callable[[Path, dict[str, Any], Any | None, Section], None] | None = None
    store_meta_keys: Sequence[str] = ("names", "type_map")
    store_meta_detail: Mapping[str, str] = field(default_factory=dict)
    index_check: str = "dataset index"
    required_display_limit: int = 8
    extend_parser: Callable[[Any], None] | None = None
    theme_name: str | None = None
    distribution: str | None = None
    accent: str = ACCENT
    messages: Messages = ENGLISH

    # Paths ---------------------------------------------------------------------

    def resources(self, explicit: str | None) -> str | None:
        """``--resources`` or the environment variable; None when unset or blank."""
        value = explicit if explicit is not None else os.environ.get(self.resources_env)
        return value if value is not None and value.strip() else None

    def store_dir(self, resources: str) -> Path:
        return Path(resources).expanduser().resolve(strict=False).joinpath(*self.store_directory)

    def feature_root_dir(self, resources: str) -> Path:
        return Path(resources).expanduser().resolve(strict=False).joinpath(*self.feature_root)

    def resolve_store_dir(self, output_dir: str | None) -> Path:
        """``--output-dir`` as given, or the store directory under the resources root."""
        if output_dir is not None:
            if not output_dir.strip():
                raise ValueError("--output-dir must not be empty")
            return Path(output_dir).expanduser().resolve(strict=False)
        resources = self.resources(None)
        if resources is None:
            raise ValueError(f"{self.resources_env} is not set")
        return self.store_dir(resources)

    def open_dataset(self, root: Path | str | None, resources: str | None, console: Console) -> Any:
        """Run ``load_dataset`` under a spinner with its warnings silenced."""
        assert self.load_dataset is not None
        with (
            console.status(
                f"[dim]{self.messages.indexing_dataset}[/]",
                spinner="dots",
                spinner_style=self.accent,
            )
            if console.is_terminal
            else nullcontext(),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            return self.load_dataset(root, resources)

    # Output helpers ------------------------------------------------------------

    def _section(self, console: Console, title: str, body: Any, note: str | None = None) -> None:
        from .render import print_section

        print_section(console, title, body, note=note, accent=self.accent)

    def error(self, message: str, hint: str | None = None) -> int:
        """Print ``error: message`` (and a hint) on stderr; return exit code 1."""
        from .render import new_console, print_error

        print_error(
            new_console(stderr=True),
            message,
            hint,
            prefix=self.messages.error_prefix,
            hint_prefix=self.messages.hint_prefix,
        )
        return EXIT_FINDINGS

    # Commands ------------------------------------------------------------------

    def paths_document(self, root: str | None, resources_arg: str | None) -> dict[str, Any]:
        """Resolution of both roots as a JSON-ready document with an ``ok`` flag."""
        dataset: dict[str, Any] = {
            "configured": root if root is not None else os.environ.get(self.dataset_env),
            "source": "--root" if root is not None else f"env {self.dataset_env}",
            "resolved": None,
            "error": None,
            "missing": [],
        }
        try:
            resolved = self.resolve_root(root)
        except Exception as exc:
            dataset["error"] = str(exc)
        else:
            dataset["resolved"] = str(resolved)
            dataset["missing"] = [
                entry for entry in self.required_entries if not (resolved / entry).exists()
            ]
        resources = self.resources(resources_arg)
        resources_doc: dict[str, Any] = {
            "configured": resources,
            "source": "--resources" if resources_arg is not None else f"env {self.resources_env}",
            "feature_root": None,
            "feature_stores": None,
        }
        if resources is not None:
            for key, path in (
                ("feature_root", self.feature_root_dir(resources)),
                ("feature_stores", self.store_dir(resources)),
            ):
                resources_doc[key] = {"path": str(path), "exists": path.is_dir()}
        ok = dataset["error"] is None and not dataset["missing"]
        return {"ok": ok, "dataset_root": dataset, "resources_root": resources_doc}

    def cmd_paths(self, args: argparse.Namespace) -> int:
        document = self.paths_document(args.root, args.resources)
        if args.json:
            _json_dump(document)
            return EXIT_OK if document["ok"] else EXIT_FINDINGS

        from rich.text import Text

        from .render import label_grid, new_console, status_text

        messages = self.messages
        console = new_console()

        def line(grid: Any, status: Any, label: str, detail: str) -> None:
            style = "" if status == "OK" else "yellow" if status == "WARN" else "bold red"
            grid.add_row(
                status_text(console, status), Text(label, style=style), Text(detail, style="dim")
            )

        dataset = document["dataset_root"]
        grid = label_grid()
        grid.add_column(overflow="fold")
        grid.add_row(
            Text(" "),
            Text("configured", style="dim"),
            Text(dataset["configured"] or messages.unset, style="dim"),
        )
        if dataset["error"] is not None:
            line(grid, "FAIL", "resolved", dataset["error"])
        else:
            line(grid, "OK", "resolved", dataset["resolved"])
            if dataset["missing"]:
                text = messages.missing_entries.format(entries=", ".join(dataset["missing"]))
                line(grid, "FAIL", "required", text)
            else:
                text = cap_list(self.required_entries, self.required_display_limit)
                line(grid, "OK", "required", text)
        console.print()
        self._section(console, messages.dataset_root_title, grid, note=dataset["source"])

        resources = document["resources_root"]
        grid = label_grid()
        grid.add_column(overflow="fold")
        grid.add_row(
            Text(" "),
            Text("configured", style="dim"),
            Text(resources["configured"] or messages.unset, style="dim"),
        )
        if resources["configured"] is None:
            line(grid, "WARN", "feature paths", messages.resources_unset_paths)
        else:
            for label, key in (
                ("feature_root", "feature_root"),
                ("feature stores", "feature_stores"),
            ):
                entry = resources[key]
                if entry["exists"]:
                    line(grid, "OK", label, entry["path"])
                else:
                    line(grid, "WARN", label, f"{entry['path']}{messages.missing_suffix}")
        self._section(console, messages.resources_root_title, grid, note=resources["source"])
        return EXIT_OK if document["ok"] else EXIT_FINDINGS

    def cmd_dataset(self, args: argparse.Namespace) -> int:
        messages = self.messages
        if not is_interactive():
            return self.error(
                messages.interactive_only.format(command="dataset"), messages.hint_terminal
            )
        if self.load_dataset is None or self.browse is None:
            return self.error(
                messages.dataset_open_failed.format(error="no explorer is configured")
            )
        from rich.console import Console

        try:
            dataset = self.open_dataset(args.root, self.resources(args.resources), Console())
        except Exception as exc:
            return self.error(
                messages.dataset_open_failed.format(error=f"{type(exc).__name__}: {exc}"),
                messages.hint_root.format(env=self.dataset_env),
            )
        from .tui.browser import BrowserApp

        BrowserApp(
            self.browse(dataset),
            title=messages.explorer_title.format(name=self.name),
            theme_name=self.theme_name or self.name.lower(),
            accent=self.accent,
            messages=messages,
        ).run()
        return EXIT_OK

    def run_features_explorer(self, store_dir: Path) -> int:
        from .tui.stores import StoreExplorerApp

        StoreExplorerApp(
            store_dir,
            profile=self.profile,
            title=self.messages.store_explorer_title.format(name=self.name),
            theme_name=self.theme_name or self.name.lower(),
            accent=self.accent,
            messages=self.messages,
        ).run()
        return EXIT_OK

    def cmd_features(self, args: argparse.Namespace) -> int:
        messages = self.messages
        try:
            store_dir = self.resolve_store_dir(args.output_dir)
        except ValueError as exc:
            return self.error(
                messages.store_dir_unresolved.format(error=exc),
                messages.hint_resources.format(env=self.resources_env),
            )
        stores = find_stores(store_dir, args.name)
        interactive = not (args.list or args.detail or args.json or args.name is not None)
        if interactive and stores and is_interactive():
            return self.run_features_explorer(store_dir)
        if not stores and args.name is not None:
            return self.error(messages.no_matching_store.format(name=args.name))
        if args.json:
            if args.detail:
                summaries = [
                    describe_store(store, with_entries=not args.no_entries, with_meta=True)
                    for store in stores
                ]
            else:
                summaries = describe_stores(stores, with_entries=not args.no_entries)
            _json_dump([_summary_document(summary, detail=args.detail) for summary in summaries])
            return EXIT_OK

        from rich.padding import Padding
        from rich.text import Text

        from .render import new_console, supports_unicode
        from .views import feature_section, store_grid, store_table

        console = new_console()
        title = messages.feature_stores_title
        note = str(store_dir)
        console.print()
        if not store_dir.is_dir():
            self._section(console, title, Text(messages.store_dir_missing, style="dim"), note)
            return EXIT_OK
        if not stores:
            self._section(console, title, Text(messages.no_stores, style="dim"), note)
            return EXIT_OK
        if args.detail:
            for store in stores:
                with console.status(
                    f"[dim]{messages.reading_store.format(name=store.name)}[/]",
                    spinner="dots",
                    spinner_style=self.accent,
                ):
                    summary = describe_store(
                        store, with_entries=not args.no_entries, with_meta=True, messages=messages
                    )
                grid = store_grid(
                    summary, self.profile, messages=messages, accent=self.accent, expand=False
                )
                self._section(console, summary.name, grid)
                if summary.meta is not None:
                    section = feature_section(summary, self.profile, console, messages=messages)
                    console.print(Padding(section, (0, 0, 1, 2)))
            return EXIT_OK
        summaries = describe_stores(stores, with_entries=not args.no_entries)
        table = store_table(summaries, safe_box=not supports_unicode(console), messages=messages)
        self._section(console, title, table, note)
        return EXIT_OK

    def run_doctor(
        self,
        *,
        root: str | None,
        resources: str | None,
        deep: bool,
        console: Console | None = None,
        as_json: bool = False,
    ) -> int:
        """Run every check and print the report (or JSON); return the exit code."""
        from .doctor import (
            Section,
            check_store_layout,
            check_store_meta,
            doctor_progress,
            render_report,
            report_document,
        )
        from .render import new_console

        started = time.perf_counter()
        messages = self.messages
        if as_json:
            from rich.console import Console as RichConsole

            # A console that is not a terminal shows no spinner or progress bar.
            console = RichConsole(file=io.StringIO())
        console = console or new_console()
        paths = Section(messages.paths_section)
        dataset_section = Section(messages.dataset_section)
        store_section = Section(messages.stores_section)

        resolved: Path | None = None
        try:
            resolved = self.resolve_root(root)
        except Exception as exc:
            paths.add("dataset root", "FAIL", str(exc))
        else:
            paths.add("dataset root", "OK", str(resolved))
            missing = [entry for entry in self.required_entries if not (resolved / entry).exists()]
            if missing:
                paths.add("required structure", "FAIL", f"missing: {cap_list(missing)}")
            else:
                paths.add("required structure", "OK", cap_list(self.required_entries))

        store_dir: Path | None = None
        if resources is None or not resources.strip():
            paths.add(
                "resources root", "WARN", messages.resources_unset.format(env=self.resources_env)
            )
        else:
            feature_root = self.feature_root_dir(resources)
            store_dir = self.store_dir(resources)
            for label, path in (("feature_root", feature_root), ("feature stores", store_dir)):
                exists = path.is_dir()
                paths.add(
                    label,
                    "OK" if exists else "WARN",
                    f"{path}{'' if exists else messages.missing_suffix}",
                )

        dataset: Any = None
        if self.load_dataset is not None:
            if resolved is None:
                dataset_section.add(self.index_check, "SKIP", messages.root_unresolved)
            else:
                try:
                    dataset = self.open_dataset(resolved, resources, console)
                except Exception as exc:
                    dataset_section.add(self.index_check, "FAIL", f"{type(exc).__name__}: {exc}")
                else:
                    if self.inspect_dataset is not None:
                        self.inspect_dataset(dataset, dataset_section)

        stores = find_stores(store_dir) if store_dir is not None else []
        with doctor_progress(console, accent=self.accent) as progress:
            if deep and self.deep_check_dataset is not None:
                if dataset is None:
                    dataset_section.add("deep check", "SKIP", messages.index_unavailable)
                else:
                    self.deep_check_dataset(dataset, dataset_section, progress)
            if store_dir is None:
                store_section.add("feature stores", "SKIP", messages.resources_root_unset)
            elif not stores:
                store_section.add("feature stores", "WARN", messages.no_stores)
            else:
                for store in stores:
                    check_store_layout(store, store_section, messages=messages)
                if deep:
                    task = progress.add_task(messages.store_checks_progress, total=len(stores))
                    for store in stores:
                        meta = check_store_meta(
                            store,
                            store_section,
                            required_keys=self.store_meta_keys,
                            describe=self.store_meta_detail,
                        )
                        if meta is not None and self.check_store is not None:
                            self.check_store(store, meta, dataset, store_section)
                        progress.advance(task)

        sections = [paths, dataset_section, store_section]
        if not dataset_section.checks:
            sections.remove(dataset_section)
        elapsed = time.perf_counter() - started
        if as_json:
            document = report_document(sections, elapsed=elapsed)
            _json_dump(document)
            return EXIT_OK if document["ok"] else EXIT_FINDINGS
        ok = render_report(
            console, sections, messages=messages, accent=self.accent, elapsed=elapsed
        )
        return EXIT_OK if ok else EXIT_FINDINGS

    def cmd_doctor(self, args: argparse.Namespace) -> int:
        return self.run_doctor(
            root=args.root,
            resources=self.resources(args.resources),
            deep=bool(args.all),
            as_json=bool(args.json),
        )

    # Parser --------------------------------------------------------------------

    def _formatter(self, rich_help: bool) -> type[argparse.HelpFormatter]:
        if not rich_help:
            return argparse.HelpFormatter
        from rich_argparse import RichHelpFormatter

        accent = self.accent
        messages = self.messages

        class Formatter(RichHelpFormatter):
            styles = {  # noqa: RUF012
                **RichHelpFormatter.styles,
                "argparse.args": accent,
                "argparse.groups": f"bold {accent}",
                "argparse.metavar": "dim",
                "argparse.prog": "bold",
                "argparse.help": "default",
                "argparse.syntax": "bold",
            }
            usage_markup = True
            group_name_formatter = str

            def add_usage(self, usage: Any, actions: Any, groups: Any, prefix: Any = None) -> None:
                # argparse passes prefix="" to build subcommand names; keep that untouched.
                if prefix is None:
                    prefix = f"{messages.usage_heading}: "
                super().add_usage(usage, actions, groups, prefix=prefix)

        return Formatter

    def version_text(self) -> str:
        """``<prog> <version> (amber-lmdb <version>)``."""
        package = _package_version(self.distribution or self.prog)
        return f"{self.prog} {package} (amber-lmdb {_package_version('amber-lmdb')})"

    def build_parser(self, *, rich_help: bool = False) -> argparse.ArgumentParser:
        """Build the parser; ``rich_help`` formats help with Rich (loaded only then)."""
        messages = self.messages
        formatter = self._formatter(rich_help)

        def localize(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
            parser._positionals.title = messages.arguments_heading  # pyright: ignore[reportPrivateUsage]
            parser._optionals.title = messages.options_heading  # pyright: ignore[reportPrivateUsage]
            for action in parser._actions:  # pyright: ignore[reportPrivateUsage]
                if isinstance(action, argparse._HelpAction):  # pyright: ignore[reportPrivateUsage]
                    action.help = messages.help_help
            return parser

        parser = localize(
            argparse.ArgumentParser(
                prog=self.prog,
                description=messages.description.format(name=self.name),
                formatter_class=formatter,
            )
        )
        parser.add_argument(
            "--version", action="version", version=self.version_text(), help=messages.version_help
        )
        commands = parser.add_subparsers(
            dest="command", required=True, metavar="COMMAND", title=messages.commands_heading
        )

        def add_command(name: str, help_text: str) -> argparse.ArgumentParser:
            return localize(
                commands.add_parser(
                    name, help=help_text, description=help_text, formatter_class=formatter
                )
            )

        def add_roots(command: argparse.ArgumentParser) -> None:
            command.add_argument("--root", default=None, metavar="DIR", help=messages.root_help)
            command.add_argument(
                "--resources", default=None, metavar="DIR", help=messages.resources_help
            )

        def add_json(command: argparse.ArgumentParser) -> None:
            command.add_argument("--json", action="store_true", help=messages.json_help)

        doctor = add_command("doctor", messages.doctor_help)
        doctor.add_argument("--all", action="store_true", help=messages.deep_help)
        add_roots(doctor)
        add_json(doctor)
        doctor.set_defaults(func=self.cmd_doctor)

        dataset = add_command("dataset", messages.dataset_help)
        add_roots(dataset)
        dataset.set_defaults(func=self.cmd_dataset)

        paths = add_command(
            "paths",
            messages.paths_help.format(
                dataset_env=self.dataset_env, resources_env=self.resources_env
            ),
        )
        add_roots(paths)
        add_json(paths)
        paths.set_defaults(func=self.cmd_paths)

        features = add_command("features", messages.features_help)
        features.add_argument(
            "name", nargs="?", default=None, metavar="NAME", help=messages.name_filter_help
        )
        features.add_argument("--list", action="store_true", help=messages.list_help)
        features.add_argument("--detail", action="store_true", help=messages.detail_help)
        features.add_argument(
            "--output-dir", default=None, metavar="DIR", help=messages.output_dir_help
        )
        features.add_argument("--no-entries", action="store_true", help=messages.no_entries_help)
        add_json(features)
        features.set_defaults(func=self.cmd_features)

        if self.extend_parser is not None:
            self.extend_parser(commands)
        return parser

    def main(self, argv: Sequence[str] | None = None) -> int:
        """Parse ``argv`` and run the command; return its exit code.

        Exit codes: 0 success, 1 a check failed or the command could not run, 2 invalid
        usage, 130 interrupted.
        """
        arguments = list(sys.argv[1:] if argv is None else argv)
        parser = self.build_parser(rich_help=_wants_help(arguments) or not arguments)
        if not arguments:
            parser.print_help()
            return EXIT_USAGE
        try:
            args = parser.parse_args(arguments)
            return int(args.func(args))
        except KeyboardInterrupt:
            print(self.messages.interrupted, file=sys.stderr)
            return EXIT_INTERRUPTED
        except BrokenPipeError:
            # The reader went away (e.g. `| head`); silence the flush at exit.
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            return EXIT_OK


__all__ = ["DatasetCLI"]
