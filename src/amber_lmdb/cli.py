"""Command-line inspection, validation, generation management and cache cleanup.

Every command prints one JSON document on stdout. Exit codes: 0 success; 1 the
command ran and found something (problems, differences, refused deletions, busy
cache entries); 2 invalid usage; 3 the command could not complete (missing or
unreadable store, invalid layout, I/O failure), with ``{"error": ...}`` on stdout
and the message on stderr; 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import lmdb
import numpy as np

from . import __version__
from .errors import AmberLMDBError
from .fingerprints import diff_stores, fingerprint_store
from .generations import delete_generations, inspect_generations, prune_generations, set_current
from .indexes import DerivedIndexCache, _namespaces
from .inspection import StoreInfo, inspect_store
from .keys import decode_key
from .lmdb import LMDBReader
from .physical import get_backend_info, verify_physical
from .progress import Progress
from .scanning import ValidationIssue, array_statistics, verify_store

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2
EXIT_ERROR = 3


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"float": repr(value)}
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, np.ndarray):
        return {"type": "numpy.ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _json_value(item) for key, item in value.items()}
        return {"mapping": [[_json_value(key), _json_value(item)] for key, item in value.items()]}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, complex):
        return {"real": _json_value(value.real), "imag": _json_value(value.imag)}
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return {"type": "torch.Tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    return str(value)


def _print(document: Any) -> None:
    print(json.dumps(_json_value(document), ensure_ascii=True, allow_nan=False, indent=2))


def _key(location: bytes | None) -> dict[str, Any] | None:
    """Decoded ``{"key", "index"}`` of an encoded location, or None if not decodable."""
    if location is None:
        return None
    try:
        key, index = decode_key(location)
    except ValueError:
        return None
    return {"key": key, "index": index}


def _report(info: StoreInfo) -> dict[str, Any]:
    # asdict would deep-copy metadata, including potentially large metadata arrays.
    return {
        "snapshot": asdict(info.snapshot),
        "entries": info.entries,
        "data_entries": info.data_entries,
        "used_bytes": info.used_bytes,
        "page_size": info.page_size,
        "lib_version": info.lib_version,
        "meta": info.meta,
        "meta_error": info.meta_error,
        "samples": [asdict(entry) for entry in info.samples],
        "next_after": None if info.next_after is None else info.next_after.hex(),
    }


def _issue(issue: ValidationIssue) -> dict[str, Any]:
    return {
        "location": issue.location,
        "key": issue.key,
        "index": issue.index,
        "category": issue.category,
        "message": issue.message,
    }


def _named(name: str) -> Callable[[Callable[[str], Any]], Callable[[str], Any]]:
    """Name an argument type for argparse messages ("invalid <name> value: ...")."""

    def decorate(function: Callable[[str], Any]) -> Callable[[str], Any]:
        function.__name__ = name
        return function

    return decorate


@_named("integer from 0 to 1024")
def _entry_count(value: str) -> int:
    number = int(value)
    if not 0 <= number <= 1024:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 1024")
    return number


@_named("hexadecimal token")
def _after_token(value: str) -> bytes:
    return bytes.fromhex(value)


@_named("positive integer")
def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


@_named("nonnegative integer")
def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


_DURATION_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_SIZE_UNITS = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}


@_named("duration")
def _duration(value: str) -> float:
    unit = _DURATION_UNITS.get(value[-1:].lower())
    try:
        number = float(value[:-1] if unit else value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("duration must be seconds or end in s, m, h or d") from exc
    seconds = number * (unit or 1)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("duration must be finite and nonnegative")
    return seconds


@_named("size")
def _size(value: str) -> int:
    unit = _SIZE_UNITS.get(value[-1:].lower())
    try:
        number = float(value[:-1] if unit else value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must be bytes or end in K, M, G or T") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("size must be finite and nonnegative")
    return int(number * (unit or 1))


def _progress_printer(enabled: bool) -> Any:
    """Return a callback writing progress to stderr: one updating line on a terminal."""
    if not enabled:
        return None
    terminal = sys.stderr.isatty()

    def show(report: Progress) -> None:
        total = "?" if report.entries_total is None else str(report.entries_total)
        line = (
            f"{report.operation}: {report.entries_done}/{total} entries, "
            f"{report.bytes_done} bytes, {report.elapsed_seconds:.1f} s"
        )
        if terminal:
            end = "\n" if report.finished else ""
            sys.stderr.write(f"\r{line}{end}")
        else:
            sys.stderr.write(line + "\n")
        sys.stderr.flush()

    return show


def _add_reading(parser: argparse.ArgumentParser, *, generation: bool = True) -> None:
    if generation:
        parser.add_argument(
            "--generation",
            default=None,
            metavar="NAME",
            help="Read this generation (v-<32 hex>) instead of the one CURRENT selects.",
        )
    parser.add_argument(
        "--read-only-media",
        action="store_true",
        help=(
            "Take no locks, for stores on media nothing can change (read-only mounts, "
            "snapshots, archives); unsafe while a writer or cleanup may run."
        ),
    )


def _add_progress(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--progress", action="store_true", help="Report progress on stderr.")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; see the module docstring for exit codes."""
    parser = argparse.ArgumentParser(
        prog="amber-lmdb",
        description="Inspect, validate and manage amber-lmdb stores. Output is JSON.",
        epilog=(
            "Exit codes: 0 ok; 1 problems, differences or refusals found; 2 invalid "
            "usage; 3 the command could not complete; 130 interrupted."
        ),
    )
    parser.add_argument("--version", action="version", version=f"amber-lmdb {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    inspect = commands.add_parser(
        "inspect", help="Show generation statistics and a bounded sample of keys and headers."
    )
    doctor = commands.add_parser(
        "doctor",
        help=(
            "Check a store: a quick sample by default, every entry with --full, native "
            "pages of a private copy with --physical."
        ),
    )
    for subparser in (inspect, doctor):
        subparser.add_argument("path", type=Path, help="Dataset root.")
        subparser.add_argument(
            "--entries",
            type=_entry_count,
            default=20,
            help="Entries to sample, 0 to 1024 (default 20).",
        )
        subparser.add_argument(
            "--after",
            type=_after_token,
            default=None,
            metavar="TOKEN",
            help="Continue the sample after this next_after token.",
        )
        subparser.add_argument(
            "--meta", action="store_true", help="Decode the metadata dictionary entry."
        )
        subparser.add_argument(
            "--meta-key",
            default="meta",
            metavar="KEY",
            help="Key of the metadata dictionary for --meta (default 'meta').",
        )
        subparser.add_argument(
            "--max-copy-bytes",
            type=_positive_int,
            default=64 * 1024**2,
            help="Bytes the sample may copy in total (default 64 MiB).",
        )
        _add_reading(subparser)
    modes = doctor.add_mutually_exclusive_group()
    modes.add_argument(
        "--full",
        action="store_true",
        help="Visit every entry, verify checksums and decode values within the limits.",
    )
    modes.add_argument(
        "--physical",
        action="store_true",
        help="Check native LMDB pages of a private, immutable copy without opening it.",
    )
    doctor.add_argument(
        "--max-issues",
        type=_nonnegative_int,
        default=100,
        help="Issue details to list with --full or --physical; counts stay complete.",
    )
    doctor.add_argument(
        "--batch-size", type=_positive_int, default=512, help="Entries per page, 1 to 1024."
    )
    doctor.add_argument(
        "--max-value-bytes",
        type=_positive_int,
        default=64 * 1024**2,
        help="Largest value --full verifies; larger ones are reported unverified.",
    )
    doctor.add_argument(
        "--max-batch-bytes",
        type=_positive_int,
        default=None,
        help="Bytes one page may copy (default: enough for --max-value-bytes).",
    )
    doctor.add_argument("--progress", action="store_true", help="Report --full progress on stderr.")

    statistics = commands.add_parser(
        "stats", help="Group array headers by writer tag, dtype and rank."
    )
    statistics.add_argument("path", type=Path, help="Dataset root.")
    statistics.add_argument(
        "--batch-size", type=_positive_int, default=512, help="Entries per page, 1 to 1024."
    )
    statistics.add_argument(
        "--max-groups", type=_positive_int, default=1024, help="Fail beyond this many groups."
    )
    _add_reading(statistics)
    _add_progress(statistics)

    fingerprint = commands.add_parser(
        "fingerprint", help="Digest every key and integrity envelope without reading bodies."
    )
    fingerprint.add_argument("path", type=Path, help="Dataset root.")
    _add_reading(fingerprint)
    _add_progress(fingerprint)

    difference = commands.add_parser(
        "diff", help="List entries added, removed or changed between two generations."
    )
    difference.add_argument("before_path", metavar="BEFORE", type=Path, help="Dataset root.")
    difference.add_argument("after_path", metavar="AFTER", type=Path, help="Dataset root.")
    difference.add_argument(
        "--before-generation", default=None, metavar="NAME", help="Generation of BEFORE."
    )
    difference.add_argument(
        "--after-generation", default=None, metavar="NAME", help="Generation of AFTER."
    )
    difference.add_argument(
        "--max-listed",
        type=_nonnegative_int,
        default=100,
        help="Keys to list per category; counts stay complete (default 100).",
    )
    difference.add_argument(
        "--compare-bodies",
        action="store_true",
        help="Read bodies whose envelopes use different digest algorithms.",
    )
    _add_reading(difference, generation=False)
    _add_progress(difference)

    manifest = commands.add_parser(
        "manifest", help="Show the manifest the writer recorded, or null."
    )
    manifest.add_argument("path", type=Path, help="Dataset root.")
    _add_reading(manifest)

    commands.add_parser("backend", help="Show py-lmdb and available engine versions.")

    generations = commands.add_parser(
        "generations",
        help="Inventory generations, CURRENT, history, trash and unfinished staging.",
    )
    generations.add_argument("path", type=Path, help="Dataset root.")

    selection = commands.add_parser(
        "set-current", help="Point CURRENT at an existing generation (roll back or restore)."
    )
    selection.add_argument("path", type=Path, help="Dataset root.")
    selection.add_argument("name", help="Generation name (v-<32 hex>).")
    selection.add_argument(
        "--verify", action="store_true", help="Verify the generation fully first."
    )

    cleanup = commands.add_parser(
        "delete-generations",
        help="Preview (or --apply) deletion of named or all but the newest old generations.",
    )
    cleanup.add_argument("path", type=Path, help="Dataset root.")
    cleanup.add_argument("names", nargs="*", metavar="NAME", help="Generations to delete.")
    cleanup.add_argument(
        "--keep-last",
        type=_nonnegative_int,
        default=None,
        metavar="N",
        help="Instead of names: delete all but the N newest non-current generations.",
    )
    cleanup.add_argument("--apply", action="store_true", help="Actually delete.")
    cleanup.add_argument(
        "--same-host",
        action="store_true",
        help="Affirm every access cooperates on this host; never for NFS or cross-host use.",
    )

    listing = commands.add_parser(
        "list", help="Show statistics of every store in a directory without reading values."
    )
    listing.add_argument("directory", type=Path, help="Directory holding dataset roots.")
    listing.add_argument("--name", default=None, help="Only names containing this text.")

    for command in ("cache-list", "cache-prune"):
        cache = commands.add_parser(
            command,
            help=(
                "List derived-cache files, reading headers only."
                if command == "cache-list"
                else "Preview (or --apply) removal of invalid, expired or excess cache files."
            ),
        )
        cache.add_argument("directory", type=Path, help="Cache directory.")
        cache.add_argument(
            "--namespace",
            default=None,
            help="Only this namespace ('' for unnamespaced files); default: every namespace.",
        )
        cache.add_argument(
            "--suffix", default=".typed-index", help="Cache file suffix (default .typed-index)."
        )
        cache.add_argument(
            "--magic-hex",
            default=None,
            help="Envelope magic as hex; the library default if omitted.",
        )
        if command == "cache-prune":
            cache.add_argument(
                "--older-than",
                type=_duration,
                default=None,
                help="Remove entries unused for this long, e.g. 3600, 12h, 30d.",
            )
            cache.add_argument(
                "--max-total-bytes",
                type=_size,
                default=None,
                help="Remove the oldest entries beyond this total, e.g. 500M, 10G (per namespace).",
            )
            cache.add_argument(
                "--max-entries",
                type=_nonnegative_int,
                default=None,
                help="Remove the oldest entries beyond this count (per namespace).",
            )
            cache.add_argument(
                "--temporary-grace",
                type=_duration,
                default=3600.0,
                help="Age after which abandoned temporaries are removed (default 1h).",
            )
            cache.add_argument("--apply", action="store_true", help="Actually remove files.")
    return parser


def _caches(args: argparse.Namespace) -> list[DerivedIndexCache[bytes]]:
    """Cache handles for listing and pruning; payloads are never decoded here."""
    if not args.directory.is_dir():
        raise FileNotFoundError(f"cache directory not found: {args.directory}")
    options: dict[str, Any] = {}
    if args.magic_hex is not None:
        options["magic"] = bytes.fromhex(args.magic_hex)
    names = (
        _namespaces(args.directory, args.suffix) if args.namespace is None else (args.namespace,)
    )
    return [
        DerivedIndexCache(
            args.directory,
            encode=bytes,
            decode=bytes,
            namespace=name,
            suffix=args.suffix,
            **options,
        )
        for name in names
    ]


def _is_store(path: Path) -> bool:
    try:
        return path.is_dir() and (
            path.name.endswith(".lmdb") or (path / "FORMAT").exists() or (path / "CURRENT").exists()
        )
    except OSError:
        return False


def _run(args: argparse.Namespace) -> int:
    reading: dict[str, Any] = {}
    if "read_only_media" in args:
        reading["cooperative"] = not args.read_only_media
    if getattr(args, "generation", None) is not None:
        reading["generation"] = args.generation
    command = args.command
    if command == "cache-list":
        entries = []
        for cache in _caches(args):
            entries.extend({"namespace": cache.namespace, **asdict(e)} for e in cache.entries())
        _print(entries)
        return EXIT_OK
    if command == "cache-prune":
        reports = {
            cache.namespace: cache.prune(
                older_than=args.older_than,
                max_total_bytes=args.max_total_bytes,
                max_entries=args.max_entries,
                dry_run=not args.apply,
                temporary_grace=args.temporary_grace,
            )
            for cache in _caches(args)
        }
        _print(
            {
                "dry_run": not args.apply,
                "namespaces": {name: asdict(report) for name, report in reports.items()},
            }
        )
        return EXIT_FINDINGS if any(report.in_use for report in reports.values()) else EXIT_OK
    if command == "delete-generations":
        if args.keep_last is not None:
            results = prune_generations(
                args.path,
                keep_last=args.keep_last,
                dry_run=not args.apply,
                same_host=args.same_host,
            )
        else:
            results = delete_generations(
                args.path, args.names, dry_run=not args.apply, same_host=args.same_host
            )
        _print({"dry_run": not args.apply, "results": [asdict(item) for item in results]})
        refused = ("current", "in_use", "unexpected_files")
        return EXIT_FINDINGS if any(item.status in refused for item in results) else EXIT_OK
    if command == "set-current":
        _print(asdict(set_current(args.path, args.name, verify=args.verify)))
        return EXIT_OK
    if command == "generations":
        _print(asdict(inspect_generations(args.path)))
        return EXIT_OK
    if command == "backend":
        _print(asdict(get_backend_info()))
        return EXIT_OK
    if command == "manifest":
        with LMDBReader(args.path, **reading) as reader:
            stored = reader.manifest()
            snapshot = reader.get_snapshot()
        _print(
            {
                "snapshot": asdict(snapshot),
                "manifest": None if stored is None else asdict(stored),
            }
        )
        return EXIT_OK
    if command == "doctor" and args.physical:
        result = verify_physical(args.path, max_issues=args.max_issues)
        _print(
            {
                **asdict(result),
                "ok": result.ok,
                "coverage": "offline native pages; excludes typed values and supplied lock files",
            }
        )
        return EXIT_OK if result.ok else EXIT_FINDINGS
    if command == "fingerprint":
        fingerprint = fingerprint_store(
            args.path, progress=_progress_printer(args.progress), **reading
        )
        _print(asdict(fingerprint))
        return EXIT_OK
    if command == "diff":
        difference = diff_stores(
            args.before_path,
            args.after_path,
            before_generation=args.before_generation,
            after_generation=args.after_generation,
            cooperative=not args.read_only_media,
            max_listed=args.max_listed,
            compare_bodies=args.compare_bodies,
            progress=_progress_printer(args.progress),
        )
        _print({**asdict(difference), "identical": difference.identical})
        return EXIT_OK if difference.identical else EXIT_FINDINGS
    if command == "stats":
        result = array_statistics(
            args.path,
            batch_size=args.batch_size,
            max_groups=args.max_groups,
            progress=_progress_printer(args.progress),
            **reading,
        )
        report = asdict(result)
        for group, row in zip(result.groups, report["groups"], strict=True):
            row["smallest_key"] = _key(group.smallest_location)
            row["largest_key"] = _key(group.largest_location)
        _print(report)
        return EXIT_OK
    if command == "doctor" and args.full:
        result = verify_store(
            args.path,
            batch_size=args.batch_size,
            max_value_bytes=args.max_value_bytes,
            max_batch_bytes=args.max_batch_bytes,
            max_issues=args.max_issues,
            progress=_progress_printer(args.progress),
            **reading,
        )
        report = asdict(result)
        report["issues"] = [_issue(issue) for issue in result.issues]
        report.update(
            ok=result.ok,
            complete=result.complete,
            full_value_validation=result.complete,
            coverage=(
                "every key and integrity envelope; checksums and decoding of values "
                "within the byte limits and available codecs"
            ),
        )
        _print(report)
        return EXIT_OK if result.ok else EXIT_FINDINGS
    if command == "list":
        if not args.directory.is_dir():
            raise FileNotFoundError(f"directory not found: {args.directory}")
        paths = sorted(
            path
            for path in args.directory.iterdir()
            if _is_store(path) and (args.name is None or args.name in path.name)
        )
        reports = []
        failed = False
        for path in paths:
            try:
                reports.append({"name": path.name, **_report(inspect_store(path))})
            except (OSError, ValueError, lmdb.Error, AmberLMDBError) as exc:
                failed = True
                reports.append({"name": path.name, "error": f"{type(exc).__name__}: {exc}"})
        _print(reports)
        return EXIT_FINDINGS if failed else EXIT_OK
    info = inspect_store(
        args.path,
        with_meta=args.meta,
        entry_limit=args.entries,
        after=args.after,
        max_batch_bytes=args.max_copy_bytes,
        metadata_key=args.meta_key,
        **reading,
    )
    report = _report(info)
    errors = [entry.error for entry in info.samples if entry.error is not None]
    if info.meta_error is not None:
        errors.append(info.meta_error)
    if command == "doctor":
        report.update(
            sample_ok=not errors,
            full_value_validation=False,
            coverage="layout, statistics, optional meta and sampled keys, tags and headers",
            hint="run doctor --full to verify every entry",
        )
    _print(report)
    return EXIT_FINDINGS if errors else EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return its exit code (see the module docstring)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in ("stats", "doctor") and not 1 <= args.batch_size <= 1024:
        parser.error("--batch-size must be between 1 and 1024")
    if args.command == "doctor":
        if (args.full or args.physical) and (args.after is not None or args.meta):
            parser.error("--full/--physical cannot be combined with --after or --meta")
        if args.progress and not args.full:
            parser.error("--progress requires --full")
        if args.physical and (args.generation is not None or args.read_only_media):
            parser.error(
                "--physical reads files directly; --generation and --read-only-media do not apply"
            )
    if args.command == "delete-generations":
        if (args.keep_last is None) == (not args.names):
            parser.error("name generations or pass --keep-last, not both")
        if args.apply and not args.same_host:
            parser.error("--apply requires --same-host and only cooperating amber-lmdb clients")
    try:
        return _run(args)
    except (OSError, ValueError, lmdb.Error, AmberLMDBError) as exc:
        message = f"{type(exc).__name__}: {exc}"
        _print({"error": {"type": type(exc).__name__, "message": str(exc)}})
        print(message, file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


__all__ = ["build_parser", "main"]
