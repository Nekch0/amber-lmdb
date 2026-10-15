"""Rich store views, doctor reports and the dataset command line."""

import io
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from amber_lmdb import LMDBWriter
from amber_lmdb.cli import main as amber_main
from amber_lmdb.ui import (
    ENGLISH,
    DatasetCLI,
    Section,
    StoreProfile,
    describe_store,
    environment_rows,
    feature_nodes,
    meta_rows,
    store_grid,
)
from amber_lmdb.ui.stores import (
    default_summary,
    feature_rows,
    find_stores,
    shape_range,
    split_store_name,
)

META = {
    "dataset": "toy",
    "n_utterances": 2,
    "names": ("logmel", "label"),
    "type_map": {"logmel": np.ndarray, "label": int},
    "dtype_map": {"logmel": "<f4"},
    "shape_map": {"logmel": ((2, 4), (2, 9))},
    "count_map": {"logmel": 2, "label": 2},
    "python_version": "3.12.1",
}


def make_store(directory: Path, name: str = "logmel_a1b2c3d4.lmdb", meta=META) -> Path:
    store = directory / name
    with LMDBWriter(store) as writer:
        if meta is not None:
            writer.write_dict(meta, key="meta")
        writer.write_ndarray(np.zeros((2, 4), dtype=np.float32), key="logmel", index=0)
        writer.commit()
    return store


def render(renderable) -> str:
    console = Console(file=io.StringIO(), width=200, color_system=None)
    console.print(renderable)
    return console.file.getvalue()


def test_split_store_name():
    assert split_store_name("logmel_a1b2c3d4.lmdb") == ("logmel", "a1b2c3d4")
    assert split_store_name("plain.lmdb") == ("plain", "-")


def test_describe_store_reads_meta_and_convention(tmp_path):
    store = make_store(tmp_path)
    summary = describe_store(store, with_meta=True)
    assert summary.typename == "logmel"
    assert summary.n_versions == 1
    assert summary.entries is not None and summary.entries > 0
    assert summary.meta is not None and summary.meta["dataset"] == "toy"
    assert summary.shapes == {"logmel": ([2, 4], [2, 9])}
    assert summary.counts == {"logmel": 2, "label": 2}
    assert feature_rows(summary)[0] == (
        "logmel",
        "numpy.ndarray",
        "<f4",
        "ndarray",
        "(2, 4..9)",
        "2",
    )
    assert shape_range(([1, 2], [1, 3])) == "(1, 2..3)"


def test_describe_store_without_current(tmp_path):
    store = tmp_path / "broken.lmdb"
    store.mkdir()
    messages = replace(ENGLISH, current_unreadable="no CURRENT", meta_unreadable="No meta.{reason}")
    summary = describe_store(store, with_meta=True, messages=messages)
    assert summary.version == "-"
    assert summary.meta is None
    assert summary.meta_error == "no CURRENT"
    assert "No meta. (no CURRENT)" in render(store_grid(summary, StoreProfile(), messages=messages))


def test_default_profile_summary(tmp_path):
    summary = describe_store(make_store(tmp_path), with_meta=True)
    rows = default_summary(summary)
    assert ("meta", "dataset", "toy") in rows
    assert all(key not in {"names", "type_map"} for _, key, _ in rows)
    assert ("environment", "python", "3.12.1") in rows
    text = render(store_grid(summary, StoreProfile()))
    assert "logmel_a1b2c3d4.lmdb" in text and "toy" in text


def test_custom_profile(tmp_path):
    def summary_rows(summary):
        meta = summary.meta or {}
        return [
            *meta_rows(meta, "summary", ["dataset", ("utterances", "n_utterances")]),
            *environment_rows(meta),
        ]

    profile = StoreProfile(
        summary=summary_rows,
        contents=lambda s: [
            ("meta", "store metadata", ()),
            ("utterance[i]", "", feature_nodes(s.meta or {}, limit=1)),
        ],
        usage=lambda s: f"load({str(s.path)!r})",
    )
    summary = describe_store(make_store(tmp_path), with_meta=True)
    text = render(store_grid(summary, profile))
    assert "utterances" in text and "2" in text
    assert "└── utterance[i]" in text
    assert "... (1 more)" in text
    assert "cuda" in text  # environment rows render "-" for absent keys


def test_show_command(tmp_path, capsys):
    store = make_store(tmp_path)
    assert amber_main(["show", str(store)]) == 0
    out = capsys.readouterr().out
    assert "FEATURE" in out and "logmel" in out and "LMDBReader" in out
    assert amber_main(["show", str(tmp_path / "missing.lmdb")]) == 3


class Toy:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records = sorted(path.name for path in (root / "wav").iterdir())


def toy_cli(**changes) -> DatasetCLI:
    def resolve_root(root: str | None) -> Path:
        import os

        value = root if root is not None else os.environ.get("TOY_DATASET_PATH")
        if not value:
            raise ValueError("TOY_DATASET_PATH is not set")
        path = Path(value)
        if not (path / "wav").is_dir():
            raise ValueError(f"{path} has no wav directory")
        return path

    def inspect(dataset: Toy, section: Section) -> None:
        section.add(
            "record index", "OK" if dataset.records else "FAIL", f"{len(dataset.records)} records"
        )

    def check_store(store, meta, dataset, section) -> None:
        if dataset is not None:
            same = meta["n_utterances"] == len(dataset.records)
            section.add(f"{store.name} records", "OK" if same else "WARN", "")

    config = DatasetCLI(
        name="TOY",
        prog="dataset-toy",
        dataset_env="TOY_DATASET_PATH",
        resources_env="TOY_RESOURCES_PATH",
        resolve_root=resolve_root,
        required_entries=("wav",),
        feature_root=("dataset", "toy"),
        store_directory=("dataset", "toy", "utterance_feature"),
        load_dataset=lambda root, resources: Toy(Path(root)),
        inspect_dataset=inspect,
        check_store=check_store,
        store_meta_keys=("names", "type_map", "n_utterances"),
        store_meta_detail={"utterances": "n_utterances"},
        index_check="record index",
    )
    return replace(config, **changes)


@pytest.fixture
def toy_env(tmp_path, monkeypatch):
    root = tmp_path / "toy"
    (root / "wav").mkdir(parents=True)
    for name in ("a.wav", "b.wav"):
        (root / "wav" / name).write_bytes(b"")
    resources = tmp_path / "resources"
    store_dir = resources / "dataset" / "toy" / "utterance_feature"
    store_dir.mkdir(parents=True)
    monkeypatch.setenv("TOY_DATASET_PATH", str(root))
    monkeypatch.setenv("TOY_RESOURCES_PATH", str(resources))
    return root, store_dir


def test_doctor_passes(toy_env, capsys):
    _, store_dir = toy_env
    make_store(store_dir)
    assert toy_cli().main(["doctor", "--all"]) == 0
    out = capsys.readouterr().out
    assert "2 records" in out
    assert "utterances=2" in out
    assert "All checks passed" in out


def test_doctor_reports_failures(toy_env, capsys, monkeypatch):
    monkeypatch.setenv("TOY_DATASET_PATH", str(toy_env[0] / "missing"))
    messages = replace(ENGLISH, root_unresolved="Root is unknown.")
    assert toy_cli(messages=messages).main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "1 FAIL result(s) found" in out
    assert "Root is unknown." in out


def test_doctor_missing_meta_key(toy_env, capsys):
    _, store_dir = toy_env
    make_store(store_dir, meta={"names": ()})
    assert toy_cli().main(["doctor", "--all"]) == 1
    assert "missing keys: type_map, n_utterances" in capsys.readouterr().out


def test_paths(toy_env, capsys, monkeypatch):
    assert toy_cli().main(["paths"]) == 0
    out = capsys.readouterr().out
    assert "resolved" in out and "feature stores" in out
    monkeypatch.delenv("TOY_RESOURCES_PATH")
    assert toy_cli().main(["paths"]) == 0
    assert "(unset)" in capsys.readouterr().out


def test_features_list_and_detail(toy_env, capsys):
    _, store_dir = toy_env
    make_store(store_dir)
    make_store(store_dir, "other_00000000.lmdb")
    assert toy_cli().main(["features", "--list"]) == 0
    out = capsys.readouterr().out
    assert "logmel_a1b2c3d4.lmdb" in out and "other_00000000.lmdb" in out
    assert toy_cli().main(["features", "other", "--detail"]) == 0
    out = capsys.readouterr().out
    assert "other_00000000.lmdb" in out and "logmel_a1b2c3d4.lmdb" not in out
    assert "FEATURE" in out
    assert toy_cli().main(["features", "nothing"]) == 1


def test_features_without_resources(toy_env, capsys, monkeypatch):
    monkeypatch.delenv("TOY_RESOURCES_PATH")
    assert toy_cli().main(["features", "--list"]) == 1
    assert "TOY_RESOURCES_PATH" in capsys.readouterr().err


def test_dataset_requires_terminal(toy_env, capsys):
    assert toy_cli().main(["dataset"]) == 1
    err = capsys.readouterr().err
    assert "error: The dataset command needs an interactive terminal." in err
    assert "hint:" in err


def test_extend_parser(toy_env):
    calls = []

    def extend(commands) -> None:
        command = commands.add_parser("hello")
        command.set_defaults(func=lambda args: calls.append(args.command) or 0)

    assert toy_cli(extend_parser=extend).main(["hello"]) == 0
    assert calls == ["hello"]


def test_find_stores(tmp_path):
    make_store(tmp_path, "b_00000000.lmdb")
    make_store(tmp_path, "a_00000000.lmdb")
    (tmp_path / "not-a-store").mkdir()
    assert [path.name for path in find_stores(tmp_path)] == ["a_00000000.lmdb", "b_00000000.lmdb"]
    assert find_stores(tmp_path / "missing") == []


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        toy_cli(distribution="amber-lmdb").main(["--version"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("dataset-toy ") and "(amber-lmdb " in out


def test_no_arguments_prints_help(capsys):
    assert toy_cli().main([]) == 2
    out = capsys.readouterr().out
    assert "Usage: dataset-toy" in out and "Commands" in out and "doctor" in out


def test_subcommand_help_is_rich_and_customizable(capsys):
    with pytest.raises(SystemExit):
        toy_cli(messages=replace(ENGLISH, options_heading="Flags")).main(["features", "--help"])
    out = capsys.readouterr().out
    assert out.count("Usage:") == 1
    assert "--json" in out and "Flags" in out


def test_doctor_json(toy_env, capsys):
    import json

    _, store_dir = toy_env
    make_store(store_dir)
    assert toy_cli().main(["doctor", "--all", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["ok"] is True
    assert document["counts"]["fail"] == 0
    titles = [section["title"] for section in document["sections"]]
    assert titles == ["Paths", "Dataset", "Feature stores"]
    assert {"name": "record index", "status": "OK", "detail": "2 records"} in document["sections"][
        1
    ]["checks"]


def test_paths_json_reports_failure(toy_env, capsys, monkeypatch):
    import json

    monkeypatch.setenv("TOY_DATASET_PATH", str(toy_env[0] / "missing"))
    assert toy_cli().main(["paths", "--json"]) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["ok"] is False
    assert "no wav directory" in document["dataset_root"]["error"]
    assert document["resources_root"]["feature_stores"]["exists"] is True


def test_features_json(toy_env, capsys):
    import json

    _, store_dir = toy_env
    make_store(store_dir)
    assert toy_cli().main(["features", "--json"]) == 0
    [listed] = json.loads(capsys.readouterr().out)
    assert listed["name"] == "logmel_a1b2c3d4.lmdb" and listed["hash"] == "a1b2c3d4"
    assert "meta" not in listed
    assert toy_cli().main(["features", "--json", "--detail"]) == 0
    [detailed] = json.loads(capsys.readouterr().out)
    assert detailed["meta"]["dataset"] == "toy"
    assert detailed["features"][0] == {
        "name": "logmel",
        "type": "numpy.ndarray",
        "dtype": "<f4",
        "kind": "ndarray",
        "shape": "(2, 4..9)",
        "entries": "2",
    }


def test_interrupt_exits_130(toy_env, capsys):
    def interrupted(dataset, section):
        raise KeyboardInterrupt

    assert toy_cli(inspect_dataset=interrupted).main(["doctor"]) == 130
    assert "Interrupted." in capsys.readouterr().err


def test_parsing_does_not_load_rich_or_textual():
    import subprocess
    import sys

    code = "; ".join(
        [
            "import sys",
            "from amber_lmdb.ui import DatasetCLI, ENGLISH, StoreProfile, Listing, Detail",
            "cli = DatasetCLI(name='T', prog='t', dataset_env='A', resources_env='B', "
            "resolve_root=lambda r: None, messages=ENGLISH, profile=StoreProfile())",
            "cli.build_parser().parse_args(['paths'])",
            "print(sorted(m for m in ('rich', 'textual', 'rich_argparse') if m in sys.modules))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_short_version():
    from amber_lmdb.ui._common import short_version

    assert short_version("v-0123456789abcdef") == "v-012345678…"
    assert short_version("v-0123456789abcdef", ellipsis="...") == "v-0123456..."
    assert short_version("v-1") == "v-1"
