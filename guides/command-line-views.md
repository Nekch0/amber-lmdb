# Command-line views and explorers

The JSON commands of `amber-lmdb` are for scripts. For people there are two more
commands, and the same views are a library, `amber_lmdb.ui`, that packages built on
amber-lmdb customize instead of copying.

```console
amber-lmdb show ./stores/logmel_a1b2c3d4.lmdb   # file, metadata summary, features, usage
amber-lmdb browse ./stores                      # interactive explorer (needs a terminal)
```

`browse` lists the `*.lmdb` stores of a directory. Enter opens a store with its
metadata, feature table, source and usage example; `c` copies the path and `d`
deletes the store after two confirmations. Escape goes back and `q` quits. Dragging
over text copies it; double-clicking a cell of a plain table copies the cell.

## Feature-store convention

The views read an optional convention from the `meta` dictionary entry: `names`
(feature names), `type_map`, `dtype_map`, `shape_map` (per-dimension minimums and
maximums) and `count_map`. With `names` present the views show a feature table; a
`code.log` file in the store directory is shown as the source that wrote it.
Environment keys (`python_version`, `torch_version`, `cuda_version`, `cudnn_version`,
`device`) appear as their own group.

```python
import tempfile
from pathlib import Path

import numpy as np

from amber_lmdb import LMDBWriter
from amber_lmdb.ui import StoreProfile, describe_store, store_grid

directory = Path(tempfile.mkdtemp())
store = directory / "logmel_a1b2c3d4.lmdb"
with LMDBWriter(store) as writer:
    writer.write_dict(
        {
            "names": ("logmel",),
            "type_map": {"logmel": np.ndarray},
            "dtype_map": {"logmel": "<f4"},
            "shape_map": {"logmel": ((80, 12), (80, 300))},
            "count_map": {"logmel": 1},
            "n_utterances": 1,
        },
        key="meta",
    )
    writer.write_ndarray(np.zeros((80, 12), dtype=np.float32), key="logmel", index=0)
    writer.commit()

summary = describe_store(store, with_meta=True)
assert summary.typename == "logmel" and summary.digest == "a1b2c3d4"
assert summary.shapes == {"logmel": ([80, 12], [80, 300])}
grid = store_grid(summary, StoreProfile())  # a rich renderable
```

## Profiles

A `StoreProfile` decides what the detail views show below the file information:
`summary` returns `(group, key, value)` rows, `contents` returns a tree describing
the stored layout, `usage` returns example code and `source_file` names the file
shown as source. Helpers build the usual rows.

```python
from amber_lmdb.ui import (
    StoreProfile,
    environment_rows,
    feature_nodes,
    feature_usage_comment,
    meta_rows,
)


def summary_rows(summary):
    meta = summary.meta or {}
    return [
        *meta_rows(meta, "summary", ["record_type", ("utterances", "n_utterances")]),
        *environment_rows(meta),
    ]


def contents(summary):
    meta = summary.meta or {}
    return [
        ("meta", "store metadata", ()),
        ("utterance[i]", "one entry per utterance", feature_nodes(meta)),
    ]


def usage(summary):
    feature, comment = feature_usage_comment(summary)
    return (
        "from my_dataset import FeatureLoader\n\n"
        f'loader = FeatureLoader(r"{summary.path}")\n'
        f'x = loader.get_features(0)["{feature}"]  # {comment}'
    )


profile = StoreProfile(summary=summary_rows, contents=contents, usage=usage)
```

## Browsing a dataset

`Listing` describes one list screen and `Detail` one detail screen. Selecting a row
runs `open` in a worker thread, so parsing annotations never freezes the interface;
`open` returns the next `Listing` or a `Detail`.

```python
from amber_lmdb.ui.tui import BrowserApp, Detail, Listing

sessions = {"S01": ["hello", "world"], "S02": ["good morning"]}


def open_session(name):
    return Detail(
        title="Session",
        info=[("id", name), ("utterances", len(sessions[name]))],
        table_title="Utterances",
        columns=("index", "text"),
        rows=[(str(index), text) for index, text in enumerate(sessions[name])],
    )


root = Listing(
    title="Sessions",
    columns=("session", "utterances"),
    items=sorted(sessions),
    row=lambda name: (name, str(len(sessions[name]))),
    key=lambda name: name,
    filter_text=lambda name: name,
    open=open_session,
)
app = BrowserApp(root, title="My Dataset Explorer")  # app.run() starts it
```

## A dataset command line

`DatasetCLI` assembles four commands for a dataset package: `doctor` (paths, the
dataset and every feature store; `--all` reads the data), `dataset` (the browser),
`paths` (how the roots resolve) and `features` (the store explorer, or `--list` and
`--detail` output). The package supplies its names, root resolution and hooks;
`dataclasses.replace(ENGLISH, ...)` changes single strings of the text.

```python
import os
import tempfile
from pathlib import Path

from dataclasses import replace

from amber_lmdb.ui import ENGLISH, DatasetCLI, Section, StoreProfile
from amber_lmdb.ui.tui import Listing


def resolve_root(root):
    value = root if root is not None else os.environ.get("MY_DATASET_PATH")
    if not value or not Path(value, "wav").is_dir():
        raise ValueError("set MY_DATASET_PATH to the directory holding wav/")
    return Path(value)


def inspect_dataset(dataset, section: Section):
    section.add("record index", "OK" if dataset else "FAIL", f"{len(dataset)} records")


cli = DatasetCLI(
    name="MY",
    prog="dataset-my",
    dataset_env="MY_DATASET_PATH",
    resources_env="MY_RESOURCES_PATH",
    resolve_root=resolve_root,
    required_entries=("wav",),
    feature_root=("dataset", "my"),
    store_directory=("dataset", "my", "utterance_feature"),
    profile=StoreProfile(),
    load_dataset=lambda root, resources: sorted(Path(root, "wav").iterdir()),
    browse=lambda files: Listing(
        title="Files", columns=("name",), items=files, row=lambda path: (path.name,)
    ),
    inspect_dataset=inspect_dataset,
    store_meta_keys=("names", "type_map", "n_utterances"),
    messages=replace(ENGLISH, deep_help="Also decode every wav file."),
)

empty = Path(tempfile.mkdtemp())
assert cli.main(["features", "--list", "--output-dir", str(empty)]) == 0  # no stores yet
assert cli.main(["paths", "--root", str(empty)]) == 1  # wav/ is missing
```

`doctor`, `paths` and `features` accept `--json` for scripts; `--version` prints the
package and amber-lmdb versions. Without arguments the command prints its help. Errors
go to stderr as `error: ...` followed by a `hint: ...`. Exit codes: 0 success, 1 a
check failed or the command could not run, 2 invalid usage, 130 interrupted. Colors
follow `NO_COLOR` and `FORCE_COLOR`. Rich and Textual load only when a command prints
or opens something, so `--help`-free parsing and `--version` stay fast.

Expose `main = cli.main` as the package's console script. `extend_parser` adds
commands of your own; give each a `func` default that returns an exit code.
