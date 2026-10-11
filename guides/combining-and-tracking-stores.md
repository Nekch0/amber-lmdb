# Combine stores and track what they contain

These APIs answer questions about a store without reading array bodies: which keys
exist, whether two generations hold the same data, what a generation contained when
it was written, and how far a long scan has got. An overlay reader also lets new
data live in its own store instead of rewriting an existing one.

## Check, count and list keys

`contains`, `count`, `index_range`, `iter_indices` and `iter_keys` look only at the
integrity metadata of the opened generation. All indices of one logical key are
stored next to each other, so `index_range` takes two seeks and a listing takes one
seek per logical key, whatever the number of indices.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from amber_lmdb import LMDBReader, LMDBWriter

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "sessions.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_str("corpus", key="name")
        for index in (0, 1, 2, 5):
            writer.write_primitive(index * 10, key=("s01", "pitch"), index=index)
        writer.write_primitive(7, key=("s02", "pitch"), index=0)
    with LMDBReader(path) as reader:
        assert reader.contains(("s01", "pitch"), 5)
        assert not reader.contains(("s01", "pitch"), 3)
        assert reader.count(("s01", "pitch")) == 4
        assert reader.index_range(("s01", "pitch")) == (0, 5)
        groups = [group for page in reader.iter_keys(prefix=("s01",)) for group in page]
        assert [group.key for group in groups] == [("s01", "pitch")]
        assert (groups[0].first_index, groups[0].last_index) == (0, 5)
```

Listings come in encoded key order: string keys first, then tuple keys by component
count, and each component compares by byte length before content. `count` walks the
key's index range, so it takes time proportional to the count; a writer manifest
(below) records counts without a scan.

## Read a base store and new features as one

`LMDBOverlayReader` reads several published stores as one keyspace. Each key and
index must exist in exactly one layer; a key may still be split across layers by
index. Reads, `contains`, counts and listings raise `OverlayConflictError` when an
entry they touch is in more than one layer, and `check_disjoint()` checks the whole
overlay up front. Nothing is copied or modified, and every layer stays pinned to the
generation it opened.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from amber_lmdb import LMDBOverlayReader, LMDBWriter, OverlayConflictError

with TemporaryDirectory() as tmp:
    base, extra, clash = (Path(tmp) / name for name in ("base", "extra", "clash"))
    with LMDBWriter(base) as writer:
        writer.write_str("corpus", key="name")
        for index in range(3):
            writer.write_ndarray(np.full(4, index, np.float32), key="features", index=index)
    with LMDBWriter(extra) as writer:
        for index in range(3):
            writer.write_ndarray(np.full(2, -index, np.int16), key="pitch", index=index)
    with LMDBOverlayReader([base, extra]) as reader:
        assert reader.check_disjoint() == 7
        features, pitch = reader.read_many(["features", "pitch"], indices=[2, 2])
        assert features.tolist() == [2.0] * 4 and pitch.tolist() == [-2, -2]

    with LMDBWriter(clash) as writer:
        writer.write_str("other", key="name")
    with LMDBOverlayReader([base, clash]) as reader:
        try:
            reader.read_as_str("name")
        except OverlayConflictError as error:
            assert error.layers == (base, clash)
        else:
            raise AssertionError("a repeated entry must not be read")
```

Batches look up integrity metadata in every layer and then copy each layer's share
in one short transaction per layer; byte and item budgets cover the whole batch.
`get_snapshot()` returns one snapshot per layer, and `RecordTable` and `ValueView`
accept an overlay like a single reader. `verify_checksums` applies to every layer.

## Derive a new generation from a published one

When one self-contained store is needed, or entries must be removed,
`LMDBWriter(path, base="current")` (or a generation name) starts the new generation
from a compact copy of a published one, then applies writes and deletions before
publishing as usual. The copy holds the base generation's shared lease, so
cooperative deletion cannot remove it meanwhile; the base stays until it is deleted
explicitly. Copied entries keep their envelopes, the engine follows the base, and a
copied manifest is dropped (`manifest=True` rebuilds it and records the base).

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from amber_lmdb import LMDBReader, LMDBWriter, diff_stores

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "store.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_str("corpus", key="name")
        for index in range(3):
            writer.write_ndarray(np.zeros(4, np.float32), key="old", index=index)
    base = (path / "CURRENT").read_text().strip()
    with LMDBWriter(path, base="current", manifest=True) as writer:
        writer.write_ndarray(np.ones(2, np.int16), key="pitch", index=0)
        assert writer.delete_key("old") == 3
    with LMDBReader(path) as reader:
        assert reader.read_as_str("name") == "corpus"
        assert not reader.contains("old", 0)
        assert reader.manifest().base == base
    difference = diff_stores(path, path, before_generation=base)
    assert difference.added == (("pitch", 0),) and difference.removed_count == 3
```

The copy costs one sequential pass over the base data file. An overlay costs
nothing up front; prefer it when nothing has to be removed.

## Fold layers into one store

`merge_stores(paths, destination)` publishes one store holding every entry of
several disjoint stores, for example a base and the feature stores read through an
overlay. It checks disjointness first (`OverlayConflictError` otherwise, before
anything is written), then copies entries as stored, bodies with their integrity
envelopes, so chunked and compressed arrays and every value type survive without
decoding. Layer manifests are not copied; `manifest=True` writes a new one.
`verify_checksums=True` (the default) checks every body while copying, and
`progress`/`cancel` work as for scans; a cancelled merge publishes nothing.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from amber_lmdb import LMDBReader, LMDBWriter, merge_stores

with TemporaryDirectory() as tmp:
    base, extra, merged = (Path(tmp) / name for name in ("base", "extra", "merged"))
    with LMDBWriter(base) as writer:
        writer.write_ndarray(np.zeros((100, 4), np.float32), key="features", chunk_rows=16)
    with LMDBWriter(extra) as writer:
        writer.write_primitive(3, key="count")
    assert merge_stores([base, extra], merged, manifest=True) == 2
    with LMDBReader(merged) as reader:
        assert reader.read("count") == 3
        assert reader.read_array_rows("features", start=20, stop=24).shape == (4, 4)
        assert reader.manifest().entries == 2
```

## Identify content and compare generations

`fingerprint_store` digests every key and its integrity envelope in key order. Each
envelope binds key, writer tag, body length and body digest, so equal fingerprints
mean equal content written with the same digest algorithm, regardless of write
order, LMDB engine or page layout. `diff_stores` lists what was added, removed or
changed, at most `max_listed` keys per category (the counts stay complete). Pass a
generation name to look at an older generation of the same root.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from amber_lmdb import LMDBWriter, diff_stores, fingerprint_store

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "store.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_str("v1", key="release")
        writer.write_primitive(3, key="count")
    first = fingerprint_store(path)
    old_generation = (path / "CURRENT").read_text().strip()
    with LMDBWriter(path, overwrite=True) as writer:
        writer.write_str("v1", key="release")
        writer.write_primitive(4, key="count")
        writer.write_str("new", key="notes")
    assert fingerprint_store(path, generation=old_generation).digest == first.digest
    difference = diff_stores(path, path, before_generation=old_generation)
    assert difference.changed == (("count", None),)
    assert difference.added == (("notes", None),)
    assert difference.unchanged == 1 and not difference.identical
```

Fingerprints read integrity metadata only; `verify_store` checks bodies against it.
Entries whose envelopes use different digest algorithms (SHA-256 and BLAKE2b-256)
with equal tag and length are reported as undetermined unless
`compare_bodies=True` reads and compares those bodies.

## Record a manifest at write time

`LMDBWriter(..., manifest=True)` stores a summary as the last entry before
publication: entry and byte totals, envelope algorithms, the content fingerprint,
and per logical key its writer tags, unindexed entry, index count and bounds, and
array dtypes with per-axis minimum and maximum shapes. `provenance` adds your own
mapping. The manifest is an ordinary dictionary under a reserved key that writers
refuse for user data; fingerprints, diffs, listings and overlay conflict checks
ignore it.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from amber_lmdb import LMDBReader, LMDBWriter, fingerprint_store

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "store.lmdb"
    with LMDBWriter(path, manifest=True, provenance={"source": "extract.py"}) as writer:
        for index, length in enumerate((5, 9, 7)):
            writer.write_ndarray(np.zeros((length, 80), np.float32), key="mel", index=index)
    with LMDBReader(path) as reader:
        manifest = reader.manifest()
    assert manifest is not None and manifest.entries == 3
    assert manifest.fingerprint == fingerprint_store(path).digest
    assert manifest.provenance == {"source": "extract.py"}
    (group,) = manifest.groups
    (arrays,) = group.arrays
    assert (group.indexed, group.first_index, group.last_index) == (3, 0, 2)
    assert (arrays.min_shape, arrays.max_shape) == ((5, 80), (9, 80))
```

Building the manifest scans the staging generation once before publication, so it
is off by default. At most `manifest_max_groups` (10,000) key groups are itemized;
beyond that only totals are kept and `groups_truncated` is true.

## Follow and stop long scans

`verify_store`, `array_statistics`, `fingerprint_store`, `diff_stores`,
`merge_stores` and `LMDBOverlayReader.check_disjoint` accept `progress`, a callback
that receives a
`Progress` report between pages (never inside a transaction, at most every 0.2 s,
and once at the end), and `cancel`, any object with `is_set()` such as a
`threading.Event`. A set event raises `OperationCancelled` at the next page boundary.

```python
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

from amber_lmdb import LMDBWriter, OperationCancelled, verify_store

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "store.lmdb"
    with LMDBWriter(path) as writer:
        for index in range(100):
            writer.write_primitive(index, key="values", index=index)
    reports = []
    assert verify_store(path, batch_size=10, progress=reports.append).ok
    assert reports[-1].finished and reports[-1].entries_done == 100

    cancel = threading.Event()
    cancel.set()
    try:
        verify_store(path, cancel=cancel)
    except OperationCancelled as stopped:
        assert stopped.partial.cancelled and not stopped.partial.ok
```

Only `verify_store` attaches a partial report, and it is marked cancelled, so it is
never complete or `ok`. On the command line, `--progress` on `doctor --full`, `stats`,
`fingerprint` and `diff` writes progress to stderr while stdout keeps the JSON.
