# Define a record store without a dataset framework

Use a schema when every record must have the same field names and exact Python
classes. Use a record factory when dictionaries should become application objects.
Neither API adds session conventions, random sampling or padding.

## Read dictionaries as application records

This example stores three invoices. The factory runs after each short read
transaction closes. Duplicate indices keep their order and produce separate rows.

```python
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from amber_lmdb import LMDBReader, LMDBWriter, RecordTable, infer_schema


@dataclass(frozen=True)
class Invoice:
    number: int
    paid: bool


with TemporaryDirectory() as temporary:
    path = Path(temporary) / "invoices.lmdb"
    schema = infer_schema({"number": 0, "paid": False})
    with LMDBWriter(path) as writer:
        for index in range(3):
            writer.write_dict(
                {"number": 100 + index, "paid": index == 1}, key="invoices", index=index
            )
        writer.commit()
    with LMDBReader(path) as reader:
        table = RecordTable(reader, "invoices", lambda row: Invoice(**row), count=3, schema=schema)
        assert table.get_many([1, 0, 1]) == [
            Invoice(101, True),
            Invoice(100, False),
            Invoice(101, True),
        ]
        assert list(table.iter_records(offset=1, limit=1)) == [Invoice(101, True)]
```

`ValueSchema` can also be constructed explicitly, including `ordered=False` for
mappings whose key order is irrelevant. Schemas check exact classes, nested
supported types, object dtypes, cycles and nesting depth. They do not infer fixed
array shapes or dtypes, execute custom encoders, or replace domain validation.
The encoder still enforces its own item and byte budgets. For optional/nullable,
shape, dtype, numeric and length rules, use [RecordSchema](bounded-values-and-schemas.md).

`RecordTable(reader, key, factory=None, *, count=None)` reads rows written with
`write_dict` at indices `0..count-1`. Without `count` it counts the key's indices
(a key-only scan) and requires them to be exactly `0..count-1`; pass `count`
explicitly to read a prefix or to avoid the scan. Without a factory, rows are
returned as dictionaries. A table is a read-only sequence: `len(table)`,
`table[i]` (negative `i` counts from the end), `table[a:b]` (a list) and iteration.
`get(i)` is the strict form: negative and out-of-range indices raise `IndexError`;
missing physical rows raise `KeyNotFoundError` (a `KeyError`). Reads are paged by
`batch_size` and `max_bytes` (64 MiB by default, `None` for no limit) for encoded
bodies and integrity metadata. Decoded allocations and retained results are
separate. Snapshot changes between batches or pages raise `SourceChangedError`. No
transaction spans a yield.

## Read named values under a namespace

A `ValueView` defines the logical field names and physical key layout. Its default
keys include the storage category, matching existing feature-store keys. Pass a
`key_factory` for an application-specific layout. Neither option changes the
binary encoding of keys.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from amber_lmdb import LMDBReader, LMDBWriter, ValueView, infer_schema

with TemporaryDirectory() as temporary:
    path = Path(temporary) / "account.lmdb"
    schema = infer_schema({"balance": 0, "name": ""})
    with LMDBWriter(path) as writer:
        writer.write(42, key=("account", "primitive", "balance"), index=0)
        writer.write("Ada", key=("account", "str", "name"), index=0)
        writer.commit()
    with LMDBReader(path) as reader:
        view = ValueView(reader, schema, namespace=("account",), index=0)
        assert view.get_as("balance", int) == 42
        assert view.get_as("name", str) == "Ada"
```

`index=None` means an unindexed key, not a padding row. Missing fields are errors.
Custom classes require registration, `schema.allow_custom=True`, and an explicit
`allow_custom=True` read. No stored class name triggers arbitrary module imports.
Decoded arrays reference owned bytes and survive reader closure.

## Share a derived index between processes

`DerivedIndexCache` coordinates builders; it does not decide what an index means.
Supply its source identity, codec, builder and semantic validator. Include every
input affecting the result: the opened snapshot, schema, builder revision and
relevant codec/package versions. A filename or `CURRENT` pointer alone is not a
sufficient source identity.

```python
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from amber_lmdb import DerivedIndexCache, LMDBReader, LMDBWriter, RecordTable
from amber_lmdb.codec import any_to_bytes, bytes_to_any

with TemporaryDirectory() as temporary:
    root = Path(temporary)
    path = root / "invoices.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_dict({"paid": False}, key="invoices", index=0)
        writer.write_dict({"paid": True}, key="invoices", index=1)
        writer.commit()
    with LMDBReader(path) as reader:
        table = RecordTable(reader, "invoices", lambda row: row, count=2)
        snapshot = reader.get_snapshot()
        identity = {
            "snapshot": asdict(snapshot),
            "builder": "paid-count/1",
            "schema": "invoice/1",
            "codec_package": version("amber-lmdb"),
        }
        cache = DerivedIndexCache(
            root / "indexes",
            namespace="paid-count",
            encode=any_to_bytes,
            decode=bytes_to_any,
            lock_timeout=30,
        )
        calls = []

        def build():
            calls.append("build")
            return sum(row["paid"] for row in table.iter_records())

        def validate(value):
            if type(value) is not int or not 0 <= value <= len(table):
                raise ValueError("invalid paid count")

        def source_check():
            if reader.get_snapshot() != snapshot:
                raise ValueError("invoice source changed")

        assert (
            cache.get_or_build(identity, build, validate=validate, source_check=source_check) == 1
        )
        assert (
            cache.get_or_build(identity, build, validate=validate, source_check=source_check) == 1
        )
        assert calls == ["build"]
```

Identities are str-keyed mappings of JSON values (None, bool, int, finite float,
str, lists, nested mappings) or dataclass instances such as `LMDBSnapshot`, which
are converted field by field. Tuples and non-string keys raise `TypeError`, because
JSON would store them as lists and strings and the identity read back would differ.

Ready hits do not acquire a builder lock. Misses take the identity's builder lock
stripe (one of a fixed number of retained files in `<directory>/.locks`) and recheck
before building. The first builder records the stripe count in `.locks/STRIPES`
(64 unless `lock_stripes=` says otherwise); later caches on the directory adopt it,
and an explicit different `lock_stripes` raises `ValueError`, because processes
using different counts would not exclude each other. Corrupt, incompatible or incomplete cache files rebuild;
permission and other I/O failures propagate. Publication uses bounded headers and
payloads, a checksum, fsync and atomic replacement. Callbacks must use safe codecs;
custom executable decoding is not automatically enabled.

`directory=None` disables persistence. `lock_timeout=None` waits indefinitely;
a live stuck builder can block successors. A timeout only bounds lock acquisition,
not source reads or builder execution; `cancel=` (any object with `is_set()`) stops
this caller while it waits, before it builds and before it publishes, raising
`OperationCancelled` with nothing published. Stopping a runaway builder needs a
separate process. Killing a builder releases its OS lock; unfinished temporary files
are ignored and later removed by `prune`. Cached `None` is supported.
`lookup(identity)` returns `CacheResult(hit, value)`; inspect `hit` to distinguish
None from a miss. `get_or_build` uses hit status and never rebuilds a valid None.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from amber_lmdb import CacheResult, DerivedIndexCache
from amber_lmdb.codec import any_to_bytes, bytes_to_any

with TemporaryDirectory() as temporary:
    cache = DerivedIndexCache(Path(temporary), encode=any_to_bytes, decode=bytes_to_any)
    identity = {"builder": "empty-result/1"}
    assert cache.lookup(identity) == CacheResult(False)
    assert cache.get_or_build(identity, lambda: None) is None
    assert cache.lookup(identity) == CacheResult(True, None)

    def unexpected_rebuild():
        raise AssertionError("None was already cached")

    assert cache.get_or_build(identity, unexpected_rebuild) is None
```

Caches may contain restricted source data. The caller chooses permissions and
licensing. Cache suffixes ending in `.lock`, regardless of case, are reserved for
builder locks. Lock stripe files are never removed.

## Keep a cache directory from growing

An identity that includes the opened snapshot creates a new file whenever the source
is regenerated. `entries()` lists a cache's files from their headers alone, and
`prune` removes invalid ones, those a caller predicate rejects, those unchanged for
`older_than` seconds and then the oldest beyond `max_total_bytes`/`max_entries`. It
is a dry run unless `dry_run=False`. Each removal takes the entry's builder lock stripe
without waiting, so an entry being rebuilt is reported `in_use` and kept; a reader
that opens a removed entry afterwards simply rebuilds it. `touch_on_hit=True` makes
hits refresh the file time (at most hourly), so age means time since last use.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from amber_lmdb import DerivedIndexCache
from amber_lmdb.codec import any_to_bytes, bytes_to_any

with TemporaryDirectory() as temporary:
    cache = DerivedIndexCache(
        Path(temporary), namespace="counts", encode=any_to_bytes, decode=bytes_to_any
    )
    live = {"generation": "v-new"}
    for generation in ("v-old", "v-new"):
        cache.get_or_build({"generation": generation}, lambda: 42)
    assert sorted(entry.identity["generation"] for entry in cache.entries()) == ["v-new", "v-old"]

    def stale(entry):  # The caller knows which identities are still reachable.
        return entry.identity != live

    preview = cache.prune(remove=stale)
    assert preview.dry_run and len(preview.removed) == 1
    report = cache.prune(remove=stale, dry_run=False)
    assert report.reasons == {"predicate": 1} and not report.in_use
    assert [entry.identity for entry in cache.entries()] == [live]
```

`amber-lmdb cache-list DIR` and `amber-lmdb cache-prune DIR --older-than 30d
--max-total-bytes 10G [--apply]` do the same from the command line. Both cover every
namespace in the directory unless `--namespace NAME` selects one (`--namespace ""`
for files without a namespace); size and count limits apply to each namespace
separately, and a missing directory is an error (exit status 3). Abandoned
temporaries of crashed builders are removed once older than `temporary_grace` (one
hour) with their stripe free. Cleanup is same-host only: no NFS or cross-host
guarantee.
