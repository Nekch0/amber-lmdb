# amber-lmdb

Store Python values and NumPy arrays by key in LMDB. Every write builds a complete
new generation and publishes it with an atomic `CURRENT` switch; readers stay on
the generation they opened until they ask for a newer one.

Python 3.12 or later is required.

## Install

```console
pip install amber-lmdb
amber-lmdb --help
```

The base installation depends on LMDB, NumPy, MessagePack and frozendict, plus
Rich, Textual and pyperclip for the human-oriented views; the storage API never
imports the latter three. It does not install or import PyTorch. Install `amber-lmdb[torch]` for tensor conversion
and `amber-lmdb[dtypes]` for NumPy extension dtypes that require ml-dtypes.

## Concepts

- **Dataset root**: the directory you pass as `path`. It holds a `FORMAT` marker,
  the `versions/` directory with one subdirectory per generation, the `CURRENT`
  pointer, the `HISTORY` log and `.coordination` lock files.
- **Generation**: one complete LMDB database (`versions/v-<32 hex>`). A writer
  stages a new generation beside the root and publishes it by rewriting `CURRENT`;
  nothing is ever modified in place. `HISTORY` records every switch in order.
- **Pinning**: a reader opens the generation `CURRENT` selects at its first read and
  keeps reading it, even after newer publications, until `refresh()`. Old
  generations stay on disk until you delete them explicitly.
- **Keys**: a string, a path (stored in POSIX form) or a tuple of strings, plus an
  optional index in `[0, 2**64)`. `"a"` and `("a",)` are different keys. All
  entries of one key with indices form a sequence you can count and list.
- **Writer tags and typed readers**: each entry records which writer stored it, and
  each reader method accepts only matching tags, so a value comes back with the type
  it was written with. `read` chooses the decoder from the tag.
- **Integrity envelopes**: every entry has a digest bound to its key, tag and
  length. Reads verify bodies before decoding; `verify_store` checks everything.

## Write and read

```python
from pathlib import Path
import numpy as np
from amber_lmdb import LMDBReader, LMDBWriter

path = Path("example.lmdb")
with LMDBWriter(path) as writer:
    writer.write_dict({"name": "example", "items": 2}, key="meta")
    writer.write_ndarray(np.arange(8, dtype=np.float32).reshape(2, 4), key="features", index=0)
    writer.write_primitive(3, key=("session", "count"))
    # A successful exit commits the remaining writes and publishes the generation.

with LMDBReader(path) as reader:
    array = reader.read_as_ndarray("features", index=0)
    records = reader.read_as_dict_many(["meta", "meta"])
    snapshot = reader.get_snapshot()
```

Writers take the value first and the key and index as keywords (`key=`, `index=`),
so a value and a key can never be swapped silently. Readers take the key first.
`commit()` is optional; it only bounds the size of one transaction. Batch reads copy
owned bytes during one bounded transaction and decode after it closes; single reads
also close their transactions before returning.

## Supported values

| Writer | Values | Readers |
|---|---|---|
| `write_primitive` | `bool`, `int` (any size), `float`, `complex` | `read_as_primitive`, `read_as_any` |
| `write_str` | `str` (UTF-8) | `read_as_str`, `read_as_str_many` |
| `write_sequence` | `list`, `tuple` of supported values | `read_as_list`, `read_as_tuple`, `read_as_any` |
| `write_set` | `set`, `frozenset` | `read_as_set`, `read_as_frozenset`, `read_as_any` |
| `write_dict` | `dict`, `frozendict` | `read_as_dict`, `read_as_frozendict`, `read_as_any` |
| `write_any` | the above nested in each other (arrays too), `bytes`, `None`, built-in types, registered classes | `read_as_any` |
| `write_ndarray` | NumPy arrays of any non-object dtype, optionally chunked or compressed | `read_as_ndarray`, `read_as_tensor`, `read_array_rows` |
| `write_tensor` | PyTorch tensors (CPU copy; device and gradients are dropped) | `read_as_tensor` |
| `write` | chooses one of the writers above from the value's type | `read` |

MessagePack values (everything except strings and arrays) are limited to 64 MiB
encoded, 1,000,000 nodes and nesting depth 100 when written, so whatever is stored
can be read back. NumPy scalars are refused by the value writers with a hint
(`.item()` gives a Python number); `write_ndarray` stores them as 0-d arrays.
Masked arrays are refused because their mask would be lost. Typed reads never
convert: a stored tuple is not returned by `read_as_list`.

Arrays come back read-only, owning the bytes copied from LMDB; pass
`writable=True` for a writable array without a second copy. A `dtype=` conversion
follows NumPy's `"same_kind"` casting by default, so float-to-integer truncation
needs `casting="unsafe"`. `write_ndarray(..., compression="zlib" | "lzma" | "zstd")`
compresses values (zstd needs Python 3.14); reads decompress transparently.

## Update a store

| Goal | How |
|---|---|
| Replace everything | `LMDBWriter(path, overwrite=True)` writes a new generation from scratch |
| Change, add or delete some entries | `LMDBWriter(path, base="current")` copies the current generation, then `write_*`, `delete`, `delete_key` |
| Add new features without rewriting | write a separate store and read both with `LMDBOverlayReader([base, features])` |
| Fold layers into one store | `merge_stores([base, features], destination)` copies entries as stored |
| Roll back | `set_current(path, "v-...")` or `amber-lmdb set-current PATH NAME` |
| Free disk space | `prune_generations(path, keep_last=N)` or `delete_generations(path, names)`; dry runs by default |

Existing readers keep their generation through all of these. Deletion requires
`same_host=True` (CLI `--apply --same-host`): it is safe only when every access goes
through this library on one host, and it refuses generations that a reader holds.
A writer lock rejects a second concurrent writer to the same root, and `CURRENT`
is never removed automatically: a root that lost `CURRENT` is refused until you
restore it with `set_current`.

## Readers, processes and threads

A reader opens nothing until its first read, then pins that generation (see
Concepts). `refresh()` releases it so the next read follows `CURRENT`; `close()`
releases it for good, and a closed reader raises `ReaderClosedError`. A reader
dropped without closing releases its reference when it is garbage collected.

Readers can be shared by threads. Pickling a reader, as `DataLoader` workers do,
carries its pinned generation, so every copy reads the same data; call
`reader.get_snapshot()` before handing an unopened reader to workers so they all
pin one generation. Copies open their own environments. See
[PyTorch data loading](https://github.com/Nekch0/amber-lmdb/blob/main/guides/pytorch.md).

Within a process, readers of one generation share one LMDB environment; its tuning
(map size, readahead) follows the first opener. `LMDBReader(path, generation=...)`
reads a named generation. `cooperative=False` reads stores on media nothing can
change, such as read-only mounts or snapshots, without lock files; never use it
while a writer or a cleanup may touch the store.

## Choose an API for the task

| Task | API | Boundary |
|---|---|---|
| Restore many values | `read_many`, `read_as_*_many`, `iter_read_pages` | Up to 2**20 items; byte budget 64 MiB by default, `max_bytes=None` removes it |
| Bound a single read | `read(..., max_bytes=)` and every `read_as_*` | Refused before copying when the body is larger |
| Inspect array shapes/dtypes | `read_array_headers_many` | No body copies; repeated headers share parsing within a call |
| Read row windows of large arrays | `read_array_rows`, `read_array_rows_many`; `write_ndarray(..., chunk_rows=)` | Chunked arrays copy and verify only touched chunks |
| Feed PyTorch | `read_as_tensor`, `read_as_tensor_many` (tensors and arrays), `as_tensor` | Writable tensors without the read-only warning |
| Construct indexed records | `RecordTable` | A sequence of rows `0..count-1`; optional factory and schema |
| Check mapping fields | `ValueSchema`, `RecordSchema` | Exact types or explicit shape/dtype/range/optional rules; no coercion |
| Share a derived result | `DerivedIndexCache.get_or_build`, `lookup` | Caller identity/codec; `CacheResult.hit` distinguishes a stored `None` |
| Keep a cache directory bounded | `DerivedIndexCache.prune`; CLI `cache-list`, `cache-prune` | Dry run by default; busy entries kept |
| Bound decoding of untrusted values | `LMDBReader(..., value_limits=ValueLimits(...))`, `max_decoded_nodes=` | Lowers MessagePack limits only; arrays unaffected |
| Verify large batches on idle cores | `LMDBReader(..., verify_threads=N)` | Bodies and chunks of at least 64 KiB; off by default |
| Read cold batches from network storage | `LMDBReader(..., read_threads=N)` | Fetches the bodies of a batch concurrently; off by default |
| Check, count or list keys | `contains`, `count`, `index_range`, `iter_indices`, `iter_keys` | Integrity metadata only; listing costs one seek per logical key |
| Read several stores as one | `LMDBOverlayReader` | Nothing copied; an entry in more than one layer raises `OverlayConflictError` |
| Manage generations | `inspect_generations`, `set_current`, `prune_generations`, `delete_generations`, `read_history` | Deletion is explicit, cooperative and same-host |
| Diagnose a store | `inspect_store`, `verify_store`, `verify_physical` | Sampled, typed-value and physical coverage are different |
| Identify or compare content | `fingerprint_store`, `diff_stores` | Integrity envelopes only; bodies are read only to resolve algorithm changes |
| Summarize a generation at write time | `LMDBWriter(..., manifest=True)`, `reader.manifest()` | Opt-in; one reserved dictionary entry, excluded from fingerprints |
| Follow or stop long scans | `progress=`, `cancel=` on scans | Called between pages; a cancelled `verify_store` report is never `ok` |

Scans (`verify_store`, `array_statistics`, `inspect_store`, `fingerprint_store`,
`diff_stores`) also take `generation=` and `cooperative=`.

## Errors

Every library error derives from `AmberLMDBError` and also from the built-in
exception it refines, so existing `except KeyError:` code keeps working:
`KeyNotFoundError` (`KeyError`, with `.key` and `.index`), `ReaderMismatchError`
(`ValueError`, with the stored writer tag and the readers that accept it),
`StoreNotFoundError` (`FileNotFoundError`), `ReaderClosedError`, `ReadLimitError`,
`IntegrityError`, `OverlayConflictError`, `SourceChangedError` and
`OperationCancelled`. Messages name keys in their decoded form, such as
`key ('s01', 'pitch') index 3`.

## Command line

```console
amber-lmdb list ./stores
amber-lmdb inspect ./example.lmdb --meta --entries 20
amber-lmdb doctor ./example.lmdb                 # quick sample
amber-lmdb doctor ./example.lmdb --full --progress
amber-lmdb stats ./example.lmdb
amber-lmdb generations ./example.lmdb
amber-lmdb set-current ./example.lmdb v-<32 hex>
amber-lmdb delete-generations ./example.lmdb --keep-last 2   # preview
amber-lmdb fingerprint ./example.lmdb --generation v-<32 hex>
amber-lmdb diff ./example.lmdb ./example.lmdb --before-generation v-<32 hex>
amber-lmdb manifest ./example.lmdb
amber-lmdb show ./example.lmdb                   # rich text for people
amber-lmdb browse ./stores                       # interactive explorer
python -m amber_lmdb --version
```

Every command except `show` and `browse` prints one JSON document. Exit codes: 0 success; 1 the command ran
and found something (problems, differences, refused deletions, busy cache entries);
2 invalid usage; 3 the command could not complete (missing or unreadable store,
invalid layout, I/O error), with `{"error": {"type", "message"}}` on stdout and the
message on stderr; 130 interrupted. `--read-only-media` and `--generation` select
the reading mode and generation of the reading commands.

`doctor` without `--full` checks the layout and a bounded sample of keys, tags and
headers and reports `sample_ok`; it is **not** a full check. `doctor --full` visits
every entry, verifies checksums and decodes values within the byte limits, and
reports `ok` and `complete`. `--max-issues` caps listed issue details (with their
decoded keys), never the number of checked entries. Pass the previous `next_after`
token to `inspect --after` for another page; each call opens a fresh transaction.
Metadata with non-string dictionary keys is shown as a `mapping` list of pairs;
arrays are summarized by type, shape and dtype. The CLI never runs custom decoders.

`show` and `browse` are the human-oriented views of `amber_lmdb.ui`, which packages
built on amber-lmdb customize (store profiles, Japanese text, a dataset browser and a
ready-made `doctor`/`dataset`/`paths`/`features` command line); see
[Command-line views and explorers](guides/command-line-views.md).

The same inspection is available in Python:

```python
from amber_lmdb import inspect_store

info = inspect_store("example.lmdb", with_meta=True, entry_limit=20)
# Select another unindexed dictionary key with metadata_key="description".
print(info.snapshot, info.data_entries, info.samples)
```

`data_entries` excludes the integrity database reference that `entries` (LMDB's
own count) includes. `used_bytes` counts allocated pages of the inspected
generation; it is not the total disk footprint.

## Backend engines and physical checks

py-lmdb 3 is required. New generations default to LMDB 1.0; pass
`LMDBWriter(..., lib_version=0)` for 0.9. Readers detect the existing file's engine.
`get_backend_info()` / `amber-lmdb backend` report available engines, and
`reader.get_lib_version()` reports the one actually opened.

`verify_physical(path)` / `doctor PATH --physical` use the upstream offline Python
page verifier without opening the native engine. Use a private immutable copy;
lock files are not verified. `LMDBWriter(..., verify_on_publish=True)` optionally
checks its exclusively owned staging data before publication. See
[py-lmdb 3](https://github.com/Nekch0/amber-lmdb/blob/main/guides/lmdb-3.md) for engine formats, trust boundaries and cost.

## Define records and named fields

Use `ValueSchema` / `infer_schema` / `validate_schema` for exact field types.
`RecordSchema` / `FieldSpec` add nullable/optional fields, exact-type unions,
shape/dtype/range/length constraints and trusted cross-field validation; invalid
specifications, such as a NumPy scalar type or an unknown dtype name, fail when the
schema is built. `ValueView` reads declared fields under a namespace; `RecordTable`
is a read-only sequence of indexed dictionaries (`len`, `table[i]`, slices,
iteration in bounded pages) with an optional factory applied after transactions
close. Neither API introduces padding, filtering, dataset session conventions or RNG.

`DerivedIndexCache` shares caller-built results under an explicit identity: a
str-keyed mapping of JSON values or a dataclass such as `LMDBSnapshot`; tuples and
non-string keys are refused because JSON would change them. Builders share a fixed
set of lock stripes whose count is recorded in the cache directory, and `prune`
removes invalid, unwanted, expired or excess files safely.

See [bounded values and schemas](https://github.com/Nekch0/amber-lmdb/blob/main/guides/bounded-values-and-schemas.md) for byte-aware
streaming, field constraints and checksum coverage, [records and
indexes](https://github.com/Nekch0/amber-lmdb/blob/main/guides/records-and-indexes.md) for runnable invoice examples, and
[connection lifetimes](https://github.com/Nekch0/amber-lmdb/blob/main/guides/connection-lifecycle.md) for raw `open_environment`
handles, generation retention and deletion.

## Storage format compatibility

This library has no dependency on a dataset framework or on its consumer packages.
Applications supply their own schemas, keys, constructors and cache identities.

The store marker is `amber-lmdb-v1\n`, with key/tag/length-bound integrity
envelopes. New entries use BLAKE2b-256 (envelope version 2) by default;
`LMDBWriter(..., checksum="sha256")` writes SHA-256 (version 1). Readers verify
both, per entry. `LMDBReader(path, verify_checksums=False)` skips only the body
digest for one reader, for stores whose operator takes responsibility that they are
undamaged and unaltered; tags, lengths and decoders are still checked, and
`verify_store`/`doctor` still verify. The historical `mmfabric-lmdb-v2` marker is
rejected explicitly; there is no automatic migration. This library reads its own
typed format, not arbitrary LMDB layouts.

Compressed arrays use array format version 3; readers older than this release
refuse them with an "unsupported array version" error instead of misreading them.
Uncompressed arrays keep version 2. `HISTORY` is a JSON-lines file appended after
each `CURRENT` switch; roots written before it existed are read normally and gain
one at their next publication.

Tuple keys whose first component is `"amber-lmdb"` are reserved: the optional
manifest and the chunks of arrays written with `chunk_rows` live there. Writers refuse
them for user data, and listings, fingerprints, diffs and overlay checks skip them.
A chunked array's header entry lists its chunks' digests, so its own envelope covers
the whole array and each chunk must carry exactly the listed digest.

Custom codecs require explicit registration (`register_bytes_codec`, undone with
`unregister_bytes_codec`) and `allow_custom=True` when reading. The CLI never enables
custom decoding, and no module is imported from stored type names.

## Scan arrays or validate all stored values

`array_statistics(path)` scans headers and groups arrays by writer tag, dtype and
rank. Axis minima are bounds, not necessarily real sample shapes; smallest and
largest sample shapes are reported separately, with their keys.

`verify_store(path)` visits all data and tag entries, decodes values after each
short transaction, and detects malformed keys/tags/values and orphan tags. Values
over the copy limit (`max_value_bytes`, 64 MiB by default), unavailable optional
codecs and blocked custom decoders are reported as **unverified**, not certified.
`report.ok` requires complete coverage and no detected errors. Checksums detect
corruption, not authenticity. In-place updates abort a scan rather than mixing
transactions. See [scanning stores](https://github.com/Nekch0/amber-lmdb/blob/main/guides/scanning-stores.md).

## Operational limits

Normal LMDB locks remain enabled. No network filesystem safety guarantee is made;
the filesystem must satisfy LMDB's locking and memory-mapping requirements.
Generation deletion assumes that every access cooperates through this library on
one host; never use it for NFS or cross-host access. Metadata can contain
transcripts or other restricted information; use `--meta` only when appropriate.
Record definitions, index builders, selection and application semantics belong in
the consumer.

## Development

Clone the repository and run `uv sync`. Type checking requires the optional
dependencies, so use `uv sync --all-extras` before those checks. Keep a separate
base test run without PyTorch.

```console
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv build
```

Tests with PyTorch skip when the optional dependency is absent. CI tests Python
3.12, 3.13 and 3.14 without PyTorch, and tests the tensor extra separately.

Use the [synthetic benchmark](https://github.com/Nekch0/amber-lmdb/blob/main/benchmarks/README.md) to compare batch sizes and
engines without touching existing stores.

See [CONTRIBUTING.md](https://github.com/Nekch0/amber-lmdb/blob/main/CONTRIBUTING.md) for compatibility and release checks.
