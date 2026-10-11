# Read headers, summarize arrays, or validate a store

These APIs work without a consumer framework. Header operations do not import
PyTorch; full tensor decoding requires the optional tensor dependency.

## Read several headers and summarize variable shapes

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from amber_lmdb import LMDBReader, LMDBWriter, array_statistics, verify_store

with TemporaryDirectory() as temporary:
    path = Path(temporary) / "measurements.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_ndarray(np.zeros((2, 8), dtype=np.float32), key="a", index=0)
        writer.write_ndarray(np.ones((4, 3), dtype=np.float32), key="b", index=0)
    with LMDBReader(path) as reader:
        headers = reader.read_array_headers_many(["b", "a", "b"], indices=[0, 0, 0])
    assert [header.shape for header in headers] == [(4, 3), (2, 8), (4, 3)]
    assert headers[0].payload_bytes == 48
    stats = array_statistics(path, batch_size=1)
    group = stats.groups[0]
    assert group.min_shape == (2, 3)
    assert group.smallest_shape == (4, 3)
    assert group.largest_shape == (2, 8)
    assert group.count == 2
    assert verify_store(path, batch_size=1).ok
```

The axis-wise minimum `(2, 3)` does not appear in this store. The smallest actual
sample has 12 elements and shape `(4, 3)`. Statistics group arrays by writer tag,
dtype and rank, so different ranks never merge into misleading shape bounds.
Equal-size extrema select the first sample in binary key order. Scalar and empty
arrays count as samples. Only top-level array/tensor writer tags participate;
arrays embedded inside dictionaries are not included in header-only statistics.

Header batches take up to 1024 items by default and `max_items=` up to 2**20.
Metadata has a per-entry 64 KiB default limit, adjustable on
`read_array_headers_many`. Statistics use that default limit, short pages of at
most 1024 entries, and a default maximum of 1024 groups. More
groups or malformed entries raise `ValueError` instead of silently truncating.
Header memory scales with page size and the header limit, not array body size.

Tensor names unknown to this version remain inspectable with `itemsize=None`.
Their payload lengths cannot be certified by header inspection alone. Array
statistics do not authenticate numeric contents, verify full-body checksums or
verify non-array bodies.

## Validate every key, tag and value

These operations open the native engine and expect trusted files. For a received
or untrusted database, first use the [offline page verifier](lmdb-3.md) on a private
immutable copy. Read-only opening and disabled custom callbacks do not make an
untrusted native file safe. Supplied lock files also require separate handling.

```console
amber-lmdb doctor measurements.lmdb --full --batch-size 512 --max-issues 20
amber-lmdb stats measurements.lmdb --max-groups 1024
```

`verify_store` visits all entries in the opened generation and all writer tags.
It checks malformed keys, missing/unknown tags, array headers and payload lengths,
codec validity, key/tag/length-bound checksums, and tags pointing to missing values. It copies bounded payloads
inside short transactions and runs decoders afterward. No read transaction stays
open between pages. An in-place transaction-ID change raises `SourceChangedError`;
a new `CURRENT` generation does not change the already pinned scan.

`max_value_bytes` (64 MiB by default) is the largest body that is verified; larger
values are reported unverified. Pages are sized to hold such a value by default
(`max_batch_bytes=None`: 64 MiB, or `max_value_bytes` plus 1 MiB if larger), so
raising `max_value_bytes` alone verifies larger arrays. An explicit
`max_batch_bytes` too small even for bounded integrity metadata raises
`ReadLimitError`, rather than silently exceeding the copy budget. These are copy
limits, not limits on allocations or runtime inside a decoder. Issue details are
capped by `max_issues` (CLI `--max-issues`), while total error/skipped counts
continue to accumulate; each issue carries the decoded `key` and `index` of its
location.
`report.unverified_by_reason` counts every unverified value by reason (for example
`"optional dependency unavailable: torch"` or `"custom decoding not allowed"`) and
`report.entries_by_tag` counts checked entries by writer tag; `doctor --full` prints
both. `value_limits=` applies lowered MessagePack decoding limits to the scan.

`report.ok` requires zero errors and zero skipped values. `report.complete` reports
coverage, not success. Missing optional packages, oversized values and blocked
custom decoders prevent complete coverage. Full CLI scans never enable custom
callbacks. Python callers may explicitly set `allow_custom=True`; registered
callbacks then run and must validate their inputs. Never enable this for untrusted
stores. Application schemas and business rules require separate validation.

`verify_store`, `array_statistics`, `inspect_store`, `fingerprint_store` and
`diff_stores` accept `generation="v-..."` to scan a named generation instead of
`CURRENT`, and `cooperative=False` for stores on media nothing can change (CLI
`--generation`, `--read-only-media`).

Checksums detect numeric body changes, including same-length changes, but are not
proof of authenticity. A writer can replace both value and checksum. This validator
neither repairs physical LMDB pages nor establishes NFS safety. Ordinary `doctor` without `--full` still checks only a bounded sample.

## Inventory generations without deleting them

`inspect_generations(path)` and `amber-lmdb generations PATH` report the generation
`CURRENT` selects (or why it cannot be read), every recognized generation in
publication order with its `HISTORY` sequence and publication time, logical
data/lock-file sizes, the modification time of `data.mdb`, and damaged generations
with an `error` instead of failing. They also list generations left in `.trash` by an
interrupted deletion, unfinished staging directories beside the root, and the
recorded history. `list_generations(path)` returns just the generations. These are
filesystem observations, not live-data size or allocated disk usage. A CURRENT
change during the inventory raises `SourceChangedError`.

Non-current directories may still be used by another process. The inventory
grants no permission to delete them: `delete_generations` (explicit names) and
`prune_generations(path, keep_last=N)` (all but the newest N non-current ones) use
same-host cooperative lifetime locks and are dry runs by default; the CLI equivalent
is `amber-lmdb delete-generations PATH NAME...` or `--keep-last N`. See
[connection lifetimes](connection-lifecycle.md) for prerequisites and recovery.

Long scans accept `progress` and `cancel`; content fingerprints, generation diffs
and key listings are described in
[combining and tracking stores](combining-and-tracking-stores.md).
