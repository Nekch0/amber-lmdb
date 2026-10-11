# Bound owned reads and declare record constraints

A batch defaults to 1024 items and 64 MiB of encoded body plus integrity metadata
copies; `max_items` goes up to 2**20 and `max_bytes=None` removes the byte budget.
Single reads take `max_bytes=` too, and refuse a larger body before copying it. An array-header batch counts owned array headers and integrity
metadata, not array bodies. Duplicate requests count separately. Limits are checked
using transaction-local views before copying bodies. No mapping-backed view escapes
its transaction.

## Stream pages when the request is larger than one copy budget

Ordinary batch methods raise `ReadLimitError` if the full request exceeds its
budget. `iter_read_pages` instead splits by both items and bytes. A single entry
larger than `max_bytes` still raises the error. Empty input opens nothing.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from amber_lmdb import LMDBReader, LMDBWriter, ReadLimitError

with TemporaryDirectory() as temporary:
    path = Path(temporary) / "messages.lmdb"
    with LMDBWriter(path, map_size=1024**2) as writer:
        for index in range(4):
            writer.write_str("x" * 20, key="message", index=index)
    with LMDBReader(path) as reader:
        try:
            reader.read_many(["message"] * 4, indices=range(4), max_bytes=150)
        except ReadLimitError as error:
            assert error.kind == "bytes" and error.limit == 150
        else:
            raise AssertionError("request should exceed the copy budget")
        pages = reader.iter_read_pages(
            ["message"] * 5,
            indices=[3, 1, 1, 0, 2],
            max_bytes=150,
        )
        assert [len(page) for page in pages] == [2, 2, 1]
```

Each page closes its transaction before verifying checksums, decoding and yielding.
`iter_dict_pages` applies the strict dictionary-reader contract. `RecordTable`
uses byte-aware pages with caller-selected `max_bytes` and `batch_size`.
The opened generation and transaction ID must stay unchanged across pages. A
reader refresh or an in-place source update raises `SourceChangedError`.

These limits do not bound decoder allocations, native RSS, hashing/decoding runtime,
or the combined results a caller retains. Ordinary single-value body reads do not
have the batch's aggregate limit. Use batches/pages with an explicit budget when
that limit is needed. Full scans include integrity metadata in `max_batch_bytes`.
A scan budget too small for even bounded metadata raises `ReadLimitError`.

## Read row windows of large arrays

`read_array_rows` and `read_array_rows_many` return rows `[start, stop)` along axis
0, clamped to the row count like slicing. Writing with
`write_ndarray(..., chunk_rows=N)` stores the array as a header entry plus chunks of
`N` rows, each with its own integrity envelope; a row window then copies and
verifies only the chunks it touches. Full reads, header reads, statistics and
`verify_store` treat chunked arrays like any other array.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from amber_lmdb import LMDBReader, LMDBWriter, ReadLimitError

frames = np.arange(4096 * 64, dtype=np.float32).reshape(4096, 64)  # 256 B rows, 1 MiB
with TemporaryDirectory() as temporary:
    path = Path(temporary) / "frames.lmdb"
    with LMDBWriter(path, map_size=64 * 1024**2) as writer:
        writer.write_ndarray(frames, key="utterance", index=0, chunk_rows=512)  # 128 KiB chunks
        writer.write_ndarray(frames, key="utterance", index=1)  # unchunked
    with LMDBReader(path) as reader:
        assert reader.read_array_headers_many(["utterance"], indices=[0])[0].chunk_rows == 512
        window = reader.read_array_rows("utterance", 0, start=1000, stop=1200)
        assert np.array_equal(window, frames[1000:1200]) and not window.flags.writeable
        assert np.array_equal(reader.read_as_ndarray("utterance", 0), frames)
        windows = reader.read_array_rows_many(
            ["utterance", "utterance"], indices=[0, 1], starts=[0, 4000], stops=[10, 5000]
        )
        assert [item.shape for item in windows] == [(10, 64), (96, 64)]
        try:
            reader.read_array_rows_many(
                ["utterance"], indices=[0], starts=[0], stops=[4096], max_bytes=256 * 1024
            )
        except ReadLimitError:
            pass
        else:
            raise AssertionError("a whole 1 MiB array exceeds a 256 KiB budget")
```

Chunks of about 64-256 KiB suit windows; on a CPU without SHA instructions, a
verified window of 10 % of a 2 MiB array took about a sixth of the whole-array read,
while full reads of chunked arrays cost a few percent more than unchunked ones. An
unchunked array works too: a reader with `verify_checksums=False` copies only the
requested rows, but a verifying reader must copy and verify the whole body first.
Returned rows own their memory, not the rest of the array. Chunks live under the
reserved key namespace `("amber-lmdb", ...)`; the header lists their digests, so
fingerprints and diffs report a changed chunk as its changed array. Tensors and
arrays without dimensions are not chunked.

## Declare optional, nullable, array and numeric fields

`ValueSchema` retains its exact-name/type behavior. `RecordSchema` adds explicit
field constraints. There is no coercion, inferred default or insertion of missing
optional fields. Nullable `None` is distinct from an absent optional field.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from amber_lmdb import (
    Dimension,
    FieldSpec,
    LMDBReader,
    LMDBWriter,
    RecordSchema,
    RecordTable,
    SchemaConstraintError,
    validate_schema,
)


def check_interval(record):
    if record["end"] < record["start"]:
        raise ValueError("end must be at least start")


schema = RecordSchema(
    (
        FieldSpec("id", (int, str)),
        FieldSpec(
            "values",
            (np.ndarray,),
            shape=(Dimension(1, 8), 2),
            dtypes=("float32",),
            finite=True,
            minimum=0,
            maximum=1,
        ),
        FieldSpec("start", (int,), minimum=0),
        FieldSpec("end", (int,), minimum=0),
        FieldSpec("note", (str,), required=False, nullable=True, max_length=40),
    ),
    validator=check_interval,
)
record = {
    "id": "one",
    "values": np.ones((3, 2), dtype=np.float32),
    "start": 1,
    "end": 2,
    "note": None,
}
validate_schema(record, schema)
try:
    validate_schema({**record, "id": True}, schema)
except SchemaConstraintError as error:
    assert error.field == "id" and error.constraint == "type"
else:
    raise AssertionError("bool is not the declared exact int type")

with TemporaryDirectory() as temporary:
    path = Path(temporary) / "records.lmdb"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_dict(record, key="records", index=0)
    with LMDBReader(path) as reader:
        table = RecordTable(reader, "records", lambda row: row["id"], count=1, schema=schema)
        assert table.get(0) == "one"
```

Shape axes may be fixed nonnegative integers, unconstrained `None`, or inclusive
`Dimension` ranges. An empty shape tuple declares a scalar. Dtype strings accept
NumPy dtype representations and Tensor names with an optional `torch.` prefix.
Numeric bounds apply to each array element and reject NaN; `finite=True` also
rejects infinity. Length bounds use Python `len`, not byte length or element count.
Tensor full-value checks may synchronize the device.

Required/extra/order checks and field constraints run before the cross-field
validator. Caller callback exceptions propagate. Custom classes still require
registration and explicit schema consent; built-in validation never calls their
encoders. `ValueView` keeps its exact `ValueSchema` namespace/type contract.

## Store format and checksum coverage

The store marker is now exactly `amber-lmdb-v1\n`. Metadata lives in the named
`amber-lmdb.entries.v1` database under each logical binary key. Each envelope is:

| Bytes | Meaning |
|---|---|
| `TLMI` followed by version byte `02` or `01` | Integrity domain/version |
| One unsigned byte | ASCII writer-tag length, 1 through 63 |
| Eight big-endian unsigned bytes | Encoded body length |
| Declared tag bytes | Writer method |
| 32 bytes | BLAKE2b-256 digest (version 2) or SHA-256 digest (version 1) |

Writers use version 2 by default; `LMDBWriter(..., checksum="sha256")` selects
version 1. Readers accept both in the same store and verify each entry with the
algorithm its version names, so a store need not use one version throughout and
older generations stay readable after a default changes. Both are standard
functions of their input bytes: every CPU code path (SHA instructions, SIMD or plain
integer code) yields the same digest, so verification never depends on the reading
CPU. BLAKE2b-256 is the default because it hashes about 1.7x faster than SHA-256 on
CPUs without SHA instructions. `inspect_store` samples report each entry's
`checksum`.

The digest input is the domain/version, a four-byte big-endian binary-key length,
the binary key, tag/body lengths as above, the tag, and the entire encoded body.
Moving an envelope between keys, changing its tag/body or relabelling its version
byte invalidates the digest.
A normal body read verifies owned bytes after its short transaction and before
calling a decoder. `verify_store` also checks bounded bodies. Header-only reads,
statistics and sampled inspection do not verify complete body checksums. Optional
`inspect_store(..., with_meta=True)` does verify the copied metadata body.

`LMDBReader(path, verify_checksums=False)` turns off only the digest comparison of
that reader's body reads, including `RecordTable` and `ValueView` reads through it.
Use it only when whoever operates the store guarantees that the files are undamaged
and unaltered, for example a store verified once with `verify_store` and then kept
read-only. Writer tags, the stored length, budgets and decoder validation still
apply, so a same-length change is returned silently while a length or tag change
still fails. The option survives pickling and `refresh()`; `verify_store`, `doctor`
and inspection never use it.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from amber_lmdb import LMDBReader, LMDBWriter, verify_store

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "trusted.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_ndarray(np.arange(4, dtype=np.float32), key="features", index=0)
    assert verify_store(path).ok  # verify once, then read without per-read digests
    with LMDBReader(path, verify_checksums=False) as reader:
        assert reader.read_as_ndarray("features", index=0).tolist() == [0, 1, 2, 3]
```

Verification can also run in parallel instead of being skipped:
`LMDBReader(path, verify_threads=4)` checks bodies and chunks of at least 64 KiB in a
pool of four threads when a read has two or more of them (hashlib releases the GIL).
On a busy 14-core CPU without SHA instructions, reading 36 verified 2 MiB arrays took
0.55 of the sequential time with four threads (the copies stay sequential; hashing
alone took about a third). The default 0 verifies in the calling thread,
which suits data-loader workers that already use every core. Errors are reported
for the first corrupt entry in request order, as without threads.

## Lower decoding limits for untrusted stores

MessagePack values (dictionaries, sequences, sets, `write_any`) decode into Python
objects that peak at roughly 17-25 times their encoded size, so copy budgets do not
bound decoding memory. `ValueLimits` lowers the per-value limits (encoded bytes,
schema nodes, nesting depth) for every MessagePack read of a reader, including
`RecordTable` pages; `verify_store(..., value_limits=...)` applies them to a whole
store. `max_decoded_nodes` bounds the schema nodes a batch, or each page of a stream,
may decode and raises `ReadLimitError(kind="nodes")` beyond it. Every scalar,
container and mapping key counts as one node; a single container longer than
`max_nodes` is already rejected while unpacking ("invalid MessagePack value"). Arrays
are unaffected, and custom decoders remain caller code.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from amber_lmdb import LMDBReader, LMDBWriter, ReadLimitError, ValueLimits

with TemporaryDirectory() as tmp:
    path = Path(tmp) / "untrusted.lmdb"
    with LMDBWriter(path) as writer:
        # 1 + 1 + 1 + 200 * 3 = 603 nodes: the dict, its key, the list, 200 pairs.
        writer.write_dict({"pairs": [[i, i] for i in range(200)]}, key="document", index=0)
        writer.write_dict({"pairs": [[1, 2]]}, key="document", index=1)  # 6 nodes
    with LMDBReader(path, value_limits=ValueLimits(max_nodes=500)) as reader:
        assert reader.read_as_dict("document", 1) == {"pairs": [[1, 2]]}
        try:
            reader.read_as_dict("document", 0)
        except ValueError as error:
            assert "maximum item count" in str(error)
        else:
            raise AssertionError("603 nodes exceed max_nodes=500")
    with LMDBReader(path) as reader:
        try:
            reader.read_many(["document"] * 2, indices=[0, 1], max_decoded_nodes=605)
        except ReadLimitError as error:
            assert error.kind == "nodes" and error.limit == 605
        else:
            raise AssertionError("609 nodes exceed the batch budget of 605")
```

Python cannot enforce a byte-accurate memory limit in process; for hostile inputs,
decode in a separate process with operating-system limits.

The checksum detects corruption, not authenticity. A writer who can replace both body
and envelope can forge a matching checksum. Native physical verification and
application-schema validation remain separate checks. No checksum makes an
untrusted native LMDB file safe to open.

The historical `mmfabric-lmdb-v2` store marker is explicitly rejected. Its golden
fixture is retained unchanged as evidence. Envelope version 1 has its own golden
fixture, and pinned version 1 and 2 digests, including multi-block bodies, are in
`tests/fixtures/amber_lmdb_integrity_v2.json`.
There is no implicit migration or legacy fallback. Rebuild old stores from their
source with an application-specific, reviewed process into a separate path before
switching consumers. Merely rewriting `FORMAT` is not conversion. Never rewrite or
remove a production store based on this guide's temporary examples.
