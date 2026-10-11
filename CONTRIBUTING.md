# Contributing

Run `uv sync` to install the base package and development tools. The base test
suite must run without PyTorch. Test tensor support separately with
`uv sync --all-extras`, then run the suite again.

Before a change is ready:

```console
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pyright --verifytypes amber_lmdb --ignoreexternal
uv lock --check
uv build
```

The type checks above need the optional dependencies installed. Write docstrings,
comments, diagnostic messages and project documentation in English. Keep Unicode
sample data in tests and compatibility fixtures unchanged.

Public APIs must have annotations and documentation. Stored dictionary contents
are intentionally opaque and use `Any`; omitted generic arguments are not an
acceptable substitute.

Keep read transactions bounded and close them before decoding owned bytes. Do not
turn off LMDB locks, add global readahead changes or copy array bodies in
header-only inspection. Whole-value validation must be explicit and honor its
body/integrity-metadata copy budgets; unavailable or blocked codecs remain unverified.
Checksum verification must finish after transaction closure and before decoders; only
a reader built with `verify_checksums=False` may skip the digest, never its tag and
length checks, and store-level verification must not honor that option. Header-only
operations must not claim full-body integrity. Cooperative deletion requires retained
shared OS locks for native environment lifetimes and a gate around acquisition,
publication and retirement. Never replace or unlink the root gate or a lock file that
a live generation may use; a generation's own lease file is removed only by
deletion, under the root gate, after the generation has left both `versions` and
`.trash`, so no acquisition can reach it. Dry runs create and remove nothing.

Generation history is a safety contract. `HISTORY` is appended after every `CURRENT`
switch (publication and `set_current`), and a writer never removes a generation that
`HISTORY` lists: a root without `CURRENT` is refused with a hint, never repaired by
deletion; only generations absent from an existing `HISTORY` count as unpublished.
Readers pin the generation of their first read; `refresh()` releases it, `close()`
is final, a dropped reader releases its reference through a finalizer, and pickling
carries the pinned generation (an explicit `generation=` survives `refresh()`).
Lazy opening must stay safe for threads. These are compatibility contracts, not
implementation details. `cooperative=False` must remain read-only and must never
create or take coordination locks or LMDB lock files.

Key listing, counts, fingerprints and diffs read integrity metadata only; keep them
free of body copies. An overlay must never return or count an entry that exists in
more than one layer. Tuple keys starting with `"amber-lmdb"` are reserved for the
manifest `("amber-lmdb", "manifest")` and array chunks: writers refuse them for user
data, and fingerprints, diffs, listings and overlay conflict checks exclude them, so
neither a manifest nor a chunk changes content identity on its own. A chunked header
must list every chunk's envelope digest; readers compare each chunk's envelope with
it even when `verify_checksums=False`, writers remove a chunked array's chunks
whenever they replace or delete it, and `verify_store` checks the links both ways. A
derived writer copies its base while holding the base's shared lease.

Cache builders lock one of a fixed number of stripe files in `<directory>/.locks`;
never unlink them. The stripe count is recorded once in `.locks/STRIPES` and every
cache on the directory must use it; a different explicit count is an error, never a
silent remapping. Cache identities are validated and canonicalized before hashing;
tuples and non-string keys are refused rather than converted. `prune` removes an entry, or an abandoned temporary, only while
holding its stripe without waiting, and a builder publishes only under its stripe, so
a busy entry is skipped rather than removed. A lock timeout, a cooperative `cancel`
and process isolation are different stops; none may publish partial output.
`ValueLimits` may only lower the built-in decoding limits, and a node budget must
raise rather than truncate. The typed-value decoder must keep the previous
implementation's results and error messages; `tests/test_value_limits.py` compares
them on valid and mutated nodes. Threaded digest verification must report the first
corrupt entry in request order, like sequential verification. Progress callbacks run between pages, outside transactions, and a
cancelled report must never be `ok`. Tensor bodies are copied once into a private
bytearray that the returned tensor owns; never let a tensor alias LMDB memory or a
caller's buffer. The same holds for arrays read with `writable=True` and for
`read_as_tensor` on stored arrays.

Compressed NumPy arrays use array format version 3 and name their codec in the
header metadata; uncompressed arrays keep version 2, so older readers refuse
compressed arrays instead of misreading them. A compressed chunked header lists every
chunk's stored size before the digests. Decompression must be bounded by the size the
header implies and must reject any other outcome; integrity digests cover the stored
(compressed) bytes.

Value writers take the value first and the key and index only as keywords. The
command line prints one JSON document per command and keeps its exit codes: 0
success, 1 findings, 2 usage errors, 3 the command could not complete (with an
`{"error": ...}` document), 130 interrupted.

The frozen fixture in `tests/fixtures/mmfabric_0_2_2.json` was generated by the
untouched mmfabric 0.2.2 writer before extraction. Tests reconstruct a database from
its raw key/value/tag bytes without using the current writer. Do not regenerate
this fixture to make a format change pass. Legacy-format support is not a permanent
requirement. A deliberate breaking format needs a new identifier, a stated rebuild
or migration policy and separate fixtures/tests; it never authorizes automatic
conversion or removal of application stores. The current marker is `amber-lmdb-v1`;
old `mmfabric-lmdb-v2` stores are deliberately rejected. Rebuild from application
sources into a separately reviewed output. Golden envelope metadata is in
`tests/fixtures/amber_lmdb_integrity_v1.json` (SHA-256, version 1) and
`tests/fixtures/amber_lmdb_integrity_v2.json` (pinned version 1 and 2 digests).
These pinned digests are what make a store verifiable on any CPU; never regenerate
them to make a change pass, and keep every published envelope version readable. See
[formats and limits](guides/bounded-values-and-schemas.md) for the envelope contract.

This repository must not import or depend on consumers such as mmfabric or
dataset-cejc. Downstream compatibility modules must reuse this implementation,
not copy codecs or create separate environment/custom-codec registries.
The independence test blocks consumer imports while exercising the standalone API.
Frozen historical fixtures remain evidence of their original producer, rather than
being rewritten to match a new implementation.

## Releasing

The package is published on PyPI as `amber-lmdb` from
https://github.com/Nekch0/amber-lmdb. A release bumps `version` in
`pyproject.toml`, passes CI on Linux and Windows for the pushed commit, and is
tagged `v<version>`. Build the wheel and source distribution from a clean checkout
of that tag, install the wheel into a fresh environment and run the tests against
it, then upload both files and attach them to a GitHub release. A version number
is never reused, even after a deleted upload.

Consumers replace local path sources with the published dependency only after both
files are available from PyPI; do not publish consumer wheels that depend on an
unavailable version.

No application data, private metadata or generated benchmark results should be
included in a distribution or source commit.
