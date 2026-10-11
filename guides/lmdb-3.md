# Use py-lmdb 3 engines and offline verification

py-lmdb 3.0.0 ships LMDB 0.9.36 and LMDB 1.0.2 in its bundled build.
The Python binding version is separate from the native engine version and from
this library's typed-value format. LMDB 0.9 uses native data format v1, while
LMDB 1.0 uses native data format v3. Older engines cannot read the other's format.

## Select an engine when building a new generation

`LMDBWriter` defaults to `lib_version=1`, selecting LMDB 1.0. Set `lib_version=0`
for LMDB 0.9. The selection is explicit, unaffected by the process-wide
`LMDB_DEFAULT_LIB_VERSION` setting, and applies only to the new staging generation.
Existing generations are not converted. Readers detect their opened file using
a bounded Python header probe and pass the detected selector to py-lmdb. This
works around the 3.0.0 CPython auto-probe's `fopen` failure on Windows Unicode paths;
it does not substitute for validating the file.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from amber_lmdb import (
    LMDBReader,
    LMDBWriter,
    get_backend_info,
    inspect_store,
    open_environment,
    verify_physical,
)

assert get_backend_info().binding_version.startswith("3.")
with TemporaryDirectory() as temporary:
    path = Path(temporary) / "numbers.lmdb"
    with LMDBWriter(path, lib_version=1, verify_on_publish=True) as writer:
        writer.write_primitive(42, key="answer")
    with LMDBReader(path) as reader:
        assert reader.get_lib_version()[0] == 1
        assert reader.read_as_primitive("answer") == 42
    with open_environment(path) as handle:
        assert handle.environment.dbs()
    assert inspect_store(path).lib_version[0] == 1
    assert verify_physical(path).ok
```

A system-library build may supply only one engine. `get_backend_info()` reports
available versions; select `lib_version=0` explicitly if it only supplies 0.9.
Requesting an unavailable engine fails instead of silently substituting another.
Normal locks and non-writemap operation remain enabled. Read transaction caching
is explicitly disabled with `max_spare_txns=0` for consistent fork/lifetime behavior.
Writer completion uses `sync(force=True)` before closing and publishing.

LMDB 1.0 includes a trailing NUL in main-database named-DB descriptors. Typed
inspection and full/header scans handle both descriptor encodings. This is not an
extra application value. `inspect_store().lib_version` reports the actual engine.
The new `Environment.dbs()` lists named databases through an exact public handle;
it returns normalized names rather than raw main-database descriptor keys.

## Check physical pages before opening an untrusted file

```console
amber-lmdb backend
amber-lmdb doctor /path/to/private-copy.lmdb --physical
```

`verify_physical` uses py-lmdb 3's pure-Python byte walker. It never opens the file
with the native engine. It checks native page structure, tree ordering/counters,
free-page relationships and dirty-state invariants. A typed root selects CURRENT;
a raw LMDB directory or a single data file is also accepted. `report.ok` requires
no problems and no unsupported/invalid-input error. Detail limits do not bound
upstream verifier memory or execution time.

Use a private immutable copy, not a file an untrusted writer can replace during
or after verification. The wrapper's before/after identity/size/mtime comparison
only detects ordinary changes. The verifier supports native-byte-order 64-bit
files, not portable arbitrary architectures. It does not validate a supplied lock
file. Before later opening an inactive private copy, discard an untrusted supplied
lock file and let LMDB recreate it. Never remove live readers' lock files.

The CLI keeps `--physical` separate from `--full`, so an offline check cannot
silently open a native environment afterward. Physical validity does not establish
typed-value validity, numeric authenticity or safe concurrent writers. Use the
existing `doctor --full` only after the file and its later writers are trusted.
No repair, encryption setup or filesystem/NFS safety guarantee is provided.

## Verify private output before publication

`verify_on_publish=True` performs physical verification after commits, forced sync,
environment closure and staging fsync, but before switching CURRENT. Failed
verification leaves the existing published generation unchanged and removes the
unpublished staging output. This optional whole-file pass is off by default
because it may be expensive for very large stores. The staging directory is
exclusively managed by the writer; the check does not create a long read transaction.

Native `lmdb.Error` subclasses and the 3.0 `what`, `code`, `reason` attributes pass
through normal storage calls. The binding also supplies fixes for transaction
abort deadlocks, parent/child lifetime handling and CFFI cursor closure. No nested
transaction or asynchronous API is introduced here; lifetime guarantees still
require this library's explicit connection ownership and process coordination.

References: [3.0.0 changelog](https://github.com/jnwatson/py-lmdb/blob/py-lmdb_3.0.0/ChangeLog)
and [versioned upstream documentation](https://github.com/jnwatson/py-lmdb/blob/py-lmdb_3.0.0/docs/index.rst).
