# Keep connections and generations alive for their actual users

Readers use one process-local Environment per resolved generation. Requests with
the same access mode share that connection even when their tuning differs.
Readahead, map size and maximum readers follow the **first opener** until all
references close. Later acquisitions do not silently resize the map or change
readahead. Different readonly modes, or cooperative and non-cooperative opens of
one generation, fail with `ValueError`; close every reference before changing them.

## Reader lifetime: pin, refresh, close

A reader opens nothing until its first read. That read opens the generation
`CURRENT` selects at that moment and pins it: later publications do not affect the
reader. `refresh()` releases the generation so the next read follows `CURRENT`
again; the reader stays usable. `close()` releases it for good, and further reads
raise `ReaderClosedError`. A reader dropped without `close()` releases its
reference when it is garbage collected, but closing (or a `with` block) releases it
at a predictable time, which matters for deletion.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from amber_lmdb import LMDBReader, LMDBWriter, ReaderClosedError

with TemporaryDirectory() as temporary:
    path = Path(temporary) / "pinned.lmdb"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_primitive(1, key="value")
    reader = LMDBReader(path)
    assert reader.read("value") == 1
    first = reader.generation
    with LMDBWriter(path, overwrite=True, map_size=1024**2) as writer:
        writer.write_primitive(2, key="value")
    assert reader.read("value") == 1  # still pinned
    reader.refresh()
    assert reader.read("value") == 2 and reader.generation != first
    with LMDBReader(path, generation=first) as old:
        assert old.read("value") == 1  # a named generation stays pinned across refresh()
    reader.close()
    try:
        reader.read("value")
    except ReaderClosedError:
        pass
```

One reader may be shared by threads; concurrent first reads open one reference.
Pickling a reader carries its pinned generation, so worker processes read exactly
what the original reads; an unopened reader carries none, and each copy pins on its
own first read. Call `get_snapshot()` before pickling to give every copy the same
generation. Copies open their own environments; prefer the spawn start method.

`LMDBReader(path, cooperative=False)` (and `cooperative=False` on the scans) reads
stores on media nothing can change, such as read-only mounts, snapshots or archive
copies. It takes neither the coordination locks below nor LMDB's own lock file, so
it works where nothing may be written. Never use it while a writer, `set_current`
or a deletion may touch the store: nothing then protects the open generation.

## Use an exact handle for raw LMDB access

Prefer `LMDBReader` for typed reads. For raw transactions, use a context-managed
handle instead of balancing references by path.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from amber_lmdb import LMDBWriter, open_environment

with TemporaryDirectory() as temporary:
    path = Path(temporary) / "store.lmdb"
    with LMDBWriter(path) as writer:
        writer.write_primitive(1, key="number")
    with open_environment(path) as handle:
        assert handle.environment.stat()["entries"] == 2
```

A handle releases its exact generation reference even if `CURRENT` switches.
Repeated close is harmless. It cannot be pickled or used after a PID change. Do
not call `close()` directly on its raw Environment; the shared owner manages that
lifetime. `open_environment(path, generation=...)` opens a named generation.

## Retain generations, record publications, roll back

A successful replacement switches `CURRENT` but does not delete older published
generations. Process-local reference counts cannot establish whether another
process still uses an old generation, and retention lets a pinned reader keep
reading while a writer publishes. Each switch is appended to the root's `HISTORY`
file (`read_history(path)`), which orders generations for `inspect_generations`
and `prune_generations` without relying on random names or access times.

`set_current(path, name)` points `CURRENT` at an existing generation, to roll back
or to restore a lost pointer; `verify=True` runs `verify_store` on it first. It takes
the writer lock, so it fails with `BlockingIOError` while a writer is active, and it
records the switch in `HISTORY`. Open readers keep their generation.

A root whose `CURRENT` is missing is never cleaned up automatically. A writer
removes only generations that `HISTORY` proves were never published (output of a
writer interrupted before its switch); if any listed generation was published, or
the root has no `HISTORY`, the writer refuses with a hint to restore one with
`set_current` or to delete generations explicitly.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from amber_lmdb import LMDBReader, LMDBWriter, read_history, set_current

with TemporaryDirectory() as temporary:
    path = Path(temporary) / "rollback.lmdb"
    for value in (1, 2):
        with LMDBWriter(path, overwrite=value > 1, map_size=1024**2) as writer:
            writer.write_primitive(value, key="value")
    history = read_history(path)
    assert history is not None and [item.sequence for item in history] == [1, 2]
    set_current(path, history[0].generation, verify=True)
    with LMDBReader(path) as reader:
        assert reader.read("value") == 1
```

## Delete only explicitly, cooperatively, on one host

Every shared native Environment holds a same-host shared generation OS lock until
its last exact reference closes. A retained root gate serializes acquisitions,
publication and deletion. Live, idle or paused holders keep their locks; process
death releases them. No PID, TTL, stale record or expiry authorizes removal.

`delete_generations(path, names)` accepts only explicit recognized names;
`prune_generations(path, keep_last=N)` selects all but the newest `N` non-current
generations (by `HISTORY`). Both default to a dry run that creates and removes
nothing. Applying requires `dry_run=False, same_host=True`, affirming that every
access cooperates through this library on the same host. Current or in-use
candidates remain untouched, and a generation directory holding unexpected files
is reported as `unexpected_files` and kept. Eligible candidates move into private
`.trash` before removal, with their exclusive generation lock held; repeating an
explicit request resumes an interrupted removal. Once a generation is gone from
both `versions` and `.trash`, its coordination lock file is removed too, under the
root gate: no acquisition can reach a generation that no longer exists, so the
file cannot be in use. No automatic GC is performed.

Raw noncooperating LMDB opens, older clients, `cooperative=False` readers, NFS,
cross-host use and malicious filesystem changes are outside this deletion
guarantee. Do not delete if such users might exist.

`LMDBSnapshot` identifies the opened generation and observed transaction ID; it is
not a retained read transaction. Raw writable handles can modify a generation, so
derived caches and multi-page reads must check identities.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from amber_lmdb import LMDBReader, LMDBWriter, delete_generations, prune_generations

with TemporaryDirectory() as temporary:
    path = Path(temporary) / "replace.lmdb"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_primitive(1, key="value")
    reader = LMDBReader(path)
    old = Path(reader.get_snapshot().generation).name
    with LMDBWriter(path, overwrite=True, map_size=1024**2) as writer:
        writer.write_primitive(2, key="value")
    assert delete_generations(path, [old])[0].status == "in_use"
    assert reader.read("value") == 1
    reader.close()
    assert [row.name for row in prune_generations(path, keep_last=0)] == [old]
    result = delete_generations(path, [old], dry_run=False, same_host=True)
    assert result[0].status == "deleted"
```

`amber-lmdb delete-generations PATH NAME...` (or `--keep-last N`) previews the same
checks and exits with status 1 when a candidate is refused. Applying additionally
requires both `--apply` and `--same-host`. A dry-run result is a transient
observation; the actual operation always rechecks locks and `CURRENT`.
Coordination files require caller-managed permissions for all local cooperating
users. Storage creates `.coordination` on first use, so the root must permit
creation or have these files provisioned with suitable permissions. Never remove
them by hand while users may exist. Do not fork while writers or
acquisition/publication/deletion operations are active.

Filesystem locking and memory-mapping semantics remain the caller's responsibility.
Retaining generations and sharing process-local connections do not establish NFS
safety, cross-host coordination or a performance guarantee.
