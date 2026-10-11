# Compare owned reads on synthetic stores

Run from the repository checkout. No mmfabric, dataset framework, Torch or extra
benchmark dependency is needed. The script creates and removes only its own private
temporary stores; it never accepts an existing store as input.

```console
uv run python benchmarks/benchmark_storage.py --output .audit/storage-baseline.json
uv run python benchmarks/benchmark_storage.py --rows 64 --array-items 8 --array-bytes 256 --repeats 2 --trials 1 --output .audit/storage-smoke.json
```

The output must be a new filename. An existing report is never overwritten.
`--temporary-directory DIRECTORY` chooses scratch placement under an existing
caller-selected directory. Use local storage initially. `--engines 1` restricts
the run to LMDB 1.0; `--engines 0` selects 0.9. Both are tested by default.

## Choose a workload before comparing results

Default data is 1024 indexed synthetic records, 1024 integers and 256 float32
arrays of 16 KiB each. Records contain strings, lists, tuples and nested mappings;
arrays vary by index to detect order errors. Requests are shuffled deterministically
and include duplicates. Record count is bounded at 100000, arrays at 4096, total
array body bytes at 128 MiB. An explicit modest writer map avoids the Windows 0.9
engine's huge logically sparse default-size file. This is not a real-size training
workload or a controlled cold-storage experiment. Array-body batches and full scans
receive explicit copy budgets large enough for the bounded fixture's duplicate
requests; each worker records `copy_limits`. Small/default cases keep 64 MiB.
These overrides are benchmark policy, not a change to library read defaults.

Cases compare single reads, generic/typed batches, RecordTable materialization,
codec-only decoding, array bodies/headers and trusted-store scans. Native reads
copy bytes but do not check writer tags, verify integrity envelopes/SHA-256, enforce
aggregate copy budgets or decode types. Their lower time measures
a different contract and is not a supported replacement for typed reads.

`--batch-sizes 32 256 1024` changes page sizes. Defaults run three fresh-process
trials per engine, five samples per case after correctness checks and warmup.
Cases run in deterministic shuffled order for each repeat. `--workers 1` keeps
trials serial; `--workers 4` overlaps independent trials and measures a different
contention condition. Do not pool serial and parallel samples.

## Add adverse shapes, large values, read-path parts and lifecycles

```console
uv run python benchmarks/benchmark_storage.py --extended --trials 5 --output .audit/storage-extended.json
```

`--extended` keeps every default case unchanged and adds to the same fixture:

- `shapes`: `--unique-items` (256) 1-D float32 arrays whose lengths are all distinct
  and average `--array-bytes`. Compare `shapes.headers/N` with `arrays.headers/N` to
  see the cost when per-request header reuse cannot help.
- `large`: `--large-items` (32) `(rows, 1024)` float32 matrices averaging
  `--large-bytes` (2 MiB), with distinct row counts like variable-length features.
  The batch case uses an explicit copy budget; `large.stream` keeps the default
  64 MiB and therefore pages by bytes.
- `<family>.parts.{lookup,copy,checksum,decode}/N` for records, arrays, shapes and
  large at the largest batch size. `lookup` performs the native metadata and body
  gets without copying, `copy` adds the owned copies inside the same transactions,
  `checksum` verifies pre-copied entries with the fixture's envelope digest, and
  `decode` decodes them. The public read minus `copy`, `checksum` and `decode` is
  location construction, metadata parsing, tag/budget checks and bookkeeping. Parts
  call private reader/integrity helpers so they follow the code path of `read_many`;
  they are diagnostics, not API.
- `<family>.trusted/N` repeats the largest-batch public read (`records.generic/N`,
  `arrays.batch/N`, ...) through a reader built with `verify_checksums=False`, with
  the same requests and copy budgets, so the ratio to the verified case shows what
  skipping the digest saves on this CPU.
- `<family>.stream` (`iter_read_pages`) and `records.dict_stream` (`iter_dict_pages`)
  with default limits; streams also take a snapshot per page.
- `lifecycle.reader_shared` opens a reader, reads one value and closes it 16 times
  while another reference holds the generation. `lifecycle.reader_first` repeats
  that after the trial's own reader and handle are closed, so every cycle acquires
  the root gate, the shared generation lease and a native environment. It is timed
  in a separate loop after the other cases, not interleaved with them.

`--checksum sha256` writes the fixture with envelope version 1 instead of the default
BLAKE2b-256 version 2; run both with otherwise identical settings to compare digests.
SHA-256 throughput depends heavily on whether the CPU has SHA instructions; record
the CPU model with any checksum conclusion. Parts are measured in isolation and do
not sum exactly to the public read: cache residency and frequency differ between a
hash over freshly copied bytes and a hash over bytes copied earlier.

## Read the JSON without overstating coverage

- `samples` records uninstrumented wall and process-CPU nanoseconds per call.
  Imports, reader setup, correctness checks and full output destruction are outside
  those call samples. Windows CPU timer granularity can give zero for short calls.
- `summary` combines per-process sample medians, not individual pooled samples.
  Inspect trial medians and all raw samples; one median can hide large variance.
- `whole_run_wall_ns` includes setup, spawn/import, correctness, all diagnostics,
  fixture construction and temporary cleanup. It is not a pure throughput measure.
- Allocation peaks use a separate tracemalloc call, never the timed call. They
  describe Python-tracked allocations, not process RSS or all native/mapped memory.
  Allocation passes can make the complete benchmark much slower than its samples.
- A separate record-read cProfile records Python self/cumulative time. Profiling
  overhead does not belong in throughput comparisons. Stage messages show progress.
- Provenance records Python/platform/packages/native engines, source and benchmark
  hashes, configuration and the actual opened generation. No page cache is cleared.
- Expected explicit transaction counts are structural predictions, not instrumented
  counts; LMDB's internal named-DB transactions are excluded.

Every case is value-checked outside its timed samples. Array checks inspect all
numeric bodies, not only shapes. No random-global state is changed. Unit tests
check the protocol and values without wall-time pass/fail thresholds.

Keep identical fixture/settings/tool hashes when comparing revisions. A format or
checksum-contract change is not an equivalent no-checksum throughput comparison. Profile the
measured workload before changing code. For a small improvement, alternate old/new
implementations in the same process, repeat in fresh processes and retain raw pairs.
Report unique-shape or other adverse cases as well as repeated-header improvements.
A stopped run is censored; it is not a completed duration or library timeout.

## Scope not yet established

This benchmark does not measure cold page cache, network filesystems, remote
coherence, production-size stores, concurrent writers, simultaneous training,
optional tensor decoding, cache-builder contention or safe generation deletion.
The extended large arrays are bounded synthetic matrices, not a real feature store.
Local timings cannot justify claims about those cases. Keep generated JSON/logs
outside source distributions and commits. Rust requires a measured remaining
bottleneck and the same ownership/format conformance tests as Python.
