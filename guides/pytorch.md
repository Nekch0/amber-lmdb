# Feed PyTorch from a store

amber-lmdb never imports PyTorch on its own; install the `torch` extra to use the
tensor functions. A store opened by a `Dataset` behaves well under `DataLoader`
without a `worker_init_fn`: readers open lazily, can be pickled into workers and
pin one generation, so every worker reads the same data even while a writer
publishes a new generation.

## A map-style dataset

```python
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from amber_lmdb import LMDBReader, LMDBWriter


class Features(Dataset):
    """One tensor per index of the key "features"."""

    def __init__(self, path):
        self.reader = LMDBReader(path)
        self.reader.get_snapshot()  # pin one generation for this process and all workers
        self.length = self.reader.count("features")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.reader.read_as_tensor("features", index)


with TemporaryDirectory() as temporary:
    path = Path(temporary) / "features.lmdb"
    with LMDBWriter(path) as writer:
        for index in range(6):
            writer.write_ndarray(
                np.full((2, 3), index, dtype=np.float32), key="features", index=index
            )
    dataset = Features(path)
    worker_reader = pickle.loads(pickle.dumps(dataset.reader))  # what each worker receives
    assert worker_reader.generation == dataset.reader.generation
    batches = list(DataLoader(dataset, batch_size=4, num_workers=0))
    assert batches[0].shape == (4, 2, 3) and batches[1].shape == (2, 2, 3)
    dataset.reader.close()
    worker_reader.close()
```

`read_as_tensor` accepts entries written with `write_tensor` and NumPy arrays
written with `write_ndarray` (plain, chunked or compressed). The tensor owns the
bytes copied from LMDB, is writable, and costs no second copy, so PyTorch never
warns about non-writable memory. With `num_workers > 0`, define the dataset class in
an importable module (spawn, the default on Windows and macOS, imports it in each
worker) and keep the `if __name__ == "__main__":` guard in the script.

## Read whole batches in one transaction

Per-item reads open one short transaction each. For larger batches, let the sampler
produce index lists and read each list in one call:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler

from amber_lmdb import LMDBReader, LMDBWriter, as_tensor


class FeatureBatches(Dataset):
    def __init__(self, path):
        self.reader = LMDBReader(path)
        self.reader.get_snapshot()
        self.length = self.reader.count("features")

    def __len__(self):
        return self.length

    def __getitem__(self, indices):
        arrays = self.reader.read_as_ndarray_many(["features"] * len(indices), indices=indices)
        return as_tensor(np.stack(arrays))


with TemporaryDirectory() as temporary:
    path = Path(temporary) / "features.lmdb"
    with LMDBWriter(path) as writer:
        for index in range(10):
            writer.write_ndarray(np.full(3, index, dtype=np.float32), key="features", index=index)
    dataset = FeatureBatches(path)
    sampler = BatchSampler(SequentialSampler(range(len(dataset))), batch_size=4, drop_last=False)
    loader = DataLoader(dataset, sampler=sampler, batch_size=None)
    sizes = [batch.shape[0] for batch in loader]
    assert sizes == [4, 4, 2]
    dataset.reader.close()
```

`as_tensor` turns any array into a tensor without the read-only warning: it shares
memory with a writable array and copies a read-only one once. Arrays read with
`writable=True` are shared, which suits `torch.from_numpy` as well. Batch reads
take up to 1024 keys by default; raise `max_items` (up to 2**20) and `max_bytes`
(64 MiB by default, `None` for no limit) for larger batches.

## Practical settings

- Keep `verify_threads=0` (the default) in worker readers; workers already use the
  cores. Verifying digests can be skipped with `verify_checksums=False` only for
  stores whose operator vouches that they are intact.
- On network storage (NFS) a cold read waits for every page in turn. Read a batch
  with `read_as_ndarray_many` or `read_as_tensor_many` and give the reader
  `read_threads=8` (or more): the bodies are fetched concurrently before they are
  copied. On a 1 GbE NFS mount, reading 26 values of about 155 KB at a time rose
  from 14 MB/s to 54 MB/s with eight threads and 74 MB/s with 26; four processes
  together rose from 50 MB/s to 82 MB/s. The threads wait on I/O and use little
  CPU, so they also suit data-loader workers. Turning on `readahead=True` helps
  only when the whole store is read in file order.
- Large arrays read in windows: write them with `chunk_rows=` and read
  `read_array_rows(key, index, start=, stop=)`; only the touched chunks are copied.
- Compression (`write_ndarray(..., compression="zstd")`) trades CPU in the workers
  for smaller stores; measure it for your data before choosing it.
- Call `reader.refresh()` between epochs if each epoch should see the newest
  generation; otherwise the loader keeps reading the pinned one.
- Close readers when the dataset is discarded; a reader that is only dropped is
  released when it is garbage collected.
