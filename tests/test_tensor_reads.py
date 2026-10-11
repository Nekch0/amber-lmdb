"""Reader tensor results use their owned body in place: one copy, still verified."""

import pytest

torch = pytest.importorskip("torch")

import amber_lmdb.codec as codec
from amber_lmdb import (
    IntegrityError,
    LMDBOverlayReader,
    LMDBReader,
    LMDBWriter,
    get_loc,
    open_environment,
)

TENSORS = {
    "float32": torch.arange(12, dtype=torch.float32).reshape(3, 4),
    "bfloat16": torch.linspace(-1, 1, 7).to(torch.bfloat16),
    "float8": torch.tensor([0.5, -1.0, 2.0]).to(torch.float8_e4m3fn),
    "int64": torch.tensor([[1, -2], [3, 2**40]], dtype=torch.int64),
    "bool": torch.tensor([True, False, True]),
    "empty": torch.empty(0, 5),
    "scalar": torch.tensor(3.5, dtype=torch.float64),
}


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "tensors.lmdb"
    with LMDBWriter(path, map_size=1024**2) as writer:
        for name, tensor in TENSORS.items():
            writer.write_tensor(tensor, key=name)
    return path


def _same(a, b):
    return (
        a.dtype == b.dtype
        and a.shape == b.shape
        and torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
    )


def test_every_read_path_restores_values_without_a_second_copy(store, monkeypatch):
    def copying(*args, **kwargs):
        pytest.fail("reader used the copying decoder")

    monkeypatch.setattr(codec, "bytes_to_tensor", copying)
    names = list(TENSORS)
    with LMDBReader(store) as reader:
        assert all(_same(reader.read_as_tensor(n), TENSORS[n]) for n in names)
        assert all(
            _same(v, TENSORS[n]) for v, n in zip(reader.read_many(names), names, strict=True)
        )
        pages = [v for page in reader.iter_read_pages(names) for v in page]
        assert all(_same(v, TENSORS[n]) for v, n in zip(pages, names, strict=True))
        assert _same(reader.read("int64"), TENSORS["int64"])


def test_results_are_writable_and_independent(store):
    with LMDBReader(store) as reader:
        first = reader.read_as_tensor("float32")
        second = reader.read_as_tensor("float32")
        assert first.data_ptr() != second.data_ptr()
        first.add_(100)
        assert torch.equal(second, TENSORS["float32"])
        assert torch.equal(reader.read_as_tensor("float32"), TENSORS["float32"])


def test_tensor_bodies_are_still_verified(store):
    with (
        open_environment(store, readonly=False) as handle,
        handle.environment.begin(write=True) as txn,
    ):
        raw = txn.get(get_loc("float32"))
        txn.put(get_loc("float32"), raw[:-1] + bytes([raw[-1] ^ 1]))
    with LMDBReader(store) as reader, pytest.raises(IntegrityError):
        reader.read_as_tensor("float32")
    with LMDBReader(store, verify_checksums=False) as reader:
        assert not torch.equal(reader.read_as_tensor("float32"), TENSORS["float32"])


def test_overlay_reads_use_the_same_path(store, tmp_path):
    other = tmp_path / "other.lmdb"
    with LMDBWriter(other) as writer:
        writer.write_tensor(torch.ones(2, 2), key="ones")
    with LMDBOverlayReader([store, other]) as reader:
        values = reader.read_many(["ones", "float32"])
        assert torch.equal(values[0], torch.ones(2, 2)) and _same(values[1], TENSORS["float32"])


def test_public_decoder_still_copies_the_callers_buffer():
    buffer = bytearray(codec.tensor_to_bytes(torch.zeros(4)))
    tensor = codec.bytes_to_tensor(buffer)
    buffer[-4:] = b"\xff\xff\xff\xff"
    assert torch.equal(tensor, torch.zeros(4))
