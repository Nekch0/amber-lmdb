"""Tensor round-trips for the optional PyTorch dependency."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from frozendict import frozendict

from amber_lmdb.lmdb import LMDBReader, LMDBWriter

torch = pytest.importorskip("torch")


@dataclass(frozen=True)
class _LMDBPoint:
    x: int
    y: int


@pytest.fixture
def ds_path(tmp_path) -> Path:
    return tmp_path / "dataset.lmdb"


@pytest.fixture(autouse=True)
def _clean_env_cache():
    """Close cached environments after each test to prevent connection leaks."""
    yield
    import amber_lmdb.lmdb as db

    db._clear_env_cache()


class TestArrayRoundTrips:
    @pytest.mark.parametrize("shape", [(), (5,), (2, 3), (0,)])
    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.int64, torch.bfloat16, torch.float8_e4m3fn, torch.bool],
    )
    def test_tensor(self, ds_path, shape, dtype):
        n = 1 if shape == () else int(np.prod(shape))
        t = torch.arange(n, dtype=torch.float32).to(dtype).reshape(shape)
        with LMDBWriter(ds_path) as w:
            w.write_tensor(t, key="x")
            w.commit()
        with LMDBReader(ds_path) as r:
            out = r.read_as_tensor("x")
            assert out.dtype == dtype
            assert tuple(out.shape) == shape
            a = out.reshape(-1).contiguous().view(torch.uint8)
            b = t.reshape(-1).contiguous().view(torch.uint8)
            assert torch.equal(a, b)

    def test_tensor_detaches_and_contiguous(self, ds_path):
        t = torch.randn(3, 4, requires_grad=True).t()  # Noncontiguous input with gradients.
        with LMDBWriter(ds_path) as w:
            w.write_tensor(t, key="x")
            w.commit()
        with LMDBReader(ds_path) as r:
            out = r.read_as_tensor("x")
            assert not out.requires_grad
            assert tuple(out.shape) == (4, 3)
            assert torch.equal(out, t.detach().contiguous())

    def test_read_cast_dtype(self, ds_path):
        a = np.arange(6, dtype=np.int32).reshape(2, 3)
        t = torch.arange(6, dtype=torch.int32)
        with LMDBWriter(ds_path) as w:
            w.write_ndarray(a, key="a")
            w.write_tensor(t, key="t")
            w.commit()
        with LMDBReader(ds_path) as r:
            assert r.read_as_ndarray("a", dtype=np.float64).dtype == np.float64
            assert r.read_as_tensor("t", dtype=torch.float16).dtype == torch.float16


class TestShapeDtypeReaders:
    def test_tensor_shape_and_dtype(self, ds_path):
        t = torch.zeros(2, 3, 4, dtype=torch.bfloat16)
        with LMDBWriter(ds_path) as w:
            w.write_tensor(t, key="x")
            w.commit()
        with LMDBReader(ds_path) as r:
            assert r.read_as_shape("x") == (2, 3, 4)
            dt = r.read_as_dtype("x")
            assert dt == torch.bfloat16 and isinstance(dt, torch.dtype)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize(
    "value",
    [
        None,
        b"payload",
        int,
        True,
        3,
        2.5,
        1j,
        "abc",
        [1, None],
        (1, 2),
        {"a": [1]},
        frozendict({"b": 2}),
        {1, 2},
        frozenset({2}),
        np.arange(4),
        torch.arange(3),
    ],
)
def test_generic_write_and_bounded_read_return_original_types(ds_path, value):
    with LMDBWriter(ds_path, map_size=2 * 1024**2) as writer:
        writer.write(value, key=("feature", "a"), index=0)
    with LMDBReader(ds_path) as reader:
        single = reader.read(("feature", "a"), index=0)
        batch = reader.read_many([("feature", "a")], indices=[0], max_items=1)
    for restored in (single, batch[0]):
        assert type(restored) is type(value)
        if isinstance(value, torch.Tensor):
            assert torch.equal(restored, value)
        elif isinstance(value, np.ndarray):
            np.testing.assert_array_equal(restored, value)
        else:
            assert restored == value


def test_read_as_tensor_many_matches_single_reads(ds_path):
    with LMDBWriter(ds_path, map_size=16 * 1024**2) as writer:
        for index in range(3):
            writer.write_tensor(torch.full((2, 3), float(index)), key="hidden", index=index)
        writer.write_ndarray(np.arange(4, dtype=np.int32), key="array")
        writer.commit()
    keys = ["hidden", "array", "hidden", "hidden"]
    indices = [2, None, 0, 2]
    with LMDBReader(ds_path, read_threads=2) as reader:
        tensors = reader.read_as_tensor_many(keys, indices=indices)
        singles = [reader.read_as_tensor(k, index=i) for k, i in zip(keys, indices, strict=True)]
        halves = reader.read_as_tensor_many(keys[:1], indices=[1], dtype=torch.float16)
        assert reader.read_as_tensor_many([]) == []
    for tensor, single in zip(tensors, singles, strict=True):
        assert torch.equal(tensor, single)
    assert tensors[0] is not tensors[3]
    tensors[0].add_(1)  # Every result owns writable memory.
    assert torch.equal(tensors[3], torch.full((2, 3), 2.0))
    assert halves[0].dtype == torch.float16
