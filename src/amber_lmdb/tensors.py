"""Convert arrays read from a store into PyTorch tensors without aliasing surprises."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor


def as_tensor(array: NDArray[Any]) -> Tensor:
    """Return a CPU tensor holding ``array``, sharing its memory only when that is safe.

    Reads return read-only arrays by default, and ``torch.from_numpy`` warns about
    those because the tensor could write to memory NumPy treats as immutable. This
    shares memory with a writable, native-byte-order array (as ``torch.from_numpy``
    does) and copies once otherwise, so it never warns. ``read_as_tensor`` already
    returns tensors for stored arrays and tensors; use this for arrays from batch or
    row reads (or read them with ``writable=True`` and call ``torch.from_numpy``).

    Raises:
        TypeError: ``array`` is not a NumPy array, or its dtype has no PyTorch
            equivalent (for example structured or string dtypes).
        ImportError: PyTorch is not installed.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"array must be a numpy.ndarray, got {type(array).__name__}")
    import torch

    if not array.dtype.isnative:
        array = array.astype(array.dtype.newbyteorder("="))
    elif not array.flags.writeable:
        array = array.copy()
    try:
        return torch.from_numpy(array)
    except TypeError as exc:
        raise TypeError(f"dtype {array.dtype} has no PyTorch equivalent: {exc}") from None


__all__ = ["as_tensor"]
