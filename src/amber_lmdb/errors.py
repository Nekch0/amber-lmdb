"""Exception hierarchy: every exception amber-lmdb raises itself derives from AmberLMDBError.

Each class also derives from the built-in exception it refines (``KeyError``,
``ValueError``, ``FileNotFoundError`` ...), so existing ``except`` clauses keep working.
"""

from __future__ import annotations


class AmberLMDBError(Exception):
    """Base class of the exceptions amber-lmdb raises itself.

    Errors passed through from py-lmdb (``lmdb.Error``) and the operating system keep
    their own types.
    """


class KeyNotFoundError(AmberLMDBError, KeyError):
    """A requested entry does not exist.

    Attributes:
        key: Logical key, or None when the location could not be decoded.
        index: Index of the entry, or None.
        location: Encoded LMDB key.
    """

    def __init__(
        self,
        message: str,
        *,
        key: object = None,
        index: int | None = None,
        location: bytes | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.key = key
        self.index = index
        self.location = location

    def __str__(self) -> str:
        # KeyError would quote its argument; show the message as written.
        return self.message


class StoreNotFoundError(AmberLMDBError, FileNotFoundError, ValueError):
    """The dataset root does not exist."""


class ReaderMismatchError(AmberLMDBError, ValueError):
    """An entry was read with a method that cannot read its writer tag.

    Attributes:
        location: Encoded LMDB key.
        writer_tag: Writer tag stored with the entry.
        read_method: Method that was called.
        valid_readers: Methods that can read the entry.
    """

    def __init__(
        self,
        message: str,
        *,
        location: bytes,
        writer_tag: str,
        read_method: str,
        valid_readers: tuple[str, ...],
    ) -> None:
        super().__init__(message)
        self.location = location
        self.writer_tag = writer_tag
        self.read_method = read_method
        self.valid_readers = valid_readers


class ReaderClosedError(AmberLMDBError, ValueError):
    """A reader was used after ``close()``; ``refresh()`` releases without closing."""


__all__ = [
    "AmberLMDBError",
    "KeyNotFoundError",
    "ReaderClosedError",
    "ReaderMismatchError",
    "StoreNotFoundError",
]
