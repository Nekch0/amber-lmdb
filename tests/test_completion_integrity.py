"""Envelope conformance and corruption checks before any application decoder."""

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import amber_lmdb.codec as codec
import amber_lmdb.integrity as integrity
from amber_lmdb import (
    IntegrityError,
    LMDBReader,
    LMDBWriter,
    get_loc,
    inspect_store,
    open_environment,
    verify_store,
)
from amber_lmdb.lmdb import _FORMAT_MARKER, _METADATA_DB_NAME

FIXTURES = Path(__file__).parent / "fixtures"
CHECKSUMS = ["blake2b-256", "sha256"]
NAMES = {"blake2b-256": "BLAKE2b-256", "sha256": "SHA-256"}
# Masks SSSE3, AVX, AVX2, BMI1/2 and SHA so OpenSSL takes its integer code path.
OPENSSL_INTEGER_PATH = "~0x1000020000000000:~0x20000128"


def _independent_envelope(checksum, location, tag, body):
    if checksum == "blake2b-256":
        version, digest = 2, hashlib.blake2b(digest_size=32)
    else:
        version, digest = 1, hashlib.sha256()
    prefix = b"TLMI" + bytes((version,))
    lengths = struct.pack(">BQ", len(tag), len(body))
    digest.update(prefix + struct.pack(">I", len(location)) + location + lengths + tag.encode())
    digest.update(body)
    return prefix + lengths + tag.encode() + digest.digest()


def _cases():
    return json.loads((FIXTURES / "amber_lmdb_integrity_v2.json").read_text())["cases"]


def test_v1_golden_envelope_independent_digest_and_historical_fixture_unchanged(tmp_path):
    fixture = json.loads((FIXTURES / "mmfabric_0_2_2.json").read_text())
    assert fixture["format"] == "mmfabric-lmdb-v2\n"
    key = get_loc("message")
    tag, body = "write_str", "日本語".encode()
    expected_input = b"TLMI\x01" + struct.pack(">I", len(key)) + key
    expected_input += struct.pack(">BQ", len(tag), len(body)) + tag.encode()
    digest = hashlib.sha256(expected_input + body).digest()
    expected = b"TLMI\x01" + struct.pack(">BQ", len(tag), len(body)) + tag.encode() + digest
    golden = json.loads((FIXTURES / "amber_lmdb_integrity_v1.json").read_text())
    assert golden["format"] == _FORMAT_MARKER
    assert bytes.fromhex(golden["metadata_hex"]) == expected
    assert bytes.fromhex(golden["location_hex"]) == key
    assert bytes.fromhex(golden["value_hex"]) == body
    assert integrity._encode_metadata(key, tag, body, "sha256") == expected
    metadata = integrity._parse_metadata(expected)
    assert metadata.writer_tag == tag and metadata.size_bytes == len(body)
    assert metadata.checksum == "sha256"
    assert metadata.metadata_bytes == len(expected)
    integrity._verify_checksum(key, metadata, body)
    path = tmp_path / "store"
    with LMDBWriter(path, map_size=1024**2, checksum="sha256") as writer:
        writer.write_str("日本語", key="message")
    assert (path / "FORMAT").read_text() == _FORMAT_MARKER
    with open_environment(path) as handle:
        tags = handle.environment.open_db(_METADATA_DB_NAME)
        with handle.environment.begin() as txn:
            assert txn.get(key, db=tags) == expected


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_pinned_envelopes_match_independent_digests_and_implementation(case):
    location = bytes.fromhex(case["location_hex"])
    body = bytes.fromhex(case["value_hex"])
    expected = bytes.fromhex(case["metadata_hex"])
    checksum, tag = case["checksum"], case["writer_tag"]
    assert _independent_envelope(checksum, location, tag, body) == expected
    assert integrity._encode_metadata(location, tag, body, checksum) == expected
    metadata = integrity._parse_metadata(expected)
    assert (metadata.checksum, metadata.writer_tag, metadata.size_bytes) == (
        checksum,
        tag,
        len(body),
    )
    integrity._verify_checksum(location, metadata, body)


def test_default_writer_uses_blake2b_envelope_version_2(tmp_path):
    case = next(case for case in _cases() if case["name"] == "blake2b-256/short")
    path = tmp_path / "store"
    with LMDBWriter(path, map_size=1024**2) as writer:
        writer.write_str("日本語", key="message")
    with open_environment(path) as handle:
        tags = handle.environment.open_db(_METADATA_DB_NAME)
        with handle.environment.begin() as txn:
            assert txn.get(get_loc("message"), db=tags) == bytes.fromhex(case["metadata_hex"])
    assert inspect_store(path, entry_limit=1).samples[0].checksum == "blake2b-256"


def test_pinned_envelopes_hold_on_openssl_integer_code_path():
    script = (
        "import json, sys; import amber_lmdb.integrity as i\n"
        "cases = json.loads(open(sys.argv[1], encoding='utf-8').read())['cases']\n"
        "print(all(i._encode_metadata(bytes.fromhex(c['location_hex']), c['writer_tag'], "
        "bytes.fromhex(c['value_hex']), c['checksum']).hex() == c['metadata_hex'] "
        "for c in cases))\n"
    )
    environment = {**os.environ, "OPENSSL_ia32cap": OPENSSL_INTEGER_PATH}
    completed = subprocess.run(
        [sys.executable, "-c", script, str(FIXTURES / "amber_lmdb_integrity_v2.json")],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "True"


def test_relabelled_version_byte_fails_verification():
    location, body = get_loc("message"), b"payload"
    for source, target in (("sha256", 2), ("blake2b-256", 1)):
        raw = integrity._encode_metadata(location, "write_str", body, source)
        relabelled = integrity._parse_metadata(raw[:4] + bytes((target,)) + raw[5:])
        assert relabelled.checksum != source
        with pytest.raises(IntegrityError, match="checksum mismatch"):
            integrity._verify_checksum(location, relabelled, body)


@pytest.mark.parametrize("checksum", ["md5", "SHA256", "", None, 2])
def test_unknown_checksum_is_rejected_before_writing(tmp_path, checksum):
    with pytest.raises(ValueError, match="checksum must be"):
        LMDBWriter(tmp_path / "store", checksum=checksum)
    with pytest.raises(ValueError, match="checksum must be"):
        integrity._encode_metadata(b"key", "write_str", b"body", checksum)
    assert not (tmp_path / "store").exists()


def test_generations_with_different_checksums_stay_readable(tmp_path):
    path = tmp_path / "store"
    with LMDBWriter(path, map_size=1024**2, checksum="sha256") as writer:
        writer.write_str("old", key="value")
    with LMDBReader(path) as pinned:
        assert pinned.read_as_str("value") == "old"
        with LMDBWriter(path, map_size=1024**2, overwrite=True) as writer:
            writer.write_str("new", key="value")
        assert pinned.read_as_str("value") == "old"
        assert verify_store(path).ok
        with LMDBReader(path) as current:
            assert current.read_as_str("value") == "new"
    assert inspect_store(path, entry_limit=1).samples[0].checksum == "blake2b-256"


@pytest.mark.parametrize("checksum", CHECKSUMS)
@pytest.mark.parametrize(
    "damage", ["short", "version", "trailing", "tag-length", "non-ascii", "huge"]
)
def test_malformed_metadata_is_bounded_and_rejected(damage, checksum):
    raw = integrity._encode_metadata(b"key", "write_str", b"body", checksum)
    if damage == "short":
        raw = raw[:-1]
    elif damage == "version":
        raw = raw[:4] + b"\xff" + raw[5:]
    elif damage == "trailing":
        raw += b"x"
    elif damage == "tag-length":
        raw = raw[:5] + b"\x00" + raw[6:]
    elif damage == "non-ascii":
        raw = raw[:14] + b"\xff" + raw[15:]
    else:
        raw = b"x" * 1024
    with pytest.raises(IntegrityError):
        integrity._parse_metadata(raw)


@pytest.mark.parametrize("checksum", CHECKSUMS)
@pytest.mark.parametrize("engine", [0, 1])
@pytest.mark.parametrize("damage", ["body", "key", "tag", "length", "missing"])
def test_read_and_full_scan_detect_tampering_before_decode(
    tmp_path, monkeypatch, engine, damage, checksum
):
    path = tmp_path / "tampered"
    with LMDBWriter(path, map_size=1024**2, lib_version=engine, checksum=checksum) as writer:
        writer.write_str("four", key="first")
        writer.write_str("four", key="second")
    with open_environment(path, readonly=False) as handle:
        tags = handle.environment.open_db(_METADATA_DB_NAME)
        with handle.environment.begin(write=True) as txn:
            location = get_loc("second")
            frame = txn.get(location, db=tags)
            if damage == "body":
                txn.put(location, b"evil")
            elif damage == "key":
                txn.put(location, txn.get(get_loc("first"), db=tags), db=tags)
            elif damage == "tag":
                changed = integrity._parse_metadata(frame)
                replacement = integrity._encode_metadata(location, "write_any", b"four", checksum)
                txn.put(location, replacement[:-32] + changed.digest, db=tags)
            elif damage == "length":
                txn.put(location, b"three")
            else:
                txn.delete(location, db=tags)
    original_decode = codec.bytes_to_str
    monkeypatch.setattr(
        codec, "bytes_to_str", lambda *args: pytest.fail("decoder reached corrupt input")
    )
    with LMDBReader(path) as reader, pytest.raises((IntegrityError, ValueError)):
        reader.read("second")
    monkeypatch.setattr(codec, "bytes_to_str", original_decode)
    report = verify_store(path)
    assert not report.ok and report.error_count == 1


@pytest.mark.parametrize("checksum", CHECKSUMS)
def test_same_length_numeric_change_is_detected_but_header_scan_stays_header_only(
    tmp_path, checksum
):
    path = tmp_path / "array"
    with LMDBWriter(path, map_size=1024**2, checksum=checksum) as writer:
        writer.write_ndarray(np.arange(3, dtype=np.int32), key="values")
    with (
        open_environment(path, readonly=False) as handle,
        handle.environment.begin(write=True) as txn,
    ):
        raw = txn.get(get_loc("values"))
        txn.put(get_loc("values"), raw[:-1] + bytes([raw[-1] ^ 1]))
    with LMDBReader(path) as reader:
        assert reader.read_array_headers_many(["values"])[0].shape == (3,)
        with pytest.raises(IntegrityError, match=f"{NAMES[checksum]} checksum mismatch"):
            reader.read_as_ndarray("values")
    assert not inspect_store(path, entry_limit=1).samples[0].error
    assert verify_store(path).error_count == 1


@pytest.mark.parametrize("checksum", CHECKSUMS)
def test_optional_metadata_body_checksum_is_checked_before_decode(tmp_path, monkeypatch, checksum):
    path = tmp_path / "meta"
    with LMDBWriter(path, map_size=1024**2, checksum=checksum) as writer:
        writer.write_dict({"number": 3}, key="meta")
    with (
        open_environment(path, readonly=False) as handle,
        handle.environment.begin(write=True) as txn,
    ):
        raw = txn.get(get_loc("meta"))
        txn.put(get_loc("meta"), raw[:-1] + bytes([raw[-1] ^ 1]))
    monkeypatch.setattr(
        codec, "bytes_to_dict", lambda *a, **kw: pytest.fail("corrupt meta decoded")
    )
    info = inspect_store(path, with_meta=True)
    assert info.meta is None and f"{NAMES[checksum]} checksum mismatch" in info.meta_error
