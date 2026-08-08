"""Adversarial and NumPy-interoperability tests for the no-pickle NPZ codec."""
from __future__ import annotations

import importlib.util
import io
import lzma
import struct
import sys
import tracemalloc
import warnings
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / 'app' / 'services' / 'npz_transport.py'
SPEC = importlib.util.spec_from_file_location('lds_npz_transport_test', MODULE_PATH)
transport = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = transport
assert SPEC.loader is not None
SPEC.loader.exec_module(transport)


def _npy(*, descr='<f4', shape=(1,), payload=None, fortran=False,
         version=(1, 0), header=None):
    if header is None:
        header = repr({
            'descr': descr, 'fortran_order': fortran, 'shape': shape,
        }).encode('latin1')
    if payload is None:
        item_size = 4 if descr in ('<f4', '>f4', '<U1') else 8
        payload = b'\0' * (item_size * max(1, shape[0] if shape else 1))
    if version == (1, 0):
        prefix = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header))
    elif version == (2, 0):
        prefix = b'\x93NUMPY\x02\x00' + struct.pack('<I', len(header))
    else:
        prefix = b'\x93NUMPY' + bytes(version) + struct.pack('<I', len(header))
    return prefix + header + payload


def _npz(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in entries:
            archive.writestr(name, raw)
    return output.getvalue()


def _read(raw, *, max_elements=100):
    return transport.read_bytes(
        raw, max_file_bytes=1024 * 1024,
        max_uncompressed_bytes=1024 * 1024, max_elements=max_elements)


@pytest.mark.parametrize('version', ((1, 0), (2, 0)))
def test_npy_v1_and_v2_float32_are_supported(version):
    raw = _npz([('values.npy', _npy(
        version=version, shape=(2,), payload=struct.pack('<2f', 1.25, -2.5)))])
    arrays = _read(raw)
    assert arrays['values'].shape == (2,)
    assert arrays['values'].float(0) == pytest.approx(1.25)
    assert arrays['values'].float(1) == pytest.approx(-2.5)


@pytest.mark.parametrize(
    'npy',
    (
        _npy(descr='|O', payload=b'12345678'),
        _npy(descr='>f4', payload=struct.pack('>f', 1.0)),
        _npy(fortran=True, payload=struct.pack('<f', 1.0)),
        _npy(shape=(101,), payload=b'\0' * 404),
        _npy(shape=(0, 10 ** 30), payload=b''),
        _npy(payload=b'\0' * 3),
        _npy(payload=b'\0' * 5),
        _npy(version=(3, 0), payload=struct.pack('<f', 1.0)),
        _npy(header=b' ' * 8193, payload=b''),
        b'\x93NUMPY\x02\x00\x00\x00',
        b'\x93NUMPY\x02\x00\x00\x00\x00',
    ),
    ids=('object', 'big-endian', 'fortran', 'shape-budget', 'zero-huge-shape',
         'truncated', 'trailing', 'npy-v3', 'header-budget', 'v2-prefix-10',
         'v2-prefix-11'),
)
def test_unsupported_or_malformed_npy_is_rejected(npy):
    assert _read(_npz([('values.npy', npy)])) is None


def test_zip_duplicate_traversal_entry_count_and_declared_size_are_rejected():
    valid = _npy(payload=struct.pack('<f', 1.0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        duplicate = _npz([('x.npy', valid), ('x.npy', valid)])
    assert _read(duplicate) is None
    assert _read(_npz([('../x.npy', valid)])) is None
    assert _read(_npz([(f'x{i}.npy', valid) for i in range(33)])) is None

    declared_huge = bytearray(_npz([('x.npy', valid)]))
    central = declared_huge.find(b'PK\x01\x02')
    assert central >= 0
    struct.pack_into('<I', declared_huge, central + 24, 2 * 1024 * 1024)
    assert _read(bytes(declared_huge)) is None


def test_invalid_utf32_codepoint_is_never_materialized_as_a_path():
    raw = _npz([('paths.npy', _npy(
        descr='<U1', payload=struct.pack('<I', 0xD800)))])
    arrays = _read(raw)
    assert arrays is not None
    with pytest.raises(ValueError, match='invalid Unicode'):
        arrays['paths'].string(0)


def test_writer_and_numpy_are_bidirectionally_compatible(tmp_path):
    np = pytest.importorskip('numpy')
    ours = tmp_path / 'ours.npz'
    assert transport.write_atomic(ours, {
        'paths': transport.unicode(['one.jpg', 'deux.webp']),
        'values': transport.floats([1.5, -3.25], (2,)),
        'flags': transport.uint8([1, 0], (2,)),
        'hashes': transport.uint8(range(64), (2, 32)),
    }, max_file_bytes=1024 * 1024)
    with np.load(ours, allow_pickle=False) as archive:
        assert archive['paths'].tolist() == ['one.jpg', 'deux.webp']
        assert archive['values'].tolist() == pytest.approx([1.5, -3.25])
        assert archive['flags'].tolist() == [1, 0]
        assert archive['hashes'].shape == (2, 32)
        assert archive['hashes'][1].tolist() == list(range(32, 64))

    numpy_file = tmp_path / 'numpy.npz'
    np.savez_compressed(
        numpy_file,
        paths=np.array(['alpha.jpg', 'beta.webp']),
        values=np.array([2.0, 4.5], dtype='float32'),
        flags=np.array([0, 1], dtype='uint8'),
        hashes=np.arange(64, dtype='uint8').reshape(2, 32))
    decoded = transport.read(
        numpy_file, max_file_bytes=1024 * 1024,
        max_uncompressed_bytes=1024 * 1024, max_elements=100)
    assert [decoded['paths'].string(i) for i in range(2)] == [
        'alpha.jpg', 'beta.webp']
    assert decoded['values'].float(1) == pytest.approx(4.5)
    assert decoded['flags'].uint8(1) == 1
    assert decoded['hashes'].uint8_row(1) == bytes(range(32, 64))


class _ReadGuard:
    def __init__(self, raw, calls):
        self.raw = raw
        self.calls = calls

    def read(self, size=-1):
        self.calls.append(size)
        assert 0 <= size <= transport._IO_CHUNK
        return self.raw.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.raw.close()

    def __getattr__(self, name):
        return getattr(self.raw, name)


class _WriteGuard:
    def __init__(self, raw, calls):
        self.raw = raw
        self.calls = calls

    def write(self, data):
        self.calls.append(len(data))
        assert len(data) <= transport._IO_CHUNK
        return self.raw.write(data)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.raw.close()

    def __getattr__(self, name):
        return getattr(self.raw, name)


def _stored_npz(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_STORED) as archive:
        for name, raw in entries:
            archive.writestr(name, raw)
    return output.getvalue()


def test_large_reader_never_requests_over_one_mib_and_has_bounded_peak(monkeypatch):
    payload = b'\0' * (200_000 * 4 * 4)
    raw = _stored_npz([('embs.npy', _npy(
        shape=(200_000, 4), payload=payload))])
    calls = []
    original_open = transport.zipfile.ZipFile.open

    def guarded_open(archive, name, mode='r', pwd=None, *, force_zip64=False):
        opened = original_open(
            archive, name, mode=mode, pwd=pwd, force_zip64=force_zip64)
        return _ReadGuard(opened, calls) if mode == 'r' else opened

    monkeypatch.setattr(transport.zipfile.ZipFile, 'open', guarded_open)
    tracemalloc.start()
    decoded = transport.read_bytes(
        raw, max_file_bytes=5 * 1024 * 1024,
        max_uncompressed_bytes=5 * 1024 * 1024,
        max_elements=800_000)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert decoded['embs'].shape == (200_000, 4)
    assert calls and max(calls) <= 1024 * 1024
    # One retained 3.2 MiB Array plus at most one 1 MiB transport chunk; leave
    # generous interpreter/ZipFile headroom while guarding against a full
    # payload-sized second copy.
    assert peak < 12 * 1024 * 1024


def test_writer_streams_chunks_and_keeps_numpy_layout(monkeypatch, tmp_path):
    target = tmp_path / 'large.npz'
    arrays = {
        'embs': transport.Array(
            '<f4', (200_000, 4), bytearray(200_000 * 4 * 4)),
    }
    calls = []
    original_open = transport.zipfile.ZipFile.open

    def guarded_open(archive, name, mode='r', pwd=None, *, force_zip64=False):
        opened = original_open(
            archive, name, mode=mode, pwd=pwd, force_zip64=force_zip64)
        return _WriteGuard(opened, calls) if mode == 'w' else opened

    monkeypatch.setattr(transport.zipfile.ZipFile, 'open', guarded_open)
    assert transport.write_atomic(
        target, arrays, max_file_bytes=5 * 1024 * 1024)
    assert calls and max(calls) <= 1024 * 1024
    decoded = transport.read(
        target, max_file_bytes=5 * 1024 * 1024,
        max_uncompressed_bytes=5 * 1024 * 1024,
        max_elements=800_000)
    assert decoded['embs'].shape == (200_000, 4)


def test_wanted_paths_are_selected_before_row_matrices(tmp_path):
    target = tmp_path / 'runtime.npz'
    embs = [float(row * 1000 + column)
            for row in range(3) for column in range(8)]
    assert transport.write_atomic(target, {
        'paths': transport.unicode(['a.jpg', 'b.jpg', 'c.jpg']),
        'states': transport.unicode(['ok', 'ok', 'ok']),
        'embs': transport.floats(embs, (3, 8)),
        # Not row-aligned with paths, so it stays whole.
        'version': transport.uint8([1, 2], (2,)),
    }, max_file_bytes=1024 * 1024)
    decoded = transport.read(
        target, max_file_bytes=1024 * 1024,
        max_uncompressed_bytes=1024 * 1024, max_elements=100,
        wanted_paths={'b.jpg'})
    assert decoded['paths'].shape == (1,)
    assert decoded['paths'].string(0) == 'b.jpg'
    assert decoded['states'].shape == (1,)
    assert decoded['embs'].shape == (1, 8)
    assert decoded['embs'].float_row(0) == pytest.approx(
        tuple(float(1000 + column) for column in range(8)))
    assert decoded['version'].shape == (2,)


def test_crc_corruption_is_rejected():
    valid = _npy(shape=(2,), payload=struct.pack('<2f', 1.0, 2.0))
    raw = bytearray(_stored_npz([('values.npy', valid)]))
    local = raw.find(b'PK\x03\x04')
    assert local >= 0
    name_len = struct.unpack_from('<H', raw, local + 26)[0]
    extra_len = struct.unpack_from('<H', raw, local + 28)[0]
    payload_start = local + 30 + name_len + extra_len
    raw[payload_start + len(valid) - 1] ^= 0xFF
    assert _read(bytes(raw)) is None


@pytest.mark.parametrize('codec_error', (
    zlib.error('corrupt deflate stream'),
    lzma.LZMAError('corrupt lzma stream'),
))
def test_codec_errors_are_normalized_to_a_cache_miss(
        monkeypatch, codec_error):
    def fail_closed(*_args, **_kwargs):
        raise codec_error

    monkeypatch.setattr(transport, '_read_archive', fail_closed)
    assert transport.read_bytes(
        b'PK placeholder', max_file_bytes=1024,
        max_uncompressed_bytes=1024, max_elements=10) is None


def test_row_path_and_index_caps_fail_closed(monkeypatch):
    too_many_rows = _npy(
        shape=(200_001, 1), payload=b'\0' * (200_001 * 4))
    assert transport.read_bytes(
        _stored_npz([('embs.npy', too_many_rows)]),
        max_file_bytes=2 * 1024 * 1024,
        max_uncompressed_bytes=2 * 1024 * 1024,
        max_elements=300_000) is None

    monkeypatch.setattr(transport, '_MAX_PATH_CHARACTERS', 5)
    assert transport.unicode(['abc', 'def']) is None
    raw = _stored_npz([('paths.npy', _npy(
        descr='<U3', shape=(2,),
        payload=('abc'.encode('utf-32le') + 'def'.encode('utf-32le'))))])
    assert _read(raw) is None

    monkeypatch.setattr(transport, '_MAX_INDEX_BYTES', 10)
    assert _read(_stored_npz([('hashes.npy', _npy(
        descr='|u1', shape=(32,), payload=bytes(range(32))))])) is None


def test_replace_failure_keeps_old_cache_and_removes_unique_temp(monkeypatch, tmp_path):
    target = tmp_path / 'cache.npz'
    target.write_bytes(b'old-cache')
    monkeypatch.setattr(
        transport.os, 'replace',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError('fault')))
    assert not transport.write_atomic(target, {
        'values': transport.floats([1.0, 2.0], (2,)),
    }, max_file_bytes=1024 * 1024)
    assert target.read_bytes() == b'old-cache'
    assert list(tmp_path.glob(f'.{target.name}.*.tmp')) == []


def test_concurrent_writers_publish_one_complete_archive(tmp_path):
    target = tmp_path / 'shared.npz'

    def publish(seed):
        return transport.write_atomic(target, {
            'values': transport.floats([float(seed)] * 16, (4, 4)),
            'marker': transport.uint8([seed], (1,)),
        }, max_file_bytes=1024 * 1024)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(publish, range(8)))
    assert all(results)
    decoded = transport.read(
        target, max_file_bytes=1024 * 1024,
        max_uncompressed_bytes=1024 * 1024, max_elements=100)
    marker = decoded['marker'].uint8()
    assert decoded['values'].float_row(0) == pytest.approx(
        (float(marker),) * 4)
    assert list(tmp_path.glob(f'.{target.name}.*.tmp')) == []
