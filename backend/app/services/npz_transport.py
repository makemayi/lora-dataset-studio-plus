"""Small, bounded NPZ reader/writer for the Flask interpreter.

The Flask environment deliberately does not depend on NumPy, but Bank transfer
caches must remain readable by NumPy's ``allow_pickle=False`` loader.  This
module implements only the primitive NPY layouts LDS writes: little-endian
float32/int32, uint8 and fixed-width Unicode vectors/matrices.

All archive I/O is chunked.  Headers, row counts and path/index payloads have
hard limits, writers use a unique sibling tempfile, and a malformed archive
returns ``None``/``False`` without replacing an existing cache.
"""
from __future__ import annotations

import ast
import io
import lzma
import math
import os
import re
import struct
import tempfile
import threading
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path


_MAGIC = b'\x93NUMPY'
_UNICODE_RE = re.compile(r'^<U(?P<count>[1-9]\d*)$')
_FLOAT_RE = re.compile(r'^<f4$')
_INT32_RE = re.compile(r'^<i4$')
_UINT8_RE = re.compile(r'^\|u1$')
_NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_]*\.npy$')

_IO_CHUNK = 1024 * 1024
_MAX_HEADER_BYTES = 8192
_MAX_ROWS = 200_000
_MAX_PATH_CHARACTERS = 16 * 1024 * 1024
_MAX_INDEX_BYTES = 64 * 1024 * 1024
_MAX_ENTRIES = 32
_INDEX_ARRAYS = frozenset(('paths.npy', 'states.npy', 'sigs.npy', 'hashes.npy'))
_INDEX_NAMES = frozenset(name[:-4] for name in _INDEX_ARRAYS)
# Only these arrays are rows keyed by ``paths``. Model/version metadata commonly
# has shape ``(1,)`` too; treating every first dimension equal to the image count
# as row-aligned corrupts metadata whenever a one-image cache is subset.
_ROW_ALIGNED_NAMES = frozenset((
    'states', 'sigs', 'hashes', 'aes', 'nsfw', 'dets', 'bfracs', 'yaws', 'embs',
))
_MAX_WRITTEN_UNCOMPRESSED = 1024 * 1024 * 1024
_WRITE_LOCKS = tuple(threading.Lock() for _ in range(64))


def _product(shape):
    value = 1
    for part in shape:
        value *= part
    return value


def _valid_shape(shape):
    return (isinstance(shape, tuple) and 1 <= len(shape) <= 2
            and all(not isinstance(n, bool) and isinstance(n, int) and n >= 0
                    for n in shape)
            and shape[0] <= _MAX_ROWS)


@dataclass(frozen=True)
class Array:
    descr: str
    shape: tuple
    data: bytes | bytearray

    @property
    def size(self):
        return _product(self.shape)

    def _float_prefix(self):
        if not _FLOAT_RE.fullmatch(self.descr):
            raise TypeError('not float32')
        return '<'

    def float(self, *indices):
        flat = self._flat_index(indices)
        return struct.unpack_from(self._float_prefix() + 'f', self.data, flat * 4)[0]

    def float_row(self, index):
        if (not _FLOAT_RE.fullmatch(self.descr) or len(self.shape) != 2
                or not 0 <= index < self.shape[0]):
            raise TypeError('not a float32 matrix row')
        width = self.shape[1]
        start = index * width * 4
        return tuple(value[0] for value in struct.iter_unpack(
            '<f', memoryview(self.data)[start:start + width * 4]))

    def uint8(self, index=0):
        if not _UINT8_RE.fullmatch(self.descr) or not 0 <= index < self.size:
            raise (TypeError if not _UINT8_RE.fullmatch(self.descr) else IndexError)
        return self.data[index]

    def int32(self, index=0):
        if not _INT32_RE.fullmatch(self.descr) or not 0 <= index < self.size:
            raise (TypeError if not _INT32_RE.fullmatch(self.descr) else IndexError)
        return struct.unpack_from('<i', self.data, index * 4)[0]

    def uint8_row(self, index):
        if (not _UINT8_RE.fullmatch(self.descr) or len(self.shape) != 2
                or not 0 <= index < self.shape[0]):
            raise TypeError('not a uint8 matrix row')
        width = self.shape[1]
        start = index * width
        return bytes(memoryview(self.data)[start:start + width])

    def string(self, index):
        match = _UNICODE_RE.fullmatch(self.descr)
        if not match or len(self.shape) != 1 or not 0 <= index < self.shape[0]:
            raise TypeError('not a Unicode vector')
        width = int(match.group('count'))
        start = index * width * 4
        values = struct.iter_unpack(
            '<I', memoryview(self.data)[start:start + width * 4])
        chars = []
        padding = False
        try:
            for (value,) in values:
                if value == 0:
                    padding = True
                    continue
                if padding or value > 0x10FFFF or 0xD800 <= value <= 0xDFFF:
                    raise ValueError
                chars.append(chr(value))
            return ''.join(chars)
        except (ValueError, OverflowError) as exc:
            raise ValueError('invalid Unicode codepoint/padding') from exc

    def _flat_index(self, indices):
        if len(indices) != len(self.shape):
            raise IndexError
        flat = 0
        for index, extent in zip(indices, self.shape):
            if not isinstance(index, int) or not 0 <= index < extent:
                raise IndexError
            flat = flat * extent + index
        return flat


def _dtype_size(descr):
    match = _UNICODE_RE.fullmatch(descr)
    if match:
        count = int(match.group('count'))
        return count * 4 if count <= 4096 else None
    if _FLOAT_RE.fullmatch(descr):
        return 4
    if _INT32_RE.fullmatch(descr):
        return 4
    if _UINT8_RE.fullmatch(descr):
        return 1
    return None


def _read_into(entry, view):
    """Fill a writable byte view without requesting more than 1 MiB at once."""
    offset = 0
    while offset < len(view):
        want = min(_IO_CHUNK, len(view) - offset)
        chunk = entry.read(want)
        if not chunk:
            raise ValueError('truncated NPY payload')
        if len(chunk) > want:
            raise ValueError('archive reader exceeded requested size')
        view[offset:offset + len(chunk)] = chunk
        offset += len(chunk)


def _read_exact(entry, size):
    """Read exactly ``size`` bytes, never asking the stream for over 1 MiB."""
    if size < 0:
        raise ValueError('negative read')
    out = bytearray(size)
    _read_into(entry, memoryview(out))
    return out


def _drain_exact(entry, size):
    remaining = size
    while remaining:
        want = min(_IO_CHUNK, remaining)
        chunk = entry.read(want)
        if not chunk:
            raise ValueError('truncated NPY payload')
        if len(chunk) > want:
            raise ValueError('archive reader exceeded requested size')
        remaining -= len(chunk)


def _parse_header(entry, declared_size, *, max_elements):
    prefix = _read_exact(entry, 8)
    if bytes(prefix[:6]) != _MAGIC:
        raise ValueError('bad NPY magic')
    version = tuple(prefix[6:8])
    if version == (1, 0):
        raw_len = _read_exact(entry, 2)
        header_len = struct.unpack('<H', raw_len)[0]
        prefix_size = 10
    elif version == (2, 0):
        raw_len = _read_exact(entry, 4)
        header_len = struct.unpack('<I', raw_len)[0]
        prefix_size = 12
    else:
        raise ValueError('unsupported NPY version')
    if header_len > _MAX_HEADER_BYTES or prefix_size + header_len > declared_size:
        raise ValueError('bad NPY header size')
    raw_header = _read_exact(entry, header_len)
    try:
        header = ast.literal_eval(bytes(raw_header).decode('latin1').strip())
    except (ValueError, SyntaxError, UnicodeError, RecursionError, MemoryError) as exc:
        raise ValueError('bad NPY header') from exc
    if not isinstance(header, dict) or set(header) != {
            'descr', 'fortran_order', 'shape'}:
        raise ValueError('bad NPY header fields')
    descr = header['descr']
    shape = header['shape']
    if (not isinstance(descr, str) or header['fortran_order'] is not False
            or not _valid_shape(shape)):
        raise ValueError('unsupported NPY layout')
    elements = _product(shape)
    item_size = _dtype_size(descr)
    if (item_size is None or isinstance(max_elements, bool)
            or not isinstance(max_elements, int) or max_elements < 0
            or elements > max_elements
            or any(extent > max(max_elements, 1) for extent in shape)):
        raise ValueError('unsupported NPY dtype/size')
    payload_size = elements * item_size
    if prefix_size + header_len + payload_size != declared_size:
        raise ValueError('NPY payload length mismatch')
    return descr, shape, item_size, payload_size


def _selected_ranges(indices):
    if not indices:
        return ()
    ranges = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append((start, previous + 1))
        start = previous = index
    ranges.append((start, previous + 1))
    return tuple(ranges)


def _read_payload(entry, payload_size, *, rows, row_size, selected_rows=None):
    if selected_rows is None:
        data = _read_exact(entry, payload_size)
    else:
        selected_rows = tuple(selected_rows)
        if any(not isinstance(i, int) or i < 0 or i >= rows
               for i in selected_rows) or tuple(sorted(set(selected_rows))) != selected_rows:
            raise ValueError('invalid row selection')
        data = bytearray(len(selected_rows) * row_size)
        destination = memoryview(data)
        source_row = 0
        target_offset = 0
        for start, stop in _selected_ranges(selected_rows):
            _drain_exact(entry, (start - source_row) * row_size)
            block_size = (stop - start) * row_size
            _read_into(
                entry, destination[target_offset:target_offset + block_size])
            target_offset += block_size
            source_row = stop
        _drain_exact(entry, (rows - source_row) * row_size)
    # Forces the ZipExtFile to verify EOF/CRC and catches undeclared trailing data.
    if entry.read(1):
        raise ValueError('trailing NPY payload')
    return data


def _decode_entry(archive, info, *, max_elements, selected_rows=None):
    with archive.open(info) as entry:
        descr, shape, item_size, payload_size = _parse_header(
            entry, info.file_size, max_elements=max_elements)
        row_size = item_size * (_product(shape[1:]) if len(shape) > 1 else 1)
        data = _read_payload(
            entry, payload_size, rows=shape[0], row_size=row_size,
            selected_rows=selected_rows)
    selected_shape = ((len(selected_rows),) + shape[1:]
                      if selected_rows is not None else shape)
    return Array(descr, selected_shape, data)


def _path_character_count(array):
    match = _UNICODE_RE.fullmatch(array.descr)
    if not match or len(array.shape) != 1:
        raise ValueError('paths must be a Unicode vector')
    width = int(match.group('count'))
    total = 0
    for index in range(array.shape[0]):
        start = index * width * 4
        for (value,) in struct.iter_unpack(
                '<I', memoryview(array.data)[start:start + width * 4]):
            # Keep Unicode validation lazy in ``Array.string`` for backwards
            # compatibility: callers can inspect an archive structurally and
            # still get a precise error only if they materialise the bad path.
            # Non-zero units nevertheless count against the hard index budget.
            total += int(value != 0)
            if total > _MAX_PATH_CHARACTERS:
                raise ValueError('path index is too large')
    return total


def _subset_array(array, indices):
    row_size = _dtype_size(array.descr) * (
        _product(array.shape[1:]) if len(array.shape) > 1 else 1)
    out = bytearray(len(indices) * row_size)
    target = memoryview(out)
    source = memoryview(array.data)
    for target_index, source_index in enumerate(indices):
        target[target_index * row_size:(target_index + 1) * row_size] = \
            source[source_index * row_size:(source_index + 1) * row_size]
    return Array(array.descr, (len(indices),) + array.shape[1:], out)


def _read_archive(source, *, max_uncompressed_bytes, max_elements,
                  wanted_paths=None):
    """Decode one size-bounded ZIP source, optionally subsetting row caches."""
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if (len(infos) > _MAX_ENTRIES or len(names) != len(set(names))
                or any(not _NAME_RE.fullmatch(name) for name in names)
                or any(info.file_size < 0 for info in infos)
                or sum(info.file_size for info in infos) > max_uncompressed_bytes
                or sum(info.file_size for info in infos
                       if info.filename in _INDEX_ARRAYS) > _MAX_INDEX_BYTES):
            return None
        by_name = {info.filename: info for info in infos}
        arrays = {}
        selected_rows = None
        row_count = None
        paths_info = by_name.get('paths.npy')
        if paths_info is not None:
            paths = _decode_entry(
                archive, paths_info, max_elements=max_elements)
            _path_character_count(paths)
            row_count = paths.shape[0]
            if wanted_paths is not None:
                wanted = {str(path) for path in wanted_paths}
                wanted_canonical = {
                    os.path.normcase(os.path.realpath(path)) for path in wanted}
                selected_rows = tuple(
                    index for index in range(row_count)
                    if ((stored := paths.string(index)) in wanted
                        or os.path.normcase(os.path.realpath(stored))
                        in wanted_canonical))
                paths = _subset_array(paths, selected_rows)
            arrays['paths'] = paths
        for info in infos:
            name = info.filename[:-4]
            if name == 'paths':
                continue
            selection = None
            if selected_rows is not None and name in _ROW_ALIGNED_NAMES:
                # Parse once to learn whether this array is row-aligned. Opening
                # twice costs a tiny header but prevents retaining an unwanted
                # 200k x 768 matrix in memory.
                with archive.open(info) as entry:
                    _descr, shape, _item_size, _payload = _parse_header(
                        entry, info.file_size, max_elements=max_elements)
                if shape[0] == row_count:
                    selection = selected_rows
            arrays[name] = _decode_entry(
                archive, info, max_elements=max_elements,
                selected_rows=selection)
        return arrays


def _valid_limits(max_file_bytes, max_uncompressed_bytes, max_elements):
    return all(not isinstance(value, bool) and isinstance(value, int) and value >= 0
               for value in (max_file_bytes, max_uncompressed_bytes, max_elements))


def read(path, *, max_file_bytes, max_uncompressed_bytes, max_elements,
         wanted_paths=None):
    path = Path(path)
    if not _valid_limits(max_file_bytes, max_uncompressed_bytes, max_elements):
        return None
    try:
        if not path.is_file() or path.stat().st_size > max_file_bytes:
            return None
        return _read_archive(
            path, max_uncompressed_bytes=max_uncompressed_bytes,
            max_elements=max_elements, wanted_paths=wanted_paths)
    except (OSError, EOFError, RuntimeError, ValueError, TypeError, struct.error,
            zlib.error, lzma.LZMAError,
            MemoryError, OverflowError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return None


def read_bytes(data, *, max_file_bytes, max_uncompressed_bytes, max_elements,
               wanted_paths=None):
    """Decode immutable in-memory NPZ bytes, useful across archive seams."""
    if (not isinstance(data, (bytes, bytearray, memoryview))
            or not _valid_limits(
                max_file_bytes, max_uncompressed_bytes, max_elements)):
        return None
    try:
        if len(data) > max_file_bytes:
            return None
        # Sidecars are capped at 64 KiB by their caller.  BytesIO is the only
        # in-memory seam here; payload arrays themselves are still streamed.
        source = io.BytesIO(data if isinstance(data, bytes) else bytes(data))
        return _read_archive(
            source, max_uncompressed_bytes=max_uncompressed_bytes,
            max_elements=max_elements, wanted_paths=wanted_paths)
    except (OSError, EOFError, RuntimeError, ValueError, TypeError, struct.error,
            zlib.error, lzma.LZMAError,
            MemoryError, OverflowError, zipfile.BadZipFile, zipfile.LargeZipFile):
        return None


def _constructor_shape(shape):
    try:
        shape = tuple(shape)
    except (TypeError, MemoryError):
        return None
    return shape if _valid_shape(shape) else None


def floats(values, shape):
    shape = _constructor_shape(shape)
    if shape is None:
        return None
    count = _product(shape)
    try:
        data = bytearray(count * 4)
        iterator = iter(values)
        for index in range(count):
            number = float(next(iterator))
            if not math.isfinite(number) and not math.isnan(number):
                raise ValueError
            struct.pack_into('<f', data, index * 4, number)
        try:
            next(iterator)
            raise ValueError
        except StopIteration:
            pass
        return Array('<f4', shape, data)
    except (StopIteration, TypeError, ValueError, OverflowError, struct.error,
            MemoryError):
        return None


def int32(values, shape):
    shape = _constructor_shape(shape)
    if shape is None:
        return None
    count = _product(shape)
    try:
        data = bytearray(count * 4)
        iterator = iter(values)
        for index in range(count):
            value = next(iterator)
            if isinstance(value, bool):
                raise ValueError
            struct.pack_into('<i', data, index * 4, int(value))
        try:
            next(iterator)
            raise ValueError
        except StopIteration:
            pass
        return Array('<i4', shape, data)
    except (StopIteration, TypeError, ValueError, OverflowError, struct.error,
            MemoryError):
        return None


def uint8(values, shape):
    shape = _constructor_shape(shape)
    if shape is None:
        return None
    count = _product(shape)
    try:
        data = bytearray(count)
        iterator = iter(values)
        for index in range(count):
            value = next(iterator)
            if isinstance(value, bool):
                value = int(value)
            else:
                value = int(value)
            if not 0 <= value <= 255:
                raise ValueError
            data[index] = value
        try:
            next(iterator)
            raise ValueError
        except StopIteration:
            pass
        return Array('|u1', shape, data)
    except (StopIteration, TypeError, ValueError, OverflowError, MemoryError):
        return None


def unicode(values):
    try:
        strings = []
        width = 1
        characters = 0
        for raw in values:
            if len(strings) >= _MAX_ROWS:
                raise ValueError
            value = str(raw)
            characters += len(value)
            if characters > _MAX_PATH_CHARACTERS:
                raise ValueError
            width = max(width, len(value))
            if width > 4096:
                raise ValueError
            strings.append(value)
        data = bytearray(len(strings) * width * 4)
        for row, value in enumerate(strings):
            offset = row * width * 4
            for column, char in enumerate(value):
                codepoint = ord(char)
                if codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
                    raise ValueError
                struct.pack_into('<I', data, offset + column * 4, codepoint)
        return Array(f'<U{width}', (len(strings),), data)
    except (TypeError, ValueError, OverflowError, struct.error, MemoryError):
        return None


def _npy_header(array):
    header = repr({
        'descr': array.descr,
        'fortran_order': False,
        'shape': array.shape,
    })
    prefix_len = 10
    padding = (-((prefix_len + len(header) + 1) % 64)) % 64
    encoded = (header + (' ' * padding) + '\n').encode('latin1')
    if len(encoded) > min(65535, _MAX_HEADER_BYTES):
        raise ValueError('NPY header too large')
    return _MAGIC + bytes((1, 0)) + struct.pack('<H', len(encoded)) + encoded


def _validated_array(array):
    if not isinstance(array, Array) or not _valid_shape(array.shape):
        return False
    item_size = _dtype_size(array.descr)
    return (item_size is not None
            and len(array.data) == _product(array.shape) * item_size)


def write_atomic(path, arrays, *, max_file_bytes):
    if (isinstance(max_file_bytes, bool) or not isinstance(max_file_bytes, int)
            or max_file_bytes < 0 or not isinstance(arrays, dict) or not arrays
            or len(arrays) > _MAX_ENTRIES
            or any(not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name)
                   or not _validated_array(array)
                   for name, array in arrays.items())):
        return False
    try:
        if sum(len(array.data) for array in arrays.values()) \
                > _MAX_WRITTEN_UNCOMPRESSED:
            return False
        if sum(len(array.data) for name, array in arrays.items()
               if name in _INDEX_NAMES) > _MAX_INDEX_BYTES:
            return False
        if 'paths' in arrays:
            _path_character_count(arrays['paths'])
    except (TypeError, ValueError, OverflowError, MemoryError):
        return False
    target = Path(path)
    write_lock = _WRITE_LOCKS[
        hash(os.path.normcase(os.path.abspath(target))) % len(_WRITE_LOCKS)]
    tmp = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent)
        tmp = Path(tmp_name)
        with os.fdopen(fd, 'w+b') as raw:
            with zipfile.ZipFile(raw, 'w', compression=zipfile.ZIP_DEFLATED,
                                 allowZip64=True) as archive:
                for name, array in arrays.items():
                    with archive.open(f'{name}.npy', 'w', force_zip64=True) as entry:
                        entry.write(_npy_header(array))
                        view = memoryview(array.data)
                        for offset in range(0, len(view), _IO_CHUNK):
                            entry.write(view[offset:offset + _IO_CHUNK])
            raw.flush()
            os.fsync(raw.fileno())
        if tmp.stat().st_size > max_file_bytes:
            return False
        # Windows can reject two simultaneous ReplaceFile operations on the
        # same destination even though both sibling tempfiles are complete.
        # A bounded striped lock serialises only the publication seam.
        with write_lock:
            os.replace(tmp, target)
        tmp = None
        return True
    except (OSError, ValueError, TypeError, RuntimeError, struct.error,
            MemoryError, OverflowError, zipfile.BadZipFile,
            zipfile.LargeZipFile):
        return False
    finally:
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
