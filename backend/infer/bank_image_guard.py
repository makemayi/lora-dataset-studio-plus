"""Self-contained input guard for live Image Bank inference workers.

The Bank source directory is deliberately live: a file accepted by the Flask
parent can be replaced before its path reaches a dedicated ML interpreter.
Workers therefore must consume *only* the bounded bytes returned here, never
re-open the original path for decoding.  This module intentionally depends on
Pillow plus the standard library only; the infer environments do not import the
Flask app or its services.
"""
from __future__ import annotations

import io
import os
import stat
import warnings

from PIL import Image, UnidentifiedImageError


MAX_FILE_BYTES = 96 * 1024 * 1024
MAX_SIDE = 8192
MAX_PIXELS = 16 * 1024 * 1024
STATIC_FORMATS = frozenset(("JPEG", "PNG", "WEBP", "BMP"))


class BankImageGuardError(ValueError):
    """The supplied live Bank path cannot safely enter an infer worker."""


def _read_bounded_regular_file(path: str) -> bytes:
    """Return exactly one regular file's bytes without retaining over 96 MiB.

    The descriptor is checked rather than the path alone, so an atomic
    replacement between a parent's preflight and this worker is harmless: the
    bytes inspected below are the only bytes later given to cv2/Pillow.  A
    one-byte post-read probe catches a file that grew after ``fstat`` without
    appending that byte to the retained payload.
    """
    try:
        with open(path, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise BankImageGuardError("bank image is not a regular file")
            if info.st_size > MAX_FILE_BYTES:
                raise BankImageGuardError(
                    f"bank image is too large (max {MAX_FILE_BYTES // (1024 * 1024)} MiB)")
            payload = source.read(MAX_FILE_BYTES)
            # A concurrent append can invalidate the fstat result.  Do not add
            # this sentinel byte to ``payload``: peak retained image data stays
            # at the published 96 MiB ceiling.
            if source.read(1):
                raise BankImageGuardError(
                    f"bank image is too large (max {MAX_FILE_BYTES // (1024 * 1024)} MiB)")
    except BankImageGuardError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise BankImageGuardError("bank image is unavailable") from exc
    return payload


def _validate_image_bytes(payload: bytes) -> None:
    """Validate a raster header/content without materialising its pixels."""
    try:
        # Pillow's default decompression-bomb policy is process-global.  Infer
        # workers must not alter it for another task in the same interpreter, so
        # promote the warning only inside this short header-validation scope.
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                fmt = (image.format or "").upper()
                if fmt not in STATIC_FORMATS:
                    raise BankImageGuardError(
                        f"bank image has unsupported format: {fmt or 'unknown'}")
                if getattr(image, "n_frames", 1) != 1:
                    raise BankImageGuardError("bank image must be a static image")
                width, height = image.size
                if (not isinstance(width, int) or not isinstance(height, int)
                        or width <= 0 or height <= 0
                        or width > MAX_SIDE or height > MAX_SIDE
                        or width * height > MAX_PIXELS):
                    raise BankImageGuardError(
                        f"bank image rejects images above {MAX_SIDE} px per side or "
                        f"{MAX_PIXELS} pixels (got {width}x{height})")
                # ``verify`` validates container integrity without asking a
                # caller to decode a second, path-raced file.
                image.verify()
    except BankImageGuardError:
        raise
    except (OSError, SyntaxError, ValueError, UnidentifiedImageError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise BankImageGuardError("bank image is invalid") from exc


def read_validated_bank_image(path: str) -> bytes:
    """Read and validate one live Bank path, returning its safe decode snapshot.

    Callers must pass the returned bytes to their decoder (``cv2.imdecode`` or
    ``Image.open(BytesIO(...))``).  Re-opening ``path`` afterwards would restore
    the time-of-check/time-of-use race this guard exists to remove.
    """
    payload = _read_bounded_regular_file(path)
    _validate_image_bytes(payload)
    return payload
