"""face_score_infer's face-sharpness helper, tested without onnxruntime.

The infer/ scripts are standalone (no __init__.py, run with a dedicated
interpreter), so the module is loaded by path and only the pure helper is
exercised — the onnxruntime path stays out of the unit suite.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

_INFER = pathlib.Path(__file__).resolve().parents[1] / 'infer' / 'face_score_infer.py'


@pytest.fixture(scope='module')
def infer_mod():
    spec = importlib.util.spec_from_file_location('face_score_infer', _INFER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lap_var_bgr_sharpness_ordering(infer_mod):
    """A smooth face crop must score FAR below a textured one — the whole point
    of the face_blurry gate is telling those two apart."""
    lap_var_bgr = infer_mod.lap_var_bgr

    smooth = np.full((64, 64, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(7)
    noisy = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

    v_smooth = lap_var_bgr(smooth)
    v_noisy = lap_var_bgr(noisy)
    assert v_smooth < v_noisy
    assert v_noisy > 10.0          # textured: definitely nonzero


def test_lap_var_bgr_handles_gray_crops(infer_mod):
    """A single-channel crop must not crash the helper (the caller converts)."""
    gray = np.full((32, 32), 100, dtype=np.uint8)
    assert infer_mod.lap_var_bgr(gray) == 0.0


def test_lap_var_bgr_tiny_crop_returns_zero(infer_mod):
    """A crop too small for the 3x3 kernel is unmeasurable, not a crash."""
    assert infer_mod.lap_var_bgr(np.zeros((2, 2, 3), dtype=np.uint8)) == 0.0
