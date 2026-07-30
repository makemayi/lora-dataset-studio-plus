"""Regression coverage for standalone inference helpers that must not use VRAM."""
from __future__ import annotations

import importlib.util
import io
import json
import pathlib
import sys
import types


INFER = pathlib.Path(__file__).resolve().parents[1] / 'infer'


def _load_infer(name: str):
    spec = importlib.util.spec_from_file_location(f'{name}_cpu_test', INFER / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_face_score_uses_cpu_provider_and_negative_context_id(monkeypatch):
    captured = {}

    class FaceAnalysis:
        def __init__(self, **kwargs):
            captured['init'] = kwargs

        def prepare(self, **kwargs):
            captured['prepare'] = kwargs

        def get(self, _image):
            return []

    numpy = types.ModuleType('numpy')
    onnxruntime = types.ModuleType('onnxruntime')
    onnxruntime.get_available_providers = lambda: []
    cv2 = types.ModuleType('cv2')
    cv2.BORDER_CONSTANT = 0
    cv2.imread = lambda _path: types.SimpleNamespace(shape=(640, 640, 3))
    cv2.copyMakeBorder = lambda image, *_args, **_kwargs: image
    insightface = types.ModuleType('insightface')
    insightface.__path__ = []
    insightface_app = types.ModuleType('insightface.app')
    insightface_app.FaceAnalysis = FaceAnalysis

    monkeypatch.setitem(sys.modules, 'numpy', numpy)
    monkeypatch.setitem(sys.modules, 'onnxruntime', onnxruntime)
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    monkeypatch.setitem(sys.modules, 'insightface', insightface)
    monkeypatch.setitem(sys.modules, 'insightface.app', insightface_app)
    module = _load_infer('face_score_infer')
    monkeypatch.setattr(module, '_repair_nested_antelopev2', lambda _root: None)
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({
        'ref': 'reference.jpg', 'images': ['candidate.jpg'], 'models_root': '/models',
    })))

    assert module.main() == 1  # no fake face is intentional; initialization still ran
    assert captured['init'] == {
        'name': 'antelopev2',
        'root': '/models',
        'providers': ['CPUExecutionProvider'],
    }
    assert captured['prepare']['ctx_id'] == -1


def test_person_mask_creates_rembg_session_with_cpu_provider(monkeypatch, tmp_path):
    captured = {}
    rembg = types.ModuleType('rembg')

    def new_session(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return object()

    rembg.new_session = new_session
    rembg.remove = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, 'rembg', rembg)
    module = _load_infer('mask_infer')
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({
        'images': [], 'out_dir': str(tmp_path / 'masks'),
    })))

    assert module.main() == 0
    assert captured == {
        'args': ('u2net',),
        'kwargs': {'providers': ['CPUExecutionProvider']},
    }
