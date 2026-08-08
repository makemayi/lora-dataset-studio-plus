"""Regression coverage for the live-Bank input guard used by infer children."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import pathlib
import struct
import sys
import types

import pytest
from PIL import Image


INFER = pathlib.Path(__file__).resolve().parents[1] / 'infer'


def _load_infer(name: str):
    """Load a standalone infer script as its own module, like its ML venv does."""
    if str(INFER) not in sys.path:
        sys.path.insert(0, str(INFER))
    spec = importlib.util.spec_from_file_location(f'{name}_guard_test', INFER / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _guard():
    return _load_infer('bank_image_guard')


def _compact_bmp(path: pathlib.Path, width: int, height: int) -> None:
    """A header-only BMP: tiny on disk, but it advertises the supplied raster."""
    file_header = struct.pack('<2sIHHI', b'BM', 54, 0, 0, 54)
    dib_header = struct.pack('<IiiHHIIiiII', 40, width, height, 1, 24, 0,
                             0, 0, 0, 0, 0)
    path.write_bytes(file_header + dib_header)


def _small_image(path: pathlib.Path, fmt: str) -> None:
    Image.new('RGB', (16, 12), (20, 30, 40)).save(path, fmt)


def _last_json(capsys):
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert lines
    return json.loads(lines[-1])


def test_compact_9000px_bmp_is_rejected_before_pillow_decode(tmp_path, monkeypatch):
    guard = _guard()
    bomb = tmp_path / 'compact.bmp'
    _compact_bmp(bomb, 9000, 9000)

    loaded = []
    original_load = Image.Image.load

    def tracking_load(self, *args, **kwargs):
        loaded.append(self)
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(Image.Image, 'load', tracking_load)
    with pytest.raises(guard.BankImageGuardError, match='above 8192 px'):
        guard.read_validated_bank_image(str(bomb))
    assert not loaded                  # header budget ran before a pixel decode


def test_guard_rejects_invalid_bytes_and_keeps_small_bmp_jpeg_usable(tmp_path):
    guard = _guard()
    invalid = tmp_path / 'invalid.jpg'
    invalid.write_bytes(b'not an image')
    with pytest.raises(guard.BankImageGuardError, match='invalid'):
        guard.read_validated_bank_image(str(invalid))

    for suffix, fmt, expected in (('bmp', 'BMP', 'BMP'), ('jpg', 'JPEG', 'JPEG')):
        path = tmp_path / f'valid.{suffix}'
        _small_image(path, fmt)
        snapshot = guard.read_validated_bank_image(str(path))
        with Image.open(io.BytesIO(snapshot)) as decoded:
            assert decoded.format == expected
            assert decoded.size == (16, 12)


def test_replacement_after_parent_preflight_is_rejected_before_worker_decode(tmp_path):
    """A path that was safe when scanned cannot smuggle in a compact bomb later."""
    guard = _guard()
    live = tmp_path / 'live.jpg'
    _small_image(live, 'JPEG')
    assert guard.read_validated_bank_image(str(live))  # equivalent safe preflight

    # Same filename, now a different unsafe file before the child reads it.
    _compact_bmp(live, 9000, 9000)
    with pytest.raises(guard.BankImageGuardError, match='above 8192 px'):
        guard.read_validated_bank_image(str(live))


def test_face_embed_marks_rejected_snapshot_error_without_cv2_decode(
        tmp_path, monkeypatch, capsys):
    pytest.importorskip('numpy')
    bad = tmp_path / 'late-replaced.bmp'
    _compact_bmp(bad, 9000, 9000)

    cv2 = types.ModuleType('cv2')
    cv2.IMREAD_COLOR = 1
    cv2.calls = 0

    def imdecode(*_args, **_kwargs):
        cv2.calls += 1
        return None

    cv2.imdecode = imdecode
    onnxruntime = types.ModuleType('onnxruntime')
    onnxruntime.get_available_providers = lambda: []
    insightface = types.ModuleType('insightface')
    insightface.__path__ = []
    insightface_app = types.ModuleType('insightface.app')

    class FaceAnalysis:
        def __init__(self, **_kwargs):
            pass

        def prepare(self, **_kwargs):
            pass

    insightface_app.FaceAnalysis = FaceAnalysis
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    monkeypatch.setitem(sys.modules, 'onnxruntime', onnxruntime)
    monkeypatch.setitem(sys.modules, 'insightface', insightface)
    monkeypatch.setitem(sys.modules, 'insightface.app', insightface_app)

    module = _load_infer('face_embed_infer')
    monkeypatch.setattr(module, '_repair_nested_antelopev2', lambda _root: None)
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({
        'images': [str(bad)], 'models_root': str(tmp_path / 'models'), 'device': 'cpu'})))
    assert module.main() == 0
    out = _last_json(capsys)
    assert out['results'][str(bad)]['state'] == 'error'
    assert cv2.calls == 0


def test_face_embed_does_not_commit_result_when_file_changes_during_model_call(
        tmp_path, monkeypatch, capsys):
    np = pytest.importorskip('numpy')
    live = tmp_path / 'live.jpg'
    _small_image(live, 'JPEG')
    cache_path = tmp_path / 'face-cache.npz'

    cv2 = types.ModuleType('cv2')
    cv2.IMREAD_COLOR = 1
    cv2.imdecode = lambda *_args, **_kwargs: np.zeros((12, 16, 3), dtype='uint8')
    onnxruntime = types.ModuleType('onnxruntime')
    onnxruntime.get_available_providers = lambda: []
    insightface = types.ModuleType('insightface')
    insightface.__path__ = []
    insightface_app = types.ModuleType('insightface.app')

    face = types.SimpleNamespace(
        bbox=np.array([1, 1, 12, 10]), det_score=0.95,
        pose=np.array([0.0, 3.0, 0.0]),
        normed_embedding=np.zeros(512, dtype='float32'))

    class FaceAnalysis:
        def __init__(self, **_kwargs):
            pass

        def prepare(self, **_kwargs):
            pass

        def get(self, _image):
            # Same path, different identity while the model is working.
            live.write_bytes(live.read_bytes() + b'replacement')
            return [face]

    insightface_app.FaceAnalysis = FaceAnalysis
    monkeypatch.setitem(sys.modules, 'cv2', cv2)
    monkeypatch.setitem(sys.modules, 'onnxruntime', onnxruntime)
    monkeypatch.setitem(sys.modules, 'insightface', insightface)
    monkeypatch.setitem(sys.modules, 'insightface.app', insightface_app)

    module = _load_infer('face_embed_infer')
    monkeypatch.setattr(module, '_repair_nested_antelopev2', lambda _root: None)
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({
        'images': [str(live)], 'models_root': str(tmp_path / 'models'),
        'device': 'cpu', 'cache': str(cache_path),
    })))
    assert module.main() == 0
    out = _last_json(capsys)
    assert out['results'][str(live)]['state'] == 'error'
    assert not cache_path.exists()
    assert module._load_cache(str(cache_path)) == {}


def test_bank_score_does_not_commit_embedding_when_file_changes_during_clip(
        tmp_path, monkeypatch, capsys):
    np = pytest.importorskip('numpy')
    live = tmp_path / 'live-score.jpg'
    _small_image(live, 'JPEG')
    cache_path = tmp_path / 'score-cache.npz'

    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    torch.no_grad = NoGrad
    open_clip = types.ModuleType('open_clip')

    class Input:
        def unsqueeze(self, _axis):
            return self

        def to(self, _device):
            return self

    class Embedding:
        def __init__(self):
            self.value = np.ones((1, 768), dtype='float32')

        def norm(self, **_kwargs):
            return 1.0

        def __truediv__(self, _other):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Clip:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def encode_image(self, _input):
            live.write_bytes(live.read_bytes() + b'replacement')
            return Embedding()

    open_clip.create_model_and_transforms = (
        lambda *_args, **_kwargs: (Clip(), None, lambda _image: Input()))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'open_clip', open_clip)

    module = _load_infer('bank_score_infer')
    monkeypatch.setattr(module, '_load_aesthetic_head',
                        lambda *_args: (None, False, 'offline'))
    monkeypatch.setattr(module, '_load_nsfw',
                        lambda *_args: (None, False, 'offline'))
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({
        'images': [str(live)], 'models_root': str(tmp_path / 'models'),
        'cache': str(cache_path),
    })))
    assert module.main() == 0
    out = _last_json(capsys)
    assert out['results'][str(live)]['state'] == 'error'
    assert not cache_path.exists()
    assert module._load_cache(str(cache_path)) == {}


def test_infer_caches_store_sha256_as_compact_uint8_rows(tmp_path):
    np = pytest.importorskip('numpy')
    digest = hashlib.sha256(b'exact image payload').digest()

    score = _load_infer('bank_score_infer')
    score_path = tmp_path / 'score.npz'
    score._save_cache(str(score_path), {
        'one.jpg': ('ok', 7.0, 0.1, np.zeros(768, dtype='float32'),
                    '123:456', digest),
    })
    with np.load(score_path, allow_pickle=False) as archive:
        assert archive['hashes'].dtype == np.dtype('uint8')
        assert archive['hashes'].shape == (1, 32)
    assert score._load_cache(str(score_path))['one.jpg'][5] == digest

    face = _load_infer('face_embed_infer')
    face_path = tmp_path / 'face.npz'
    face._save_cache(str(face_path), {
        'one.jpg': ('scorable', 0.9, 0.2,
                    np.zeros(512, dtype='float32'), 3.0, '123:456', digest),
    })
    with np.load(face_path, allow_pickle=False) as archive:
        assert archive['hashes'].dtype == np.dtype('uint8')
        assert archive['hashes'].shape == (1, 32)
    assert face._load_cache(str(face_path))['one.jpg'][6] == digest


def test_bank_score_marks_rejected_snapshot_error_without_second_pillow_open(
        tmp_path, monkeypatch, capsys):
    pytest.importorskip('numpy')
    bad = tmp_path / 'late-replaced.bmp'
    _compact_bmp(bad, 9000, 9000)

    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    torch.no_grad = NoGrad
    open_clip = types.ModuleType('open_clip')

    class Clip:
        def to(self, _device):
            return self

        def eval(self):
            return self

    open_clip.create_model_and_transforms = lambda *_args, **_kwargs: (Clip(), None, lambda im: im)
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'open_clip', open_clip)

    opens = []
    real_open = Image.open

    def tracking_open(fp, *args, **kwargs):
        opens.append(fp)
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(Image, 'open', tracking_open)
    module = _load_infer('bank_score_infer')
    monkeypatch.setattr(module, '_load_aesthetic_head',
                        lambda *_args: (None, False, 'URLError: unreachable'))
    monkeypatch.setattr(module, '_load_nsfw',
                        lambda *_args: (None, False, 'OSError: unreachable'))
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({
        'images': [str(bad)], 'models_root': str(tmp_path / 'models')})))
    assert module.main() == 0
    out = _last_json(capsys)
    # Every result carries the exact-byte authority field. A source rejected
    # before hashing has no authority, explicitly, rather than an absent key
    # that an older parent could mistake for a legacy result.
    assert out['results'][str(bad)] == {
        'state': 'error', 'fingerprint': None}
    assert len(opens) == 1 and isinstance(opens[0], io.BytesIO)


def test_joycaption_reports_rejected_snapshot_in_errors(tmp_path, monkeypatch, capsys):
    bad = tmp_path / 'late-replaced.bmp'
    _compact_bmp(bad, 9000, 9000)

    torch = types.ModuleType('torch')
    torch.bfloat16 = 'bfloat16'

    class Linear:
        def __init__(self, *_args, **_kwargs):
            pass

    torch.nn = types.SimpleNamespace(Linear=Linear)

    transformers = types.ModuleType('transformers')
    transformers.__version__ = '4.50.0'

    class Tokenizer:
        def convert_tokens_to_ids(self, token):
            return {'<|end_header_id|>': 10, '<|eot_id|>': 11}[token]

        def apply_chat_template(self, *_args, **_kwargs):
            return 'prompt'

        def encode(self, *_args, **_kwargs):
            return [1]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return Tokenizer()

    class BitsAndBytesConfig:
        def __init__(self, **_kwargs):
            pass

    weight = types.SimpleNamespace(dtype='bfloat16', device='vision')
    vision = types.SimpleNamespace(
        vision_model=types.SimpleNamespace(
            head=types.SimpleNamespace(attention=types.SimpleNamespace(embed_dim=1)),
            embeddings=types.SimpleNamespace(
                patch_embedding=types.SimpleNamespace(weight=weight))))
    language = types.SimpleNamespace(
        get_input_embeddings=lambda: types.SimpleNamespace(weight=types.SimpleNamespace(device='language')))

    class Model:
        def __init__(self):
            self.device = 'model'
            self.config = types.SimpleNamespace(image_token_index=99, image_seq_length=1)
            self.model = types.SimpleNamespace(vision_tower=vision, language_model=language)

        def eval(self):
            return self

    class LlavaForConditionalGeneration:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return Model()

    transformers.AutoTokenizer = AutoTokenizer
    transformers.BitsAndBytesConfig = BitsAndBytesConfig
    transformers.LlavaForConditionalGeneration = LlavaForConditionalGeneration

    torchvision = types.ModuleType('torchvision')
    torchvision.__path__ = []
    transforms = types.ModuleType('torchvision.transforms')
    transforms.__path__ = []
    functional = types.ModuleType('torchvision.transforms.functional')
    transforms.functional = functional
    torchvision.transforms = transforms
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    monkeypatch.setitem(sys.modules, 'torchvision', torchvision)
    monkeypatch.setitem(sys.modules, 'torchvision.transforms', transforms)
    monkeypatch.setitem(sys.modules, 'torchvision.transforms.functional', functional)

    opens = []
    real_open = Image.open

    def tracking_open(fp, *args, **kwargs):
        opens.append(fp)
        return real_open(fp, *args, **kwargs)

    monkeypatch.setattr(Image, 'open', tracking_open)
    module = _load_infer('joycaption_infer')
    monkeypatch.setattr(module, '_model_is_cached', lambda: True)
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps({'images': [str(bad)]})))
    assert module.main() == 0
    out = _last_json(capsys)
    assert out['captions'] == {}
    assert str(bad) in out['errors']
    assert len(opens) == 1 and isinstance(opens[0], io.BytesIO)
