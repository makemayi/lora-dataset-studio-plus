"""SigLIP2 Bank semantics: provenance, cache safety and paired workers.

All ML objects are stubs.  These tests never resolve or download a checkpoint.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest
from PIL import Image

from app.services import bank_semantic_engine as semantic
from app.services import bank_semantic_models as models


np = pytest.importorskip('numpy')
INFER_DIR = Path(__file__).resolve().parents[1] / 'infer'


def test_semantic_python_prefers_its_key_then_legacy_score_fallback(app):
    from app import config
    with app.app_context():
        assert config.DEFAULTS['bank_semantic']['python'] == ''
        config.save_config({
            'bank_scoring': {'python': '/legacy/score/python'},
            'bank_semantic': {'python': '/semantic/python'},
        })
        assert models.semantic_python() == '/semantic/python'
        config.save_config({'bank_semantic': {'python': ''}})
        assert models.semantic_python() == '/legacy/score/python'
        config.save_config({'bank_scoring': {'python': ''}})
        assert models.semantic_python() == sys.executable


def _load_script(name: str):
    path = INFER_DIR / f'{name}.py'
    spec = importlib.util.spec_from_file_location(f'{name}_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def isolated_semantic(tmp_path, monkeypatch):
    banks = tmp_path / 'banks'
    banks.mkdir()
    monkeypatch.setattr(semantic.cfg, 'banks_root', lambda: banks)
    monkeypatch.setattr(models, 'models_root', lambda: tmp_path / 'models')
    semantic.reset_memo()
    yield banks
    semantic.reset_memo()


def _unit(dimension: int, index: int = 0):
    vector = np.zeros(dimension, dtype='float32')
    vector[index] = 1.0
    return vector


def test_cache_provenance_shape_and_cross_engine_refusal(isolated_semantic, tmp_path):
    worker = _load_script('bank_semantic_infer')
    bank = SimpleNamespace(id=7, semantic_engine='siglip2',
                           source_path=str(tmp_path))
    image = tmp_path / 'image.bin'
    image.write_bytes(b'exact image generation')
    signature = worker._file_sig(str(image))
    digest = worker._file_hash(str(image), signature)
    contract = semantic.semantic_contract()
    cache_path = semantic.semantic_cache_path(bank, 'siglip2')
    cache_path.parent.mkdir(parents=True)
    worker._save_cache(str(cache_path), {
        str(image): ('ok', _unit(models.DIMENSION), signature, digest),
    }, contract)

    with np.load(cache_path, allow_pickle=False) as saved:
        assert set(saved.files) == worker._CACHE_KEYS
        for key in ('version', 'engine', 'model_id', 'revision',
                    'model_key', 'dimension'):
            assert saved[key].shape == (1,)
        assert saved['version'].dtype == np.dtype('int32')
        assert saved['dimension'].dtype == np.dtype('int32')

    loaded = semantic.load_semantic_embeddings(bank)
    assert list(loaded) == [str(image)]
    assert semantic.embedding_fingerprint(str(image), 'siglip2') == digest.hex()
    counts = semantic.semantic_counts(bank, total=1)
    assert counts['engine'] == 'siglip2'
    assert counts['model_key'] == models.MODEL_KEY
    assert counts['source'] == 'semantic_cache'
    assert counts['ready'] is True and counts['complete'] is True
    assert counts['needs_index'] is False and counts['ok'] == 1

    # A valid NPZ structure from another key/dimension is not a partial hit: the
    # whole cache is refused so incomparable vectors never meet.
    wrong = {**contract, 'model_key': 'another-space', 'dimension': 4}
    worker._save_cache(str(cache_path), {
        str(image): ('ok', _unit(4), signature, digest),
    }, wrong)
    semantic.reset_memo()
    assert semantic.load_semantic_embeddings(bank) == {}
    refused = semantic.semantic_counts(bank, total=1)
    assert refused['ready'] is False and refused['complete'] is False
    assert refused['needs_index'] is True and refused['error']


def test_historical_clip_cache_stays_readable(isolated_semantic, tmp_path):
    bank = SimpleNamespace(id=8, semantic_engine='clip', source_path=str(tmp_path))
    image = tmp_path / 'legacy.jpg'
    image.write_bytes(b'legacy score bytes')
    path = semantic.semantic_cache_path(bank, 'clip')
    path.parent.mkdir(parents=True)
    # Earliest score caches had neither provenance nor stat/SHA arrays.
    np.savez_compressed(
        path,
        paths=np.asarray([str(image)]),
        states=np.asarray(['ok']),
        embs=np.stack([_unit(semantic.CLIP_DIMENSION)]).astype('float32'),
    )
    loaded = semantic.load_semantic_embeddings(bank)
    assert list(loaded) == [str(image)]
    info = semantic.semantic_counts(bank, total=1)
    assert info['source'] == 'score'
    assert info['model_key'] == 'clip-vit-l-14-openai'
    # The pure loader keeps the pre-hash format readable, but production/poll
    # readiness is strict: an entry with no stored SHA has no transfer authority.
    assert info['ready'] is False and info['complete'] is False


def test_counts_never_hash_image_bytes(isolated_semantic, tmp_path, monkeypatch):
    worker = _load_script('bank_semantic_infer')
    bank = SimpleNamespace(id=81, semantic_engine='siglip2')
    image = tmp_path / 'poll.bin'
    image.write_bytes(b'polling must not reopen this payload')
    signature = worker._file_sig(str(image))
    digest = worker._file_hash(str(image), signature)
    contract = semantic.semantic_contract()
    cache = semantic.semantic_cache_path(bank, 'siglip2')
    cache.parent.mkdir(parents=True)
    worker._save_cache(str(cache), {
        str(image): ('ok', _unit(contract['dimension']), signature, digest),
    }, contract)
    monkeypatch.setattr(semantic, '_sha256_path',
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            AssertionError('semantic_counts hashed image bytes')))
    monkeypatch.setattr(np, 'isfinite',
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            AssertionError('semantic_counts scanned embeddings')))
    info = semantic.semantic_counts(bank, total=1)
    assert info['ready'] is True and info['ok'] == 1
    monkeypatch.setattr(semantic, '_parsed_cache',
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            AssertionError('fast-count memo reparsed the NPZ')))
    again = semantic.semantic_counts(bank, total=1)
    assert again['ok'] == 1 and again['stale'] == 0


def test_counts_intersect_live_eligible_paths_and_memo_never_leaks_ok_over_total(
        isolated_semantic, tmp_path):
    worker = _load_script('bank_semantic_infer')
    bank = SimpleNamespace(id=82, semantic_engine='siglip2')
    paths = []
    entries = {}
    contract = semantic.semantic_contract()
    for index in range(2):
        path = tmp_path / f'eligible-{index}.bin'
        path.write_bytes(f'payload-{index}'.encode())
        signature = worker._file_sig(str(path))
        entries[str(path)] = (
            'ok', _unit(contract['dimension'], index), signature,
            worker._file_hash(str(path), signature))
        paths.append(str(path))
    cache = semantic.semantic_cache_path(bank, 'siglip2')
    cache.parent.mkdir(parents=True)
    worker._save_cache(str(cache), entries, contract)

    one = semantic.semantic_counts(
        bank, total=1, eligible_paths=[paths[0]])
    assert (one['total'], one['cached'], one['ok'], one['stale']) == (1, 1, 1, 0)
    none = semantic.semantic_counts(bank, total=0, eligible_paths=[])
    assert (none['total'], none['cached'], none['ok'], none['stale']) == (0, 0, 0, 0)
    assert none['complete'] is True and none['ready'] is False


@pytest.mark.parametrize('bad', [None, True, False, 0, -1, '7', 'bank',
                                 '../7', Path('7')])
def test_bank_id_is_a_positive_integer_only(isolated_semantic, bad):
    with pytest.raises(ValueError):
        semantic.semantic_cache_path(bad, 'siglip2')

    assert semantic.semantic_cache_path(7, 'siglip2').parent.name == '7'
    assert semantic.semantic_cache_path(
        SimpleNamespace(id=8), 'siglip2').parent.name == '8'


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value, dtype='float32')

    def norm(self, dim=None, keepdim=False):
        return _Tensor(np.linalg.norm(self.value, axis=-1, keepdims=keepdim))

    def __truediv__(self, other):
        return _Tensor(self.value / other.value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Batch(dict):
    def to(self, device):
        self.device = device
        return self


def _install_ml_stubs(monkeypatch, dimension, *, cancel_file=None):
    calls = {'processor_load': [], 'model_load': [], 'processor': [],
             'image_features': 0, 'text_features': 0}
    torch = types.ModuleType('torch')
    torch.cuda = SimpleNamespace(is_available=lambda: False)

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    torch.no_grad = _NoGrad

    class _Processor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls['processor_load'].append((model_id, kwargs))
            return cls()

        def __call__(self, **kwargs):
            calls['processor'].append(kwargs)
            return _Batch()

    class _Model:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls['model_load'].append((model_id, kwargs))
            return cls()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def get_image_features(self, **_kwargs):
            calls['image_features'] += 1
            if cancel_file and calls['image_features'] == 1:
                Path(cancel_file).write_text('1', encoding='utf-8')
            return SimpleNamespace(
                pooler_output=_Tensor(_unit(dimension)[None, :]))

        def get_text_features(self, **_kwargs):
            calls['text_features'] += 1
            return SimpleNamespace(
                pooler_output=_Tensor(_unit(dimension, 1)[None, :]))

    transformers = types.ModuleType('transformers')
    transformers.AutoProcessor = _Processor
    # AutoModel, because that is what the workers import: the pinned SigLIP 2
    # checkpoint is a FIXED-RESOLUTION variant whose config declares
    # `model_type: siglip`, so naming Siglip2Model by hand made transformers
    # refuse the weights. A double that stubs a class production no longer calls
    # is a double that stops proving anything — it would leave every assertion
    # below green while the real worker died on load.
    transformers.AutoModel = _Model
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'transformers', transformers)
    return calls


def _run_image_worker(worker, monkeypatch, request):
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(request)))
    stdout = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', stdout)
    code = worker.main()
    lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    return code, json.loads(lines[-1])


def test_image_worker_resumes_and_invalidates_with_stat_plus_sha(
        isolated_semantic, tmp_path, monkeypatch):
    worker = _load_script('bank_semantic_infer')
    contract = semantic.semantic_contract()
    cache = isolated_semantic / '9' / semantic.SIGLIP2_CACHE_FILENAME
    images = []
    for index in range(2):
        path = tmp_path / f'{index}.bmp'
        Image.new('RGB', (16, 16), (30 + index, 40, 50)).save(path, 'BMP')
        images.append(str(path))
    calls = _install_ml_stubs(monkeypatch, contract['dimension'])
    base = {**contract, 'cache': str(cache), 'cancel_file': str(cache) + '.cancel',
            'device': 'cpu'}

    code, first = _run_image_worker(
        worker, monkeypatch, {**base, 'images': images[:1]})
    assert code == 0 and first['ok'] and calls['image_features'] == 1
    code, resumed = _run_image_worker(worker, monkeypatch, {**base, 'images': images})
    assert code == 0 and resumed['computed'] == 1 and resumed['reused'] == 1
    assert calls['image_features'] == 2
    assert Path(str(cache) + '.count').read_text(encoding='utf-8') == '2'

    # Change pixels while preserving size and mtime: a stat-only resume would
    # silently reuse the old vector; the exact SHA must force one re-embed.
    changed = Path(images[0])
    before = changed.stat()
    payload = bytearray(changed.read_bytes())
    payload[-1] ^= 1
    changed.write_bytes(payload)
    os.utime(changed, ns=(before.st_atime_ns, before.st_mtime_ns))
    code, invalidated = _run_image_worker(
        worker, monkeypatch, {**base, 'images': images})
    assert code == 0 and invalidated['computed'] == 1
    assert invalidated['reused'] == 1 and calls['image_features'] == 3

    for _model_id, kwargs in calls['processor_load'] + calls['model_load']:
        assert kwargs == {
            'revision': contract['revision'],
            'cache_dir': contract['models_root'],
            'local_files_only': True,
        }
    assert not list(cache.parent.glob('*.tmp.npz'))


def test_image_worker_cancel_flushes_an_atomic_resumable_cache(
        isolated_semantic, tmp_path, monkeypatch):
    worker = _load_script('bank_semantic_infer')
    contract = semantic.semantic_contract()
    cache = isolated_semantic / '10' / semantic.SIGLIP2_CACHE_FILENAME
    cancel = str(cache) + '.cancel'
    paths = []
    for index in range(2):
        path = tmp_path / f'cancel-{index}.bmp'
        Image.new('RGB', (16, 16), (index, 10, 20)).save(path, 'BMP')
        paths.append(str(path))
    calls = _install_ml_stubs(
        monkeypatch, contract['dimension'], cancel_file=cancel)
    code, result = _run_image_worker(worker, monkeypatch, {
        **contract, 'images': paths, 'cache': str(cache),
        'cancel_file': cancel, 'device': 'cpu',
    })
    assert code == 0 and result['cancelled'] is True
    assert result['computed'] == 1 and result['cached'] == 1
    assert result['ready'] == 1 and result['failed'] == 0
    assert calls['image_features'] == 1 and cache.is_file()
    loaded = worker._load_cache(str(cache), contract)
    assert list(loaded) == paths[:1]
    assert not list(cache.parent.glob('*.tmp.npz'))


def test_image_worker_retries_a_cached_error(
        isolated_semantic, tmp_path, monkeypatch):
    worker = _load_script('bank_semantic_infer')
    contract = semantic.semantic_contract()
    cache = isolated_semantic / '11' / semantic.SIGLIP2_CACHE_FILENAME
    image = tmp_path / 'retry.bmp'
    Image.new('RGB', (16, 16), (1, 2, 3)).save(image, 'BMP')
    signature = worker._file_sig(str(image))
    digest = worker._file_hash(str(image), signature)
    worker._save_cache(str(cache), {
        str(image): ('error', np.zeros(contract['dimension'], dtype='float32'),
                     signature, digest),
    }, contract)
    failed = worker._result_payload(
        contract, [str(image)], worker._load_cache(str(cache), contract),
        computed=1, reused=0)
    assert failed['ready'] == 0 and failed['failed'] == 1
    assert failed['remaining'] == 1 and 'results' not in failed
    calls = _install_ml_stubs(monkeypatch, contract['dimension'])
    code, result = _run_image_worker(worker, monkeypatch, {
        **contract, 'images': [str(image)], 'cache': str(cache), 'device': 'cpu',
    })
    assert code == 0 and result['computed'] == 1 and result['reused'] == 0
    assert result['ready'] == 1 and result['failed'] == 0
    assert calls['image_features'] == 1


def test_success_payload_is_bounded_for_a_200k_bank(isolated_semantic):
    worker = _load_script('bank_semantic_infer')
    contract = semantic.semantic_contract()
    images = [f'C:/bank/{index}.jpg' for index in range(200_000)]
    entry = ('ok', _unit(contract['dimension']), '1:1', b'x' * 32)
    cache = dict.fromkeys(images, entry)
    payload = worker._result_payload(
        contract, images, cache, computed=200_000, reused=0)
    encoded = json.dumps(payload)
    assert payload['ready'] == 200_000 and payload['failed'] == 0
    assert 'results' not in payload and 'C:/bank/' not in encoded
    assert len(encoded) < 1024


def test_text_worker_is_local_cpu_and_uses_the_paired_token_contract(
        isolated_semantic, monkeypatch):
    worker = _load_script('siglip2_text_infer')
    contract = semantic.text_worker_handshake()
    calls = _install_ml_stubs(monkeypatch, contract['dimension'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'test-visible')
    stdin = io.StringIO(json.dumps(contract) + '\n' + json.dumps({'text': 'red dress'}) + '\n')
    stdout = io.StringIO()
    monkeypatch.setattr(sys, 'stdin', stdin)
    monkeypatch.setattr(sys, 'stdout', stdout)
    assert worker.main() == 0
    messages = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert messages[0]['ready'] is True
    assert messages[0]['model_key'] == contract['model_key']
    assert len(messages[1]['vector']) == contract['dimension']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    for _model_id, kwargs in calls['processor_load'] + calls['model_load']:
        assert kwargs['revision'] == contract['revision']
        assert kwargs['cache_dir'] == contract['models_root']
        assert kwargs['local_files_only'] is True
    query = calls['processor'][0]
    assert query['text'] == ['red dress']
    assert query['padding'] == 'max_length'
    assert query['truncation'] is True and query['max_length'] == 64
    assert calls['text_features'] == 1
