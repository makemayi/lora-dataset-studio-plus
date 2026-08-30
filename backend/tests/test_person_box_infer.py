"""The local-first model loader for the Crop-to-person detector.

The policy the operator asked for, pinned here as tests: load from the LOCAL
cache and NEVER touch the network for a model that is already on disk (a machine
that cannot reach huggingface.co answers every HEAD request with a connection
timeout, and the default from_pretrained retries it forever, hanging the pass).
Only a genuinely-absent model falls back to the network, bounded, and a network
failure there is a clear error — never a silent unbounded retry.
"""
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
SCRIPT = os.path.abspath(os.path.join(HERE, '..', 'infer', 'person_box_infer.py'))


class FakeProcessor:
    pass


class FakeModel:
    pass


class _FakeTorch:
    cuda = None

    @staticmethod
    def set_num_threads(*_a):
        return None

    @staticmethod
    def is_available():
        return True


def _load_script(monkeypatch):
    """Load person_box_infer.py as a module under a private name. The script
    imports torch at module top; the test venv has no torch, so a stub is
    planted in sys.modules first (the real detector runs in the bank-scoring
    interpreter, which does)."""
    import types
    fake = types.ModuleType('torch')
    fake.cuda = _FakeTorch
    fake.set_num_threads = staticmethod(lambda *_a: None)
    fake.cuda.is_available = staticmethod(lambda: True)
    monkeypatch.setitem(sys.modules, 'torch', fake)

    spec = importlib.util.spec_from_file_location('_pbi_under_test', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub_transformers(monkeypatch, *, local_ok, net_ok):
    """Install a fake transformers into sys.modules; record the kwargs each
    from_pretrained call received."""
    calls = []

    def _from_pretrained(which):
        def fn(model_id, cache_dir=None, local_files_only=False, **_k):
            calls.append((which, model_id, cache_dir, local_files_only))
            if local_files_only and not local_ok:
                raise OSError('local cache miss')
            if not local_files_only and not net_ok:
                raise ConnectionError('cannot reach huggingface.co')
            return FakeProcessor() if which == 'processor' else FakeModel()
        return staticmethod(fn)

    fake = type('transformers', (), {
        'AutoProcessor': type('AP', (), {'from_pretrained': _from_pretrained('processor')}),
        'AutoModelForZeroShotObjectDetection': type(
            'AM', (), {'from_pretrained': _from_pretrained('model')}),
    })
    monkeypatch.setitem(sys.modules, 'transformers', fake)
    return calls


def test_a_model_in_the_local_cache_loads_without_any_network_call(monkeypatch):
    """The happy path: the model is cached, so BOTH from_pretrained calls run
    local_files_only=True and never hit the network."""
    calls = _stub_transformers(monkeypatch, local_ok=True, net_ok=False)
    mod = _load_script(monkeypatch)
    processor, model = mod._load_model('/some/cache/dir')

    assert calls and all(c[3] is True for c in calls), \
        'every load must be local_files_only when the model is cached'
    assert all(c[1] == 'IDEA-Research/grounding-dino-tiny' for c in calls)
    assert (processor, model) is not None


def test_a_missing_model_falls_back_to_network_then_reports_loudly(monkeypatch):
    """Not local -> the network attempt runs, and a network failure is a clear
    RuntimeError naming the model and the cause — not an unbounded retry."""
    calls = _stub_transformers(monkeypatch, local_ok=False, net_ok=False)
    mod = _load_script(monkeypatch)
    with pytest.raises(RuntimeError, match='could not be downloaded'):
        mod._load_model('/some/cache/dir')
    # The processor (loaded first) tried local-only, then the network; the
    # network failure became a loud error, never an unbounded retry.
    assert any(c[3] for c in calls), 'a local_files_only attempt ran'
    assert any(not c[3] for c in calls), 'the network fallback ran'
    assert len(calls) >= 2


def test_a_missing_model_that_downloads_succeeds(monkeypatch):
    """Not local, but the network is reachable: the download attempt succeeds."""
    calls = _stub_transformers(monkeypatch, local_ok=False, net_ok=True)
    mod = _load_script(monkeypatch)
    processor, model = mod._load_model('/some/cache/dir')
    assert processor is not None and model is not None
    assert any(not c[3] for c in calls), 'the network fallback ran'
