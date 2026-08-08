"""Every .npz cache writer is wired to the retrying/salvaging helper.

The helper's own behaviour is proved in test_npz_atomic_lock_and_salvage.py.
What is proved HERE is that each cache actually goes through it — the failure
mode of a shared helper is not that it is wrong, it is that one call site quietly
kept its own ``os.replace``. So each test locks the rename for real (by making
``os.replace`` refuse) and asserts on the CACHE, not on the plumbing.
"""
import importlib.util
import os
import pathlib
import sys

import pytest

np = pytest.importorskip('numpy')

INFER = pathlib.Path(__file__).resolve().parents[1] / 'infer'
if str(INFER) not in sys.path:
    sys.path.append(str(INFER))
import npz_atomic  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f'{name}_cachelock_test', INFER / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    monkeypatch.setattr(npz_atomic, '_sleep', lambda _delay: None)


def _lock_replace(monkeypatch, times, code=5):
    """Hold the destination for `times` attempts. Returns a mutable state whose
    'left' can be zeroed to release the lock — a targeted release, because
    monkeypatch.undo() would also revert the log/sleep patches a test set."""
    real = os.replace
    state = {'left': times, 'seen': []}

    def flaky(src, dst, *args, **kwargs):
        state['seen'].append(str(dst))
        if state['left'] > 0:
            state['left'] -= 1
            error = PermissionError(code, 'Access is denied')
            error.winerror = code
            raise error
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(npz_atomic.os, 'replace', flaky)
    return state


def _score_entry(value):
    return ('ok', 5.0, 0.1, np.full(8, value, dtype='float32'), '', b'')


def _face_entry(value):
    return ('ok', 0.9, 0.2, np.full(8, value, dtype='float32'), 0.0, '', b'')


# --- ✨ Score: the pass that actually died in the field ----------------------------
def test_score_cache_write_survives_a_transient_lock(tmp_path, monkeypatch):
    score = _load('bank_score_infer')
    destination = str(tmp_path / 'score_cache.npz')
    _lock_replace(monkeypatch, 2)

    score._save_cache(destination, {'a.jpg': _score_entry(1)})

    assert set(score._load_cache(destination)) == {'a.jpg'}


def test_score_pass_does_not_die_on_a_lock_that_never_clears(tmp_path, monkeypatch):
    """The incident: a nine-hour pass killed at image 1849 by a held file.

    The flush now reports and returns; the work stays on disk as a temporary.
    """
    score = _load('bank_score_infer')
    destination = str(tmp_path / 'score_cache.npz')
    said = []
    monkeypatch.setattr(score, '_log', said.append)
    _lock_replace(monkeypatch, 999)

    score._flush_cache(destination, {f'{i}.jpg': _score_entry(i) for i in range(30)})

    assert not os.path.exists(destination)
    orphans = npz_atomic.orphan_temporaries(destination)
    assert len(orphans) == 1
    assert len(score._load_cache(orphans[0])) == 30       # the work is intact
    assert any('another program is holding it open' in line for line in said)


def test_score_pass_salvages_the_temporary_left_by_the_previous_run(
        tmp_path, monkeypatch):
    """End to end: run 1 loses the rename, run 2 starts and finds its work."""
    score = _load('bank_score_infer')
    destination = str(tmp_path / 'score_cache.npz')
    score._save_cache(destination, {f'{i}.jpg': _score_entry(i) for i in range(1800)})

    lock = _lock_replace(monkeypatch, 999)
    score._flush_cache(destination, {f'{i}.jpg': _score_entry(i) for i in range(1850)})
    lock['left'] = 0
    assert len(score._load_cache(destination)) == 1800    # rename lost, as expected

    said = []
    monkeypatch.setattr(score, '_log', said.append)
    score._salvage_cache(destination)

    assert len(score._load_cache(destination)) == 1850
    assert not npz_atomic.orphan_temporaries(destination)
    assert any('recovered 1850 entries' in line for line in said)


def test_score_pass_refuses_a_temporary_truncated_by_a_kill(tmp_path, monkeypatch):
    score = _load('bank_score_infer')
    destination = str(tmp_path / 'score_cache.npz')
    score._save_cache(destination, {f'{i}.jpg': _score_entry(i) for i in range(40)})

    orphan = pathlib.Path(destination + '.tmp.npz')
    score._save_cache(str(tmp_path / 'donor.npz'),
                      {f'{i}.jpg': _score_entry(i) for i in range(500)})
    whole = (tmp_path / 'donor.npz').read_bytes()
    orphan.write_bytes(whole[:len(whole) // 2])

    said = []
    monkeypatch.setattr(score, '_log', said.append)
    score._salvage_cache(destination)

    assert len(score._load_cache(destination)) == 40      # untouched
    assert not orphan.exists()
    assert any('discarded a leftover temporary' in line for line in said)


# --- faces ------------------------------------------------------------------------
def test_face_cache_write_survives_a_transient_lock(tmp_path, monkeypatch):
    face = _load('face_embed_infer')
    destination = str(tmp_path / 'face_cache.npz')
    _lock_replace(monkeypatch, 3)

    face._save_cache(destination, {'a.jpg': _face_entry(1)})

    assert set(face._load_cache(destination)) == {'a.jpg'}


def test_face_pass_salvages_its_orphan(tmp_path, monkeypatch):
    face = _load('face_embed_infer')
    destination = str(tmp_path / 'face_cache.npz')
    face._save_cache(destination, {'a.jpg': _face_entry(1)})
    lock = _lock_replace(monkeypatch, 999)
    face._flush_cache(destination,
                      {'a.jpg': _face_entry(1), 'b.jpg': _face_entry(2)})
    lock['left'] = 0

    face._salvage_cache(destination)
    assert set(face._load_cache(destination)) == {'a.jpg', 'b.jpg'}


# --- semantic: the writer that used to DELETE its own rescue ----------------------
CONTRACT = {'cache_version': 1, 'engine': 'siglip2', 'model_id': 'x/y',
            'revision': 'r1', 'model_key': 'k1', 'dimension': 4,
            'models_root': 'root'}


def _semantic_cache(count):
    unit = np.zeros(4, dtype='float32')
    unit[0] = 1.0
    return {f'{i}.jpg': ('ok', unit, '', b'\0' * 32) for i in range(count)}


def test_semantic_lost_rename_keeps_the_temporary_instead_of_deleting_it(
        tmp_path, monkeypatch):
    """This writer wrapped its rename in a ``finally`` that unlinked the
    temporary, so a lost rename destroyed the finished archive on the way out."""
    semantic = _load('bank_semantic_infer')
    destination = str(tmp_path / 'semantic_cache.npz')
    monkeypatch.setattr(semantic, '_log', lambda _m: None)
    lock = _lock_replace(monkeypatch, 999)

    semantic._flush_cache(destination, _semantic_cache(12), CONTRACT)
    lock['left'] = 0

    orphans = npz_atomic.orphan_temporaries(destination)
    assert len(orphans) == 1
    semantic._salvage_cache(destination, CONTRACT)
    assert len(semantic._load_cache(destination, CONTRACT)) == 12


def test_semantic_refuses_an_orphan_from_another_model(tmp_path, monkeypatch):
    """Salvage inherits the loader's provenance check, so a temporary written by
    a DIFFERENT semantic engine can never be promoted over this one's cache."""
    semantic = _load('bank_semantic_infer')
    destination = str(tmp_path / 'semantic_cache.npz')
    monkeypatch.setattr(semantic, '_log', lambda _m: None)
    semantic._save_cache(destination, _semantic_cache(3), CONTRACT)

    other = dict(CONTRACT, model_key='k2', revision='r2')
    semantic._save_cache(str(tmp_path / 'donor.npz'), _semantic_cache(99), other)
    pathlib.Path(destination + '.tmp.npz').write_bytes(
        (tmp_path / 'donor.npz').read_bytes())

    semantic._salvage_cache(destination, CONTRACT)

    assert len(semantic._load_cache(destination, CONTRACT)) == 3
    assert not pathlib.Path(destination + '.tmp.npz').exists()
