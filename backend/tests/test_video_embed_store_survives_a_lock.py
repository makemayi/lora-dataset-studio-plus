"""🔎 The video bank's vector store also survives a held file, and salvages one.

A frame vector is CLIP over a decoded video frame — the most expensive thing that
module produces — so the same two rules apply as on the image side: the rename
retries, and a temporary an interrupted run left behind is checked, then claimed.
"""
import os
import pathlib

import pytest

np = pytest.importorskip('numpy')

from app.services import atomic_npz          # noqa: E402
from app.services import video_clip_search as vcs   # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    monkeypatch.setattr(atomic_npz.npz_atomic, '_sleep', lambda _delay: None)


def _lock_replace(monkeypatch, times):
    """Hold the destination for `times` attempts. Returns a mutable state whose
    'left' can be zeroed to release the lock — a targeted release, because
    monkeypatch.undo() would also revert the path and sleep patches a test set."""
    real = os.replace
    state = {'left': times, 'seen': []}

    def flaky(src, dst, *args, **kwargs):
        state['seen'].append(str(dst))
        if state['left'] > 0:
            state['left'] -= 1
            error = PermissionError(5, 'Access is denied')
            error.winerror = 5
            raise error
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(atomic_npz.npz_atomic.os, 'replace', flaky)
    return state


def _store(count):
    return {cid: [{'label': 'key', 'time_s': 1.0,
                   'vec': np.full(4, cid, dtype='float32')}]
            for cid in range(1, count + 1)}


@pytest.fixture
def store_path(tmp_path, monkeypatch):
    path = tmp_path / 'clip_embeddings.npz'
    monkeypatch.setattr(vcs, 'embed_cache_path', lambda _bank_id: path)
    vcs.forget_memory_cache()
    return path


def test_the_vector_store_survives_a_transient_lock(store_path, monkeypatch):
    _lock_replace(monkeypatch, 3)

    vcs.save_embeddings(1, _store(5))

    assert store_path.is_file()
    assert len(vcs.load_embeddings(1)) == 5


def test_a_lock_that_never_clears_keeps_the_vectors_for_the_next_run(
        store_path, monkeypatch):
    vcs.save_embeddings(1, _store(2))
    vcs.forget_memory_cache()
    lock = _lock_replace(monkeypatch, 999)

    with pytest.raises(atomic_npz.NpzReplaceLocked):
        vcs.save_embeddings(1, _store(9))
    lock['left'] = 0

    # Nothing was lost: the finished archive is on disk, and the next read
    # claims it instead of quietly re-embedding the whole bank.
    vcs.forget_memory_cache()
    assert len(vcs.load_embeddings(1)) == 9
    assert not atomic_npz.orphan_temporaries(store_path)


def test_a_truncated_temporary_is_refused_not_promoted(store_path, monkeypatch):
    vcs.save_embeddings(1, _store(3))
    donor = store_path.parent / 'donor.npz'
    monkeypatch.setattr(vcs, 'embed_cache_path', lambda _bank_id: donor)
    vcs.save_embeddings(1, _store(40))
    monkeypatch.setattr(vcs, 'embed_cache_path', lambda _bank_id: store_path)

    whole = donor.read_bytes()
    orphan = pathlib.Path(str(store_path) + '.tmp.npz')
    orphan.write_bytes(whole[:len(whole) // 2])

    vcs.forget_memory_cache()
    assert len(vcs.load_embeddings(1)) == 3     # the good store is untouched
    assert not orphan.exists()


def test_an_empty_store_is_a_valid_write(store_path):
    vcs.save_embeddings(1, {})
    assert vcs.load_embeddings(1) == {}
    assert vcs._store_entry_count(store_path) == 0
