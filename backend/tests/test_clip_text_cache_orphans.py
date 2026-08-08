"""The text-query caches also go through the atomic writer — and SWEEP orphans.

These two are the only .npz writers in the app that deliberately do NOT salvage a
leftover temporary. Rebuilding a text vector costs a handful of milliseconds
against a worker that is loaded anyway, and both loaders enforce a strict
provenance contract a promoted file would have to satisfy to be worth anything —
so promoting would be risk with no reward. Deleting, though, is a real fix: the
SigLIP2 writer names its temporary with pid + thread + nanoseconds, so before
this every process killed mid-write left one more file behind, forever.
"""
import pathlib

import pytest

np = pytest.importorskip('numpy')

from app.services import atomic_npz            # noqa: E402
from app.services import clip_text_encoder as cte   # noqa: E402


@pytest.fixture
def cache_path(tmp_path, monkeypatch):
    path = tmp_path / 'text_query_cache.npz'
    monkeypatch.setattr(cte, '_cache_path', lambda: path)
    monkeypatch.setattr(cte, '_memory', cte.OrderedDict())
    return path


def test_saving_the_text_cache_uses_a_unique_temporary_and_leaves_none(cache_path):
    cte._remember_query(cte._memory, 'a red car', np.ones(768, dtype='float32'))

    cte._save_disk_cache()

    assert cache_path.is_file()
    assert not atomic_npz.orphan_temporaries(cache_path)


def test_temporaries_left_by_killed_runs_are_swept_not_promoted(cache_path):
    cte._remember_query(cte._memory, 'a red car', np.ones(768, dtype='float32'))
    litter = [
        pathlib.Path(str(cache_path) + '.tmp.npz'),
        pathlib.Path(str(cache_path) + '.4242-99-1700000000000000000.tmp.npz'),
    ]
    for orphan in litter:
        np.savez_compressed(str(orphan), queries=np.array(['x']),
                            vecs=np.zeros((1, 768), dtype='float32'))

    cte._save_disk_cache()

    assert not any(orphan.exists() for orphan in litter)
    with np.load(str(cache_path)) as z:
        assert list(z['queries']) == ['a red car']   # ours, not the orphan's
