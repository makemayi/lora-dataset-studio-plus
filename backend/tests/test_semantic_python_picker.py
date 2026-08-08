"""The SigLIP 2 semantic index, on a GPU Python the machine already has.

Score got this first, and the detector it grew is good: it probes by DEPENDENCY,
it is read-only, it refuses what it could not prove. The semantic engine needs
exactly the same thing and a DIFFERENT answer — its worker never imports
open_clip or timm, so an interpreter Score refuses can be perfectly good here,
and one Score accepts can still be too old for `Siglip2Model`.

Two contracts are load-bearing and both are asserted below:

* the two profiles really disagree (a shared list would silently refuse the most
  common borrowed venv, and nothing would look broken);
* choosing where the index RUNS never moves where Setup INSTALLS.

No real subprocess runs here: `scoring_python._run_probe` is the single seam.
"""
from unittest.mock import patch

import pytest


def _facts(cuda=True, missing=(), siglip2=True):
    """Raw probe output: every module the union knows, minus `missing`, plus the
    symbol table. `siglip2=False` is the 2024 venv whose `transformers` imports
    fine and has no Siglip2Model."""
    from app.services import scoring_python as sp
    mods = {m: m not in missing for m in sp._DEP_MODULES}
    return {'python': '3.12.9', 'modules': mods,
            'symbols': {'transformers:Siglip2Model': bool(siglip2)},
            'cuda': cuda,
            'device_name': 'NVIDIA GeForce RTX 4090' if cuda else None,
            'torch_version': '2.9.1+cu128'}


@pytest.fixture()
def sp(app):
    from app.services import scoring_python
    scoring_python.clear_cache()
    yield scoring_python
    scoring_python.clear_cache()


# ── The two profiles must really disagree ────────────────────────────────────

def test_an_interpreter_score_refuses_can_still_run_the_semantic_index(sp, app, tmp_path):
    """A ComfyUI venv: CUDA torch, transformers, Pillow, numpy — no open_clip,
    no timm. Score is right to refuse it. Refusing it for the semantic index too
    would be a lie about a worker that never imports either package, and it is
    the single most common borrowed environment out there."""
    py = tmp_path / 'python'
    py.write_text('')
    facts = _facts(cuda=True, missing=('open_clip', 'timm'))
    scoring = sp.describe(str(py), facts, 'scoring')
    semantic = sp.describe(str(py), facts, 'semantic')
    assert scoring['status'] == 'incomplete'
    assert scoring['missing'] == ['open_clip_torch', 'timm']
    assert semantic['status'] == 'gpu_ready'
    assert semantic['usable'] is True and semantic['gpu'] is True
    assert semantic['missing'] == []
    # And the row says which question it answered.
    assert semantic['profile'] == 'semantic'
    assert [d['module'] for d in semantic['deps']] == [
        'torch', 'transformers', 'numpy', 'PIL']


def test_transformers_that_imports_but_has_no_siglip2model_is_refused(sp, app, tmp_path):
    """The false positive a module-name probe cannot see. `import transformers`
    succeeds on 4.38 and `Siglip2Model` is not in it — the worker dies at model
    load, an hour into an index. The repair line must carry the version floor,
    because a bare `pip install transformers` is a no-op on an env that already
    holds an older copy."""
    py = tmp_path / 'python'
    py.write_text('')
    verdict = sp.describe(str(py), _facts(cuda=True, siglip2=False), 'semantic')
    assert verdict['status'] == 'incomplete'
    assert verdict['usable'] is False
    assert verdict['missing'] == ['transformers']
    assert '"transformers>=4.49"' in verdict['install_command']
    # Score, which only needs the module, is unaffected by the missing symbol.
    assert sp.describe(str(py), _facts(cuda=True, siglip2=False),
                       'scoring')['status'] == 'gpu_ready'


def test_a_probe_payload_without_the_symbol_table_cannot_prove_siglip2(sp, app, tmp_path):
    """Fail safe, like everything else here. An answer that never reported
    symbols (an older cached probe) is not evidence that the class is there."""
    py = tmp_path / 'python'
    py.write_text('')
    facts = _facts(cuda=True)
    facts.pop('symbols')
    assert sp.describe(str(py), facts, 'semantic')['status'] == 'incomplete'


# ── Where it RUNS is not where we INSTALL ────────────────────────────────────

def test_selecting_writes_only_the_semantic_key_and_leaves_score_alone(sp, app, tmp_path):
    from app import config as cfg
    py = tmp_path / 'gpu-python'
    py.write_text('')
    score_py = r'D:\score\python.exe'
    with app.app_context():
        cfg.save_config({'bank_scoring': {'python': score_py},
                         'bank_semantic': {'python': ''}})
        with patch.object(sp, '_run_probe', lambda p: _facts(cuda=True)):
            result = sp.select(str(py), profile='semantic')
        assert result['selected'] == str(py)
        assert cfg.get('bank_semantic.python') == str(py)
        assert cfg.get('bank_scoring.python') == score_py
        # The resolver every SigLIP2 worker calls now lands on the choice.
        from app.services import bank_semantic_models as assets
        assert assets.semantic_python() == str(py)


def test_reverting_clears_only_the_semantic_key(sp, app):
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'bank_scoring': {'python': r'D:\score\python.exe'},
                         'bank_semantic': {'python': r'D:\borrowed\python.exe'}})
        assert sp.select('', profile='semantic')['reverted'] is True
        assert (cfg.get('bank_semantic.python') or '') == ''
        assert cfg.get('bank_scoring.python') == r'D:\score\python.exe'


def test_an_unprovable_interpreter_is_refused_and_names_the_feature(sp, app, tmp_path):
    py = tmp_path / 'python'
    py.write_text('')
    with app.app_context(), \
         patch.object(sp, '_run_probe', lambda p: _facts(cuda=True, siglip2=False)):
        with pytest.raises(sp.SelectionError) as excinfo:
            sp.select(str(py), profile='semantic')
    assert 'SigLIP 2' in str(excinfo.value)
    with app.app_context():
        from app import config as cfg
        assert (cfg.get('bank_semantic.python') or '') == ''


# ── The list ─────────────────────────────────────────────────────────────────

def test_scores_interpreter_is_offered_as_a_one_click_row(sp, app, tmp_path):
    """"Use the same Python as ✨ Score" is the answer for most machines that
    already borrowed one. It has to be a row, not a path to copy by hand."""
    from app import config as cfg
    score_py = tmp_path / 'score-python'
    score_py.write_text('')
    with app.app_context():
        cfg.save_config({'bank_scoring': {'python': str(score_py)},
                         'bank_semantic': {'python': ''}})
        rows = sp.candidates('semantic')
    entry = next(r for r in rows if r['source'] == 'scoring')
    assert entry['path'] == str(score_py)
    assert '✨ Score' in entry['label']
    # Score's own picker never offers itself that way.
    with app.app_context():
        assert not any(r['source'] == 'scoring' for r in sp.candidates('scoring'))


def test_default_python_mirrors_the_resolver_fallback(sp, app):
    """With nothing selected the index still runs in Score's interpreter — that
    read-time fallback is what old configs rely on. Saying `sys.executable` here
    would answer a question about a different machine."""
    import sys
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'bank_scoring': {'python': r'D:\score\python.exe'},
                         'bank_semantic': {'python': ''}})
        assert sp.default_python('semantic') == r'D:\score\python.exe'
        assert sp.default_python('scoring') == sys.executable
        cfg.save_config({'bank_scoring': {'python': ''}})
        assert sp.default_python('semantic') == sys.executable


def test_an_unknown_profile_is_a_value_error_not_a_silent_default(sp):
    with pytest.raises(ValueError):
        sp.get_profile('watermarks')


# ── Route contract ───────────────────────────────────────────────────────────

def test_the_endpoint_lists_and_selects_under_the_semantic_key(sp, app, client, tmp_path):
    from app import config as cfg
    py = tmp_path / 'python'
    py.write_text('')
    with patch.object(sp, '_run_probe', lambda p: _facts(cuda=True)):
        listing = client.get('/api/semantic-python').get_json()
        assert listing['profile'] == 'semantic'
        res = client.post('/api/semantic-python', json={'python': str(py)})
    assert res.status_code == 200
    assert res.get_json()['profile'] == 'semantic'
    with app.app_context():
        assert cfg.get('bank_semantic.python') == str(py)


def test_the_endpoint_refuses_a_too_old_transformers_with_the_reason(sp, app, client, tmp_path):
    py = tmp_path / 'python'
    py.write_text('')
    with patch.object(sp, '_run_probe', lambda p: _facts(cuda=True, siglip2=False)):
        res = client.post('/api/semantic-python', json={'python': str(py)})
    assert res.status_code == 400
    body = res.get_json()
    assert 'SigLIP2-capable' in body['error']
    assert body['verdict']['missing'] == ['transformers']
    with app.app_context():
        from app import config as cfg
        assert (cfg.get('bank_semantic.python') or '') == ''
