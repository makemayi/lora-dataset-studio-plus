"""Automatic masking — SAM 3 in the app's own interpreter, on ComfyUI's weights.

WHAT THESE PIN
--------------
The service was validated against ONE machine, and everything that made it work
there is a fact about that machine: a checkpoint under ComfyUI's models/sam3, a
venv Setup built, a card SAM 3 can use. Nobody else has that. So what has to
survive a different install is:

* resolution — the checkpoint is FOUND, and the Comfy-Org repackaging that
  cannot load here is never mistaken for it;
* preflight — every gap named at once, with the action that fixes it, and never
  a crash;
* the cache key — content-addressed, so a renamed file is a hit and an edited
  one is a miss;
* the two refusals that would otherwise be silent: a phrase that matched
  nothing, and a phrase that matched the whole frame.

NOTHING here loads a model or touches a GPU.
"""
import importlib
import json
import os

import pytest


def _fresh_config(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    importlib.reload(config)
    return config


def _write(path, data=b'weights'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def am(monkeypatch, tmp_path):
    """auto_mask bound to a throwaway ComfyUI tree and config."""
    config = _fresh_config(monkeypatch, tmp_path)
    base = tmp_path / 'Comfy'
    (base / 'models' / 'checkpoints').mkdir(parents=True, exist_ok=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    import app.services.auto_mask as auto_mask
    importlib.reload(auto_mask)
    yield auto_mask, base, config, tmp_path
    comfy_model_paths.clear_cache()


# --- resolution --------------------------------------------------------------

def test_the_checkpoint_is_found_under_comfyui(am):
    auto_mask, base, _config, _tmp = am
    ckpt = _write(base / 'models' / 'sam3' / 'sam3.pt')
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    assert auto_mask.resolve_checkpoint() == str(ckpt)


def test_the_comfy_repackaged_build_is_not_mistaken_for_it(am):
    """`sam3.1_multiplex_fp16.safetensors` is remapped for ComfyUI's own loader.
    Picking it up would fail deep inside torch instead of saying it is absent."""
    auto_mask, base, _config, _tmp = am
    _write(base / 'models' / 'checkpoints' / 'sam3.1_multiplex_fp16.safetensors')
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    assert auto_mask.resolve_checkpoint() is None


def test_an_absolute_pin_outside_every_comfyui_root_is_honoured(am):
    """This model is not loaded BY ComfyUI, so it does not have to live there."""
    auto_mask, _base, _config, tmp = am
    elsewhere = _write(tmp / 'somewhere' / 'sam3.pt')
    assert auto_mask.resolve_checkpoint(str(elsewhere)) == str(elsewhere)


def test_a_stale_pin_falls_back_to_the_scan(am):
    auto_mask, base, _config, tmp = am
    found = _write(base / 'models' / 'sam3' / 'sam3.pt')
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    assert auto_mask.resolve_checkpoint(str(tmp / 'gone' / 'sam3.pt')) == str(found)


# --- preflight ---------------------------------------------------------------

def test_preflight_names_every_gap_at_once(am):
    auto_mask, _base, _config, _tmp = am
    with pytest.raises(auto_mask.AutoMaskUnavailable) as e:
        auto_mask.preflight()
    message = str(e.value)
    assert 'Setup' in message                      # the interpreter is missing
    assert 'sam3.pt' in message                    # ...and so is the checkpoint
    assert e.value.actions == ['automask']
    assert e.value.files and e.value.files[0]['path'].endswith('sam3.pt')


def test_preflight_passes_once_both_halves_exist(am, tmp_path):
    auto_mask, base, config, _tmp = am
    ckpt = _write(base / 'models' / 'sam3' / 'sam3.pt')
    fake_python = _write(tmp_path / 'envs' / 'automask' / 'python.exe')
    config.save_config({'auto_mask': {'python': str(fake_python)}})
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    assert auto_mask.availability()['ok'] is True
    assert auto_mask.preflight() == (str(fake_python), str(ckpt))


def test_availability_never_raises_on_a_bare_install(am):
    auto_mask, _base, _config, _tmp = am
    state = auto_mask.availability()
    assert state['ok'] is False and state['actions'] == ['automask']


def test_a_configured_interpreter_that_is_gone_reads_as_not_installed(am):
    """A path in config is not proof of an environment: it survives the folder
    being deleted, and reporting ✓ for it would fail at the first mask."""
    auto_mask, base, config, tmp = am
    _write(base / 'models' / 'sam3' / 'sam3.pt')
    config.save_config({'auto_mask': {'python': str(tmp / 'nope' / 'python.exe')}})
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    assert auto_mask.availability()['ok'] is False


# --- cache -------------------------------------------------------------------

def test_the_cache_key_follows_the_pixels_not_the_path(am):
    auto_mask, _base, _config, _tmp = am
    key = auto_mask.cache_key(b'PNGDATA', 'head', threshold=0.5)
    assert auto_mask.cache_key(b'PNGDATA', 'head', threshold=0.5) == key
    assert auto_mask.cache_key(b'PNGDATA', 'HEAD ', threshold=0.5) == key
    assert auto_mask.cache_key(b'OTHER', 'head', threshold=0.5) != key
    assert auto_mask.cache_key(b'PNGDATA', 'hands', threshold=0.5) != key
    assert auto_mask.cache_key(b'PNGDATA', 'head', threshold=0.3) != key


def test_a_cached_mask_is_reused_without_a_run(am, monkeypatch, tmp_path):
    auto_mask, _base, _config, _tmp = am
    image = _write(tmp_path / 'tile.png', b'PIXELS')
    key = auto_mask.cache_key(b'PIXELS', 'head', threshold=0.5)
    (auto_mask.cache_dir() / f'{key}.png').write_bytes(b'MASK')

    def _never(*a, **k):
        raise AssertionError('a cached mask must not start a run')

    monkeypatch.setattr(auto_mask, 'mask_images', _never)
    assert auto_mask.mask_for(str(image), 'head').endswith(f'{key}.png')


# --- the two silent answers --------------------------------------------------

def _stub_run(auto_mask, monkeypatch, image, state, coverage=0.1, write=True):
    def _fake(images, prompt, **kw):
        out_dir = kw['out_dir']
        os.makedirs(out_dir, exist_ok=True)
        if write:
            name = os.path.splitext(os.path.basename(image))[0] + '.png'
            with open(os.path.join(out_dir, name), 'wb') as fh:
                fh.write(b'MASKBYTES')
        return {'ok': True, 'written': 1, 'cancelled': False, 'out_dir': out_dir,
                'results': {image: {'state': state, 'coverage': coverage}}}

    monkeypatch.setattr(auto_mask, 'mask_images', _fake)


def test_a_phrase_that_matched_nothing_is_reported(am, monkeypatch, tmp_path):
    """A black mask is not an error anywhere downstream: it is an edit that
    silently does nothing, which reads as the feature being broken."""
    auto_mask, _base, _config, _tmp = am
    image = str(_write(tmp_path / 'tile.png', b'PIXELS'))
    _stub_run(auto_mask, monkeypatch, image, 'no_match', write=False)
    with pytest.raises(auto_mask.NoMatch, match='nothing in this image matched'):
        auto_mask.mask_for(image, 'unicorn')


def test_a_phrase_that_matched_the_whole_frame_is_refused(am, monkeypatch, tmp_path):
    auto_mask, _base, _config, _tmp = am
    image = str(_write(tmp_path / 'tile.png', b'PIXELS'))
    _stub_run(auto_mask, monkeypatch, image, 'ok', coverage=0.97)
    with pytest.raises(auto_mask.NoMatch, match='which is not'):
        auto_mask.mask_for(image, 'photo')


def test_no_match_is_a_request_problem_not_an_install_problem(am):
    """The UI branches on it: an install banner for "try another word" would
    send the user to fix something that is not broken."""
    auto_mask, _base, _config, _tmp = am
    assert issubclass(auto_mask.NoMatch, auto_mask.AutoMaskUnavailable)
    assert auto_mask.NoMatch('x').actions == []


def test_a_good_mask_lands_in_the_cache(am, monkeypatch, tmp_path):
    auto_mask, _base, _config, _tmp = am
    image = str(_write(tmp_path / 'tile.png', b'PIXELS'))
    _stub_run(auto_mask, monkeypatch, image, 'ok', coverage=0.2)
    out = auto_mask.mask_for(image, 'head')
    assert os.path.isfile(out)
    key = auto_mask.cache_key(b'PIXELS', 'head', threshold=0.5)
    assert out == str(auto_mask.cache_dir() / f'{key}.png')
    # ...and the second call answers from it.
    monkeypatch.setattr(auto_mask, 'mask_images',
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError('re-ran')))
    assert auto_mask.mask_for(image, 'head') == out


# --- progress ----------------------------------------------------------------

def test_progress_lines_parse_into_phase_and_count(am):
    auto_mask, _base, _config, _tmp = am
    assert auto_mask.parse_progress_line('[sam3] phase=loading') == {'phase': 'loading'}
    assert auto_mask.parse_progress_line('[sam3] 3/12 ok coverage=0.211') == {
        'phase': 'masking', 'done': 3, 'total': 12}
    assert auto_mask.parse_progress_line('some unrelated torch warning') is None


def test_the_counter_carries_the_phase_with_it(am):
    """A missed `phase=masking` line must not leave a UI stuck on "Loading…"
    while images are visibly being counted."""
    auto_mask, _base, _config, _tmp = am
    assert auto_mask.parse_progress_line('[sam3] 1/4 ok')['phase'] == 'masking'


# --- the child script's contract --------------------------------------------

def test_the_child_refuses_a_payload_it_cannot_honour(am, tmp_path):
    """It answers as ONE json line on stdout, never a mute traceback — the
    parent has nothing else to report."""
    import subprocess
    import sys
    script = str(__import__('app.config', fromlist=['config']).BACKEND_DIR
                 / 'infer' / 'sam3_mask_infer.py')
    payload = json.dumps({'images': [], 'prompt': '', 'out_dir': str(tmp_path),
                          'checkpoint': str(tmp_path / 'missing.pt')})
    proc = subprocess.run([sys.executable, script], input=payload,
                          capture_output=True, text=True, timeout=120)
    last = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith('{')][-1]
    assert json.loads(last) == {'ok': False,
                                'error': 'a phrase naming the region is required'}


# --- a region is usually several objects -------------------------------------

def test_a_region_can_be_named_as_several_phrases(am):
    """'head' segments the head and leaves the glasses on it behind — to an
    open-vocabulary model those are two objects. The masks are unioned."""
    auto_mask, _base, _config, _tmp = am
    assert auto_mask.normalize_prompts('head, glasses') == ['head', 'glasses']
    assert auto_mask.normalize_prompts(['Head', ' GLASSES ']) == ['head', 'glasses']
    # Junk, blanks and repeats do not become phrases.
    assert auto_mask.normalize_prompts(' head ,, head,  ') == ['head']
    assert auto_mask.normalize_prompts('') == []
    assert auto_mask.normalize_prompts(None) == []


def test_the_cache_key_does_not_care_about_phrase_ORDER(am):
    """Otherwise re-ordering a settings field silently re-runs every mask."""
    auto_mask, _base, _config, _tmp = am
    a = auto_mask.cache_key(b'PIX', 'head, glasses', threshold=0.5)
    b = auto_mask.cache_key(b'PIX', 'glasses,Head', threshold=0.5)
    assert a == b
    assert auto_mask.cache_key(b'PIX', 'head', threshold=0.5) != a


def test_every_phrase_reaches_the_child_as_a_list(am, monkeypatch, tmp_path):
    """One `set_image` covers all of them — the image encode is nearly the whole
    cost — so the phrases have to travel together, not one run each."""
    auto_mask, _base, config, _tmp = am
    image = str(_write(tmp_path / 'tile.png', b'PIXELS'))
    _write(_write(tmp_path / 'ckpt' / 'sam3.pt').parent / 'sam3.pt')
    config.save_config({'auto_mask': {
        'python': str(_write(tmp_path / 'py' / 'python.exe')),
        'checkpoint': str(tmp_path / 'ckpt' / 'sam3.pt')}})
    sent = {}

    def _fake_run(python, script, payload, timeout, **kw):
        sent.update(json.loads(payload))
        return ('{"ok": true, "written": 0, "cancelled": false, "results": {}}',
                [], 0, False)

    monkeypatch.setattr(auto_mask, 'run_infer_script', _fake_run)
    auto_mask.mask_images([image], 'head, glasses, hat')
    assert sent['prompts'] == ['head', 'glasses', 'hat']
    assert 'prompt' not in sent
