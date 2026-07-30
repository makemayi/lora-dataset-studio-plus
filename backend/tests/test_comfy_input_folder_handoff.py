r"""The FILESYSTEM half of the ComfyUI contract must fail out loud.

Reported on Discord by nofaceman: LoRA Dataset Studio in Docker, ComfyUI in a
SECOND container. Setup went green (URL + directory both accepted), then every
generation answered a bare `500` with no detail — "probably no network access".

It was not the network. The app reaches ComfyUI two ways, and only one is HTTP:
the API URL, and the input/ folder it COPIES every source image into. Two
containers do not share that folder by default, so `shutil.copy2` raised an
OSError no route maps -> Flask 500, no body, nothing to act on.

These tests hold the line on what was actually missing: the failure must NAME the
operation, the folder and the cause, must arrive as a mapped 409 rather than a
500, and must be safe to paste in a public thread.
"""
import os
from unittest.mock import patch

import pytest
from PIL import Image

from app.utils import comfy_fs


# --- the message: names the operation, the folder, and what to do -------------

def test_missing_input_folder_is_named_not_swallowed(tmp_path):
    """RED before the fix: the copy died on a raw FileNotFoundError. Now the
    refusal happens BEFORE the copy and says which folder is missing."""
    ghost = tmp_path / 'comfy' / 'input'
    with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
        comfy_fs.ensure_input_usable(str(ghost))
    msg = str(exc.value)
    # The path is REDACTED (tmp dirs live under the home dir) — what must survive
    # is the folder itself, unambiguously named.
    assert comfy_fs.safe_path(ghost) in msg
    assert msg.endswith('folder.') or 'comfy' in msg
    assert 'input folder' in msg
    assert 'does not exist' in msg
    assert 'shared volume' in msg          # the container/remote case, named


def test_unwritable_input_folder_is_named(tmp_path, monkeypatch):
    """The Docker case proper: the folder EXISTS (it is in the image) but this
    process cannot write into it — a read-only bind mount, or a different uid.

    The denial is injected at the probe seam rather than by chmod: a read-only
    directory is not reproducible across Windows/POSIX/CI, and what is under test
    is the MESSAGE. The real probe is exercised by the round-trip test below."""
    folder = tmp_path / 'input'
    folder.mkdir()
    monkeypatch.setattr(comfy_fs, '_write_probe',
                        lambda p: 'PermissionError: [Errno 13] Permission denied')
    with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
        comfy_fs.ensure_input_usable(str(folder))
    msg = str(exc.value)
    assert comfy_fs.safe_path(folder) in msg
    assert 'not writable' in msg
    assert 'PermissionError' in msg        # the CAUSE, not just the symptom
    assert 'shared volume' in msg


def test_write_probe_round_trips_and_leaves_nothing_behind(tmp_path):
    """A usable folder is judged usable, and the probe file never survives."""
    assert comfy_fs.folder_problem('input', str(tmp_path)) == ''
    assert list(tmp_path.iterdir()) == []


def test_copy_failure_names_the_operation_and_the_folder(tmp_path):
    """Everything checks out and the copy STILL fails (disk full, a mount that
    went away between the probe and the write). Same contract."""
    folder = tmp_path / 'input'
    folder.mkdir()
    src = tmp_path / 'src.png'
    src.write_bytes(b'\x89PNG\r\n\x1a\n')
    with patch.object(comfy_fs.shutil, 'copy2', side_effect=OSError('disk full')):
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            comfy_fs.stage_input_copy(str(src), 'x.png', str(folder))
    msg = str(exc.value)
    assert 'Copying the source image into ComfyUI' in msg
    assert comfy_fs.safe_path(folder) in msg
    assert 'OSError: disk full' in msg


def test_generated_file_write_failure_is_named_too(tmp_path):
    """The watermark lane WRITES its crop instead of copying one — same guard."""
    folder = tmp_path / 'input'
    folder.mkdir()

    def _boom(path):
        raise OSError('read-only file system')

    with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
        comfy_fs.stage_input_write('crop.png', _boom, str(folder))
    assert 'Writing the source image into ComfyUI' in str(exc.value)
    assert 'read-only file system' in str(exc.value)


# --- paste-safety: these strings are written to be dropped in a public thread --

def test_messages_never_expose_a_user_home_path():
    r"""The whole point of a readable error is that people paste it. A raw
    `C:\Users\<account>\...` would leak the OS account name into #help."""
    home = r'C:\Users\somebody\ComfyUI\input'
    problem = comfy_fs.folder_problem('input', home)
    assert 'somebody' not in problem
    assert '~' in problem and 'ComfyUI' in problem

    posix = '/home/somebody/comfy/input'
    assert 'somebody' not in comfy_fs.folder_problem('input', posix)


def test_staging_error_is_redacted_too(tmp_path):
    """Redaction applies to the CAUSE as well — an exception repr routinely
    carries the destination path."""
    folder = tmp_path / 'input'
    folder.mkdir()
    src = tmp_path / 'src.png'
    src.write_bytes(b'x')
    with patch.object(comfy_fs.shutil, 'copy2',
                      side_effect=OSError(r'cannot write C:\Users\somebody\in\x.png')):
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            comfy_fs.stage_input_copy(str(src), 'x.png', str(folder))
    assert 'somebody' not in str(exc.value)


# --- it is a RuntimeError, i.e. an actionable 409, never a bare 500 -----------

def test_failure_class_maps_to_409_not_500(app):
    """`_map_error` re-raises anything it doesn't know (-> 500). Being a
    RuntimeError is what turns this into a 409 carrying the message; the test
    drives the real mapper rather than trusting the class hierarchy."""
    from app.routes._common import _map_error
    with app.app_context():
        body, status = _map_error(comfy_fs.ComfyFolderUnavailable('nope'))
        assert status == 409
        assert body.get_json()['error'] == 'nope'


# --- end to end through the engines that stage a file ------------------------

def _klein_ready(monkeypatch, keh, comfy_in):
    monkeypatch.setattr(keh, '_comfy_input_dir', lambda: str(comfy_in))
    monkeypatch.setattr(keh, 'resolve_klein_unet', lambda *a, **k: 'unet.safetensors')
    monkeypatch.setattr(keh, 'resolve_klein_vae', lambda *a, **k: 'vae.safetensors')
    monkeypatch.setattr(keh, 'resolve_klein_text_encoder', lambda *a, **k: 'te.safetensors')
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda *a, **k: [])


def test_klein_generation_refuses_with_a_reason(app, tmp_path, monkeypatch):
    """The user-facing repro: generate with an input folder that isn't there.
    Before: OSError -> 500, no detail. After: a named refusal, and NO job queued."""
    from app.services import klein_edit_helper as keh
    src = tmp_path / 'src.png'
    src.write_bytes(b'\x89PNG\r\n\x1a\n')
    queued = []
    with app.app_context():
        _klein_ready(monkeypatch, keh, tmp_path / 'not-mounted')
        monkeypatch.setattr(keh.queue_manager, 'add_job',
                            lambda **kw: queued.append(kw))
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            keh.enqueue_klein_edit(user_id='local', source_filename='src.png',
                                   source_path=str(src), edit_prompt='hi')
    assert 'not-mounted' in str(exc.value)
    assert 'shared volume' in str(exc.value)
    assert queued == []          # nothing enqueued that could only have failed


def test_klein_generation_still_works_on_a_normal_install(app, tmp_path, monkeypatch):
    """Non-regression, the half that matters most: a plain single-machine install
    stages its source exactly as before and enqueues the job."""
    from app.services import klein_edit_helper as keh
    comfy_in = tmp_path / 'input'
    comfy_in.mkdir()
    src = tmp_path / 'src.png'
    Image.new('RGB', (32, 24), (30, 120, 220)).save(src, 'PNG')
    seen = {}
    with app.app_context():
        _klein_ready(monkeypatch, keh, comfy_in)
        monkeypatch.setattr(keh.queue_manager, 'add_job', lambda **kw: seen.update(kw))
        keh.enqueue_klein_edit(user_id='local', source_filename='src.png',
                               source_path=str(src), edit_prompt='hi')
    staged = [p.name for p in comfy_in.iterdir()]
    assert len(staged) == 1 and staged[0].endswith('src.png')
    assert seen['workflow_data']['52']['inputs']['image'] == staged[0]


def test_krea_generation_refuses_with_a_reason(app, tmp_path, monkeypatch):
    """The second local engine stages the same way and must answer the same way —
    a fix that only covered Klein would leave the identical 500 one engine away."""
    from app.services import krea_edit_helper as keh2
    src = tmp_path / 'src.png'
    src.write_bytes(b'\x89PNG\r\n\x1a\n')
    with app.app_context():
        monkeypatch.setattr(keh2, '_comfy_input_dir', lambda: str(tmp_path / 'not-mounted'))
        monkeypatch.setattr(keh2, 'preflight', lambda *a, **k: None)
        monkeypatch.setattr(keh2, 'resolve_krea_unet', lambda *a, **k: 'unet.safetensors')
        monkeypatch.setattr(keh2, 'resolve_krea_text_encoder', lambda *a, **k: 'te.safetensors')
        monkeypatch.setattr(keh2, 'resolve_krea_vae', lambda *a, **k: 'vae.safetensors')
        monkeypatch.setattr(keh2, 'resolve_krea_identity_lora',
                            lambda *a, **k: ('lora.safetensors', 'x'))
        with pytest.raises(comfy_fs.ComfyFolderUnavailable) as exc:
            keh2.enqueue_krea_edit(user_id='local', source_filename='src.png',
                                   source_path=str(src), edit_prompt='hi')
    assert 'not-mounted' in str(exc.value)


def test_watermark_klein_degrades_instead_of_raising(app, tmp_path, monkeypatch):
    """The watermark lane returns (image, error) instead of raising, so its guard
    must feed that contract — 'unavailable' + the same explanatory message."""
    from PIL import Image
    from app.services import watermark_klein as wk
    monkeypatch.setattr(wk, '_comfy_input_dir', lambda: str(tmp_path / 'not-mounted'))
    monkeypatch.setattr(wk.keh, 'resolve_klein_unet', lambda *a, **k: 'unet.safetensors')
    monkeypatch.setattr(wk.keh, 'resolve_klein_vae', lambda *a, **k: 'vae.safetensors')
    monkeypatch.setattr(wk.keh, 'resolve_klein_text_encoder', lambda *a, **k: 'te.safetensors')
    monkeypatch.setattr(wk.keh, 'klein_missing_assets', lambda *a, **k: [])
    with app.app_context():
        img, err = wk._run_klein_job('local', Image.new('RGB', (8, 8)), seed=1)
    assert img is None
    assert err['kind'] == 'unavailable'
    assert 'not-mounted' in err['detail'] and 'shared volume' in err['detail']


# --- Setup / Settings: warn at configuration time, never block ----------------

def test_settings_preview_reports_a_present_but_unusable_folder(tmp_path, monkeypatch):
    from app import capabilities
    base = tmp_path / 'Comfy'
    (base / 'input').mkdir(parents=True)
    (base / 'output').mkdir()
    monkeypatch.setattr(comfy_fs, '_write_probe', lambda p: 'PermissionError: denied')
    r = capabilities.classify_comfyui_folders(str(base), {})
    assert r['input_dir']['exists'] is True          # the old check: green
    assert r['input_dir']['usable'] is False         # the other half: honest
    assert 'not writable' in r['input_dir']['problem']
    # A folder the app only READS from is not write-probed — no phantom warning.
    assert r['output_dir']['usable'] is True and r['output_dir']['problem'] == ''


def test_settings_preview_stays_quiet_when_all_is_well(tmp_path):
    from app import capabilities
    base = tmp_path / 'Comfy'
    (base / 'input').mkdir(parents=True)
    r = capabilities.classify_comfyui_folders(str(base), {})
    assert r['input_dir']['usable'] is True
    assert r['input_dir']['problem'] == ''
    # Nothing to probe when the folder isn't there: `exists` already says it.
    assert r['models_dir']['usable'] is None


def test_setup_wizard_verdict_warns_without_blocking(app, tmp_path, monkeypatch):
    """The wizard's answer stays 'valid' — someone may configure the app before
    mounting their volumes — but it now carries the input-folder verdict."""
    from app import capabilities
    comfy = tmp_path / 'ComfyUI'
    (comfy / 'models').mkdir(parents=True)
    (comfy / 'main.py').write_text('x', encoding='utf-8')
    (comfy / 'input').mkdir()
    monkeypatch.setattr(comfy_fs, '_write_probe', lambda p: 'PermissionError: denied')
    with app.app_context():
        r = capabilities.classify_comfyui_dir(str(comfy))
    assert r['status'] == 'valid'                      # NOT downgraded, not blocking
    assert r['input_check']['ok'] is False
    assert 'not writable' in r['input_check']['problem']


def test_setup_wizard_verdict_is_green_on_a_working_install(app, tmp_path):
    from app import capabilities
    comfy = tmp_path / 'ComfyUI'
    (comfy / 'models').mkdir(parents=True)
    (comfy / 'main.py').write_text('x', encoding='utf-8')
    (comfy / 'input').mkdir()
    with app.app_context():
        r = capabilities.classify_comfyui_dir(str(comfy))
    assert r['status'] == 'valid' and r['input_check']['ok'] is True


def test_setup_wizard_honours_a_saved_input_override(app, tmp_path):
    """The wizard must judge the folder the app would REALLY use, not <base>/input
    — someone whose ComfyUI runs with --input-directory already set an override."""
    from app import capabilities
    import app.config as config
    comfy = tmp_path / 'ComfyUI'
    (comfy / 'models').mkdir(parents=True)
    (comfy / 'main.py').write_text('x', encoding='utf-8')
    override = tmp_path / 'mounted-input'
    override.mkdir()
    with app.app_context():
        config.save_config({'comfyui': {'input_dir': str(override)}})
        r = capabilities.classify_comfyui_dir(str(comfy))
    # <base>/input does not even exist; the override does -> the verdict is green.
    assert not (comfy / 'input').exists()
    assert r['input_check']['ok'] is True
    assert r['input_check']['path'].endswith('mounted-input')


def test_setup_route_carries_the_input_check(client, tmp_path):
    comfy = tmp_path / 'ComfyUI'
    (comfy / 'models').mkdir(parents=True)
    (comfy / 'main.py').write_text('x', encoding='utf-8')
    r = client.get(f'/api/setup/comfyui-dir?path={comfy}')
    assert r.status_code == 200
    body = r.get_json()
    assert body['status'] == 'valid'
    assert 'input_check' in body
    # <base>/input isn't there: the wizard says so rather than certifying it.
    assert body['input_check']['ok'] is False
    assert os.sep in body['input_check']['path']
