"""face_swap_helper.enqueue_face_swap: stages the tile's current image
(target) and the dataset reference (identity source) into ComfyUI's input
dir, rewires the fixed 'face swap.json' workflow's loader/seed/output nodes,
and enqueues it. Mirrors the fixture style of test_klein_models.py."""
import io
import json
import os
import struct

import pytest
from PIL import Image

_VALID_ST = struct.pack('<Q', 2) + b'{}'


def _img(tmp_path, stem):
    """A distinct on-disk PNG, so target and reference never collide."""
    p = tmp_path / f'{stem}.png'
    p.write_bytes(_png((10, 200, 10) if stem == 'a' else (200, 10, 10)))
    return p


def _png(color=(0, 128, 255)):
    buf = io.BytesIO(); Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _install(base, *relparts, data=_VALID_ST):
    p = base.joinpath(*relparts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _comfy(tmp_path, cfg, swap_lora=True):
    """A ComfyUI tree with the shared Klein assets + (optionally) the
    fixed face-swap LoRA this workflow specifically requires."""
    base = tmp_path / 'comfyui'
    (base / 'input').mkdir(parents=True)
    (base / 'output').mkdir(parents=True)
    (base / 'main.py').write_text('# fake', encoding='utf-8')
    _install(base, 'models', 'unet', 'klein', 'flux-2-klein-9b-fp8.safetensors')
    _install(base, 'models', 'vae', 'flux2-vae.safetensors')
    _install(base, 'models', 'text_encoders', 'qwen_3_8b_fp8mixed.safetensors')
    if swap_lora:
        _install(base, 'models', 'loras', 'klein', 'Klein2-9B-SmartCharacterSwap.safetensors')
    cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


def test_enqueue_rewires_target_ref_and_loaders(app, tmp_path, monkeypatch):
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        target = tmp_path / 'target.png'; target.write_bytes(_png((10, 200, 10)))
        ref = tmp_path / 'ref.png'; ref.write_bytes(_png((200, 10, 10)))
        captured = {}
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.update(kw), kw['job_id'])[1])
        job_id = fsh.enqueue_face_swap(user_id='local', target_path=str(target),
                                       ref_path=str(ref))
        assert job_id
        wf = captured['workflow_data']
        assert wf[fsh.NODE_TARGET_IMAGE]['inputs']['image'].endswith('.png')
        assert wf[fsh.NODE_REF_IMAGE]['inputs']['image'].endswith('.png')
        assert (wf[fsh.NODE_TARGET_IMAGE]['inputs']['image']
                != wf[fsh.NODE_REF_IMAGE]['inputs']['image'])
        # No INT8 build in this fixture, so the W8A8 loader degrades to core.
        assert wf[fsh.NODE_UNET]['class_type'] == 'UNETLoader'
        assert wf[fsh.NODE_UNET]['inputs']['unet_name'] == os.path.join(
            'klein', 'flux-2-klein-9b-fp8.safetensors')
        assert wf[fsh.NODE_VAE]['inputs']['vae_name'] == 'flux2-vae.safetensors'
        assert wf[fsh.NODE_CLIP]['inputs']['clip_name'] == 'qwen_3_8b_fp8mixed.safetensors'
        assert wf[fsh.NODE_SAVE]['inputs']['filename_prefix'].startswith('local_FaceSwap_')
        assert captured['metadata']['model_name'] == 'klein_face_swap_dataset'
        assert len(captured['metadata']['staged_inputs']) == 2


def test_an_int8_build_is_preferred_and_a_few_step_one_wins(app, tmp_path, monkeypatch):
    """The graph schedules 4 and 6 steps. A full-step INT8 checkpoint at 4 steps
    is mush, not a slightly worse face — and the first version of this resolver
    picked INT8 builds alphabetically, choosing the non-turbo one over the turbo
    build sitting right beside it on the maintainer's own disk."""
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    with app.app_context():
        _comfy(tmp_path, cfg)
        models = tmp_path / 'comfy' / 'models' / 'diffusion_models'
        models.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(fsh.comfy_model_paths, 'search_roots',
                            lambda kind: [str(models)] if kind == 'diffusion_models' else [])

        # Nothing INT8 yet -> degrade to the core loader.
        (models / 'flux-2-klein-9b-fp8.safetensors').write_bytes(b'x')
        assert fsh.resolve_swap_unet()[1] is False

        # An INT8 build appears -> keep the W8A8 loader.
        (models / 'flux-2-klein-9b-int8.safetensors').write_bytes(b'x')
        name, int8 = fsh.resolve_swap_unet()
        assert int8 is True and 'int8' in name

        # A FEW-STEP INT8 build beats a full-step one, whatever the sort order.
        (models / 'aaa-klein-9b-turbo-int8.safetensors').write_bytes(b'x')
        name, int8 = fsh.resolve_swap_unet()
        assert int8 is True
        assert 'turbo' in name, f'few-step build must win, got {name!r}'


def _model_chain(wf, sink='262'):
    """Walk the model chain backwards from DifferentialDiffusion, newest first."""
    out, cur = [], wf[sink]['inputs']['model']
    while isinstance(cur, list) and len(cur) == 2 and cur[0] in wf:
        nid = cur[0]
        out.append((nid, wf[nid]['class_type'],
                    wf[nid]['inputs'].get('lora_name')))
        cur = wf[nid]['inputs'].get('model')
    return out


def test_settings_loras_are_chained_after_the_graphs_own(app, tmp_path, monkeypatch):
    """`klein.face_swap_loras` — a flat list, because face swap has no per-run
    picker for a named preset to be selected from."""
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    from app.job_queue import queue_manager
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        # Real files in the real fixture tree — patching search_roots here would
        # also blind the SHARED Klein asset lookup and fail for the wrong reason.
        _install(base, 'models', 'loras', 'mine.safetensors', data=b'x')
        _install(base, 'models', 'loras', 'other.safetensors', data=b'x')
        cfg.save_config({'klein': {'face_swap_loras': [
            {'file': 'mine.safetensors', 'strength': 0.8},
            {'file': 'other.safetensors', 'strength': 1.2},
            {'file': 'not-on-disk.safetensors', 'strength': 1.0},
        ]}})
        captured = {}
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.update(kw), kw['job_id'])[1])
        fsh.enqueue_face_swap(user_id='local', target_path=str(_img(tmp_path, 'a')),
                              ref_path=str(_img(tmp_path, 'b')))
        wf = captured['workflow_data']
        chain = _model_chain(wf)
        names = [n for _, _, n in chain if n]
        # Newest first: ours last-applied, then whatever the graph itself loads.
        assert names[0] == 'other.safetensors'
        assert names[1] == 'mine.safetensors'
        assert any('SmartCharacterSwap' in n for n in names), \
            'the graph\'s own swap LoRA must stay in the chain'
        assert 'not-on-disk.safetensors' not in names, \
            'a stale settings row must be skipped, not queued into a 400'
        for nid, node in wf.items():
            for v in node.get('inputs', {}).values():
                if isinstance(v, list) and len(v) == 2:
                    assert v[0] in wf, f'{nid} dangles after chaining'


def test_a_lora_the_graph_already_loads_is_not_chained_twice(app, tmp_path, monkeypatch):
    """Picking the swap LoRA (or the style LoRA) in Settings must NOT stack it a
    second time: both strengths sum into one delta well past what the file was
    trained for, which shows as macro-blocking. Same guard the Klein generation
    presets have — this list shipped without it."""
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    from app.job_queue import queue_manager
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        _install(base, 'models', 'loras', 'mine.safetensors', data=b'x')
        cfg.save_config({'klein': {'face_swap_loras': [
            # exactly the graph's own swap LoRA...
            {'file': 'klein\\Klein2-9B-SmartCharacterSwap.safetensors', 'strength': 1.0},
            # ...the same file with the other separator and different case,
            # which must not dodge the guard...
            {'file': 'KLEIN/klein2-9b-smartcharacterswap.safetensors', 'strength': 1.0},
            {'file': 'mine.safetensors', 'strength': 0.7},   # this one is fine
            {'file': 'mine.safetensors', 'strength': 0.9},   # ...but not twice
            {'file': 'mine.safetensors', 'strength': 0},     # row switched off
        ]}})
        captured = {}
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.update(kw), kw['job_id'])[1])
        fsh.enqueue_face_swap(user_id='local', target_path=str(_img(tmp_path, 'a')),
                              ref_path=str(_img(tmp_path, 'b')))
        wf = captured['workflow_data']
        names = [n['inputs']['lora_name'] for n in wf.values()
                 if n.get('class_type') == 'LoraLoaderModelOnly']
        swaps = [n for n in names if 'smartcharacterswap' in n.lower()]
        assert len(swaps) == 1, f'swap LoRA chained {len(swaps)}x: {names}'
        assert names.count('mine.safetensors') == 1, f'duplicate row chained: {names}'


def test_graph_own_loras_is_read_from_the_shipped_workflow(app):
    """The editor's duplicate warning is fed from here, so it must track the
    graph rather than a hardcoded copy that goes stale on the next swap."""
    from app.services import face_swap_helper as fsh
    with app.app_context():
        own = fsh.graph_own_loras()
        assert any('SmartCharacterSwap' in n for n in own), own
        assert len(own) == len(set(own))


def test_the_lora_list_is_sanitised(app, tmp_path, monkeypatch):
    """Config is hand-editable, so junk must clamp rather than reach ComfyUI."""
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    with app.app_context():
        cfg.save_config({'klein': {'face_swap_loras': [
            {'file': '  spaced.safetensors  ', 'strength': 99},
            {'file': '', 'strength': 1},
            {'file': 'no-strength.safetensors'},
            'not-a-dict',
            {'file': 'negative.safetensors', 'strength': -5},
        ] + [{'file': f'x{i}.safetensors'} for i in range(20)]}})
        rows = fsh.configured_face_swap_loras()
        assert rows[0] == {'file': 'spaced.safetensors', 'strength': 1.5}
        assert rows[1] == {'file': 'no-strength.safetensors', 'strength': 1.0}
        assert rows[2] == {'file': 'negative.safetensors', 'strength': 0.0}
        assert len(rows) == fsh.MAX_FACE_SWAP_LORAS


def test_no_configured_loras_leaves_the_graph_untouched(app, tmp_path, monkeypatch):
    from app.services import face_swap_helper as fsh
    with app.app_context():
        wf = {'262': {'class_type': 'DifferentialDiffusion',
                      'inputs': {'model': ['424:246', 0]}},
              '424:246': {'class_type': 'LoraLoaderModelOnly', 'inputs': {}}}
        before = json.dumps(wf, sort_keys=True)
        assert fsh.append_model_loras(wf, []) == []
        assert json.dumps(wf, sort_keys=True) == before


def test_both_samplers_get_a_fresh_seed(app, tmp_path, monkeypatch):
    """The graph samples twice. Two tiles of one batch must differ in BOTH
    stages — randomising only the inpaint seed left the built head identical
    across a whole run."""
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        target = tmp_path / 'target.png'; target.write_bytes(_png((10, 200, 10)))
        ref = tmp_path / 'ref.png'; ref.write_bytes(_png((200, 10, 10)))
        seen = []
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (seen.append(kw['workflow_data']),
                                          kw['job_id'])[1])
        for _ in range(2):
            fsh.enqueue_face_swap(user_id='local', target_path=str(target),
                                  ref_path=str(ref))
        a, b = seen
        assert a[fsh.NODE_SEED_INPAINT]['inputs']['seed'] \
            != b[fsh.NODE_SEED_INPAINT]['inputs']['seed']
        assert a[fsh.NODE_SEED_NOISE]['inputs']['noise_seed'] \
            != b[fsh.NODE_SEED_NOISE]['inputs']['noise_seed']


def test_the_optional_style_lora_is_dropped_when_absent(app, tmp_path, monkeypatch):
    """An install without the maintainer's private style LoRA must still swap
    faces — without its look, but without a ComfyUI validation error either."""
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    from app.job_queue import queue_manager
    with app.app_context():
        _comfy(tmp_path, cfg)
        target = tmp_path / 'target.png'; target.write_bytes(_png((10, 200, 10)))
        ref = tmp_path / 'ref.png'; ref.write_bytes(_png((200, 10, 10)))
        captured = {}
        monkeypatch.setattr(queue_manager, 'add_job',
                            lambda **kw: (captured.update(kw), kw['job_id'])[1])
        fsh.enqueue_face_swap(user_id='local', target_path=str(target),
                              ref_path=str(ref))
        wf = captured['workflow_data']
        assert fsh.NODE_STYLE_LORA not in wf, 'style LoRA is not on disk here'
        # ...and nothing is left pointing at the node that was removed.
        for nid, node in wf.items():
            for v in node.get('inputs', {}).values():
                if isinstance(v, list) and len(v) == 2:
                    assert v[0] in wf, f'{nid} dangles after the drop'
        # The REQUIRED swap LoRA is still in the chain.
        assert wf[fsh.NODE_SWAP_LORA]['inputs']['lora_name'].endswith(
            'Klein2-9B-SmartCharacterSwap.safetensors')


def test_missing_target_raises_value_error(app, tmp_path):
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    with app.app_context():
        _comfy(tmp_path, cfg)
        ref = tmp_path / 'ref.png'; ref.write_bytes(_png())
        with pytest.raises(ValueError, match='target image not found'):
            fsh.enqueue_face_swap(user_id='local',
                                  target_path=str(tmp_path / 'gone.png'),
                                  ref_path=str(ref))


def test_missing_swap_lora_raises_face_swap_lora_missing(app, tmp_path):
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    with app.app_context():
        _comfy(tmp_path, cfg, swap_lora=False)   # shared Klein assets present, swap LoRA absent
        target = tmp_path / 'target.png'; target.write_bytes(_png())
        ref = tmp_path / 'ref.png'; ref.write_bytes(_png())
        with pytest.raises(fsh.FaceSwapLoraMissing):
            fsh.enqueue_face_swap(user_id='local', target_path=str(target), ref_path=str(ref))


def test_missing_shared_klein_model_raises_klein_models_missing(app, tmp_path):
    from app import config as cfg
    from app.services import face_swap_helper as fsh
    from app.services.klein_edit_helper import KleinModelsMissing
    with app.app_context():
        base = _comfy(tmp_path, cfg)
        os.remove(str(base / 'models' / 'unet' / 'klein' / 'flux-2-klein-9b-fp8.safetensors'))
        target = tmp_path / 'target.png'; target.write_bytes(_png())
        ref = tmp_path / 'ref.png'; ref.write_bytes(_png())
        with pytest.raises(KleinModelsMissing):
            fsh.enqueue_face_swap(user_id='local', target_path=str(target), ref_path=str(ref))
