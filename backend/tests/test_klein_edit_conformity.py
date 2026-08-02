"""Klein EDITS must obey the instruction, not a style LoRA nobody picked.

THE REPORT: "Klein edits are not conformant" — the result drifts from what was
asked and from the source photo.

THE CAUSE, in the payload. Node 139 of the shipped ``improve skin.json`` is a
``LoraLoaderModelOnly`` pinned at ``strength_model: 0.8`` on
``klein/realistic.safetensors`` (dx8152 Flux2-Klein-9B-Enhanced-Details). Exactly
one lane ever overrode it — "Upscale & improve", which sets it to 0.0. The
reference edit, the variations, the regenerate and the small-image rescue passed
NOTHING, so the widget's own 0.8 stood.

For a long time that was invisible: the file shipped with neither the app nor the
Klein install, and ``enqueue_klein_edit`` BYPASSES node 139 when the LoRA is
absent — every install rendered its edits with no style LoRA at all. Then 031766f
(2026-07-22) added ``klein_enhancement_lora`` to the Setup downloads "like every
other Klein asset". From that commit on, a detail/style LoRA at 0.8 silently
joined every Klein edit on every install that ran Setup. Nothing in the UI showed
it, and no setting could turn it down.

WHAT IS PINNED HERE
  * the four non-improve lanes now pass a strength, and its default is 0.0 —
    i.e. the render every install had BEFORE the LoRA became downloadable;
  * the workflow file keeps its own 0.8 (the fallback for any caller that passes
    nothing), so this is a call-site fix, not a graph rewrite;
  * the setting reaches node 139 when raised, and degrades on junk;
  * the improve lane keeps its own separate knob.

No GPU, no ComfyUI: the assertion is on the workflow dict handed to
``queue_manager.add_job`` — the exact bytes ComfyUI would receive.
"""
import io
import json
from unittest.mock import patch

import pytest
from PIL import Image

from app.config import LOCAL_USER


@pytest.fixture(autouse=True)
def _reset_config_cache():
    import app.config as _cfg
    _cfg._cache = None
    yield
    _cfg._cache = None


# --- the payload: what node 139 actually receives ---------------------------

@pytest.fixture()
def captured(app, tmp_path):
    """Run one enqueue with a COMPLETE Klein install and return the submitted graph."""
    def run(**kwargs):
        from app import config as cfg
        from app.services import klein_edit_helper as keh
        seen = {}
        src = tmp_path / 'src.png'
        Image.new('RGB', (8, 8), (10, 20, 30)).save(src, format='PNG')
        comfy_in = tmp_path / 'comfy_input'
        comfy_in.mkdir(exist_ok=True)
        comfy = tmp_path / 'comfy'
        loras = comfy / 'models' / 'loras' / 'klein'
        loras.mkdir(parents=True, exist_ok=True)
        # BOTH LoRAs present = the state Setup leaves an install in since 031766f.
        (loras / 'Flux2-Klein-9B-consistency-V2.safetensors').write_bytes(b'0')
        (loras / 'realistic.safetensors').write_bytes(b'0')
        cfg.save_config({'comfyui': {'base_dir': str(comfy)}})
        with patch.object(keh.queue_manager, 'add_job',
                          side_effect=lambda **kw: seen.update(kw)), \
             patch.object(keh, '_comfy_input_dir', return_value=str(comfy_in)), \
             patch.object(keh, 'resolve_klein_unet', return_value='unet.safetensors'), \
             patch.object(keh, 'resolve_klein_vae', return_value='vae.safetensors'), \
             patch.object(keh, 'resolve_klein_text_encoder', return_value='te.safetensors'), \
             patch.object(keh, 'klein_missing_assets', return_value=[]):
            keh.enqueue_klein_edit(
                user_id='local', source_filename='src.png', source_path=str(src),
                edit_prompt='remove the glasses', **kwargs)
        return seen['workflow_data']
    with app.app_context():
        yield run


def _edit_lane_kwargs():
    from app.services import face_dataset_service as svc
    return {'sampler_steps': svc._generation_steps(),
            'base_lora_strength': svc._generation_base_lora_strength()}


def test_the_style_lora_no_longer_rides_on_an_edit(app, captured):
    """THE red assertion. Before the fix this was 0.8 — a detail LoRA at near-full
    strength on top of an instruction the user typed."""
    with app.app_context():
        w = captured(**_edit_lane_kwargs())
    assert w['139']['inputs']['lora_name'].endswith('realistic.safetensors')
    assert w['139']['inputs']['strength_model'] == 0.0


def test_what_the_lane_used_to_submit_is_still_reachable_on_purpose(app, captured):
    """The LoRA is not removed, it is CHOSEN. Someone who liked the 0.8 render sets
    the setting and gets exactly the graph they had."""
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'klein': {'edit_base_lora_strength': 0.8}})
        w = captured(**_edit_lane_kwargs())
    assert w['139']['inputs']['strength_model'] == 0.8


def test_the_prompt_and_the_source_still_reach_their_own_nodes(app, captured):
    """Non-regression on the rest of the conformity chain: the instruction lands
    verbatim on the text encoder and the staged source on the loader."""
    with app.app_context():
        w = captured(**_edit_lane_kwargs())
    assert w['6']['inputs']['text'] == 'remove the glasses'
    assert w['52']['inputs']['image'].startswith('edit_source_')
    assert w['92']['inputs']['latent'] == ['53', 0]      # source -> ReferenceLatent
    assert w['77']['inputs']['cfg'] == 1                 # guidance-distilled


@pytest.mark.parametrize('stored,expected', [
    (0.6, 0.6),
    (99, 2.0),            # clamped to _IMPROVE_MAX_STRENGTH
    (-1, 0.0),
    ('nonsense', 0.0),    # a hand-edited config degrades, never crashes the enqueue
    (None, 0.0),
])
def test_bad_values_degrade(app, captured, stored, expected):
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'klein': {'edit_base_lora_strength': stored}})
        w = captured(**_edit_lane_kwargs())
    assert w['139']['inputs']['strength_model'] == expected
    json.dumps(w)          # still submittable


def test_the_workflow_file_keeps_its_own_value():
    """A call-site fix: the graph still carries 0.8 for any caller passing nothing,
    so this change is visible in ONE place instead of forked into the JSON."""
    from app.services.klein_edit_helper import WORKFLOW_IMPROVE_SKIN_PATH
    wf = json.loads(WORKFLOW_IMPROVE_SKIN_PATH.read_text(encoding='utf-8'))
    assert wf['139']['inputs']['strength_model'] == 0.8


def test_the_improve_pass_keeps_its_own_knob():
    """Two passes, two intents: an improve is ALLOWED to add the detail LoRA."""
    from app.config import DEFAULTS
    assert DEFAULTS['klein']['edit_base_lora_strength'] == 0.0
    assert 'improve_base_lora_strength' in DEFAULTS['klein']


# --- the lanes: every non-improve enqueue passes it -------------------------

def _stub_enqueue(monkeypatch, calls):
    from app.services import klein_edit_helper as keh
    monkeypatch.setattr(keh, 'klein_missing_assets', lambda: [])
    monkeypatch.setattr(keh, 'klein_missing_nodes', lambda: [])
    monkeypatch.setattr(keh, 'resolve_generation_lora_preset', lambda _n: [])
    monkeypatch.setattr(keh, 'enqueue_klein_edit',
                        lambda **kw: (calls.append(kw) or 'job-1'))


def test_the_variation_lane_passes_it(app, monkeypatch):
    from app.services import face_dataset_service as svc
    calls = []
    _stub_enqueue(monkeypatch, calls)
    monkeypatch.setattr(svc, '_sync_generate_activity', lambda _d: None)
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Conform', 'zconform')
        ds.ref_filename = 'ref.png'
        svc.db.session.commit()
        svc.generate_variations(
            LOCAL_USER, ds.id,
            [{'label': 'Face · neutral', 'framing': 'face', 'prompt': 'a portrait'}],
            1, None)
    assert calls[0]['base_lora_strength'] == 0.0


def test_the_reference_edit_lane_passes_it(client, monkeypatch):
    """The lane the report came from: ✦ Edit reference on the Klein engine. Routed
    end to end through the real /ref/edit request so the assertion covers the call
    site, not a helper called by the test."""
    import contextlib
    import app.routes.datasets as dr
    from app.services import klein_edit_helper as keh
    from app.services import reference_edit_jobs as rej
    from app.services import dataset_activity
    rej.reset()
    dataset_activity.reset()
    calls = []
    monkeypatch.setattr(keh, 'enqueue_klein_edit',
                        lambda **kw: (calls.append(kw) or 'klein-job-1'))
    monkeypatch.setattr(dr, 'gpu_exclusive_vision_window', lambda: contextlib.nullcontext())
    buf = io.BytesIO()
    Image.new('RGB', (300, 300), (1, 2, 3)).save(buf, 'WEBP')
    monkeypatch.setattr(dr.svc, 'face_crop_to_square_webp',
                        lambda raw, **k: (buf.getvalue(), True))
    did = client.post('/api/dataset/create',
                      json={'name': 'RefConform', 'trigger_word': 'zrefconform'}
                      ).get_json()['id']
    png = io.BytesIO()
    Image.new('RGB', (256, 256), (120, 40, 40)).save(png, 'PNG')
    png.seek(0)
    client.post(f'/api/dataset/{did}/ref', data={'file': (png, 'r.png')},
                content_type='multipart/form-data')

    resp = client.post(f'/api/dataset/{did}/ref/edit',
                       data={'prompt': 'plain studio-grey background', 'engine': 'klein'},
                       content_type='multipart/form-data')

    assert resp.status_code == 202
    assert calls[0]['edit_prompt'] == 'plain studio-grey background'
    assert calls[0]['base_lora_strength'] == 0.0
    rej.reset()
    dataset_activity.reset()
