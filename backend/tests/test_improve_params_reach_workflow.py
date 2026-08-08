"""`klein_edit_helper.enqueue_klein_edit`'s own parameter-to-node contract:
lora_strength/sampler_steps/base_lora_strength/output_megapixels each have to
survive the call and land on the RIGHT node in `improve skin.json`. Calls the
helper DIRECTLY (the queue_manager.add_job seam), no GPU, repeatable.

Historical note: this file used to prove the ✨ Upscale & improve Settings
(`klein.improve_steps` etc.) reached ComfyUI through THIS helper — as of the
Krea2 Ostris Edit + SeedVR2 swap (see krea_hq_helper) the improve pass no
longer calls `enqueue_klein_edit` at all, so those specific Settings are
currently orphaned (nothing reads them). The fixture below still drives real
numbers through `fds._improve_float`/`_improve_int` as a convenient source of
test doubles, but the contract being pinned is `enqueue_klein_edit`'s own
argument-to-node wiring — still load-bearing for every OTHER Klein call site
(regenerate, variation restaging) — not the improve pass.

Node map of `improve skin.json`, verified against the shipped file:
  139 LoraLoaderModelOnly  -> strength_model : the enhancement LoRA (realistic detail)
  77  KSampler             -> steps
  174 ImageScaleToTotalPixels -> megapixels : the OUTPUT resolution
and lora_strength drives the consistency LoRA (klein.consistency_strength), which
anchors composition — it is NOT an identity LoRA.
"""
import json
import os
from unittest.mock import patch

import pytest
from PIL import Image


@pytest.fixture()
def captured(app, tmp_path):
    """Run an improve enqueue and hand back the workflow that would be submitted."""
    def run(enhancement_lora_installed=True, **config):
        from app import config as cfg
        from app.services import klein_edit_helper as keh
        seen = {}

        src = tmp_path / 'src.png'
        Image.new('RGB', (1, 1), (10, 20, 30)).save(src, format='PNG')
        comfy_in = tmp_path / 'comfy_input'
        comfy_in.mkdir(exist_ok=True)
        # Node 139 carries klein/realistic.safetensors, which ships with NEITHER the
        # app nor the Klein install — enqueue_klein_edit bypasses the node when the
        # file is absent. Whether it is there decides if the strength does anything
        # at all, so the fixture makes that explicit instead of incidental.
        comfy = tmp_path / 'comfy'
        loras = comfy / 'models' / 'loras' / 'klein'
        loras.mkdir(parents=True, exist_ok=True)
        (loras / 'Flux2-Klein-9B-consistency-V2.safetensors').write_bytes(b'0')
        target = loras / 'realistic.safetensors'
        if enhancement_lora_installed:
            target.write_bytes(b'0')
        elif target.exists():
            target.unlink()
        cfg.save_config({'comfyui': {'base_dir': str(comfy)}})

        if config:
            cfg.save_config({'klein': config})
        with patch.object(keh.queue_manager, 'add_job',
                          side_effect=lambda **kw: seen.update(kw)), \
             patch.object(keh, '_comfy_input_dir', return_value=str(comfy_in)), \
             patch.object(keh, 'resolve_klein_unet', return_value='unet.safetensors'), \
             patch.object(keh, 'resolve_klein_vae', return_value='vae.safetensors'), \
             patch.object(keh, 'resolve_klein_text_encoder', return_value='te.safetensors'), \
             patch.object(keh, 'klein_missing_assets', return_value=[]):
            from app.services import face_dataset_service as fds
            keh.enqueue_klein_edit(
                user_id='local', source_filename='src.png', source_path=str(src),
                edit_prompt='improve',
                lora_strength=fds._improve_float('improve_consistency_strength', 1.0, 1.5),
                sampler_steps=fds._improve_int('improve_steps', 4),
                base_lora_strength=fds._improve_float('improve_base_lora_strength', 0.0),
                output_megapixels=fds._improve_float('improve_megapixels', 2.0, 8.0),
            )
        return seen['workflow_data']
    with app.app_context():
        yield run


def test_defaults_reproduce_the_shipped_behaviour(captured):
    """An untouched install must submit exactly what it submitted before the
    settings existed — otherwise adding a knob silently changed everyone's output."""
    w = captured()
    assert w['77']['inputs']['steps'] == 4
    assert w['139']['inputs']['strength_model'] == 0.0     # enhancement LoRA off
    assert w['174']['inputs']['megapixels'] == 2.0         # the historical 2 MP


def test_every_knob_lands_on_its_own_node(captured):
    w = captured(improve_base_lora_strength=0.75, improve_steps=9, improve_megapixels=4)
    assert w['139']['inputs']['strength_model'] == 0.75
    assert w['77']['inputs']['steps'] == 9
    assert w['174']['inputs']['megapixels'] == 4.0
    # untouched knobs keep the workflow's own values — we patch, never rebuild
    assert w['77']['inputs']['sampler_name'] == 'euler'
    assert w['77']['inputs']['cfg'] == 1


def test_output_size_is_what_makes_it_an_upscale(captured):
    """Node 174 rescales the source to a pixel budget BEFORE sampling, so it is the
    result's resolution. Hardcoded at 2 MP until now, which made "Upscale" produce
    the same size whatever you asked for."""
    assert captured(improve_megapixels=1)['174']['inputs']['megapixels'] == 1.0
    assert captured(improve_megapixels=6)['174']['inputs']['megapixels'] == 6.0


def test_the_consistency_lora_is_clamped_where_the_helper_clamps_it(captured):
    """It drives klein.consistency_strength, which enqueue_klein_edit caps at 1.5.
    Offering more in Settings would be a value the engine silently pulls back."""
    from app.services import face_dataset_service as fds
    from app import config as cfg
    cfg.save_config({'klein': {'improve_consistency_strength': 99}})
    assert fds._improve_float('improve_consistency_strength', 1.0, 1.5) == 1.5


def test_a_broken_config_still_submits_a_valid_workflow(captured):
    """A hand-edited config must degrade the pass, never break the enqueue."""
    w = captured(improve_steps='nonsense', improve_megapixels=None,
                 improve_base_lora_strength=[])
    assert w['77']['inputs']['steps'] == 4
    assert w['174']['inputs']['megapixels'] == 2.0
    assert w['139']['inputs']['strength_model'] == 0.0
    json.dumps(w)          # still serialisable, i.e. still submittable


def test_a_raised_strength_with_no_lora_file_is_an_ERROR_not_a_silent_skip(captured):
    """The behaviour this whole investigation was about. Bypassing a missing LoRA is
    right at strength 0 — the node would contribute nothing. But doing it while the
    user deliberately raised the strength is what made the setting look broken: the
    job ran, the result was unchanged, and nothing said why. It is now reported as a
    missing ASSET, which is also what triggers the auto-download."""
    from app.services.klein_edit_helper import KleinModelsMissing
    with pytest.raises(KleinModelsMissing) as err:
        captured(enhancement_lora_installed=False, improve_base_lora_strength=0.8)
    assert 'klein_enhancement_lora' in err.value.missing


@pytest.mark.parametrize('sep,expected', [
    ('/',  'klein/realistic.safetensors'),      # a Linux install (Docker)
    ('\\', 'klein\\realistic.safetensors'),     # a Windows install
])
def test_a_workflow_name_is_respelled_for_the_host_that_opens_it(sep, expected):
    """`improve skin.json` was exported from a Windows ComfyUI, so node 139 carries
    ``klein\\realistic.safetensors``. On a Linux install a backslash is a legal
    FILENAME character, not a separator: the probe asked for the single file
    ``klein\\realistic.safetensors`` inside models/loras, missed the one really in
    ``klein/``, and "Upscale & improve" answered "Klein needs klein_enhancement_lora,
    I've started downloading it" — over a file already on disk, forever (reported by
    @_nofaceman, Discord). Every OTHER model reference already went through this
    normalisation (resolve_model_ref, _resolve_lora_abs_path); node 139 was the one
    hole, which is exactly why that one file looked broken while all the others
    loaded. `sep` is injectable so the Linux contract is pinned from a Windows host
    and the reverse — the defect was invisible on the OS it was written on."""
    from app.services.klein_edit_helper import normalize_rel_model_name
    assert normalize_rel_model_name('klein\\realistic.safetensors', sep=sep) == expected
    assert normalize_rel_model_name('klein/realistic.safetensors', sep=sep) == expected


def test_the_installed_base_lora_survives_at_a_raised_strength(captured):
    """The case no test covered: file present AND strength raised. Only the two
    ABSENT-file cases were pinned, so a node 139 that silently stopped resolving
    could not fail any test. Also asserts the name handed to ComfyUI is spelled with
    the host separator — a LoraLoader validates against `folder_paths`, which lists
    with os.sep, so a foreign spelling fails validation even once the file is found."""
    w = captured(enhancement_lora_installed=True, improve_base_lora_strength=0.8)
    assert w['139']['inputs']['strength_model'] == 0.8
    assert w['139']['inputs']['lora_name'] == os.path.join('klein', 'realistic.safetensors')


def test_strength_zero_with_no_lora_file_stays_a_quiet_skip(captured):
    """Nothing is lost by not loading a LoRA nobody asked to apply, so this must NOT
    become an error — otherwise a stock install could no longer improve at all."""
    w = captured(enhancement_lora_installed=False, improve_base_lora_strength=0)
    assert '139' not in w
    assert w['174']['inputs']['megapixels'] == 2.0     # the pass still runs
