"""The Klein model a dataset runs on is a CHOICE, stored on the dataset (not per
browser) — originally shipped so ✨ Upscale & improve could honour it too.

That pass no longer runs Klein at all as of the Krea2 Ostris Edit + SeedVR2
swap (see krea_hq_helper): its model choice is auto-resolved, not user-picked,
so the "the stored choice reaches enqueue_klein_edit on all three improve
lanes" pinning that used to live here is gone with it. Everything below is
about `dataset_klein_model` / `enqueue_klein_edit` as they still stand for
every OTHER Klein call site (regenerate, variation restaging).

What this file pins:

* a dataset that never chose still resolves EXACTLY the model it resolved before
  (the anti-regression that made the original migration a no-op);
* a stored model that has since left the disk is refused BY NAME instead of being
  silently swapped for another one — the old `resolve_klein_unet(selected)`
  fallback chain did the swap without a word;
* every on-disk layout the resolver accepts is also OFFERED by the picker, layout
  by layout (the list is imported from test_klein_model_locations_documented so
  the two can never drift apart);
* the choice is stored on the dataset, sanitized against path traversal.
"""
import io
import os
import struct

import pytest
from PIL import Image

from test_klein_model_locations_documented import DOCUMENTED_LAYOUTS, KLEIN_FILE

_VALID_ST = struct.pack('<Q', 2) + b'{}'
OTHER_FILE = 'flux-2-klein-32b-heavy.safetensors'


def _png():
    buf = io.BytesIO()
    Image.new('RGB', (96, 64), (25, 50, 75)).save(buf, 'PNG')
    return buf.getvalue()


# --- A model that left the disk is refused BY NAME ---------------------------
def test_a_vanished_model_is_refused_by_name_not_swapped(app, tmp_path):
    """The old chain resolved an unknown pick to the canonical file and ran the
    job on it. A user who chose a 32B model and got a 9B result had no way to
    know: the tile looks fine, it is just not what was asked for."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    with app.app_context():
        base = tmp_path / 'ComfyUI'
        (base / 'models' / 'unet' / 'klein').mkdir(parents=True)
        (base / 'input').mkdir(parents=True, exist_ok=True)
        (base / 'main.py').write_text('# fake', encoding='utf-8')
        (base / 'models' / 'unet' / 'klein' / KLEIN_FILE).write_bytes(_VALID_ST)
        cfg.save_config({'comfyui': {'base_dir': str(base)}})

        # The present file still resolves, prefix and all.
        assert keh.klein_model_on_disk(KLEIN_FILE) == os.path.join('klein', KLEIN_FILE)
        # The absent one resolves to NOTHING — never to the neighbour.
        assert keh.klein_model_on_disk(OTHER_FILE) is None

        src = tmp_path / 'src.png'
        src.write_bytes(_png())
        with pytest.raises(keh.KleinModelGone) as exc:
            keh.enqueue_klein_edit(user_id='local', source_filename='src.png',
                                   source_path=str(src), edit_prompt='improve',
                                   klein_model=OTHER_FILE)
        assert OTHER_FILE in str(exc.value)
        assert exc.value.name == OTHER_FILE


# --- 4. The offered list is as WIDE as the resolver -------------------------
# Offering fewer places than the resolver accepts is the same bug seen from the
# other end: the app would refuse to let you pick a model it can happily load.
@pytest.mark.parametrize('label,parts,expected', DOCUMENTED_LAYOUTS,
                         ids=[c[0] for c in DOCUMENTED_LAYOUTS])
def test_every_resolvable_layout_is_also_offerable(app, tmp_path, label, parts, expected):
    from app import capabilities, config as cfg
    from app.services import comfy_model_paths as cmp
    from app.services import klein_edit_helper as keh
    cmp.clear_cache()
    with app.app_context():
        base = tmp_path / 'ComfyUI'
        for sub in ('input', 'output', 'models'):
            (base / sub).mkdir(parents=True, exist_ok=True)
        (base / 'main.py').write_text('# fake', encoding='utf-8')
        target = base.joinpath('models', *parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_VALID_ST)
        cfg.save_config({'comfyui': {'base_dir': str(base)}})
        offered = capabilities._scan_models()['klein']
        assert KLEIN_FILE in offered, label
        # …and picking exactly what was offered lands on exactly what the
        # resolver would have chosen by itself.
        assert keh.klein_model_on_disk(KLEIN_FILE) == expected, label
        assert keh.resolve_klein_unet() == expected, label
    cmp.clear_cache()


# --- 5. Storing the choice -------------------------------------------------
def test_the_choice_is_stored_on_the_dataset_not_the_browser(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Storage', 'storage')
        assert svc.dataset_klein_model(ds) is None
        svc.set_dataset_klein_model(LOCAL_USER, ds.id, OTHER_FILE)
        assert svc.dataset_klein_model(ds) == OTHER_FILE
        # Clearing goes back to "auto" — the same NULL a fresh dataset has, so
        # un-choosing is a real gesture and not a stuck value.
        svc.set_dataset_klein_model(LOCAL_USER, ds.id, '')
        assert svc.dataset_klein_model(ds) is None


def test_a_path_separator_cannot_be_smuggled_into_the_choice(app):
    """The picker sends a BARE file name; the prefix is the resolver's job. A
    value carrying a separator would be a traversal attempt, not a model."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Traversal', 'traversal')
        for bad in ('../secret.safetensors', 'sub/model.safetensors',
                    'sub\\model.safetensors'):
            with pytest.raises(ValueError):
                svc.set_dataset_klein_model(LOCAL_USER, ds.id, bad)
        assert svc.dataset_klein_model(ds) is None
