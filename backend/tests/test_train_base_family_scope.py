"""The training base is FAMILY-scoped, even though it lives in one column.

Reported symptom: a style dataset showing `LORA TYPE = Krea 2` / `BASE =
Official - Krea 2` while the summary line right below said
``base "z image\\bigLove_zt3.safetensors"``, and the cloud dialog offered to
push that Z-Image file to Hugging Face for the Krea run — under a warning that
the local file was "missing" (it was not: a Z-Image merge NAME was being
resolved as if it were a Krea absolute path).

Two defects, one column: `train_base_model` is global, so switching the family
left the previous family's base attached; and the panel re-seeded itself from
that column on every mount, which is why "change the model and come back" (a
purely client-side reset) fixed it until the next reload.
"""
import json

import pytest

from app.config import LOCAL_USER

ZIMAGE_MERGE = 'z image\\bigLove_zt3.safetensors'


@pytest.fixture()
def aitoolkit(tmp_path, app):
    """/train/base-info is gated on a valid ai-toolkit install; fake one (never
    executed) so the route answers instead of 409-ing."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})
    return root


@pytest.fixture()
def style_ds(app):
    """A style dataset (permanent, no trigger) on Z-Image with a custom merge —
    the reported setup. Style is asserted because the report came from one; the
    scoping bug is kind-independent and the assertions below never read it."""
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Mass data', '', kind='style',
                                train_type='zimage')
        ds.train_base_model = ZIMAGE_MERGE
        ds.train_variant = 'turbo'
        svc.db.session.commit()
        assert svc.is_style(ds) if hasattr(svc, 'is_style') else True
        return ds.id


# --- 1) the scope itself -------------------------------------------------------

def test_switching_family_detaches_the_other_familys_base(app, style_ds):
    """RED before the fix: the Z-Image merge stayed on `train_base_model` after
    the family became Krea 2, and the panel's summary printed it."""
    from app.services import face_dataset_service as svc
    with app.app_context():
        assert svc.set_train_type(LOCAL_USER, style_ds, 'krea') is True
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        assert ds.train_type == 'krea'
        assert not ds.train_base_model, (
            'a Krea 2 run must not carry the base picked for Z-Image')


def test_coming_back_restores_the_family_own_base(app, style_ds):
    """The base is remembered, not destroyed — that is what makes detaching it
    safe enough to do without asking (option B, "wipe it", was rejected)."""
    from app.services import face_dataset_service as svc
    with app.app_context():
        svc.set_train_type(LOCAL_USER, style_ds, 'krea')
        svc.set_train_type(LOCAL_USER, style_ds, 'zimage')
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        assert ds.train_base_model == ZIMAGE_MERGE
        assert ds.train_variant == 'turbo'


def test_each_family_keeps_its_own_base(app, style_ds, tmp_path):
    """Two families, two bases, no interference — including the ABSOLUTE custom
    path shape that krea/flux/klein use."""
    from app.services import face_dataset_service as svc
    weights = tmp_path / 'krea_custom.safetensors'
    weights.write_bytes(b'0')
    with app.app_context():
        svc.set_train_type(LOCAL_USER, style_ds, 'krea')
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        ds.train_base_model = str(weights)
        ds.train_variant = 'base'
        svc.db.session.commit()
        svc.set_train_type(LOCAL_USER, style_ds, 'zimage')
        assert svc.get_dataset(LOCAL_USER, style_ds).train_base_model == ZIMAGE_MERGE
        svc.set_train_type(LOCAL_USER, style_ds, 'krea')
        back = svc.get_dataset(LOCAL_USER, style_ds)
        assert back.train_base_model == str(weights)
        assert back.train_variant == 'base'


def test_switch_to_a_never_used_family_starts_official(app, style_ds):
    from app.services import face_dataset_service as svc
    with app.app_context():
        svc.set_train_type(LOCAL_USER, style_ds, 'flux2klein')
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        assert not ds.train_base_model
        assert not ds.train_variant      # -> _default_variant_for('flux2klein')


def test_reselecting_the_same_family_changes_nothing(app, style_ds):
    from app.services import face_dataset_service as svc
    with app.app_context():
        svc.set_train_type(LOCAL_USER, style_ds, 'zimage')
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        assert ds.train_base_model == ZIMAGE_MERGE
        assert ds.train_variant == 'turbo'


# --- 2) anti-regression: a base on the RIGHT family is left alone --------------

def test_existing_dataset_with_a_coherent_base_is_untouched(app, tmp_path):
    """The migration is additive and lazy: nothing runs over existing rows. A
    dataset whose custom base matches its family must read back byte-identical,
    with an empty memory (nothing has been switched yet)."""
    from app.services import face_dataset_service as svc
    weights = tmp_path / 'my_krea.safetensors'
    weights.write_bytes(b'0')
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Legacy', 'trg', train_type='krea')
        ds.train_base_model = str(weights)
        ds.train_variant = 'base'
        ds.train_family_bases = None            # exactly what an old row holds
        svc.db.session.commit()
        again = svc.get_dataset(LOCAL_USER, ds.id)
        assert again.train_base_model == str(weights)
        assert again.train_variant == 'base'
        assert svc.family_base_memory(again) == {}
        assert svc.remembered_family_base(again, 'krea') == (None, None)


def test_a_corrupted_memory_blob_degrades_to_nothing_remembered(app, style_ds):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        ds.train_family_bases = 'not json at all'
        svc.db.session.commit()
        assert svc.family_base_memory(svc.get_dataset(LOCAL_USER, style_ds)) == {}
        # …and switching still works, from a clean slate.
        assert svc.set_train_type(LOCAL_USER, style_ds, 'krea') is True
        assert not svc.get_dataset(LOCAL_USER, style_ds).train_base_model


def test_a_foreign_base_is_not_remembered_as_the_familys_own(app, style_ds):
    """The legacy state (Z-Image merge sitting on a Krea dataset) must not be
    stashed under 'krea' on the way out — that would make the bug permanent by
    handing it back the next time Krea 2 is selected."""
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        ds.train_type = 'krea'                  # the state the report came from
        svc.db.session.commit()
        svc.set_train_type(LOCAL_USER, style_ds, 'flux')
        svc.set_train_type(LOCAL_USER, style_ds, 'krea')
        assert not svc.get_dataset(LOCAL_USER, style_ds).train_base_model


# --- 3) the panel reflects the change WITHOUT a round trip ---------------------

def test_base_info_reports_the_base_the_run_will_actually_use(app, client, aitoolkit, style_ds):
    """The panel reads /train/base-info at mount. On the reported dataset it
    answered `base = <Z-Image merge>` with `train_type = krea`, which is what
    printed the contradictory summary line on every single open — and no amount
    of client-side family toggling changed it, because nothing was persisted."""
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        ds.train_type = 'krea'
        svc.db.session.commit()
    r = client.get(f'/api/dataset/{style_ds}/train/base-info')
    assert r.status_code == 200
    body = r.get_json()
    assert body['train_type'] == 'krea'
    assert body['base'] == ''
    assert 'Krea 2' in body['base_family_mismatch']
    assert 'bigLove_zt3.safetensors' in body['base_family_mismatch']


def test_base_info_after_a_family_switch_needs_no_second_switch(app, client, aitoolkit, style_ds):
    """One POST /train-type, one GET /train/base-info — the sequence the panel
    performs — and the answer is already right. Before the fix this returned the
    Z-Image merge, and only a switch-away-and-back (client state, never
    persisted) hid it until the next reload."""
    r = client.post(f'/api/dataset/{style_ds}/train-type', json={'train_type': 'krea'})
    assert r.status_code == 200
    body = client.get(f'/api/dataset/{style_ds}/train/base-info').get_json()
    assert body['train_type'] == 'krea'
    assert body['base'] == ''
    assert body['base_family_mismatch'] is None      # detached, not merely hidden


def test_base_info_keeps_a_coherent_custom_base(app, client, aitoolkit, style_ds, tmp_path):
    weights = tmp_path / 'krea_ok.safetensors'
    weights.write_bytes(b'0')
    from app.services import face_dataset_service as svc
    with app.app_context():
        svc.set_train_type(LOCAL_USER, style_ds, 'krea')
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        ds.train_base_model = str(weights)
        svc.db.session.commit()
    body = client.get(f'/api/dataset/{style_ds}/train/base-info').get_json()
    assert body['base'] == str(weights)
    assert body['base_family_mismatch'] is None


# --- 4) the coherence predicate ------------------------------------------------

@pytest.mark.parametrize('family', ['krea', 'flux', 'flux2klein', 'anima'])
def test_relative_base_is_foreign_on_absolute_path_families(family):
    from app.services import lora_training as lt
    assert lt.foreign_base_reason(family, ZIMAGE_MERGE) == 'relative_base_on_absolute_family'


def test_absolute_base_is_foreign_on_zimage():
    from app.services import lora_training as lt
    assert lt.foreign_base_reason('zimage', 'C:\\weights\\krea.safetensors') \
        == 'absolute_base_on_zimage'


def test_coherent_selections_are_not_flagged():
    from app.services import lora_training as lt
    assert lt.foreign_base_reason('zimage', ZIMAGE_MERGE) is None
    assert lt.foreign_base_reason('krea', 'C:\\weights\\krea.safetensors') is None
    assert lt.foreign_base_reason('krea', '') is None
    assert lt.foreign_base_reason('krea', None) is None
    # SDXL is deliberately out of scope: its bases are relative basenames too,
    # and telling them apart needs a configured ComfyUI.
    assert lt.foreign_base_reason('sdxl', 'anySDXL.safetensors') is None


# --- 5) the cloud lane: no push, no pod, for another family's base -------------

def test_cloud_readiness_names_the_family_not_a_missing_file(app, style_ds):
    """The reported modal said "The local file is unavailable (missing) —
    restore it to push". The file was never missing: base_push_state resolved a
    Z-Image merge NAME as a Krea absolute path and blamed the disk."""
    from app.services import hf_base_push
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        ds.train_type = 'krea'
        svc.db.session.commit()
        st = hf_base_push.base_push_state(LOCAL_USER, style_ds, 'krea', 'base',
                                          ZIMAGE_MERGE, 'a-token')
    assert st['ready'] is False
    assert st['reason'] == 'foreign_family'
    assert 'Krea 2' in st['foreign_base_message']


def test_cloud_push_refuses_another_familys_base(app, style_ds):
    from app.services import hf_base_push
    with app.app_context():
        with pytest.raises(hf_base_push.HfPublishError) as e:
            hf_base_push.start_push(app, style_ds, 'krea', 'base', ZIMAGE_MERGE,
                                    'a-token')
    assert e.value.code == 'foreign_family'


def test_cloud_push_refuses_when_the_local_file_is_absent(app, style_ds, tmp_path):
    """Independent of the family question: an action must not be offered for a
    file that is not there. Absent locally, the one-time upload has nothing to
    send — refused synchronously, before any thread or HF call."""
    from app.services import hf_base_push
    missing = str(tmp_path / 'deleted_after_being_chosen.safetensors')
    with app.app_context():
        with pytest.raises(hf_base_push.HfPublishError) as e:
            hf_base_push.start_push(app, style_ds, 'krea', 'base', missing, 'a-token')
    assert e.value.code == 'weights_missing'
    assert 'deleted_after_being_chosen' in e.value.message


def test_cloud_launch_guard_refuses_another_familys_base(app, style_ds):
    """require_base_repo is what runs before a pod is RENTED."""
    from app.services import hf_base_push
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        with pytest.raises(ValueError, match='another model family'):
            hf_base_push.require_base_repo(ds, 'krea', 'base', ZIMAGE_MERGE,
                                           'a-token')


# --- 6) the LIST of bases is family-scoped too, not just the chosen one --------

def test_every_family_gets_its_own_base_list(app, client, aitoolkit, style_ds):
    """The panel reads `bases_by_type[family] || bases`, and `bases` is the
    Z-Image list. A family missing from the map therefore does not show an empty
    selector — it shows Z-Image's, complete with this install's Z-Image merges.

    Anima was that family: `MODEL FAMILY = Anima (~20 img)` with
    `BASE = Official - Z-Image-Turbo (recommended)`, and the summary line under
    the Train button repeating it. Parametrising over TRAIN_TYPES is the point:
    the next family added pays for the omission here, not in a screenshot."""
    from app.services.face_dataset_service import TRAIN_TYPES
    body = client.get(f'/api/dataset/{style_ds}/train/base-info').get_json()
    missing = [f for f in TRAIN_TYPES if f not in body['bases_by_type']]
    assert not missing, f'families falling back to the Z-Image list: {missing}'


def test_the_anima_selector_offers_the_anima_base_and_nothing_else(
        app, client, aitoolkit, style_ds):
    """Anima has exactly one official base and no custom lane, so its selector
    must name it. Nothing Z-Image may appear there: a Z-Image merge is a
    different architecture, and picking one is an act the run then ignores."""
    from app.services import lora_training as lt
    client.post(f'/api/dataset/{style_ds}/train-type', json={'train_type': 'anima'})
    body = client.get(f'/api/dataset/{style_ds}/train/base-info').get_json()
    anima = body['bases_by_type']['anima']
    assert [b['value'] for b in anima] == ['']
    assert lt.ANIMA_BASE_LABEL in anima[0]['label']
    assert 'Z-Image' not in anima[0]['label']


def test_a_z_image_base_left_on_anima_never_reaches_the_job_config(app, style_ds):
    """The gravity verdict, pinned. While the selector was lying, a user could
    pick a Z-Image merge under Anima — so this asserts the run itself was never
    corrupted: the builder gates on an ABSOLUTE path, so a merge NAME falls back
    to the official Anima base. The defect was the display, and this test is
    what keeps it that way."""
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        ds.train_type = 'anima'
        ds.train_base_model = ZIMAGE_MERGE
        svc.db.session.commit()
        cfg = lt._build_job_config_anima(ds, 'folder', 100, training_folder='out')
    model = cfg['config']['process'][0]['model']
    assert model['name_or_path'] == lt.ANIMA_BASE
    assert model['arch'] == 'anima'


# --- 7) the memory survives a round trip through JSON --------------------------

def test_memory_is_plain_json_on_the_row(app, style_ds):
    from app.services import face_dataset_service as svc
    with app.app_context():
        svc.set_train_type(LOCAL_USER, style_ds, 'krea')
        ds = svc.get_dataset(LOCAL_USER, style_ds)
        stored = json.loads(ds.train_family_bases)
        assert stored['zimage'] == {'base': ZIMAGE_MERGE, 'variant': 'turbo'}
