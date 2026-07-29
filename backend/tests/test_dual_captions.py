"""Dual long+short captioning (ai-toolkit short_and_long_captions).

Covers the whole seam:
  - additive caption_short migration on a LEGACY table (no column yet);
  - the dual_captions train-setting round-trip (update/effective + preset whitelist);
  - short-caption derivation from the long one (text-only stub) preserving the kind
    omission (concept ban-list scrub, style aesthetic strip), and no-op when OFF;
  - export writing the JSON caption file + the recipe pointing folder_path at it with
    short_and_long_captions on — and OFF being byte-identical to the historical export;
  - the cloud path stripping dual back to the historical folder + .txt sidecars.

Single-user extraction (LOCAL_USER). The vision seam is imported locally by the pipeline,
so it is patched at app.services.vision_ollama.*.
"""
import io
import json
import os

from PIL import Image
from sqlalchemy import text

from app.extensions import db
from app.models import FaceDataset, FaceDatasetImage
from app.services import face_dataset_service as svc
from app.services import lora_training as lt
from app.config import LOCAL_USER, save_config


def _png(w=64, h=64):
    b = io.BytesIO()
    Image.new('RGB', (w, h), (120, 40, 40)).save(b, 'PNG')
    return b.getvalue()


def _enable_dual(ds):
    ds.train_settings = json.dumps({'dual_captions': True})
    db.session.commit()


def _kept_image(ds, fn, caption=None, caption_short=None):
    img_dir = svc._dataset_dir(ds.id)
    os.makedirs(img_dir, exist_ok=True)
    Image.new('RGB', (32, 32)).save(os.path.join(img_dir, fn))
    row = FaceDatasetImage(dataset_id=ds.id, status='keep', filename=fn,
                           caption=caption, caption_short=caption_short)
    db.session.add(row)
    db.session.commit()
    return row


# --- additive migration on a legacy table ------------------------------------
def test_caption_short_migration_on_legacy_table(app):
    with app.app_context():
        from app import _apply_additive_migrations
        # Rebuild the table WITHOUT caption_short to mimic a database created before the
        # column existed, with a legacy row present.
        db.session.execute(text('DROP TABLE face_dataset_image'))
        db.session.execute(text(
            'CREATE TABLE face_dataset_image ('
            'id INTEGER PRIMARY KEY, dataset_id INTEGER, status TEXT, caption TEXT)'))
        db.session.execute(text(
            "INSERT INTO face_dataset_image (id, dataset_id, status, caption) "
            "VALUES (1, 1, 'keep', 'a long caption')"))
        db.session.commit()
        cols = {r[1] for r in db.session.execute(text('PRAGMA table_info(face_dataset_image)'))}
        assert 'caption_short' not in cols

        _apply_additive_migrations()

        cols = {r[1] for r in db.session.execute(text('PRAGMA table_info(face_dataset_image)'))}
        assert 'caption_short' in cols
        # Legacy row survived and the new column is NULL for it.
        row = db.session.execute(text(
            'SELECT caption, caption_short FROM face_dataset_image WHERE id=1')).first()
        assert row[0] == 'a long caption' and row[1] is None
        # Idempotent: a second run must not raise.
        _apply_additive_migrations()


# --- train-setting round-trip -------------------------------------------------
def test_dual_captions_setting_roundtrip(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        assert lt.effective_train_settings(ds)['dual_captions'] is False
        assert svc.dual_captions_enabled(ds) is False

        eff = lt.update_train_settings(LOCAL_USER, ds.id, {'dual_captions': True})
        assert eff['dual_captions'] is True
        assert svc.dual_captions_enabled(db.session.get(FaceDataset, ds.id)) is True

        # Falsy drops the key so OFF is byte-identical to a dataset that never set it.
        eff = lt.update_train_settings(LOCAL_USER, ds.id, {'dual_captions': False})
        assert eff['dual_captions'] is False
        stored = json.loads(db.session.get(FaceDataset, ds.id).train_settings or '{}')
        assert 'dual_captions' not in stored

        assert 'dual_captions' in lt.TRAIN_SETTING_KEYS


def test_preset_apply_roundtrips_dual_captions(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        eff, ignored, rejected = lt.apply_train_settings_dict(
            LOCAL_USER, ds.id, {'dual_captions': True})
        assert eff['dual_captions'] is True
        assert 'dual_captions' not in ignored and not rejected


# --- derivation preserves the kind omission ----------------------------------
def test_derive_short_scrubs_concept_ban_list(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept',
                                concept_desc='licking ice cream')
        _enable_dual(ds)
        img = _kept_image(ds, 'k0.png', caption='A woman on a bench with a dessert')

        # The shortener tries to reintroduce the banned concept; the mechanical scrub
        # (describe=None) must drop it, exactly like the long-caption guarantee.
        leaky = 'A woman on a bench, licking ice cream'
        n = svc.derive_short_captions(LOCAL_USER, ds.id, generate=lambda p: leaky)
        assert n == 1
        short = db.session.get(FaceDatasetImage, img.id).caption_short
        assert short
        low = short.lower()
        assert 'ice' not in low and 'cream' not in low and 'licking' not in low


def test_derive_short_strips_style_trigger(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Look', 'zstyle_look', kind='style')
        _enable_dual(ds)
        img = _kept_image(ds, 'k0.png', caption='a woman standing in a field')

        # A style short must stay content-only: a stray leading trigger is stripped.
        n = svc.derive_short_captions(LOCAL_USER, ds.id,
                                      generate=lambda p: 'zstyle_look, a woman in a field')
        assert n == 1
        short = db.session.get(FaceDatasetImage, img.id).caption_short
        assert short and not short.lower().startswith('zstyle_look')


def test_derive_short_noop_when_disabled(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')   # dual OFF
        img = _kept_image(ds, 'k0.png', caption='a long caption')
        called = {'n': 0}

        def gen(p):
            called['n'] += 1
            return 'short'

        assert svc.derive_short_captions(LOCAL_USER, ds.id, generate=gen) == 0
        assert called['n'] == 0
        assert db.session.get(FaceDatasetImage, img.id).caption_short is None


def test_derive_short_fills_only_missing_unless_forced(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        _enable_dual(ds)
        a = _kept_image(ds, 'a.png', caption='long a')
        b = _kept_image(ds, 'b.png', caption='long b', caption_short='kept short b')

        # force=False fills only the missing one, leaving an existing short untouched.
        n = svc.derive_short_captions(LOCAL_USER, ds.id, generate=lambda p: 'derived')
        assert n == 1
        assert db.session.get(FaceDatasetImage, a.id).caption_short == 'derived'
        assert db.session.get(FaceDatasetImage, b.id).caption_short == 'kept short b'

        # force=True overwrites all.
        n = svc.derive_short_captions(LOCAL_USER, ds.id, force=True, generate=lambda p: 'fresh')
        assert n == 2
        assert db.session.get(FaceDatasetImage, b.id).caption_short == 'fresh'


# --- set_image_caption short param -------------------------------------------
def test_set_image_caption_short_is_opt_in(app):
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        img = _kept_image(ds, 'k0.png', caption='old', caption_short='old short')

        # Omitting short leaves the existing short intact.
        svc.set_image_caption(LOCAL_USER, img.id, 'new long')
        row = db.session.get(FaceDatasetImage, img.id)
        assert row.caption == 'new long' and row.caption_short == 'old short'

        # Passing short updates it.
        svc.set_image_caption(LOCAL_USER, img.id, 'new long', short='new short')
        row = db.session.get(FaceDatasetImage, img.id)
        assert row.caption_short == 'new short'


# --- export writes the JSON caption file, recipe points at it -----------------
def test_export_writes_dual_json_and_recipe(app, tmp_path):
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        _enable_dual(ds)
        _kept_image(ds, 'a.png', caption='a full caption', caption_short='a short')
        _kept_image(ds, 'b.png', caption='b full caption')   # no short → fallback to long

        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False,
                                             dest_dir=str(tmp_path / 'export'))
        json_path = lt._dual_caption_json_path(out)
        assert os.path.isfile(json_path)
        data = json.loads(open(json_path, encoding='utf-8').read())
        assert len(data) == 2
        for entry in data.values():
            assert set(entry) == {'caption', 'caption_short'}
            # Character trigger prepended to BOTH the long and the short.
            assert entry['caption'].startswith('zchar_emma,')
            assert entry['caption_short'].startswith('zchar_emma,')
        # Missing short degrades to the long caption (short == long). Exported files are
        # renamed <trigger>_NNN.png, so match on the caption content, not the source name.
        b_entry = next(v for v in data.values() if 'b full caption' in v['caption'])
        assert b_entry['caption_short'] == b_entry['caption']

        # Recipe points folder_path at the JSON and turns on short_and_long_captions.
        cfg = lt.build_job_config(ds, out, steps=100)
        proc = cfg['config']['process'][0]
        assert proc['datasets'][0]['folder_path'] == json_path
        assert proc['train']['short_and_long_captions'] is True


def test_export_off_is_historical(app, tmp_path):
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')   # dual OFF
        _kept_image(ds, 'a.png', caption='a full caption', caption_short='ignored short')

        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False,
                                             dest_dir=str(tmp_path / 'export'))
        # No JSON file; the .txt sidecar is the only caption source.
        assert not os.path.isfile(lt._dual_caption_json_path(out))
        assert os.path.isfile(os.path.join(out, 'zchar_emma_000.txt'))

        cfg = lt.build_job_config(ds, out, steps=100)
        proc = cfg['config']['process'][0]
        assert proc['datasets'][0]['folder_path'] == out
        assert 'short_and_long_captions' not in proc.get('train', {})


# --- the re-caption route regenerates BOTH captions --------------------------
def test_caption_route_regenerates_both_when_dual_on(app, client, monkeypatch):
    from app.services import vision_ollama
    with app.app_context():
        save_config({'captioning': {'backend': 'ollama'}})   # skip JoyCaption
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        _enable_dual(ds)
        img = _kept_image(ds, 'k0.png')   # no caption yet
        ds_id = ds.id
        img_id = img.id

    monkeypatch.setattr(vision_ollama, 'describe_image_ollama',
                        lambda *a, **k: 'a woman standing in a park')
    monkeypatch.setattr(vision_ollama, 'generate_text_ollama',
                        lambda *a, **k: 'a woman in a park')
    monkeypatch.setattr(vision_ollama, 'unload_vision_model', lambda *a, **k: True)

    r = client.post(f'/api/dataset/{ds_id}/caption', json={'force': True})
    assert r.status_code == 200

    with app.app_context():
        row = db.session.get(FaceDatasetImage, img_id)
        assert (row.caption or '').strip()          # long written
        assert (row.caption_short or '').strip()     # short derived in the same pass


# --- cloud strips dual back to the historical shape --------------------------
def test_cloudify_strips_dual_captions(app):
    from app.services import cloud_training as ct
    with app.app_context():
        staging = 'C:/stage/dataset'
        job_config = {'job': 'extension', 'config': {'name': 'x', 'process': [{
            'type': 'sd_trainer',
            'datasets': [{'folder_path': staging + '/_captions.json', 'caption_ext': 'txt'}],
            'train': {'steps': 100, 'short_and_long_captions': True},
            'model': {},
        }]}}
        pod_settings = {'DATASETS_FOLDER': '/workspace/datasets',
                        'TRAINING_FOLDER': '/workspace/out'}
        out = ct._cloudify_job_config(job_config, 'myjob', staging, pod_settings)
        proc = out['config']['process'][0]
        # Reverted to the historical folder + .txt sidecars, dual flag dropped.
        assert proc['datasets'][0]['folder_path'] == '/workspace/datasets/myjob'
        assert proc['datasets'][0]['caption_ext'] == 'txt'
        assert 'short_and_long_captions' not in proc['train']


# --- dual captions vs families that cache their text embeddings (issue #22) ----
#
# Reported by 1Tomber: a krea run with dual captions ON died at the first step with
#     AttributeError: 'NoneType' object has no attribute 'replace'
#       ... in inject_trigger_into_prompt
# Reproduced against the installed ai-toolkit: its text-embedding caching pass reaches
# load_caption() WITHOUT the JSON caption dict (get_text_embedding_info_dict), so every
# item is filled from its .txt sidecar and `raw_caption_short` stays None; load_caption()
# then short-circuits on the real per-batch call, `caption_short` is never computed, and
# the doubled prompt list handed to inject_trigger_into_prompt contains None. Caching the
# embeddings alone is fine; dual captions alone is fine; the PAIR is what breaks — and even
# without the crash the short caption could never be encoded, because cache_text_embeddings
# encodes file_item.caption only and unload_text_encoder then removes the encoder.
_CACHING_FAMILIES = ('krea', 'anima')
_ALL_FAMILIES = ('zimage', 'sdxl', 'flux', 'flux2klein', 'krea', 'anima')


def _dual_dataset(tmp_path, family, name='Emma', monkeypatch=None):
    """A one-image dual-captions dataset of `family`, exported. Returns (ds, export_dir)."""
    ds = svc.create_dataset(LOCAL_USER, name, 'zchar_emma')
    ds.train_type = family
    if family == 'sdxl':
        # SDXL refuses to build without a checkpoint, and resolving one needs ComfyUI.
        ds.train_base_model = 'base.safetensors'
        monkeypatch.setattr(lt, '_sdxl_base_path', lambda b: f'checkpoints/{b}')
    _enable_dual(ds)
    _kept_image(ds, 'a.png', caption='a woman on a bench', caption_short='a woman')
    out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False,
                                         dest_dir=str(tmp_path / f'export_{family}'))
    return ds, out


def test_krea_dual_captions_never_emits_the_crashing_pair(app, tmp_path):
    """THE reported scenario: krea + dual captions must not produce the #22 config."""
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds, out = _dual_dataset(tmp_path, 'krea')
        proc = lt.build_job_config(ds, out, steps=100)['config']['process'][0]
        d0 = proc['datasets'][0]
        # The krea recipe is unchanged — it still caches embeddings and unloads the TE.
        assert d0['cache_text_embeddings'] is True
        assert proc['train']['unload_text_encoder'] is True
        # ...so the dual-caption pair is refused instead of emitted.
        assert 'short_and_long_captions' not in proc['train']
        assert d0['folder_path'] == out          # folder + .txt sidecars, not the JSON


def test_no_family_emits_dual_with_cached_text_embeddings(app, tmp_path, monkeypatch):
    """Family sweep: the pair must never appear, whatever the family."""
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        for fam in _ALL_FAMILIES:
            ds, out = _dual_dataset(tmp_path, fam, name=f'Emma{fam}', monkeypatch=monkeypatch)
            proc = lt.build_job_config(ds, out, steps=100)['config']['process'][0]
            cached = any(d.get('cache_text_embeddings') for d in proc['datasets'])
            unloads = bool(proc['train'].get('unload_text_encoder'))
            dual = bool(proc['train'].get('short_and_long_captions'))
            assert not (dual and (cached or unloads)), f'{fam} emits the #22 pair'
            # And the refusal is not over-broad: a family that can train it, does.
            assert dual is not (cached or unloads), f'{fam} dual={dual} cached={cached}'


def test_unsupported_family_list_matches_what_the_recipes_emit(app, tmp_path, monkeypatch):
    """The preflight warns from a constant; keep it honest against the real recipes."""
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        measured = []
        for fam in _ALL_FAMILIES:
            ds, out = _dual_dataset(tmp_path, fam, name=f'Emma{fam}', monkeypatch=monkeypatch)
            proc = lt.build_job_config(ds, out, steps=100)['config']['process'][0]
            if lt._dual_captions_unsupported_reason(proc):
                measured.append(fam)
        assert tuple(measured) == _CACHING_FAMILIES
        assert lt.DUAL_CAPTION_UNSUPPORTED_FAMILIES == _CACHING_FAMILIES


def test_krea_dual_still_trains_with_its_trigger(app, tmp_path):
    """Anti-masking guard: refusing dual must not cost the trigger.

    A `or ''` placeholder short caption would have made the crash go away while training
    a LoRA whose keyword is never injected — invisible until the weights are useless."""
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds, out = _dual_dataset(tmp_path, 'krea')
        proc = lt.build_job_config(ds, out, steps=100)['config']['process'][0]
        assert proc['trigger_word'] == 'zchar_emma'
        # The captions ai-toolkit will actually read (the .txt sidecars) carry the trigger.
        sidecar = os.path.join(out, 'zchar_emma_000.txt')
        assert open(sidecar, encoding='utf-8').read().startswith('zchar_emma,')
        # No empty/placeholder caption anywhere in the emitted config.
        assert proc['datasets'][0].get('default_caption') is None


def test_dual_captions_untouched_on_a_non_caching_family(app, tmp_path):
    """Anti-regression: dual captions alone still work exactly as before."""
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds, out = _dual_dataset(tmp_path, 'zimage')
        proc = lt.build_job_config(ds, out, steps=100)['config']['process'][0]
        assert proc['train']['short_and_long_captions'] is True
        assert proc['datasets'][0]['folder_path'] == lt._dual_caption_json_path(out)
        assert not proc['datasets'][0].get('cache_text_embeddings')


def test_text_embedding_cache_untouched_without_dual(app, tmp_path):
    """Anti-regression: embedding caching alone still works exactly as before."""
    with app.app_context():
        save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        ds.train_type = 'krea'
        db.session.commit()          # dual OFF
        _kept_image(ds, 'a.png', caption='a woman on a bench')
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False,
                                             dest_dir=str(tmp_path / 'export_plain'))
        proc = lt.build_job_config(ds, out, steps=100)['config']['process'][0]
        assert proc['datasets'][0]['cache_text_embeddings'] is True
        assert proc['train']['unload_text_encoder'] is True
        assert 'short_and_long_captions' not in proc['train']


def test_preflight_says_dual_captions_are_ignored_on_a_caching_family(app):
    """The refusal is announced BEFORE the launch, not discovered at the first step."""
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        _enable_dual(ds)
        for i in range(20):
            _kept_image(ds, f'k{i}.png', caption=f'a woman on a bench number {i} outdoors')

        rep = lt.training_preflight(LOCAL_USER, ds.id, train_type='krea')
        row = next(c for c in rep['checks'] if c['id'] == 'dual_captions')
        assert row['status'] == 'warn'
        assert 'short caption is ignored' in row['detail']
        assert any('Dual captions are ON' in w for w in rep['warnings'])
        assert not rep['blockers']       # a warning, not a wall: the run is still valid

        # A family that can train them says so instead.
        rep = lt.training_preflight(LOCAL_USER, ds.id, train_type='zimage')
        row = next(c for c in rep['checks'] if c['id'] == 'dual_captions')
        assert row['status'] == 'ok'
        assert not any('Dual captions are ON' in w for w in rep['warnings'])


def test_preflight_dual_row_stays_out_of_slider_mode(app):
    """The slider loss ignores captions; a row promising two wordings would contradict
    the 'captions are ignored by the slider loss' row already emitted above it."""
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        _enable_dual(ds)
        ds.train_slider = json.dumps({'enabled': True, 'positive': 'more detail',
                                      'negative': 'less detail', 'target_class': 'woman'})
        db.session.commit()
        for i in range(20):
            _kept_image(ds, f'k{i}.png', caption=f'a woman on a bench number {i} outdoors')
        rep = lt.training_preflight(LOCAL_USER, ds.id, train_type='zimage')
        assert not any(c['id'] == 'dual_captions' for c in rep['checks'])


def test_launch_snapshot_stamps_dual_captions_effective(app):
    """Provenance must record what the trainer SAW, not what the toggle said.

    Otherwise two krea runs that differ only by this toggle look like a dual-caption
    experiment in the run comparison, while ai-toolkit read the same captions both times."""
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        _enable_dual(ds)
        assert lt.launch_settings_snapshot(ds)['dual_captions'] is True    # zimage
        ds.train_type = 'krea'
        db.session.commit()
        assert lt.launch_settings_snapshot(ds)['dual_captions'] is False
        # The stored preference itself is untouched — the toggle still shows what was set.
        assert lt.effective_train_settings(ds)['dual_captions'] is True


def test_preflight_stays_silent_when_dual_captions_are_off(app):
    """No new row on the historical path — the readiness pill gains nothing to read."""
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Emma', 'zchar_emma')
        for i in range(20):
            _kept_image(ds, f'k{i}.png', caption=f'a woman on a bench number {i} outdoors')
        rep = lt.training_preflight(LOCAL_USER, ds.id, train_type='krea')
        assert not any(c['id'] == 'dual_captions' for c in rep['checks'])
