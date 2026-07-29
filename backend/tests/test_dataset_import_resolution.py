"""Import resolution + encoding are a CHOICE, not a fate.

Every image entering a dataset used to be resampled to 1024 px on the long side
and re-encoded as WebP q92, with no setting anywhere and no sentence saying so
(reported by Qeeyana on Reddit: "why? let me choose not to").

What these tests pin:
  - the shipped default is STILL 1024 / q92 — nobody's existing datasets change
    meaning because a setting appeared;
  - a chosen size is applied, and `0` means "store what I gave you";
  - "no downscale" is still capped, because WebP itself refuses past 16383 px —
    an uncapped "original" would turn a big panorama into a failed import;
  - encoding tiers do what they claim (lossless really is pixel-identical);
  - the guarantee that survived from the old code: it only ever SHRINKS;
  - an unusable configured value degrades to the default instead of exploding;
  - both entry lanes (photo import and the kohya ZIP/folder merge) obey it.
"""
import io
import os

from PIL import Image


def _photo(w, h, seed=3):
    """Non-uniform image: a flat color shares its dHash with every other flat
    color and would be dropped as a perceptual duplicate."""
    im = Image.new('RGB', (w, h), (250, 250, 250))
    step = max(8, w // 16)
    for i in range(0, w, step):
        im.paste(((i * seed) % 255, (i * 7) % 255, (i * 31 + seed) % 255),
                 (i, (i * 3) % max(1, h - 8), min(i + step, w), h))
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return buf.getvalue()


def _noisy(w, h):
    """A real photo does not compress like flat blocks — the encoding tiers only
    show their true cost on noise."""
    import random
    random.seed(7)
    im = Image.new('RGB', (w, h))
    im.putdata([(random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255)) for _ in range(w * h)])
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return buf.getvalue()


def _stored_sizes(ds_id, svc):
    from app.models import FaceDatasetImage
    out = []
    for row in FaceDatasetImage.query.filter_by(dataset_id=ds_id).all():
        with Image.open(os.path.join(svc._dataset_dir(ds_id), row.filename)) as im:
            out.append(im.size)
    return out


def _import_one(svc, ds_id, raw):
    from app.config import LOCAL_USER
    ids, failed = svc.import_images(LOCAL_USER, ds_id, [raw], crop=False)
    assert failed == 0 and len(ids) == 1
    return _stored_sizes(ds_id, svc)[0]


# --- the default nobody asked to change -------------------------------------

def test_default_import_stays_1024_q92(app):
    """No config → exactly the behaviour every existing install already has."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        policy = svc.import_encode_policy()
        assert policy['max_side'] == 1024
        assert policy['encoding'] == 'standard'
        assert policy['quality'] == 92
        assert policy['lossless'] is False
        ds = svc.create_dataset(LOCAL_USER, 'Def', 'def')
        assert _import_one(svc, ds.id, _photo(3000, 2000)) == (1024, 683)


def test_import_never_upscales(app):
    """`thumbnail` only shrank; a chosen bigger size must not start enlarging."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER, save_config
    with app.app_context():
        save_config({'dataset_import': {'max_side': 2048}})
        ds = svc.create_dataset(LOCAL_USER, 'Small', 'small')
        assert _import_one(svc, ds.id, _photo(512, 384)) == (512, 384)


# --- one test per value of the setting --------------------------------------

def test_chosen_max_side_is_applied(app):
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER, save_config
    with app.app_context():
        save_config({'dataset_import': {'max_side': 2048}})
        assert svc.import_encode_policy()['max_side'] == 2048
        ds = svc.create_dataset(LOCAL_USER, 'Big', 'big')
        assert _import_one(svc, ds.id, _photo(3000, 2000)) == (2048, 1365)


def test_zero_means_no_downscale_at_all(app):
    """'Let me choose not to': the stored pixels are the ones handed in."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER, save_config
    with app.app_context():
        save_config({'dataset_import': {'max_side': 0}})
        assert svc.import_encode_policy()['max_side'] == 0
        ds = svc.create_dataset(LOCAL_USER, 'Orig', 'orig')
        assert _import_one(svc, ds.id, _photo(3000, 2000)) == (3000, 2000)


def test_no_downscale_still_hits_the_hard_ceiling(app):
    """MEASURED: Pillow refuses `WEBP` past 16383 px ("Image size exceeds WebP
    limit of 16383 pixels"), so an uncapped 'original' would FAIL the import of
    a big panorama instead of storing it. The ceiling is announced, not silent."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER, save_config
    with app.app_context():
        assert svc.IMPORT_MAX_SIDE_CEILING <= 16383
        save_config({'dataset_import': {'max_side': 0}})
        ds = svc.create_dataset(LOCAL_USER, 'Huge', 'huge')
        w, h = _import_one(svc, ds.id, _photo(svc.IMPORT_MAX_SIDE_CEILING + 1200, 900))
        assert w == svc.IMPORT_MAX_SIDE_CEILING
        # a chosen size above the ceiling is clamped, and says so
        save_config({'dataset_import': {'max_side': 999999}})
        pol = svc.import_encode_policy()
        assert pol['max_side'] == svc.IMPORT_MAX_SIDE_CEILING and pol['capped'] is True


def test_encoding_tiers(app):
    """The other half of the loss: q92 is not the only option, and 'lossless'
    really is pixel-identical (not merely 'quality=100')."""
    from app.services import face_dataset_service as svc
    from app.config import save_config
    raw = _noisy(400, 300)
    with app.app_context():
        save_config({'dataset_import': {'max_side': 0, 'encoding': 'lossless'}})
        pol = svc.import_encode_policy()
        assert pol['lossless'] is True
        lossless = svc.import_encode(raw)
        save_config({'dataset_import': {'max_side': 0, 'encoding': 'standard'}})
        standard = svc.import_encode(raw)
    src = Image.open(io.BytesIO(raw)).convert('RGB')
    assert list(Image.open(io.BytesIO(lossless)).convert('RGB').getdata()) == list(src.getdata())
    assert list(Image.open(io.BytesIO(standard)).convert('RGB').getdata()) != list(src.getdata())
    # MEASURED on an 800x600 noisy frame: q92 158 KB, q100 243 KB, lossless
    # 797 KB — the ~5x the setting's help text warns about.
    assert len(lossless) > 2 * len(standard)


def test_unusable_values_fall_back_to_the_default(app):
    from app.services import face_dataset_service as svc
    from app.config import save_config
    with app.app_context():
        save_config({'dataset_import': {'max_side': 'wide', 'encoding': 'ultra'}})
        pol = svc.import_encode_policy()
        assert pol['max_side'] == 1024 and pol['encoding'] == 'standard'
        save_config({'dataset_import': {'max_side': -50}})
        assert svc.import_encode_policy()['max_side'] == 1024


# --- the other entry lane ----------------------------------------------------

def test_zip_merge_lane_obeys_the_same_policy(app):
    """Importing an existing training set went through the same hardcoded 1024."""
    import zipfile
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER, save_config
    with app.app_context():
        save_config({'dataset_import': {'max_side': 0}})
        ds = svc.create_dataset(LOCAL_USER, 'ZipRes', 'zipres')
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w') as z:
            z.writestr('10_x/a.png', _photo(2400, 1600, seed=5))
        ids, failed = svc.import_dataset_zip(LOCAL_USER, ds.id, buf.getvalue())
        assert len(ids) == 1 and failed == 0
        assert _stored_sizes(ds.id, svc)[0] == (2400, 1600)


# --- the setting is reachable from the UI ------------------------------------

def test_settings_api_round_trips_the_new_section(client):
    r = client.get('/api/settings')
    assert r.status_code == 200
    body = r.get_json()
    assert body['config']['dataset_import']['max_side'] == 1024
    assert body['config_defaults']['dataset_import']['encoding'] == 'standard'
    assert client.put('/api/settings', json={
        'config': {'dataset_import': {'max_side': 2048, 'encoding': 'high'}}}).status_code == 200
    body = client.get('/api/settings').get_json()
    assert body['config']['dataset_import'] == {'max_side': 2048, 'encoding': 'high'}


def test_capabilities_publish_the_effective_policy(client):
    """The workspace quotes the number at the point of import — it must be able
    to read it without inventing its own copy of the default."""
    r = client.get('/api/capabilities')
    assert r.status_code == 200
    di = r.get_json()['dataset_import']
    assert di['max_side'] == 1024 and di['encoding'] == 'standard'
    assert di['ceiling'] >= 1024
