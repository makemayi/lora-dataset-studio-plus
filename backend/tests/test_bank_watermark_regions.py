"""🚩 Bank watermark MASK editing — the hand-drawn correction zones.

Reported by Qeeyana (Reddit): watermark masking could only be edited from a
dataset, never from a bank. The Bank had detection, dismissal, crop, inpaint and
undo — everything EXCEPT the one thing that makes a wrong box recoverable.

The contract this file defends is not "there is an endpoint". It is:

  1. an edited mask is PERSISTED and RE-READ by the cleaning pass (an edit the
     cleaner ignores is worse than no edit at all — the user believes the fix
     landed and ships a watermarked LoRA);
  2. the Bank and the Dataset hand the SAME boxes to the same engine for the
     same mask (two mask editors that drift is the debt this project already
     pays elsewhere);
  3. an EMPTIED mask cleans nothing, on purpose, and says so — it never falls
     back to the detector's box behind the user's back.
"""
import io
import json
import os

from PIL import Image


def _photo(size=1000, value=90):
    im = Image.new('RGB', (size, size), (value, value, value))
    for y in range(0, size, 40):
        for x in range(size):
            im.putpixel((x, y), (250, 250, 250))
    return im


def _mkbank(client, tmp_path, files, name='WM'):
    src = tmp_path / 'src'
    for rel, im in files.items():
        p = src / rel
        os.makedirs(p.parent, exist_ok=True)
        im.save(str(p), 'JPEG', quality=92)
    r = client.post('/api/bank/create', json={'name': name, 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _flag(app, bank_id, bbox, *, index=0, state='detected'):
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        row = rows[index]
        row.watermark_state = state
        row.watermark_bbox = json.dumps(bbox) if bbox is not None else None
        db.session.commit()
        return row.id


def _row(app, image_id):
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        row = db.session.get(BankImage, image_id)
        return (row.watermark_state, row.watermark_clean_method)


def _regions(app, image_id):
    from app.extensions import db
    from app.models import BankImage
    with app.app_context():
        row = db.session.get(BankImage, image_id)
        return row.watermark_regions


TOP_MARK = [0.1, 0.01, 0.4, 0.05]           # croppable border band
OFFCENTER_MARK = [0.30, 0.30, 0.36, 0.36]   # LaMa's lane
# What a user draws when the detector's box missed the mark: two precise zones,
# one of them right on the subject (only a hand mask can express this).
EDITED = [[0.05, 0.02, 0.2, 0.08], [0.44, 0.46, 0.52, 0.53]]


def _capture_lama(monkeypatch, *, available=True, ok=True):
    """Stand in for the LaMa subprocess and RECORD the boxes it was handed.

    Both call shapes are captured: the bank always batches, the dataset uses the
    single-image entry point when only one row is pending."""
    from app.services import watermark_lama
    monkeypatch.setattr(watermark_lama, 'is_available', lambda: available)
    monkeypatch.setattr(watermark_lama, 'resolve_device', lambda: 'cpu')
    seen = []

    def _paint(path):
        with Image.open(path) as im:
            im.convert('RGB').resize((im.width, im.height // 2)).save(
                path, 'WEBP', quality=92)

    def fake_batch(jobs, *, device, timeout=900):
        out = {}
        for job in jobs:
            seen.append([[round(float(v), 4) for v in b] for b in job['bboxes']])
            if ok:
                _paint(job['image_path'])
                out[job['image_path']] = (True, None)
            else:
                out[job['image_path']] = (False, {'kind': 'failed', 'detail': 'boom'})
        return out

    def fake_multi(path, bboxes, **kw):
        seen.append([[round(float(v), 4) for v in b] for b in bboxes])
        if not ok:
            return False, {'kind': 'failed', 'detail': 'boom'}
        _paint(path)
        return True, None

    def fake_single(path, bbox, **kw):
        return fake_multi(path, [bbox], **kw)

    monkeypatch.setattr(watermark_lama, 'inpaint_batch', fake_batch)
    monkeypatch.setattr(watermark_lama, 'inpaint_watermarks', fake_multi)
    monkeypatch.setattr(watermark_lama, 'inpaint_watermark', fake_single)
    return seen


# --- 1. persisted, and RE-READ by the cleaning pass --------------------------
def test_edited_mask_is_persisted_and_used_by_the_repaint_pass(
        client, tmp_path, app, monkeypatch):
    """The whole point: 🧽 Repaint must act on the zones the user drew, not on
    the detector's box. Captured at the engine boundary."""
    bank_id, src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)

    r = client.put(f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
                   json={'regions': EDITED})
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert body['watermark_regions'] == EDITED
    assert body['effective_watermark_regions'] == EDITED
    assert json.loads(_regions(app, image_id)) == EDITED

    seen = _capture_lama(monkeypatch)
    before = open(src / 'a.jpg', 'rb').read()
    r = client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={'method': 'lama'})
    assert r.status_code == 202, r.get_json()

    assert seen == [EDITED]                      # the EDITED mask, not the bbox
    assert _row(app, image_id) == ('cleaned', 'lama')
    assert open(src / 'a.jpg', 'rb').read() == before   # source still untouched


def test_edited_mask_survives_a_reload(client, tmp_path, app):
    """The editor has to reopen on what was saved — the grid payload carries it."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    client.put(f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
               json={'regions': EDITED})

    img = next(i for i in client.get(f'/api/bank/{bank_id}/images')
               .get_json()['images'] if i['id'] == image_id)
    assert img['watermark_bbox'] == OFFCENTER_MARK
    assert img['watermark_regions'] == EDITED
    assert img['effective_watermark_regions'] == EDITED


def test_an_untouched_image_reopens_on_the_detected_box(client, tmp_path, app):
    """No manual override → the editor shows the detector's rectangle, and the
    stored override stays null (so 'Reset detection' has something to mean)."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    img = next(i for i in client.get(f'/api/bank/{bank_id}/images')
               .get_json()['images'] if i['id'] == image_id)
    assert img['watermark_regions'] is None
    assert img['effective_watermark_regions'] == [OFFCENTER_MARK]


def test_reset_detection_clears_the_override(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    client.put(f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
               json={'regions': EDITED})
    r = client.put(f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
                   json={'regions': None})
    assert r.status_code == 200
    assert r.get_json()['watermark_regions'] is None
    assert r.get_json()['effective_watermark_regions'] == [OFFCENTER_MARK]
    assert _regions(app, image_id) is None


# --- 2. the non-divergence contract -----------------------------------------
def test_bank_and_dataset_hand_the_same_boxes_to_the_same_engine(
        client, tmp_path, app, monkeypatch):
    """CONTRACT — same mask, same result. The Bank must not grow a second,
    slightly different interpretation of the correction zones."""
    from app.extensions import db
    from app.models import FaceDatasetImage

    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    client.put(f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
               json={'regions': EDITED})

    # The same photo, the same mask, in a dataset.
    with app.app_context():
        from app.services import face_dataset_service as svc
        dataset_id = svc.create_dataset('local', 'Same mask', 'msk').id
        buf = io.BytesIO()
        _photo().save(buf, 'JPEG', quality=92)
        ids, _dup = svc.import_images('local', dataset_id, [buf.getvalue()])
        ds_image_id = ids[0]
        row = db.session.get(FaceDatasetImage, ds_image_id)
        row.watermark_state = 'detected'
        row.watermark_bbox = json.dumps(OFFCENTER_MARK)
        db.session.commit()
    r = client.put(
        f'/api/dataset/{dataset_id}/image/{ds_image_id}/watermark-regions',
        json={'regions': EDITED})
    assert r.status_code == 200, r.get_json()

    bank_seen = _capture_lama(monkeypatch)
    client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={'method': 'lama'})
    dataset_seen = _capture_lama(monkeypatch)
    client.post(f'/api/dataset/{dataset_id}/watermarks/clean',
                json={'image_ids': [ds_image_id]})

    assert bank_seen == [EDITED]
    assert dataset_seen == bank_seen


def test_both_lanes_normalize_regions_through_the_one_validator():
    """The routing/engine code is shared by import; the VALIDATION must be too,
    or 'the same mask' stops meaning the same thing on the two sides."""
    from app.services import face_dataset_service as svc
    from app.services import image_bank_service as banks
    assert banks.normalize_watermark_regions is svc.normalize_watermark_regions


def test_an_out_of_range_or_oversized_mask_is_refused(client, tmp_path, app):
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    for bad in ([[0.5, 0.5, 0.4, 0.6]],          # x2 <= x1
                [[0.0, 0.0, 1.2, 0.5]],          # outside [0,1]
                [[0.1, 0.1, 0.1001, 0.5]],       # under the minimum side
                [[0.1, 0.1, 0.5]],               # not 4 numbers
                [[0.0, 0.0, 0.5, 0.5]] * 33):    # over the 32-zone limit
        r = client.put(f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
                       json={'regions': bad})
        assert r.status_code == 400, (bad, r.get_json())
    assert _regions(app, image_id) is None       # nothing half-stored


def test_a_cleaned_image_refuses_a_mask_edit(client, tmp_path, app):
    """Editing the mask of an already-cleaned image would silently do nothing —
    it is out of both levels' pool. Refuse loudly (409) instead."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, TOP_MARK)
    assert client.post(f'/api/bank/{bank_id}/watermark/crop').status_code == 202
    r = client.put(f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
                   json={'regions': EDITED})
    assert r.status_code == 409
    assert client.put(f'/api/bank/9999/image/{image_id}/watermark-regions',
                      json={'regions': EDITED}).status_code == 404


# --- 3. the emptied mask -----------------------------------------------------
def test_an_emptied_mask_cleans_nothing_and_never_falls_back_to_the_box(
        client, tmp_path, app, monkeypatch):
    """The user deleted every zone. That is an ANSWER ("there is nothing to
    repaint here"), not a missing value: neither level may quietly go back to
    the detector's rectangle. The image stays flagged so it is still visible."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, TOP_MARK)      # croppable → the tempting fallback
    r = client.put(f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
                   json={'regions': []})
    assert r.status_code == 200
    assert r.get_json()['watermark_regions'] == []
    assert r.get_json()['effective_watermark_regions'] == []

    seen = _capture_lama(monkeypatch)
    # ✂ Auto-crop has nothing left it may act on and SAYS so, instead of running
    # over the image and cropping the box the user just deleted.
    crop = client.post(f'/api/bank/{bank_id}/watermark/crop')
    assert crop.status_code == 400
    assert 'hand-edited mask' in crop.get_json()['error']
    assert client.post(f'/api/bank/{bank_id}/watermark/inpaint',
                       json={'method': 'lama'}).status_code == 202

    assert seen == []                             # no engine call at all
    assert _row(app, image_id) == ('detected', None)
    with app.app_context():
        from app.services import image_bank_service as banks
        assert not banks.clean_image_path(bank_id, image_id).exists()
    # ...and it is COUNTED as such rather than looking like an unhandled failure.
    levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
    assert levels['empty_masks'] == 1 and levels['croppable'] == 0


# --- the crop level defers to the hand mask ---------------------------------
def test_the_crop_level_leaves_a_hand_masked_image_to_the_repaint_level(
        client, tmp_path, app, monkeypatch):
    """A hand-drawn mask is a REPAINT instruction: it can hold several zones and
    zones on the subject, which a border crop cannot express. Same rule as the
    dataset, where manual zones always go to the inpainter."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, TOP_MARK)      # would be croppable on its own
    client.put(f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
               json={'regions': EDITED})

    levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
    assert (levels['croppable'], levels['inpaintable']) == (0, 1)
    assert levels['hand_masked'] == 1
    assert client.post(f'/api/bank/{bank_id}/watermark/crop').status_code == 400

    seen = _capture_lama(monkeypatch)
    assert client.post(f'/api/bank/{bank_id}/watermark/inpaint',
                       json={'method': 'lama'}).status_code == 202
    assert seen == [EDITED]
    assert _row(app, image_id) == ('cleaned', 'lama')


def test_the_mask_is_kept_after_a_clean_so_undo_gives_it_back(
        client, tmp_path, app, monkeypatch):
    """The Bank's undo is a first-class, advertised action (it only deletes our
    own blob). Dropping the zones on success would make undo hand back an image
    whose hand-drawn mask has to be redrawn from scratch."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, OFFCENTER_MARK)
    client.put(f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
               json={'regions': EDITED})
    _capture_lama(monkeypatch)
    client.post(f'/api/bank/{bank_id}/watermark/inpaint', json={'method': 'lama'})
    assert _row(app, image_id)[0] == 'cleaned'

    assert client.post(f'/api/bank/{bank_id}/watermark/undo', json={}).status_code == 200
    assert _row(app, image_id) == ('detected', None)
    assert json.loads(_regions(app, image_id)) == EDITED


def test_a_legacy_flag_with_no_box_becomes_cleanable_once_masked_by_hand(
        client, tmp_path, app, monkeypatch):
    """Rows flagged by an older build carry no box and no level can route them.
    Drawing the mask by hand is exactly the missing information — it must make
    them cleanable, and drop them from the 'needs a re-scan' tally."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.jpg': _photo()})
    image_id = _flag(app, bank_id, None)
    assert client.get(f'/api/bank/{bank_id}/watermark/levels'
                      ).get_json()['needs_rescan'] == 1

    client.put(f'/api/bank/{bank_id}/image/{image_id}/watermark-regions',
               json={'regions': EDITED})
    levels = client.get(f'/api/bank/{bank_id}/watermark/levels').get_json()
    assert levels['needs_rescan'] == 0 and levels['flagged'] == 1

    seen = _capture_lama(monkeypatch)
    assert client.post(f'/api/bank/{bank_id}/watermark/inpaint',
                       json={'method': 'lama'}).status_code == 202
    assert seen == [EDITED]
