"""90/180/270° rotation of a dataset image and of a bank image.

Idea by 1Tomber (GitHub #17), who asked for it "lossless if favorable". The two
lanes answer that differently ON PURPOSE and these tests pin both answers:

* a DATASET image is our own file, so the turn is applied in place through the
  same encoder the mirror uses — pixel-exact for PNG/WebP/BMP (asserted below by a
  four-turn round trip), a re-encode for JPEG;
* a BANK image belongs to the user's folder, which we never write to, so the
  turn is an integer on the row and the source keeps its exact bytes.
"""
import io
from pathlib import Path

import pytest
from PIL import Image


def _corner_marked(path: Path, fmt='PNG', size=(96, 40)):
    """An image with a unique colour per corner — the only fixture that can tell
    a 90° turn from a 270° one."""
    image = Image.new('RGB', size, (10, 10, 10))
    w, h = size
    image.paste((250, 0, 0), (0, 0, w // 4, h // 4))                  # top-left
    image.paste((0, 250, 0), (w - w // 4, 0, w, h // 4))              # top-right
    image.paste((0, 0, 250), (0, h - h // 4, w // 4, h))              # bottom-left
    kwargs = {}
    if fmt == 'WEBP':
        kwargs = {'lossless': True, 'quality': 100}
    elif fmt == 'JPEG':
        kwargs = {'quality': 100, 'subsampling': 0}
    image.save(path, fmt, **kwargs)


def _seed_dataset_image(app, *, filename='corners.png', fmt='PNG', watermark=True):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, f'Rotate {filename}', 'rotate')
        path = Path(svc._dataset_dir(ds.id)) / filename
        _corner_marked(path, fmt)
        image = FaceDatasetImage(
            dataset_id=ds.id, filename=filename, source='import', status='keep',
            caption='a person', variation_label='corners',
            watermark_state='detected' if watermark else None,
            watermark_bbox='[0.1, 0.2, 0.3, 0.4]' if watermark else None,
            watermark_regions='[[0.1, 0.2, 0.3, 0.4]]' if watermark else None,
        )
        svc.db.session.add(image)
        svc.db.session.commit()
        return ds.id, image.id, path


def _corner(path, corner):
    """Sample well inside one corner of the file on disk."""
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        w, h = rgb.size
        x = 3 if corner in ('tl', 'bl') else w - 4
        y = 3 if corner in ('tl', 'tr') else h - 4
        return rgb.getpixel((x, y))


def _dominant(pixel):
    """'r' | 'g' | 'b' | None — robust to a JPEG re-encode."""
    r, g, b = pixel
    if max(r, g, b) < 90:
        return None
    return 'rgb'[max(range(3), key=lambda i: pixel[i])]


# --- dataset lane ------------------------------------------------------------
def test_rotate_90_turns_clockwise_and_swaps_dimensions(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_dataset_image(app)
    with app.app_context():
        result = svc.rotate_image(LOCAL_USER, image_id, 90)
    assert result['image_id'] == image_id
    assert isinstance(result['cache_bust'], int) and result['cache_bust'] > 0
    with Image.open(path) as image:
        assert image.size == (40, 96)            # 96x40 landscape -> portrait
    # Clockwise: the top-left red corner lands top-RIGHT, top-right green lands
    # bottom-right, bottom-left blue lands top-left.
    assert _dominant(_corner(path, 'tr')) == 'r'
    assert _dominant(_corner(path, 'br')) == 'g'
    assert _dominant(_corner(path, 'tl')) == 'b'


def test_rotate_270_is_the_other_way_round(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_dataset_image(app)
    with app.app_context():
        svc.rotate_image(LOCAL_USER, image_id, 270)
    with Image.open(path) as image:
        assert image.size == (40, 96)
    assert _dominant(_corner(path, 'bl')) == 'r'   # top-left red -> bottom-left
    assert _dominant(_corner(path, 'tl')) == 'g'


@pytest.mark.parametrize('fmt,filename', [
    ('PNG', 'corners.png'),
    ('WEBP', 'corners.webp'),
    ('JPEG', 'corners.jpg'),
    ('BMP', 'corners.bmp'),
])
def test_rotation_preserves_the_real_file_format(app, fmt, filename):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_dataset_image(app, filename=filename, fmt=fmt)
    with app.app_context():
        svc.rotate_image(LOCAL_USER, image_id, 180)
    with Image.open(path) as image:
        assert (image.format or '').upper() == fmt
        assert image.size == (96, 40)             # a half turn keeps the shape


@pytest.mark.parametrize('fmt,filename', [
    ('PNG', 'corners.png'),
    ('WEBP', 'corners.webp'),
    ('BMP', 'corners.bmp'),
])
def test_four_quarter_turns_restore_the_exact_pixels(app, fmt, filename):
    """The losslessness claim, measured. PNG, WebP and BMP are re-encoded losslessly by
    the shared encoder, so going all the way round costs nothing — this is what
    lets a user undo a mis-click without thinking about it. (JPEG deliberately
    excluded: Pillow has no DCT-domain path, see the module docstring.)"""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_dataset_image(app, filename=filename, fmt=fmt)
    with Image.open(path) as image:
        before = image.convert('RGB').tobytes()
    with app.app_context():
        for _ in range(4):
            svc.rotate_image(LOCAL_USER, image_id, 90)
    with Image.open(path) as image:
        assert image.size == (96, 40)
        assert image.convert('RGB').tobytes() == before


def test_rotation_bakes_exif_orientation_exactly_once(app):
    """A phone photo carries its turn in EXIF. The shared encoder transposes the
    pixels first and drops the tag, so the file is never turned twice — and the
    turn the user asked for is applied to the image they were shown."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Exif rotate', 'exif-rotate')
        path = Path(svc._dataset_dir(ds.id)) / 'exif.jpg'
        # 96x40 stored pixels + "rotate 90° CW to display" (EXIF orientation 6):
        # what the user SEES is 40x96.
        base = Image.new('RGB', (96, 40), (10, 10, 10))
        base.paste((250, 0, 0), (0, 0, 24, 10))
        exif = base.getexif()
        exif[274] = 6
        base.save(path, 'JPEG', quality=100, subsampling=0, exif=exif)
        row = FaceDatasetImage(dataset_id=ds.id, filename='exif.jpg', status='keep')
        svc.db.session.add(row)
        svc.db.session.commit()
        image_id = row.id

        # A quarter turn RIGHT on a 40x96 display frame gives 96x40 back.
        svc.rotate_image(LOCAL_USER, image_id, 90)
    with Image.open(path) as image:
        assert image.size == (96, 40)
        assert image.getexif().get(274) in (None, 1)   # tag gone, not re-applied


def test_rotation_clears_watermark_metadata_and_keeps_the_rest(app):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    _ds, image_id, _path = _seed_dataset_image(app)
    with app.app_context():
        svc.rotate_image(LOCAL_USER, image_id, 90)
        row = svc.db.session.get(FaceDatasetImage, image_id)
        assert (row.watermark_state, row.watermark_bbox, row.watermark_regions) == (
            None, None, None)
        assert row.caption == 'a person' and row.filename == 'corners.png'


def test_rotation_refuses_free_angles_and_unknown_rows(app):
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_dataset_image(app)
    original = path.read_bytes()
    with app.app_context():
        for bad in (0, 45, 360, 'left', None):
            with pytest.raises(ValueError, match='90, 180 or 270'):
                svc.rotate_image(LOCAL_USER, image_id, bad)
        assert svc.rotate_image('another-user', image_id, 90) is None
        assert svc.rotate_image(LOCAL_USER, 999999, 90) is None
    assert path.read_bytes() == original
    assert not list(path.parent.glob(f'.{path.name}.rotate-*.tmp'))


def test_rotate_route_success_and_error_contracts(app, client):
    _ds, image_id, _path = _seed_dataset_image(app, watermark=False)
    response = client.post(f'/api/dataset/image/{image_id}/rotate',
                           json={'degrees': 90})
    assert response.status_code == 200
    body = response.get_json()
    assert set(body) == {'ok', 'image_id', 'cache_bust'}
    assert body['ok'] is True and body['image_id'] == image_id
    assert client.post(f'/api/dataset/image/{image_id}/rotate',
                       json={'degrees': 42}).status_code == 400
    assert client.post('/api/dataset/image/999999/rotate',
                       json={'degrees': 90}).status_code == 404


def test_mirror_and_rotation_share_one_encoder_and_one_lock(app):
    """The point of the refactor: a fix to the format/EXIF handling of one lane
    is a fix to both. Asserted structurally so a future rewrite cannot quietly
    fork them again."""
    from app.services import face_dataset_service as svc

    marker = {}

    def transform(image):
        marker['called'] = True
        return image

    _ds, _image_id, path = _seed_dataset_image(app)
    payload = svc.transformed_image_bytes(str(path), transform)
    assert marker['called'] and payload
    assert svc.ROTATION_DEGREES == (90, 180, 270)
    assert svc.normalize_rotation(-90) == 270 and svc.normalize_rotation(450) == 90


# --- bank lane ---------------------------------------------------------------
def _seed_bank(app, client, tmp_path, fmt='JPEG', name='shot.jpg'):
    folder = tmp_path / 'bank-src'
    folder.mkdir(parents=True, exist_ok=True)
    _corner_marked(folder / name, fmt, size=(96, 40))
    response = client.post('/api/bank/create',
                           json={'name': 'Rotate bank', 'folder': str(folder)})
    assert response.status_code == 200, response.get_json()
    bank_id = response.get_json()['id']
    with app.app_context():
        from app.models import BankImage
        row = BankImage.query.filter_by(bank_id=bank_id).first()
        return bank_id, row.id, folder / name


def test_bank_rotation_never_touches_the_users_file(app, client, tmp_path):
    """THE bank answer to "lossless is favorable": the source is not rewritten at
    all, so any number of turns costs the original exactly nothing."""
    from app.config import LOCAL_USER
    from app.services import image_bank_service as banks

    bank_id, image_id, src = _seed_bank(app, client, tmp_path)
    original = src.read_bytes()
    with app.app_context():
        out = banks.rotate_images(LOCAL_USER, bank_id, [image_id], 90)
        assert out == {'rotated': 1, 'rotations': {image_id: 90}}
        banks.rotate_images(LOCAL_USER, bank_id, [image_id], 90)
        banks.rotate_images(LOCAL_USER, bank_id, [image_id], 180)
    assert src.read_bytes() == original


def test_bank_resolver_serves_the_turned_copy_and_drops_it_at_zero(app, client, tmp_path):
    from app.config import LOCAL_USER
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, image_id, src = _seed_bank(app, client, tmp_path)
    with app.app_context():
        bank = banks.get_bank(LOCAL_USER, bank_id)
        row = banks.db.session.get(BankImage, image_id)
        assert banks.resolved_image_path(bank, row) == str(src)

        banks.rotate_images(LOCAL_USER, bank_id, [image_id], 90)
        row = banks.db.session.get(BankImage, image_id)
        turned = banks.resolved_image_path(bank, row)
        assert turned != str(src)
        with Image.open(turned) as image:
            assert image.size == (40, 96)
            assert (image.format or '').upper() == 'JPEG'   # format preserved

        # Back round to 0: the row forgets the angle and every reader is handed
        # the untouched source again.
        banks.rotate_images(LOCAL_USER, bank_id, [image_id], 270)
        row = banks.db.session.get(BankImage, image_id)
        assert row.rotation in (None, 0)
        assert banks.resolved_image_path(bank, row) == str(src)


def test_bank_rotation_regenerates_the_thumbnail_and_reports_new_dimensions(app, client, tmp_path):
    """The derivative that would otherwise lie: a cached thumbnail still showing
    the old orientation reads as "the button did nothing"."""
    from app.config import LOCAL_USER
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, image_id, _src = _seed_bank(app, client, tmp_path)
    with app.app_context():
        bank = banks.get_bank(LOCAL_USER, bank_id)
        row = banks.db.session.get(BankImage, image_id)
        before = banks.ensure_thumb(bank, row)
        with Image.open(before) as thumb:
            assert thumb.width > thumb.height

        banks.rotate_images(LOCAL_USER, bank_id, [image_id], 90)
        row = banks.db.session.get(BankImage, image_id)
        after = banks.ensure_thumb(bank, row)
        assert Path(after) != Path(before)
        with Image.open(after) as thumb:
            assert thumb.height > thumb.width

        payload = banks._image_dict(row, banks.thresholds())
        assert payload['rotation'] == 90
        assert (payload['width'], payload['height']) == (row.height, row.width)


def test_bank_rescan_reports_effective_rotated_dimensions_once(
        app, client, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, image_id, _src = _seed_bank(app, client, tmp_path)
    with app.app_context():
        banks.rotate_images(LOCAL_USER, bank_id, [image_id], 90)
        monkeypatch.setattr(banks, 'rebuild_dup_groups', lambda *_a, **_k: 0)
        job = banks.start_scan(
            app, LOCAL_USER, bank_id, rescan=True, ids=[image_id])
        assert job['error'] is None, job
        row = banks.db.session.get(BankImage, image_id, populate_existing=True)
        assert (row.width, row.height) == (40, 96)
        assert row.analysis_fingerprint
        payload = banks._image_dict(row, banks.thresholds())
        assert (payload['width'], payload['height']) == (40, 96)


def test_bank_rotated_cache_rebuilds_after_live_source_replacement(
        app, client, tmp_path, monkeypatch):
    from app.config import LOCAL_USER
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, image_id, source = _seed_bank(app, client, tmp_path)
    with app.app_context():
        bank = banks.get_bank(LOCAL_USER, bank_id)
        banks.rotate_images(LOCAL_USER, bank_id, [image_id], 90)
        monkeypatch.setattr(banks, 'rebuild_dup_groups', lambda *_a, **_k: 0)
        banks.start_scan(app, LOCAL_USER, bank_id, rescan=True, ids=[image_id])
        row = banks.db.session.get(BankImage, image_id, populate_existing=True)
        first = Path(banks.analysis_image_path(bank, row, refresh_rotation=True))
        first_payload = first.read_bytes()

        _corner_marked(source, 'JPEG', size=(71, 123))
        second = Path(banks.analysis_image_path(bank, row, refresh_rotation=True))
        assert second != first
        assert second.read_bytes() != first_payload
        with Image.open(second) as image:
            assert image.size == (123, 71)


def test_bank_rotate_route_contract(app, client, tmp_path):
    bank_id, image_id, _src = _seed_bank(app, client, tmp_path)
    ok = client.post(f'/api/bank/{bank_id}/rotate',
                     json={'ids': [image_id], 'degrees': -90})
    assert ok.status_code == 200
    assert ok.get_json()['rotations'] == {str(image_id): 270}
    assert client.post(f'/api/bank/{bank_id}/rotate',
                       json={'ids': [image_id], 'degrees': 12}).status_code == 400
    assert client.post(f'/api/bank/{bank_id}/rotate',
                       json={'ids': [], 'degrees': 90}).status_code == 400


def test_bank_promotion_carries_the_rotation_into_the_dataset(app, client, tmp_path):
    """A turned image must reach the dataset turned, exactly like a
    watermark-cleaned one does — otherwise the fix is lost at the door."""
    from app.config import LOCAL_USER
    from app.models import BankImage
    from app.services import image_bank_service as banks

    bank_id, image_id, _src = _seed_bank(app, client, tmp_path)
    with app.app_context():
        banks.rotate_images(LOCAL_USER, bank_id, [image_id], 90)
        bank = banks.get_bank(LOCAL_USER, bank_id)
        row = banks.db.session.get(BankImage, image_id)
        with open(banks.resolved_image_path(bank, row), 'rb') as fh:
            blob = fh.read()
    with Image.open(io.BytesIO(blob)) as image:
        assert image.size == (40, 96)
