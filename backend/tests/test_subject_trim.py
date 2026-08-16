"""Subject trim — the crop rule.

Pure by design: not one test here needs SAM3, ComfyUI, a Flask app or a file on
disk. The rules encoded below are the ones the 2026-08-16 wave got wrong, so
each test names the failure it prevents.
"""
import pytest

from app.services import subject_trim as st


def test_a_narrow_subject_is_widened_to_the_floor_and_no_further():
    # A standing person: 640x2547 in a 2160x3240 photo, measured on a real
    # Douyin photo. Keeping the subject's own 1:4 aspect is what produced the
    # 257x1024 sliver that got the first wave rejected.
    frame = st.trim_frame(2160, 3240, (922, 681, 640, 2547))
    x, y, w, h = frame
    assert h == pytest.approx(round(2547 * st.MARGIN), abs=1)
    assert w / h == pytest.approx(st.ASPECT_FLOOR, abs=0.005)
    # Widening takes background from BOTH sides — the subject stays centred.
    assert x < 922 and x + w > 922 + 640


def test_a_subject_already_wider_than_the_floor_gets_no_background():
    # A bust shot: wider than 9:16 already, so the floor must not touch it.
    frame = st.trim_frame(2000, 2000, (500, 400, 900, 1000))
    _x, _y, w, h = frame
    assert w == pytest.approx(round(900 * st.MARGIN), abs=1)
    assert h == pytest.approx(round(1000 * st.MARGIN), abs=1)


def test_the_frame_never_leaves_the_canvas():
    # Subject hard against the left edge; widening to the floor would push the
    # frame off-canvas if it were centred blindly.
    x, y, w, h = st.trim_frame(1000, 3000, (0, 0, 100, 2900))
    assert x >= 0 and y >= 0
    assert x + w <= 1000
    assert y + h <= 3000


def test_clamping_one_axis_never_shrinks_the_other():
    # THE bug of the first wave: `tight_frame` shrank the whole frame
    # proportionally when it overshot the canvas, so trimming excess width also
    # removed height — off a subject that exactly filled it. That is how people
    # and faces ended up cut in half.
    image_w, image_h = 800, 3000
    bbox = (100, 0, 600, 3000)          # subject spans the full height
    _x, _y, w, h = st.trim_frame(image_w, image_h, bbox)
    assert h == image_h, 'height must survive a width clamp intact'
    assert w == image_w


def test_the_floor_is_a_floor_not_a_target():
    # A wide subject keeps its wide frame; there is no upper ratio bound,
    # because a wide frame is simply a frame with no background added.
    _x, _y, w, h = st.trim_frame(4000, 2000, (100, 100, 3000, 800))
    assert w / h > 1.0


def test_skip_when_there_is_nothing_much_to_remove():
    # Measured subjectFrac on the test photos was 0.79-0.92: the first wave
    # cropped all of them for no gain, which read as "cropping at random".
    frame = (0, 0, 1900, 2900)
    assert st.skip_reason(2000, 3000, frame) == 'nothing-much-to-remove'


def test_skip_when_the_crop_would_be_too_small_to_use():
    # Replaces the first wave's "subject height under 15% of the image" guard,
    # which skipped the very case this feature exists for. A mis-detected speck
    # of mask cannot pass 256px; a small but sharp person can.
    frame = (0, 0, 100, 178)
    assert st.skip_reason(4000, 6000, frame) == 'subject-too-small'


def test_a_worthwhile_crop_has_no_skip_reason():
    assert st.skip_reason(2160, 3240, (400, 200, 1400, 2400)) is None


def test_nothing_much_to_remove_is_decided_before_too_small():
    # An image where both could fire: a tiny canvas whose subject fills it. The
    # honest reason is "nothing to remove", not "too small".
    assert st.skip_reason(200, 200, (0, 0, 190, 190)) == 'nothing-much-to-remove'


def test_output_size_caps_the_long_edge_and_never_upscales():
    assert st.output_size(1400, 2400) == (597, 1024)
    assert st.output_size(400, 600) == (400, 600), 'cropping in cannot create detail'


def test_an_empty_bbox_is_refused_rather_than_producing_a_zero_frame():
    with pytest.raises(ValueError):
        st.trim_frame(1000, 1000, (10, 10, 0, 500))


# --- the preview job -------------------------------------------------------
import json
import os

from PIL import Image

from app.services import subject_trim_batch as stb


class _FakeMask:
    """Stands in for auto_mask: a white rectangle at a known place."""

    def __init__(self, size, rect):
        self.size = size
        self.rect = rect

    def write(self, path):
        im = Image.new('L', self.size, 0)
        x, y, w, h = self.rect
        im.paste(255, (x, y, x + w, y + h))
        im.save(path)
        return path


def test_preview_writes_a_manifest_and_touches_no_pixels(app, tmp_path, monkeypatch):
    ds_dir = tmp_path / 'ds'
    ds_dir.mkdir()
    photo = ds_dir / 'a.png'
    Image.new('RGB', (2000, 3000), 'white').save(photo)
    before = photo.read_bytes()

    mask_file = tmp_path / 'mask.png'
    _FakeMask((2000, 3000), (800, 200, 500, 2000)).write(mask_file)
    monkeypatch.setattr(stb.auto_mask, 'is_available', lambda: True)
    monkeypatch.setattr(stb.auto_mask, 'masks_for',
                        lambda paths, prompt, **kw: {str(photo): (str(mask_file), None)})

    rows = [stb.PreviewRow(image_id=1, filename='a.png')]
    manifest = stb.build_preview(str(ds_dir), rows, progress=None)

    assert manifest['items'][0]['skip'] is None
    assert manifest['items'][0]['image'] == [2000, 3000]
    assert manifest['items'][0]['frame'][2] > 500, 'widened to the floor'
    assert photo.read_bytes() == before, 'preview must not write pixels'


def test_preview_records_the_skip_reason_instead_of_dropping_the_row(app, tmp_path, monkeypatch):
    # A dropped row is invisible; a recorded reason is reviewable. The first
    # wave only counted skips, so nobody could see WHICH images it passed over.
    ds_dir = tmp_path / 'ds'
    ds_dir.mkdir()
    photo = ds_dir / 'b.png'
    Image.new('RGB', (1000, 1000), 'white').save(photo)
    mask_file = tmp_path / 'mask2.png'
    _FakeMask((1000, 1000), (10, 10, 980, 980)).write(mask_file)
    monkeypatch.setattr(stb.auto_mask, 'is_available', lambda: True)
    monkeypatch.setattr(stb.auto_mask, 'masks_for',
                        lambda paths, prompt, **kw: {str(photo): (str(mask_file), None)})

    manifest = stb.build_preview(str(ds_dir), [stb.PreviewRow(2, 'b.png')], progress=None)
    assert manifest['items'][0]['skip'] == 'nothing-much-to-remove'
    assert manifest['items'][0]['frame'] is not None, 'the frame is still shown'


def test_preview_records_a_mask_failure_as_its_own_reason(app, tmp_path, monkeypatch):
    ds_dir = tmp_path / 'ds'
    ds_dir.mkdir()
    photo = ds_dir / 'c.png'
    Image.new('RGB', (1000, 1000), 'white').save(photo)

    monkeypatch.setattr(stb.auto_mask, 'is_available', lambda: True)
    monkeypatch.setattr(stb.auto_mask, 'masks_for',
                        lambda paths, prompt, **kw: {str(photo): (None, 'no-match')})

    manifest = stb.build_preview(str(ds_dir), [stb.PreviewRow(3, 'c.png')], progress=None)
    assert manifest['items'][0]['skip'] == 'no-subject-found'
    assert manifest['items'][0]['frame'] is None


def test_preview_records_an_engine_failure_as_failed_not_no_subject(app, tmp_path, monkeypatch):
    # Finding 6: a 'no-match' is the honest "nothing there"; anything else
    # coming back from masks_for is an engine problem and must say so.
    ds_dir = tmp_path / 'ds'
    ds_dir.mkdir()
    photo = ds_dir / 'd.png'
    Image.new('RGB', (1000, 1000), 'white').save(photo)

    monkeypatch.setattr(stb.auto_mask, 'is_available', lambda: True)
    monkeypatch.setattr(stb.auto_mask, 'masks_for',
                        lambda paths, prompt, **kw: {str(photo): (None, 'boom')})

    manifest = stb.build_preview(str(ds_dir), [stb.PreviewRow(4, 'd.png')], progress=None)
    assert manifest['items'][0]['skip'] == 'failed'
    assert manifest['items'][0]['frame'] is None


def test_preview_tells_a_stop_apart_from_an_engine_failure(app, tmp_path, monkeypatch):
    # Finding 1: 'cancelled' (the user pressed Stop) must not collapse into
    # the same 'failed' bucket as a genuine engine crash — the point being
    # tested is that they are DISTINGUISHABLE, not any particular string.
    ds_dir = tmp_path / 'ds'
    ds_dir.mkdir()
    stopped = ds_dir / 'e.png'
    crashed = ds_dir / 'f.png'
    Image.new('RGB', (1000, 1000), 'white').save(stopped)
    Image.new('RGB', (1000, 1000), 'white').save(crashed)

    monkeypatch.setattr(stb.auto_mask, 'is_available', lambda: True)
    monkeypatch.setattr(
        stb.auto_mask, 'masks_for',
        lambda paths, prompt, **kw: {str(stopped): (None, 'cancelled'),
                                     str(crashed): (None, 'boom')})

    rows = [stb.PreviewRow(5, 'e.png'), stb.PreviewRow(6, 'f.png')]
    manifest = stb.build_preview(str(ds_dir), rows, progress=None)
    stopped_skip = manifest['items'][0]['skip']
    crashed_skip = manifest['items'][1]['skip']
    assert stopped_skip == 'stopped'
    assert stopped_skip != crashed_skip, \
        'a Stop must not read as the same thing as an engine crash'


# --- start_preview -----------------------------------------------------------

from app.config import LOCAL_USER
from app.models import FaceDatasetImage
from app.services import dataset_activity as da
from app.services import face_dataset_service as fds


def _dataset_with_photos(app, count=1):
    with app.app_context():
        da.reset()
        ds = fds.create_dataset(LOCAL_USER, 'Trim me', 'trim')
        ds_dir = fds._dataset_path(ds.id)
        os.makedirs(ds_dir, exist_ok=True)
        ids = []
        for i in range(count):
            filename = f'p{i}.png'
            Image.new('RGB', (500, 500), 'white').save(os.path.join(ds_dir, filename))
            img = FaceDatasetImage(dataset_id=ds.id, filename=filename,
                                   source='import', status='keep')
            fds.db.session.add(img)
            fds.db.session.commit()
            ids.append(img.id)
        return ds.id, ids


def test_start_preview_refuses_a_second_run(app, monkeypatch):
    dataset_id, ids = _dataset_with_photos(app)
    with app.app_context():
        monkeypatch.setattr(stb.auto_mask, 'is_available', lambda: True)
        token = da.begin(dataset_id, 'trim', total=1)
        with pytest.raises(RuntimeError):
            stb.start_preview(app, LOCAL_USER, dataset_id, ids)
        da.end(token)
        da.reset()


def test_start_preview_refuses_an_unknown_dataset(app):
    with app.app_context():
        da.reset()
        with pytest.raises(ValueError):
            stb.start_preview(app, LOCAL_USER, 999999, [1])


def test_start_preview_refuses_an_empty_selection(app, monkeypatch):
    dataset_id, _ids = _dataset_with_photos(app)
    with app.app_context():
        monkeypatch.setattr(stb.auto_mask, 'is_available', lambda: True)
        with pytest.raises(ValueError):
            stb.start_preview(app, LOCAL_USER, dataset_id, [])


def test_start_preview_refuses_when_masking_is_unavailable_without_starting_activity(app, monkeypatch):
    dataset_id, ids = _dataset_with_photos(app)
    with app.app_context():
        monkeypatch.setattr(stb.auto_mask, 'is_available', lambda: False)
        with pytest.raises(stb.auto_mask.AutoMaskUnavailable):
            stb.start_preview(app, LOCAL_USER, dataset_id, ids)
        assert da.get(dataset_id) is None, 'no activity token may be begun before the refusal'


def test_start_preview_runs_inline_under_testing_and_leaves_a_readable_manifest(app, monkeypatch):
    dataset_id, ids = _dataset_with_photos(app)
    with app.app_context():
        monkeypatch.setattr(stb.auto_mask, 'is_available', lambda: True)
        monkeypatch.setattr(
            stb.auto_mask, 'masks_for',
            lambda paths, prompt, **kw: {p: (None, 'no-match') for p in paths})
        result = stb.start_preview(app, LOCAL_USER, dataset_id, ids)
        assert result == {'queued': len(ids)}
        report = stb.preview_report(dataset_id)
        assert report is not None
        assert len(report['items']) == len(ids)
        da.reset()


# --- apply and undo --------------------------------------------------------

def _seed_preview(ds_dir, dataset_id, items, monkeypatch):
    # Both names were imported INTO subject_trim_batch, so patching the module
    # attribute is what the code under test actually reads. `get_dataset` is
    # stubbed because these tests are about crop mechanics, not ownership —
    # ownership is the routes' concern and has its own tests.
    monkeypatch.setattr(stb, '_dataset_path', lambda _id: str(ds_dir))
    monkeypatch.setattr(stb, 'get_dataset', lambda user_id, dataset_id: object())
    stb._write_json(stb._preview_path(dataset_id), {'items': items})


def test_apply_crops_only_the_confirmed_rows(app, tmp_path, monkeypatch):
    ds_dir = tmp_path / 'ds'
    ds_dir.mkdir()
    for name in ('a.png', 'b.png'):
        Image.new('RGB', (2000, 3000), 'white').save(ds_dir / name)
    _seed_preview(ds_dir, 7, [
        {'image_id': 1, 'filename': 'a.png', 'image': [2000, 3000],
         'frame': [500, 100, 900, 1600], 'out': [576, 1024], 'skip': None},
        {'image_id': 2, 'filename': 'b.png', 'image': [2000, 3000],
         'frame': [500, 100, 900, 1600], 'out': [576, 1024], 'skip': None},
    ], monkeypatch)
    monkeypatch.setattr(stb.trash, 'send_to_trash',
                        lambda path, context='': str(tmp_path / os.path.basename(path)))

    counts = stb.apply_preview(1, 7, [1])
    assert counts['trimmed'] == 1
    with Image.open(ds_dir / 'a.png') as im:
        assert im.size == (576, 1024)
    with Image.open(ds_dir / 'b.png') as im:
        assert im.size == (2000, 3000), 'an unconfirmed row is untouched'


def test_apply_refuses_a_row_the_manifest_marked_as_skipped(app, tmp_path, monkeypatch):
    # The client sends ids, not coordinates, and a skipped row has no crop worth
    # applying — accepting one would let a stale UI crop something the rule
    # already rejected.
    ds_dir = tmp_path / 'ds'
    ds_dir.mkdir()
    Image.new('RGB', (1000, 1000), 'white').save(ds_dir / 'c.png')
    _seed_preview(ds_dir, 8, [
        {'image_id': 5, 'filename': 'c.png', 'image': [1000, 1000],
         'frame': [0, 0, 950, 950], 'out': [950, 950],
         'skip': 'nothing-much-to-remove'},
    ], monkeypatch)
    counts = stb.apply_preview(1, 8, [5])
    assert counts['trimmed'] == 0
    assert counts['refused'] == 1


def test_undo_restores_every_original_and_clears_the_manifest(app, tmp_path, monkeypatch):
    ds_dir = tmp_path / 'ds'
    ds_dir.mkdir()
    trashed = tmp_path / 'trashed-a.png'
    Image.new('RGB', (2000, 3000), 'white').save(trashed)
    Image.new('RGB', (576, 1024), 'black').save(ds_dir / 'a.png')
    monkeypatch.setattr(stb, '_dataset_path', lambda _id: str(ds_dir))
    monkeypatch.setattr(stb, 'get_dataset', lambda user_id, dataset_id: object())
    stb._write_json(stb._undo_path(9), {
        'entries': [{'image_id': 1, 'filename': 'a.png', 'trashed': str(trashed)}],
        'counts': {'trimmed': 1, 'refused': 0, 'failed': 0},
    })
    monkeypatch.setattr(stb.trash, 'send_to_trash',
                        lambda path, context='': str(tmp_path / 'second.png'))

    result = stb.restore_trim_batch(1, 9)
    assert result == {'restored': 1, 'gone': 0}
    with Image.open(ds_dir / 'a.png') as im:
        assert im.size == (2000, 3000)
    assert stb.trim_report(9) is None


def test_undo_counts_a_missing_original_as_gone_rather_than_failing(app, tmp_path, monkeypatch):
    ds_dir = tmp_path / 'ds'
    ds_dir.mkdir()
    monkeypatch.setattr(stb, '_dataset_path', lambda _id: str(ds_dir))
    monkeypatch.setattr(stb, 'get_dataset', lambda user_id, dataset_id: object())
    stb._write_json(stb._undo_path(10), {
        'entries': [{'image_id': 1, 'filename': 'a.png',
                     'trashed': str(tmp_path / 'never-existed.png')}],
        'counts': {},
    })
    assert stb.restore_trim_batch(1, 10) == {'restored': 0, 'gone': 1}
