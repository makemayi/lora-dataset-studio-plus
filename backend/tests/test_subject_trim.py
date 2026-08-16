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
    monkeypatch.setattr(stb.auto_mask, 'mask_for', lambda p, prompt: str(mask_file))

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
    Image.new('RGB', (1000, 1000), 'white').save(ds_dir / 'b.png')
    mask_file = tmp_path / 'mask2.png'
    _FakeMask((1000, 1000), (10, 10, 980, 980)).write(mask_file)
    monkeypatch.setattr(stb.auto_mask, 'is_available', lambda: True)
    monkeypatch.setattr(stb.auto_mask, 'mask_for', lambda p, prompt: str(mask_file))

    manifest = stb.build_preview(str(ds_dir), [stb.PreviewRow(2, 'b.png')], progress=None)
    assert manifest['items'][0]['skip'] == 'nothing-much-to-remove'
    assert manifest['items'][0]['frame'] is not None, 'the frame is still shown'


def test_preview_records_a_mask_failure_as_its_own_reason(app, tmp_path, monkeypatch):
    ds_dir = tmp_path / 'ds'
    ds_dir.mkdir()
    Image.new('RGB', (1000, 1000), 'white').save(ds_dir / 'c.png')

    def _boom(path, prompt):
        raise stb.auto_mask.NoMatch('no person here')

    monkeypatch.setattr(stb.auto_mask, 'is_available', lambda: True)
    monkeypatch.setattr(stb.auto_mask, 'mask_for', _boom)

    manifest = stb.build_preview(str(ds_dir), [stb.PreviewRow(3, 'c.png')], progress=None)
    assert manifest['items'][0]['skip'] == 'no-subject-found'
    assert manifest['items'][0]['frame'] is None
