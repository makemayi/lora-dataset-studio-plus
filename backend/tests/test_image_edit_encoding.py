"""✂️ Editing an image must not silently re-compress it.

Four operations rewrite the working image the trainer will later `copy2` verbatim:
mirror, rotate, crop and the watermark crop. Mirror and rotate already preserved the
source format; crop re-encoded EVERYTHING to lossy WEBP q92, so cropping a PNG
degraded it and left PNG-named files holding WEBP bytes. These tests pin the rule
they now share (`app.services.image_encoding`), pin that each one STATES its own
encoding policy rather than inheriting a global default, and — just as importantly —
pin that the throwaway derivatives were NOT dragged along with it.

Note on wording: crop also RESIZES (the box is normalised to a 1024 long side, and it
upscales a small box — that is what `upscale_ratio` records). Resampling is destructive
by nature, so nothing here claims a lossless CROP. What is proven lossless is the
ENCODING step: the bytes on disk decode back to exactly the crop+resize result.
"""
import io

import pytest
from PIL import Image

from app.services import image_encoding


# --- fixtures ----------------------------------------------------------------
def _photo(w=400, h=300, seed=11):
    """Detailed, non-flat content: a lossy encoder has something to lose on it.

    A flat colour survives q92 untouched, which would make every assertion below
    pass for the wrong reason.
    """
    im = Image.new('RGB', (w, h))
    px = im.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (
                (x * 7 + y * 3 + seed) % 256,
                (x * x // 5 + y * 11) % 256,
                (x * 3 + y * y // 7 + seed * 5) % 256,
            )
    return im


def _write(path, fmt):
    im = _photo()
    if fmt == 'JPEG':
        im.save(path, 'JPEG', quality=92, subsampling=0)
    elif fmt == 'WEBP':
        im.save(path, 'WEBP', quality=92)
    else:
        im.save(path, fmt)
    return im


def _reference_crop(path, box, size=1024):
    """The crop+resize result BEFORE any encoder touches it — the honest baseline."""
    with Image.open(path) as src:
        src.load()
        work = src.convert(image_encoding.resample_mode(src))
    bw, bh = box[2] - box[0], box[3] - box[1]
    if bw >= bh:
        out = (size, max(1, round(size * bh / bw)))
    else:
        out = (max(1, round(size * bw / bh)), size)
    return work.crop(box).resize(out, Image.LANCZOS)


def _decoded(path):
    with Image.open(path) as im:
        im.load()
        return (im.format or '').upper(), im.convert('RGB').tobytes(), im.size


def _seed_image(app, filename, fmt):
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, f'Crop {filename}', f'crop-{filename}')
        import os
        path = os.path.join(svc._dataset_dir(ds.id), filename)
        _write(path, fmt)
        row = FaceDatasetImage(dataset_id=ds.id, filename=filename,
                               source='import', status='keep')
        svc.db.session.add(row)
        svc.db.session.commit()
        return ds.id, row.id, path


# --- the rule ----------------------------------------------------------------
@pytest.mark.parametrize('fmt,filename', [
    ('PNG', 'shot.png'),
    ('WEBP', 'shot.webp'),
])
def test_crop_encoding_is_lossless_for_lossless_sources(app, fmt, filename):
    """A PNG cropped stays a PNG, and the bytes on disk decode back to the exact
    crop+resize result. Before the fix this wrote WEBP q92 into a .png file."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, filename, fmt)
    box = (40, 30, 280, 210)
    expected = _reference_crop(path, box)
    with app.app_context():
        assert svc.crop_image(LOCAL_USER, image_id, box[0], box[1],
                              box[2] - box[0], box[3] - box[1]) is True

    detected, pixels, size = _decoded(path)
    assert detected == fmt, f'crop changed the format to {detected}'
    assert size == expected.size
    assert pixels == expected.convert('RGB').tobytes(), 'the encoder lost pixels'


def test_crop_of_a_jpeg_stays_a_jpeg_and_loses_less_than_before(app):
    """JPEG has no lossless mode — we do NOT pretend otherwise.

    The source stays a JPEG (turning a user's JPEG into a 2.4x heavier lossless file
    to protect pixels that were already lossy is a bad trade), re-encoded at q95 with
    no chroma subsampling. What is asserted is that the residual error is far below
    what the old always-WEBP-q92 path left behind.
    """
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, 'shot.jpg', 'JPEG')
    box = (40, 30, 280, 210)
    expected = _reference_crop(path, box).convert('RGB')
    with app.app_context():
        assert svc.crop_image(LOCAL_USER, image_id, box[0], box[1],
                              box[2] - box[0], box[3] - box[1]) is True

    detected, pixels, size = _decoded(path)
    assert detected == 'JPEG'
    assert size == expected.size

    ref = expected.tobytes()
    kept = sum(abs(a - b) for a, b in zip(ref, pixels)) / len(ref)

    old = io.BytesIO()
    expected.save(old, 'WEBP', quality=92)
    with Image.open(old) as im:
        im.load()
        old_err = sum(abs(a - b) for a, b in zip(ref, im.convert('RGB').tobytes())) / len(ref)

    assert kept < old_err, (f'q95 4:4:4 JPEG should beat the old WEBP q92 path '
                            f'({kept:.3f} vs {old_err:.3f})')


def test_repeated_crops_of_a_webp_never_accumulate_damage(app):
    """The failure mode that reaches the LoRA weights: curate, re-crop, re-crop.

    Under q92 each pass re-compressed the previous compression. Lossless WEBP makes
    a second full-frame crop byte-for-byte identical to the first one's output.
    """
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, 'shot.webp', 'WEBP')
    with app.app_context():
        svc.crop_image(LOCAL_USER, image_id, 0, 0, 400, 300)
    once = open(path, 'rb').read()
    with Image.open(path) as im:
        w, h = im.size
    with app.app_context():
        svc.crop_image(LOCAL_USER, image_id, 0, 0, w, h)
    assert open(path, 'rb').read() == once, 're-cropping the full frame changed bytes'


def test_cropped_file_extension_still_matches_its_real_content(app):
    """A `.png` holding WEBP bytes is a trap for everything that guesses by name —
    and the import path writes the TRUE extension, so the trap was real."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    for filename, fmt in (('shot.png', 'PNG'), ('shot.jpg', 'JPEG'), ('shot.webp', 'WEBP')):
        _ds, image_id, path = _seed_image(app, filename, fmt)
        with app.app_context():
            svc.crop_image(LOCAL_USER, image_id, 10, 10, 200, 150)
        detected, _pixels, _size = _decoded(path)
        ext = '.' + filename.rsplit('.', 1)[1]
        assert ext in image_encoding.FORMAT_EXTENSIONS[detected], (
            f'{filename} now contains {detected} bytes')


def test_crop_still_resizes_to_a_1024_long_side_and_reports_the_ratio(app):
    """The encoding changed; the geometry must not.

    It also PINS a truth worth stating out loud: the resize is unconditional. A
    240x180 box is ENLARGED to 1024x768 — PIL only short-circuits `.resize()` when
    the box's long side is already exactly 1024. So a crop never merely cuts, and
    the enlargement invents no detail (measured: shrinking a 624->1024 upscale back
    to 624 recovers the original at PSNR 49 dB, i.e. the extra pixels carried almost
    nothing) while costing 2.3x the bytes. Whether that upscale should happen at all
    is a training-geometry question — `upscale_ratio` and its ⚠ warning exist
    because of it — and is deliberately NOT decided here."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, 'shot.webp', 'WEBP')
    with app.app_context():
        assert svc.crop_image(LOCAL_USER, image_id, 40, 30, 240, 180) is True
        row = svc.db.session.get(FaceDatasetImage, image_id)
        assert row.upscale_ratio == pytest.approx(1024 / 240)
    with Image.open(path) as im:
        assert im.size == (1024, 768)


def test_crop_of_an_exotic_format_falls_back_to_lossless_webp(app):
    """Crop must not REFUSE an image it used to handle. A stray BMP becomes lossless
    WEBP — what the pipeline would have produced anyway, minus the lossy step."""
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    import os

    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Exotic', 'exotic')
        path = os.path.join(svc._dataset_dir(ds.id), 'legacy.bmp')
        _photo().save(path, 'BMP')
        row = FaceDatasetImage(dataset_id=ds.id, filename='legacy.bmp',
                               source='import', status='keep')
        svc.db.session.add(row)
        svc.db.session.commit()
        image_id = row.id
    box = (40, 30, 280, 210)
    expected = _reference_crop(path, box)
    with app.app_context():
        assert svc.crop_image(LOCAL_USER, image_id, box[0], box[1],
                              box[2] - box[0], box[3] - box[1]) is True
    detected, pixels, _size = _decoded(path)
    assert detected == 'WEBP'
    assert pixels == expected.convert('RGB').tobytes()


def test_watermark_crop_is_byte_exact_because_it_never_resizes(app):
    """`_apply_watermark_crop` invents no pixel by design (no resize at all), so here
    'lossless' is the whole operation, not just its encoding."""
    from app.services import face_dataset_service as svc

    _ds, _image_id, path = _seed_image(app, 'shot.png', 'PNG')
    with Image.open(path) as im:
        im.load()
        expected = im.crop((20, 15, 300, 240)).convert('RGB').tobytes()
    assert svc._apply_watermark_crop(path, (20, 15, 300, 240)) is True
    detected, pixels, size = _decoded(path)
    assert detected == 'PNG' and size == (280, 225)
    assert pixels == expected


@pytest.mark.parametrize('fmt,filename', [
    ('PNG', 'turn.png'),
    ('WEBP', 'turn.webp'),
])
def test_mirror_and_rotation_come_back_byte_identical(app, fmt, filename):
    """A PUBLISHED promise, not an internal nicety: GitHub #17 was answered with
    "PNG and WEBP come back byte-identical after four quarter turns". Nothing in
    this consolidation may weaken it, so it is pinned on BYTES, not pixels — a
    future tweak aimed at the crop's encoder would have to break this test to reach
    the mirror or the rotation.

    (JPEG is excluded on purpose: Pillow has no DCT-domain path, so every turn of a
    JPEG is a re-encode. The app says so rather than pretending.)
    """
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, filename, fmt)
    with app.app_context():
        svc.mirror_image(LOCAL_USER, image_id)     # normalise through the encoder once
    settled = open(path, 'rb').read()

    with app.app_context():
        svc.mirror_image(LOCAL_USER, image_id)
        svc.mirror_image(LOCAL_USER, image_id)
    assert open(path, 'rb').read() == settled, 'two mirrors did not restore the file'

    with app.app_context():
        for _ in range(4):
            svc.rotate_image(LOCAL_USER, image_id, 90)
    assert open(path, 'rb').read() == settled, 'four quarter turns did not restore the file'


def test_every_edit_states_its_own_encoding_policy(app):
    """The policy is an argument, never a module default: mirror/rotate carry the
    byte-identity promise and crop does not, so the two must stay independently
    tunable. Pin that every call site names a policy, and that none of them silently
    picked the lossy one."""
    import pathlib
    import re

    def arguments(text, start):
        """The call's argument list, counting nested parentheses (the format is
        chosen by a nested `format_for_path(...)` at two of the call sites)."""
        depth, out = 0, []
        for ch in text[start:]:
            if ch == '(':
                depth += 1
                if depth == 1:
                    continue
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    break
            out.append(ch)
        return ''.join(out)

    root = pathlib.Path(__file__).resolve().parents[1]
    sites = []
    for rel in ('app/services/face_dataset_service.py',
                'app/services/watermark_klein.py',
                'infer/lama_infer.py'):
        text = (root / rel).read_text(encoding='utf-8')
        for call in re.finditer(r'image_encoding\.save_(?:edit|params)\s*\(', text):
            sites.append((rel, arguments(text, call.end() - 1)))

    assert len(sites) >= 4, f'expected every edit to route through the module, got {sites}'
    for rel, args in sites:
        assert 'image_encoding.LOSSLESS' in args, (
            f'{rel}: this edit does not state the LOSSLESS policy — {args!r}')
        assert 'HIGH_QUALITY' not in args, f'{rel}: an edit silently went lossy'


def test_the_escape_hatch_policy_is_real_and_unknown_policies_are_refused():
    """HIGH_QUALITY exists so an operation CAN opt out of the size cost, and the
    module docstring records the measurement that keeps it unused. Prove it is a
    working lossy branch rather than a decorative constant — and that a typo in a
    policy name fails loudly instead of silently re-compressing a dataset."""
    ref = _photo(240, 180)

    exact = io.BytesIO()
    image_encoding.save_edit(ref, exact, 'WEBP', image_encoding.LOSSLESS)
    cheap = io.BytesIO()
    image_encoding.save_edit(ref, cheap, 'WEBP', image_encoding.HIGH_QUALITY)

    with Image.open(exact) as im:
        im.load()
        assert im.convert('RGB').tobytes() == ref.tobytes()
    with Image.open(cheap) as im:
        im.load()
        assert im.convert('RGB').tobytes() != ref.tobytes(), 'HIGH_QUALITY is not lossy'
    # No size assertion here on purpose: which branch is SMALLER depends entirely on
    # the content (this synthetic fixture is high-frequency and compresses better
    # losslessly than lossy, the opposite of a photograph). The measurement that
    # matters is in the module docstring, on a real photograph.

    with pytest.raises(ValueError, match='unknown encoding policy'):
        image_encoding.save_params(ref, 'WEBP', 'lossles')


def test_mirror_and_crop_share_one_encoder(app):
    """The point of the wave: one rule, not three copies of it. Mirroring then
    cropping a PNG must never produce a format neither of them chose."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc

    _ds, image_id, path = _seed_image(app, 'shot.png', 'PNG')
    with app.app_context():
        svc.mirror_image(LOCAL_USER, image_id)
        assert _decoded(path)[0] == 'PNG'
        svc.crop_image(LOCAL_USER, image_id, 10, 10, 200, 150)
    assert _decoded(path)[0] == 'PNG'


# --- the guardrail: derivatives stay cheap -----------------------------------
def test_bank_thumbnails_stay_lossy_and_small(app, client, tmp_path):
    """Do NOT drag the derivatives into the lossless rule while doing good.

    A grid thumbnail is regenerated from the working image at will; inflating it 4x
    would cost a lot of disk for pixels nobody trains on. It must still be the q72
    WEBP it always was.
    """
    import os
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import image_bank_service as bank

    src = tmp_path / 'thumb-src'
    os.makedirs(src, exist_ok=True)
    _photo(1200, 900).save(str(src / 'a.png'))

    response = client.post('/api/bank/create', json={'name': 'Thumbs', 'folder': str(src)})
    assert response.status_code == 200, response.get_json()
    bank_id = response.get_json()['id']

    with app.app_context():
        b = db.session.get(ImageBank, bank_id)
        row = BankImage.query.filter_by(bank_id=bank_id).first()
        assert row is not None, 'the bank should have inventoried the file'
        tpath = bank.ensure_thumb(b, row)
        assert tpath is not None and tpath.is_file()

        with Image.open(tpath) as t:
            assert (t.format or '').upper() == 'WEBP'
            t.load()
            probe = io.BytesIO()
            t.convert('RGB').save(probe, 'WEBP', lossless=True, quality=100, method=4)
        assert tpath.stat().st_size * 2 < len(probe.getvalue()), (
            'the thumbnail looks lossless — a derivative was inflated by mistake')


def test_encoding_module_stays_importable_without_the_app(app):
    """`backend/infer/lama_infer.py` runs under the dedicated ML interpreter, where
    the Flask app package does not import. It loads this module by file path, so the
    module must never grow an app/db/config dependency."""
    import ast
    import pathlib

    path = pathlib.Path(image_encoding.__file__)
    tree = ast.parse(path.read_text(encoding='utf-8'))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split('.')[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or '').split('.')[0] or '<relative>')
    assert imported <= {'PIL', '__future__'}, f'unexpected dependency: {imported}'


def test_lama_worker_can_load_the_encoder_the_way_it_actually_does():
    """The worker resolves the module by FILE PATH (`load_image_encoding`), because it
    runs under an interpreter this suite never boots. Exercise that exact bootstrap in
    a clean subprocess: if the file moves, watermark inpainting would die at import
    time on a machine we cannot test here — and only there."""
    import pathlib
    import subprocess
    import sys

    worker = pathlib.Path(__file__).resolve().parents[1] / 'infer' / 'lama_infer.py'
    probe = (
        'import importlib.util, sys\n'
        f'spec = importlib.util.spec_from_file_location("w", r"{worker}")\n'
        'w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)\n'
        'e = w.load_image_encoding()\n'
        'print(e.format_for_path("a.png"), e.save_edit.__name__)\n'
        # The bootstrap must not leave the services directory on sys.path: it would
        # shadow whatever torch/simple-lama import next.
        'assert not any("services" in p for p in sys.path), sys.path\n'
    )
    done = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == 'PNG save_edit'
