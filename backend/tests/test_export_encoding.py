"""Export encoding + the disk precondition in front of it.

Context (measured 2026-08-03 on a real 6 211-image style dataset): the exporter
re-encoded every master to lossless PNG, turning 3.6 GB of .jpg/.webp masters
into 23.7 GB of staging and 24 min of CPU. That is what produced a bare
"[Errno 28] No space left on device" on a cloud launch that had already spent
twenty minutes, and what made the pod upload "12 422 files and 24 GB".

Two behaviours are locked here:
  * a master the trainer already reads, with NO EXIF block, goes through
    verbatim — no re-encode, no inflation;
  * every case that still NEEDS the re-encode keeps it. Those cases would break
    under the naive version of this fix ("copy whenever the extension is
    accepted"), which is exactly why they are asserted rather than assumed.
"""
import os
from pathlib import Path

import pytest
from PIL import Image


def _add_kept(svc, ds, filename):
    from app.models import FaceDatasetImage
    svc.db.session.add(FaceDatasetImage(dataset_id=ds.id, status='keep',
                                        filename=filename, caption='a caption'))
    svc.db.session.commit()


def _dataset_with(app, name, trigger, filename, make):
    """Create a dataset holding ONE master written by `make(path)`."""
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    ds = svc.create_dataset(LOCAL_USER, name, trigger, kind='style')
    path = os.path.join(svc._dataset_dir(ds.id), filename)
    make(path)
    _add_kept(svc, ds, filename)
    return ds, path


def _exported_images(folder):
    return sorted(p for p in folder.iterdir()
                  if p.suffix.lower() not in ('.txt', '.json'))


# --- The fix: no needless re-encode -------------------------------------------

@pytest.mark.parametrize('filename, fmt', [('master.webp', 'WEBP'),
                                           ('master.jpg', 'JPEG'),
                                           ('master.png', 'PNG')])
def test_clean_master_is_copied_byte_for_byte(app, tmp_path, filename, fmt):
    """A .jpg/.webp/.png master, RGB, no EXIF: the trainer reads it as it is
    (ai-toolkit's img_ext_list), so the export must hand over the ORIGINAL bytes
    instead of a PNG several times their size."""
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, src = _dataset_with(app, 'Clean', 'zsty_clean', filename,
                                lambda p: Image.new('RGB', (64, 48), (10, 120, 200)).save(p, fmt))
        out = tmp_path / 'export'
        lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False, dest_dir=out)
        images = _exported_images(out)
        assert len(images) == 1
        assert images[0].suffix.lower() == os.path.splitext(filename)[1].lower()
        assert images[0].read_bytes() == open(src, 'rb').read()


def test_copied_image_keeps_the_stem_of_its_caption(app, tmp_path):
    """ai-toolkit pairs a caption (and a mask) by STEM + any known extension, so
    a copied .webp must still share its stem with the .txt sidecar."""
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _ = _dataset_with(app, 'Stem', 'zsty_stem', 'm.webp',
                              lambda p: Image.new('RGB', (32, 32), (5, 5, 5)).save(p, 'WEBP'))
        out = tmp_path / 'export'
        lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False, dest_dir=out)
        image = _exported_images(out)[0]
        sidecar = next(out.glob('*.txt'))
        assert image.stem == sidecar.stem


def test_local_lane_also_copies(app, tmp_path, monkeypatch):
    """The local training lane calls the same exporter with dest_dir=None — the
    saving must not be a cloud-only privilege."""
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, src = _dataset_with(app, 'Local', 'zsty_local', 'm.jpg',
                                lambda p: Image.new('RGB', (40, 40), (9, 9, 9)).save(p, 'JPEG'))
        local_root = tmp_path / 'aitk-datasets'
        local_root.mkdir()
        monkeypatch.setattr(lt, '_datasets_dir', lambda: local_root)
        out = lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False)
        images = _exported_images(Path(out))
        assert images[0].suffix.lower() == '.jpg'
        assert images[0].read_bytes() == open(src, 'rb').read()


# --- What must STILL be re-encoded --------------------------------------------

def test_exif_orientation_is_still_baked_into_pixels(app, tmp_path):
    """The whole reason the re-encode exists: an upright JPEG that carries
    orientation 6 must not train sideways. Copying its bytes would ship the tag
    to a trainer that ignores it."""
    from app.services import lora_training as lt
    from app.config import LOCAL_USER

    def _make(p):
        im = Image.new('RGB', (60, 30), (200, 30, 30))
        exif = im.getexif()
        exif[0x0112] = 6                     # rotate 90° CW on display
        im.save(p, 'JPEG', exif=exif)

    with app.app_context():
        ds, _ = _dataset_with(app, 'Exif', 'zsty_exif', 'rot.jpg', _make)
        out = tmp_path / 'export'
        lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False, dest_dir=out)
        image = _exported_images(out)[0]
        assert image.suffix.lower() == '.png'
        with Image.open(image) as got:
            assert got.size == (30, 60)      # transposed, not merely tagged
            assert not len(got.getexif())


def test_non_rgb_master_is_still_converted(app, tmp_path):
    """A palette PNG is an accepted extension but not RGB pixels — it keeps the
    conversion, or the trainer receives a mode it never asked for."""
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _ = _dataset_with(
            app, 'Palette', 'zsty_pal', 'pal.png',
            lambda p: Image.new('RGB', (24, 24), (7, 90, 7)).convert('P').save(p, 'PNG'))
        out = tmp_path / 'export'
        lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False, dest_dir=out)
        image = _exported_images(out)[0]
        assert image.suffix.lower() == '.png'
        with Image.open(image) as got:
            assert got.mode == 'RGB'


def test_unreadable_extension_is_still_converted(app, tmp_path):
    """A BMP master: ai-toolkit's img_ext_list does not list it, so copying its
    bytes would silently drop the image from the training set."""
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _ = _dataset_with(app, 'Bmp', 'zsty_bmp', 'm.bmp',
                              lambda p: Image.new('RGB', (16, 16), (1, 2, 3)).save(p, 'BMP'))
        out = tmp_path / 'export'
        lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False, dest_dir=out)
        assert _exported_images(out)[0].suffix.lower() == '.png'


# --- The precondition: refuse BEFORE writing ----------------------------------

def test_export_refuses_up_front_when_the_disk_cannot_hold_it(app, tmp_path, monkeypatch):
    """Live twice (cloud runs that died on a bare '[Errno 28] No space left on
    device' after twenty minutes of work): the exporter wrote until the disk
    said no. It must now refuse first, name the size it wanted and the space
    there was, and leave nothing behind."""
    import shutil as _shutil
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _ = _dataset_with(app, 'Full', 'zsty_full', 'm.webp',
                              lambda p: Image.new('RGB', (32, 32), (4, 4, 4)).save(p, 'WEBP'))
        out = tmp_path / 'export'
        monkeypatch.setattr(lt.shutil, 'disk_usage',
                            lambda _p: _shutil._ntuple_diskusage(100, 100, 1024))
        with pytest.raises(ValueError) as err:
            lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False, dest_dir=out)
        message = str(err.value)
        assert 'not enough free disk space' in message
        assert 'GB needed' in message and 'GB free' in message
        assert _exported_images(out) == []          # nothing half-written


def test_export_proceeds_when_the_disk_has_room(app, tmp_path, monkeypatch):
    """The other half of the precondition: a measurable, ample disk must never
    turn into a refusal (a guard that fires on a healthy machine is a bug)."""
    import shutil as _shutil
    from app.services import lora_training as lt
    from app.config import LOCAL_USER
    with app.app_context():
        ds, _ = _dataset_with(app, 'Room', 'zsty_room', 'm.webp',
                              lambda p: Image.new('RGB', (32, 32), (4, 4, 4)).save(p, 'WEBP'))
        out = tmp_path / 'export'
        monkeypatch.setattr(lt.shutil, 'disk_usage',
                            lambda _p: _shutil._ntuple_diskusage(10 ** 12, 0, 10 ** 12))
        lt.export_dataset_to_aitoolkit(LOCAL_USER, ds.id, masked=False, dest_dir=out)
        assert len(_exported_images(out)) == 1


def test_estimate_counts_a_copy_at_its_own_size_and_a_re_encode_much_higher(app, tmp_path):
    """The estimate is what makes the refusal honest: a copied master costs its
    file size, a re-encoded one is bounded by its raw RGB size. Reading them the
    same way is how a 3.6 GB dataset was allowed to ask for 24 GB unnoticed."""
    from app.services import lora_training as lt
    clean = tmp_path / 'clean.webp'
    Image.new('RGB', (256, 256), (33, 90, 120)).save(clean, 'WEBP')
    bmp = tmp_path / 'other.bmp'
    Image.new('RGB', (256, 256), (33, 90, 120)).save(bmp, 'BMP')
    assert lt._export_image_bytes(str(clean)) == os.path.getsize(clean)
    assert lt._export_image_bytes(str(bmp)) == 256 * 256 * 3
