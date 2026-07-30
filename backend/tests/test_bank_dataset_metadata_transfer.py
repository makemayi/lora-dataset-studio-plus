"""Strict v2 metadata transport between image banks and datasets."""
import hashlib
import json
import os
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image


SOURCE_METADATA = {
    'platform': 'pexels',
    'source_url': 'https://www.pexels.com/photo/12345/',
    'photographer': 'Jane Doe',
    'photographer_url': 'https://www.pexels.com/@jane-doe/',
}
WATERMARK_BBOX = '[0.1,0.1,0.3,0.3]'
WATERMARK_REGIONS = '[[0.1,0.1,0.3,0.3]]'

# Deliberately implausible historical Bank values. Bank -> Dataset and Bank ->
# Bank calculate deterministic values again from copied output and never carry
# these ML values.
HISTORICAL_ANALYSIS = {
    'quality_state': 'ok',
    'blur_score': 13.5,
    'noise_score': 2.25,
    'uniformity_score': 41.0,
    'dhash': '0123456789abcdef',
    'face_state': 'scorable',
    'face_det': 0.91,
    'aesthetic_score': 7.2,
    'nsfw_score': 0.03,
    'detail_ratio': 0.88,
    'bars_ratio': 0.01,
    'jpeg_quality': 92.0,
    'origin': 'camera',
    'origin_evidence': 'exif-camera',
}


def _write_photo(path: Path, size=(160, 96), *, quality=35):
    """Write a small, asymmetric patterned image in the format from ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new('RGB', size)
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            pixels[x, y] = ((17 * x + 3 * y) % 256,
                            (5 * x + 19 * y) % 256,
                            (11 * x + 7 * y) % 256)
    suffix = path.suffix.lower()
    if suffix in ('.jpg', '.jpeg'):
        image.save(path, 'JPEG', quality=quality)
    elif suffix == '.webp':
        image.save(path, 'WEBP', quality=80)
    else:
        image.save(path, 'PNG')


def _dimensions(path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _row_values(row, fields) -> dict:
    return {name: getattr(row, name) for name in fields}


def _final_deterministic_analysis(path) -> dict:
    """The oracle shared by every strict transfer assertion in this module."""
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets

    analysis = datasets.bank_deterministic_analysis(path)
    return {name: analysis[name] for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS}


def _assert_v2_snapshot_matches_final(snapshot_text, final_path):
    from app.services import bank_transfer_metadata as transfer

    snapshot = transfer.parse_snapshot(snapshot_text)
    assert snapshot is not None
    assert snapshot['v'] == 2
    assert set(snapshot['analysis']) == set(transfer.DETERMINISTIC_ANALYSIS_FIELDS)
    assert set(snapshot['analysis']).isdisjoint(transfer.MODEL_ANALYSIS_FIELDS)
    assert snapshot['fingerprint'] == hashlib.sha256(Path(final_path).read_bytes()).hexdigest()
    assert snapshot['analysis'] == _final_deterministic_analysis(final_path)
    return snapshot


def _assert_model_analysis_is_empty(row):
    from app.services import bank_transfer_metadata as transfer

    assert _row_values(row, transfer.MODEL_ANALYSIS_FIELDS) == {
        name: None for name in transfer.MODEL_ANALYSIS_FIELDS
    }


def _valid_v2_snapshot_payload():
    """One independently valid payload to mutate for rejection cases."""
    return {
        'v': 2,
        'fingerprint': 'a' * 64,
        'analysis': {
            'quality_state': 'ok',
            'blur_score': 13.5,
            'noise_score': 2.25,
            'uniformity_score': 41.0,
            'dhash': '0123456789abcdef',
            'detail_ratio': 0.88,
            'bars_ratio': 0.01,
            'jpeg_quality': 92.0,
            'origin': 'camera',
            'origin_evidence': 'exif-camera',
        },
    }


def _bank_with_analysed_image(app, tmp_path, *, filename='one.jpg'):
    """Seed a Bank row with history that must not become transfer truth."""
    from app.extensions import db
    from app.models import BankImage, ImageBank

    folder = tmp_path / 'bank-source'
    source = folder / filename
    _write_photo(source)
    bank = ImageBank(user_id='local', name='Source bank', source_path=str(folder))
    db.session.add(bank)
    db.session.flush()
    image = BankImage(
        bank_id=bank.id, relpath=filename, file_size=source.stat().st_size,
        width=160, height=96, **HISTORICAL_ANALYSIS,
        # These are intentionally non-null: no transfer may recycle a relation
        # id that only has meaning inside this particular Bank.
        dup_group=7, semantic_dup_group=8, face_cluster=4, style_cluster=3,
        watermark_state='detected', watermark_bbox=WATERMARK_BBOX,
        watermark_regions=WATERMARK_REGIONS, caption='caption from the bank',
        source_metadata=json.dumps(SOURCE_METADATA), framing='body', status='keep',
    )
    db.session.add(image)
    db.session.commit()
    return bank, image


def _promote_to_dataset(app, bank, image, *, name='Transfer target'):
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks

    dataset = datasets.create_dataset('local', name, 'transfer')
    banks.start_promote(app, 'local', bank.id, [image.id], dataset.id)
    return dataset


def _make_source_transform(bank, image, kind):
    """Materialize a transform, returning the dimensions its destination owns."""
    from app.extensions import db
    from app.services import image_bank_service as banks

    if kind == 'direct':
        return (160, 96)
    if kind == 'rotation':
        image.rotation = 90
        db.session.commit()
        return (96, 160)
    if kind == 'clean':
        clean = banks.clean_image_path(bank.id, image.id)
        _write_photo(clean, size=(83, 57))
        image.watermark_clean_method = 'crop'
        image.watermark_state = 'cleaned'
        db.session.commit()
        return (83, 57)
    raise AssertionError(f'unknown transform {kind!r}')


@pytest.mark.parametrize(
    ('transform', 'expected_size', 'expected_format', 'expected_suffix'),
    (('direct', (160, 96), 'JPEG', '.jpg'),
     ('rotation', (96, 160), 'JPEG', '.jpg'),
     ('clean', (83, 57), 'WEBP', '.webp')),
)
def test_bank_to_dataset_snapshot_describes_the_final_preserved_file(
        app, tmp_path, transform, expected_size, expected_format, expected_suffix):
    """Direct/rotated Bank masters retain JPEG; a cleaned Bank derivative remains
    WebP. In every case the snapshot describes the final Dataset file, never the
    old Bank row."""
    from app.models import FaceDatasetImage
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        assert _make_source_transform(source_bank, source_image, transform) == expected_size
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        final_path = Path(dataset_path(dataset.id)) / dataset_image.filename

        with Image.open(final_path) as stored:
            assert stored.format == expected_format
            assert stored.size == expected_size
        assert dataset_image.filename.endswith(expected_suffix)
        snapshot = _assert_v2_snapshot_matches_final(
            dataset_image.bank_analysis_snapshot, final_path)
        # A clean derivative is WebP and carries no JPEG estimate. The helper
        # above already proves that a preserved JPEG's value is measured from its
        # final bytes, not copied from Bank history.
        if expected_format == 'WEBP':
            assert snapshot['analysis']['jpeg_quality'] is None
        assert source_image.jpeg_quality == HISTORICAL_ANALYSIS['jpeg_quality']


def test_dataset_to_bank_restores_v2_deterministic_data_and_current_user_metadata(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        dataset_file = Path(dataset_path(dataset.id)) / dataset_image.filename
        snapshot = _assert_v2_snapshot_matches_final(
            dataset_image.bank_analysis_snapshot, dataset_file)

        # The Dataset is now the owner of user-facing choices.
        dataset_image.caption = 'caption edited in the dataset'
        dataset_image.framing = 'face'
        dataset_image.watermark_state = 'dismissed'
        dataset_image.watermark_regions = '[]'
        db.session.commit()

        returned_bank_id = banks.start_dataset_import(app, 'local', dataset.id, 'Returned')
        returned = BankImage.query.filter_by(bank_id=returned_bank_id).one()
        returned_bank = db.session.get(ImageBank, returned_bank_id)
        returned_file = Path(returned_bank.source_path) / returned.relpath
        assert returned.caption == 'caption edited in the dataset'
        assert returned.framing == 'face'
        assert returned.watermark_state == 'dismissed'
        assert returned.watermark_regions == '[]'
        assert json.loads(returned.source_metadata) == SOURCE_METADATA
        assert returned.status == 'keep'
        assert _row_values(returned, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == snapshot['analysis']
        _assert_model_analysis_is_empty(returned)
        assert (returned.width, returned.height) == _dimensions(returned_file)
        assert (returned.dup_group, returned.semantic_dup_group,
                returned.face_cluster, returned.style_cluster) == (None, None, None, None)


def test_dataset_to_bank_without_preservation_never_uses_a_snapshot_and_sets_dimensions(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        dataset_image.caption = 'dataset-owned caption'
        db.session.commit()

        # This is stronger than merely seeing nulls: the compatibility parser
        # is not even consulted when the caller explicitly asks for a fresh Bank.
        with patch.object(banks.bank_transfer_metadata, 'compatible_analysis',
                          side_effect=AssertionError('snapshot must not be read')):
            returned_bank_id = banks.start_dataset_import(
                app, 'local', dataset.id, 'Fresh', preserve_analysis=False)
        returned = BankImage.query.filter_by(bank_id=returned_bank_id).one()
        returned_bank = db.session.get(ImageBank, returned_bank_id)
        returned_file = Path(returned_bank.source_path) / returned.relpath
        assert returned.caption == 'dataset-owned caption'
        assert returned.framing == 'body'
        assert returned.watermark_state == 'detected'
        assert _row_values(returned, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == {
            name: None for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS
        }
        _assert_model_analysis_is_empty(returned)
        assert (returned.width, returned.height) == _dimensions(returned_file)


def test_dataset_to_bank_rejects_a_same_size_middle_byte_mutation(app, tmp_path):
    """The transfer signature is a full SHA-256, not sampled first/last bytes."""
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        dataset_file = Path(dataset_path(dataset.id)) / dataset_image.filename
        snapshot = _assert_v2_snapshot_matches_final(
            dataset_image.bank_analysis_snapshot, dataset_file)

        # Keep the image readable by changing an ignored trailing byte, but make
        # it a byte in the MIDDLE of a >8 MiB fingerprinted file and preserve
        # its size.  The old sampled implementation skipped this middle range.
        original = dataset_file.read_bytes()
        padding = b'P' * (8 * 1024 * 1024 + 17)
        signed = original + padding
        dataset_file.write_bytes(signed)
        dataset_image.bank_analysis_snapshot = json.dumps({
            'v': 2,
            'fingerprint': hashlib.sha256(signed).hexdigest(),
            'analysis': snapshot['analysis'],
        })
        db.session.commit()
        with dataset_file.open('r+b') as fh:
            fh.seek(len(original) + len(padding) // 2)
            fh.write(b'X')
        assert _dimensions(dataset_file) == (160, 96)

        returned_bank_id = banks.start_dataset_import(app, 'local', dataset.id, 'Changed')
        returned = BankImage.query.filter_by(bank_id=returned_bank_id).one()
        assert _row_values(returned, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == {
            name: None for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS
        }
        _assert_model_analysis_is_empty(returned)


def test_v1_snapshot_is_rejected_even_when_its_fingerprint_matches(app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets
    from app.services import image_bank_service as banks
    from app.services.dataset_storage import dataset_path

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source_image)
        dataset_image = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        dataset_file = Path(dataset_path(dataset.id)) / dataset_image.filename
        v2 = _assert_v2_snapshot_matches_final(dataset_image.bank_analysis_snapshot, dataset_file)
        legacy = json.dumps({
            'v': 1,
            'fingerprint': hashlib.sha256(dataset_file.read_bytes()).hexdigest(),
            'analysis': v2['analysis'],
        })
        assert transfer.parse_snapshot(legacy) is None
        dataset_image.bank_analysis_snapshot = legacy
        db.session.commit()

        # Backup canonicalization must omit the rejected v1 payload too, rather
        # than shipping an opaque historical object to another installation.
        archive = datasets.build_backup_zip('local', dataset.id)
        restored = datasets.import_backup_zip('local', archive)
        restored_image = FaceDatasetImage.query.filter_by(dataset_id=restored.id).one()
        assert restored_image.bank_analysis_snapshot is None

        returned_bank_id = banks.start_dataset_import(app, 'local', dataset.id, 'Legacy')
        returned = BankImage.query.filter_by(bank_id=returned_bank_id).one()
        assert _row_values(returned, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == {
            name: None for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS
        }
        _assert_model_analysis_is_empty(returned)


@pytest.mark.parametrize(
    'overrides',
    (
        {'detail_ratio': -0.001},
        {'detail_ratio': 1.001},
        {'bars_ratio': -0.001},
        {'bars_ratio': 1.001},
        {'jpeg_quality': 0},
        {'jpeg_quality': 101},
        {'blur_score': -0.001},
        # The Laplacian has range [-4*255, +4*255], so its variance cannot
        # exceed (4*255)^2; see image_quality.quality_metrics.
        {'blur_score': float((4 * 255) ** 2) + 0.001},
        {'noise_score': -0.001},
        # Noise is RMS over an 8-bit difference image, bounded by 255.
        {'noise_score': 255.001},
        {'uniformity_score': -0.001},
        {'uniformity_score': 128.001},
        {'quality_state': 'unreadable'},
        {'origin': 'unknown', 'origin_evidence': 'claimed-source'},
        {'origin': 'ai', 'origin_evidence': None},
        {'origin': 'camera', 'origin_evidence': None},
    ),
    ids=(
        'detail_ratio-below-zero',
        'detail_ratio-above-one',
        'bars_ratio-below-zero',
        'bars_ratio-above-one',
        'jpeg_quality-below-one',
        'jpeg_quality-above-100',
        'blur-negative',
        'blur-above-physical-laplacian-bound',
        'noise-negative',
        'noise-above-physical-rms-bound',
        'uniformity-negative',
        'uniformity-above-128',
        'quality-state-not-ok',
        'unknown-origin-with-evidence',
        'ai-origin-without-evidence',
        'camera-origin-without-evidence',
    ),
)
def test_strict_v2_snapshot_validation_rejects_invalid_analysis_in_parser_and_backup_path(
        overrides):
    """Both direct reads and backup canonicalization reject invalid v2 claims."""
    from app.services import bank_transfer_metadata as transfer

    payload = _valid_v2_snapshot_payload()
    payload['analysis'].update(overrides)
    serialized = json.dumps(payload)
    assert transfer.parse_snapshot(serialized) is None
    # Backup import passes persisted JSON through this canonicalizer, so it must
    # have the same fail-closed policy as the public parser.
    assert transfer.normalized_snapshot_storage(serialized) is None


def test_snapshot_parser_rejects_deeply_nested_json_without_raising():
    from app.services import bank_transfer_metadata as transfer

    nested = '[' * 1500 + '0' + ']' * 1500
    payload = ('{"v":2,"fingerprint":"' + '0' * 64 +
               '","analysis":' + nested + '}')
    assert len(payload.encode('utf-8')) < 8 * 1024
    assert transfer.parse_snapshot(payload) is None


def test_bank_deterministic_analysis_skips_oversized_header_before_load(monkeypatch):
    """The local Bank-analysis guard must reject pixels before decoding them."""
    from app.services import face_dataset_service as datasets

    class OversizedImage:
        size = (datasets.BANK_ANALYSIS_MAX_SIDE,
                datasets.BANK_ANALYSIS_MAX_SIDE)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def draft(self, *_args):
            raise AssertionError('oversized header reached draft before guard')

        def load(self):
            raise AssertionError('oversized header reached load before guard')

    monkeypatch.setattr(datasets.Image, 'open', lambda *_args, **_kwargs: OversizedImage())
    assert datasets.bank_deterministic_analysis('oversized-image') is None


def test_bank_deterministic_analysis_skips_pillow_bombs_without_filter_mutation(
        monkeypatch):
    """Bomb errors/warnings degrade safely without changing global filters."""
    from app.services import face_dataset_service as datasets

    filters_before = list(warnings.filters)

    def bomb_error(*_args, **_kwargs):
        raise datasets.Image.DecompressionBombError('too many pixels')

    monkeypatch.setattr(datasets.Image, 'open', bomb_error)
    assert datasets.bank_deterministic_analysis('bomb-error') is None

    def bomb_warning(*_args, **_kwargs):
        raise datasets.Image.DecompressionBombWarning('too many pixels')

    monkeypatch.setattr(datasets.Image, 'open', bomb_warning)
    assert datasets.bank_deterministic_analysis('bomb-warning') is None
    assert warnings.filters == filters_before


def test_bank_to_bank_promotion_survives_pillow_bomb_in_both_destination_reads(
        app, tmp_path, monkeypatch):
    """A bomb must not kill the later dimensions read after analysis skips it."""
    from app.models import BankImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        filters_before = list(warnings.filters)

        def bomb(*_args, **_kwargs):
            raise banks.Image.DecompressionBombError('too many pixels')

        # ``banks.Image`` and the Dataset analysis share Pillow's module, so
        # this simulates both destination opens in the real B -> B job.
        monkeypatch.setattr(banks.Image, 'open', bomb)
        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source_image.id], 'Bomb-safe copy')
        copied = BankImage.query.filter_by(bank_id=destination_id).one()
        assert _row_values(copied, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == {
            name: None for name in transfer.DETERMINISTIC_ANALYSIS_FIELDS
        }
        _assert_model_analysis_is_empty(copied)
        assert (copied.width, copied.height) == (None, None)
        assert copied.caption == 'caption from the bank'
        assert json.loads(copied.source_metadata) == SOURCE_METADATA
        assert copied.status == 'pending'
        assert warnings.filters == filters_before


def test_direct_bank_to_bank_copy_recalculates_analysis_and_resets_container_state(
        app, tmp_path):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source_image.id], 'Bank copy')
        copied = BankImage.query.filter_by(bank_id=destination_id).one()
        destination = db.session.get(ImageBank, destination_id)
        copied_file = Path(destination.source_path) / copied.relpath
        assert copied.caption == 'caption from the bank'
        # Image-specific review state must be redone for every new Bank, even
        # for a direct byte copy; only caption and provenance are safe to keep.
        assert copied.framing is None
        assert copied.watermark_state is None
        assert copied.watermark_bbox is None
        assert copied.watermark_regions is None
        assert json.loads(copied.source_metadata) == SOURCE_METADATA
        assert _row_values(copied, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == (
            _final_deterministic_analysis(copied_file))
        # The historical source row is deliberately implausible: even the
        # direct route must take its technical truth from the copied bytes.
        assert _row_values(copied, transfer.DETERMINISTIC_ANALYSIS_FIELDS) != _row_values(
            source_image, transfer.DETERMINISTIC_ANALYSIS_FIELDS)
        _assert_model_analysis_is_empty(copied)
        assert (copied.width, copied.height) == _dimensions(copied_file)
        assert copied.status == 'pending'
        assert (copied.dup_group, copied.semantic_dup_group,
                copied.face_cluster, copied.style_cluster) == (None, None, None, None)


@pytest.mark.parametrize(
    ('transform', 'expected_size'),
    (('rotation', (96, 160)), ('clean', (83, 57))),
)
def test_transformed_bank_to_bank_copy_recalculates_deterministic_analysis_only(
        app, tmp_path, transform, expected_size):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_transfer_metadata as transfer
    from app.services import image_bank_service as banks

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        assert _make_source_transform(source_bank, source_image, transform) == expected_size
        destination_id = banks.start_bank_promote(
            app, 'local', source_bank.id, [source_image.id], f'{transform} copy')
        copied = BankImage.query.filter_by(bank_id=destination_id).one()
        destination = db.session.get(ImageBank, destination_id)
        copied_file = Path(destination.source_path) / copied.relpath
        assert _dimensions(copied_file) == expected_size
        assert (copied.width, copied.height) == expected_size
        assert _row_values(copied, transfer.DETERMINISTIC_ANALYSIS_FIELDS) == (
            _final_deterministic_analysis(copied_file))
        _assert_model_analysis_is_empty(copied)
        assert copied.status == 'pending'
        assert copied.framing is None
        assert copied.watermark_state is None
        assert copied.watermark_bbox is None
        assert copied.watermark_regions is None
        assert copied.watermark_clean_method is None
        assert copied.rotation is None


def test_from_dataset_route_defaults_analysis_to_true_and_accepts_false(client, app):
    from app.services import image_bank_service as banks

    with patch.object(banks, 'start_dataset_import', return_value=42) as start:
        response = client.post('/api/bank/from-dataset', json={
            'dataset_id': 3, 'name': 'Copy', 'preserve_analysis': False,
        })
    assert response.status_code == 202
    assert start.call_args.kwargs['preserve_analysis'] is False
    with patch.object(banks, 'start_dataset_import', return_value=43) as start:
        response = client.post('/api/bank/from-dataset', json={
            'dataset_id': 3, 'name': 'Default copy',
        })
    assert response.status_code == 202
    assert start.call_args.kwargs['preserve_analysis'] is True
    assert client.post('/api/bank/from-dataset', json={
        'dataset_id': 3, 'name': 'Copy', 'preserve_analysis': 'false',
    }).status_code == 400


def test_dataset_backup_roundtrip_keeps_a_valid_v2_snapshot(app, tmp_path):
    from app.models import FaceDatasetImage
    from app.services import bank_transfer_metadata as transfer
    from app.services import face_dataset_service as datasets

    with app.app_context():
        source_bank, source_image = _bank_with_analysed_image(app, tmp_path)
        dataset = _promote_to_dataset(app, source_bank, source_image)
        original = FaceDatasetImage.query.filter_by(dataset_id=dataset.id).one()
        original_snapshot = transfer.parse_snapshot(original.bank_analysis_snapshot)
        archive = datasets.build_backup_zip('local', dataset.id)
        restored = datasets.import_backup_zip('local', archive)
        restored_image = FaceDatasetImage.query.filter_by(dataset_id=restored.id).one()
        assert transfer.parse_snapshot(restored_image.bank_analysis_snapshot) == original_snapshot


def test_transfer_metadata_migrations_upgrade_legacy_tables_idempotently(app, tmp_path):
    """Both columns must appear on a DB that already has the two image tables."""
    from sqlalchemy import text
    from app import _SCHEMA_ADDITIONS, _apply_additive_migrations
    from app.extensions import db
    from app.models import BankImage, FaceDatasetImage, ImageBank
    from app.services import face_dataset_service as datasets

    assert ('face_dataset_image', 'bank_analysis_snapshot', 'TEXT') in _SCHEMA_ADDITIONS
    assert ('bank_image', 'source_metadata', 'TEXT') in _SCHEMA_ADDITIONS
    with app.app_context():
        dataset = datasets.create_dataset('local', 'Legacy', 'legacy-transfer')
        dataset_image = FaceDatasetImage(
            dataset_id=dataset.id, filename='legacy.webp', status='keep',
            caption='keep this caption')
        legacy_source = tmp_path / 'legacy-source'
        legacy_source.mkdir()
        bank = ImageBank(user_id='local', name='Legacy bank', source_path=str(legacy_source))
        db.session.add_all((dataset_image, bank))
        db.session.flush()
        bank_image = BankImage(bank_id=bank.id, relpath='legacy.png', caption='keep bank caption')
        db.session.add(bank_image)
        db.session.commit()
        dataset_image_id, bank_image_id = dataset_image.id, bank_image.id

        db.session.execute(text(
            'ALTER TABLE face_dataset_image DROP COLUMN bank_analysis_snapshot'))
        db.session.execute(text('ALTER TABLE bank_image DROP COLUMN source_metadata'))
        db.session.commit()

        def columns(table):
            return {r[1] for r in db.session.execute(text(f'PRAGMA table_info({table})'))}

        assert 'bank_analysis_snapshot' not in columns('face_dataset_image')
        assert 'source_metadata' not in columns('bank_image')
        _apply_additive_migrations()
        _apply_additive_migrations()       # runs on every boot: must remain a no-op
        assert 'bank_analysis_snapshot' in columns('face_dataset_image')
        assert 'source_metadata' in columns('bank_image')

        legacy_dataset = db.session.execute(text(
            'SELECT caption, bank_analysis_snapshot FROM face_dataset_image WHERE id=:id'),
            {'id': dataset_image_id}).one()
        legacy_bank = db.session.execute(text(
            'SELECT caption, source_metadata FROM bank_image WHERE id=:id'),
            {'id': bank_image_id}).one()
        assert legacy_dataset == ('keep this caption', None)
        assert legacy_bank == ('keep bank caption', None)
