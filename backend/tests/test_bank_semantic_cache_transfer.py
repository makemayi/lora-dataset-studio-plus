"""Portable SigLIP2 cache transport stays distinct from the CLIP score lane."""
from pathlib import Path


def _semantic_payload():
    from app.services import bank_semantic_models as assets
    return {
        'state': 'ok',
        'engine': 'siglip2',
        'model_id': assets.MODEL_ID,
        'revision': assets.REVISION,
        'model_key': assets.MODEL_KEY,
        'dimension': assets.DIMENSION,
        'embedding': (1.0,) + (0.0,) * (assets.DIMENSION - 1),
    }


def test_siglip2_asset_contract_is_pinned_and_local_only(app, tmp_path):
    from app import config
    from app.services import bank_semantic_models as assets

    root = tmp_path / 'models'
    with app.app_context():
        config.save_config({'bank_semantic': {'models_root': str(root)}})
        kwargs = assets.model_kwargs()
        assert kwargs['revision'] == assets.REVISION
        assert kwargs['cache_dir'] == str(root)
        assert kwargs['local_files_only'] is True
        assert assets.weights_present() is False
        snapshot = assets.snapshot_dir()
        snapshot.mkdir(parents=True)
        for filename in assets.FILES:
            (snapshot / filename).write_bytes(b'x')
        assert assets.weights_present() is True


def test_source_engine_restore_is_bound_to_exact_promoted_bytes(app):
    from app.services import bank_transfer_metadata as transfer

    fingerprint = 'a' * 64
    metadata = transfer.capture_transfer_metadata(
        bank={'semantic_engine': 'siglip2'}, bank_fingerprint=fingerprint)
    assert transfer.bank_semantic_engine_for_fingerprint(
        metadata, fingerprint) == 'siglip2'
    assert transfer.bank_semantic_engine_for_fingerprint(
        metadata, 'b' * 64) is None


def test_existing_bank_schema_defaults_to_clip(app, tmp_path):
    from app import _SCHEMA_ADDITIONS
    from app.extensions import db
    from app.models import ImageBank

    assert ('image_bank', 'semantic_engine',
            "VARCHAR(16) NOT NULL DEFAULT 'clip'") in _SCHEMA_ADDITIONS
    with app.app_context():
        bank = ImageBank(
            user_id='local', name='Legacy-compatible', source_path=str(tmp_path))
        db.session.add(bank)
        db.session.commit()
        db.session.refresh(bank)
        assert bank.semantic_engine == 'clip'


def test_siglip2_runtime_cache_roundtrip_is_hash_bound(app, tmp_path):
    from app.services import bank_transfer_metadata as transfer

    image = tmp_path / 'image.jpg'
    image.write_bytes(b'exact-image-bytes')
    cache = tmp_path / 'semantic_siglip2_cache.npz'
    payload = _semantic_payload()

    counts = transfer.write_runtime_caches(
        None, None, {str(image): {'semantic': payload}},
        semantic_path=cache,
        expected_fingerprints={
            str(image): transfer.content_fingerprint_path(image),
        })
    assert counts == {'score': 0, 'face': 0, 'semantic': 1}

    index = transfer.load_runtime_cache_index(
        semantic_path=cache, wanted_paths=[str(image)])
    bundle = transfer.cache_bundle_for_transfer(
        index, image, image.read_bytes())
    assert set(bundle) == {'semantic'}
    assert bundle['semantic'] == payload

    # The cache entry is authoritative only for the exact bytes the model saw.
    assert transfer.cache_bundle_for_transfer(
        index, image, b'different-image-bytes') == {}


def test_siglip2_sidecar_roundtrip_keeps_provenance(app, tmp_path):
    from app.services import bank_transfer_metadata as transfer

    payload = _semantic_payload()
    ref = transfer.write_cache_sidecar(tmp_path, {'semantic': payload})
    assert transfer.is_content_addressed_cache_ref(ref)
    assert transfer.read_cache_sidecar(tmp_path, ref) == {'semantic': payload}

    wrong_model = {**payload, 'model_key': 'clip-vit-l-14-openai'}
    assert transfer.write_cache_sidecar(
        tmp_path, {'semantic': wrong_model}) is None
    unnormalised = {**payload, 'embedding': (0.0,) * payload['dimension']}
    assert transfer.write_cache_sidecar(
        tmp_path, {'semantic': unnormalised}) is None


def test_clip_group_never_bypasses_populated_score_scalar_mismatch(app):
    from app.services import image_bank_service as banks

    score = {
        'state': 'ok', 'aesthetic': 9.1, 'nsfw': 0.03,
        'embedding': (1.0,) + (0.0,) * 767,
    }
    analysis = {
        'semantic_dup_group': 8, 'clip_semantic_dup_group': 8,
        'aesthetic_score': 7.2, 'nsfw_score': 0.03,
        'face_state': None, 'face_det': None, 'face_yaw': None,
    }
    assert banks._cache_bundle_matches_analysis(
        {'score': score}, analysis, semantic_engine='clip') == {}

    # Group-only legacy rows may lack optional scalars, but still require an OK
    # embedding cache; that narrow exception must not weaken populated values.
    analysis['aesthetic_score'] = analysis['nsfw_score'] = None
    assert set(banks._cache_bundle_matches_analysis(
        {'score': score}, analysis, semantic_engine='clip')) == {'score'}
    assert banks._cache_bundle_matches_analysis(
        {'score': {**score, 'state': 'error'}}, analysis,
        semantic_engine='clip') == {}


def test_one_row_subset_does_not_subset_semantic_metadata(app, tmp_path):
    """Regression: metadata also has shape (1,), just like a one-image cache."""
    from app.services import bank_transfer_metadata as transfer

    image = tmp_path / 'single.webp'
    image.write_bytes(b'single')
    cache = tmp_path / 'semantic_siglip2_cache.npz'
    assert transfer.write_runtime_caches(
        None, None, {str(image): {'semantic': _semantic_payload()}},
        semantic_path=cache)['semantic'] == 1

    index = transfer.load_runtime_cache_index(
        semantic_path=cache, wanted_paths=[Path(image)])
    assert str(image) in index
    assert 'semantic' in index[str(image)]


def test_int32_transport_is_bounded_and_roundtrips(tmp_path):
    from app.services import npz_transport

    path = tmp_path / 'ints.npz'
    array = npz_transport.int32([1, 768, -2], (3,))
    assert array is not None
    assert npz_transport.write_atomic(path, {'values': array}, max_file_bytes=4096)
    decoded = npz_transport.read(
        path, max_file_bytes=4096, max_uncompressed_bytes=4096,
        max_elements=8)
    assert [decoded['values'].int32(i) for i in range(3)] == [1, 768, -2]
    assert npz_transport.int32([2**31], (1,)) is None
    assert npz_transport.int32([True], (1,)) is None
