"""Dataset services must stay inert while Dataset -> Bank owns the snapshot."""

import pytest


_OPERATIONS = (
    ('classify', lambda c: c['svc'].classify_images('local', c['dataset_id'])),
    ('caption', lambda c: c['svc'].caption_images('local', c['dataset_id'])),
    ('faces', lambda c: c['svc'].analyze_faces('local', c['dataset_id'])),
    ('watermark_detect', lambda c: c['svc'].detect_watermarks(
        'local', c['dataset_id'])),
    ('watermark_regions', lambda c: c['svc'].set_watermark_regions(
        'local', c['dataset_id'], c['imported_id'], [[0.1, 0.1, 0.3, 0.3]])),
    ('watermark_dismiss', lambda c: c['svc'].dismiss_watermarks(
        'local', c['dataset_id'], [c['imported_id']])),
    ('reference_start', lambda c: c['svc'].start_reference_edit(
        c['app'], 'local', c['dataset_id'], 'klein', 'change the lighting')),
    ('reference_discard', lambda c: c['svc'].discard_reference_edit(
        c['dataset_id'])),
    ('reference_mutation', lambda c: c['svc'].reference_mutation(
        c['dataset_id'])),
    ('generate_klein', lambda c: c['svc'].generate_variations(
        'local', c['dataset_id'], [{'prompt': 'portrait'}], 1)),
    ('generate_krea', lambda c: c['svc'].generate_variations_krea(
        'local', c['dataset_id'], [{'prompt': 'portrait'}], 1)),
    ('generate_api', lambda c: c['svc'].generate_variations_nanobanana(
        c['app'], 'local', c['dataset_id'], [{'prompt': 'portrait'}], 1)),
    ('improve_one', lambda c: c['svc'].improve_existing_image(
        'local', c['imported_id'])),
    ('reimprove_one', lambda c: c['svc'].reimprove_image(
        'local', c['generated_id'])),
    ('improve_bulk', lambda c: c['svc'].start_bulk_improve(
        c['app'], 'local', c['dataset_id'], [c['imported_id']])),
    ('regenerate', lambda c: c['svc'].regenerate_image(
        'local', c['generated_id'], app=c['app'])),
)

_EXCLUSIVE_DATASET_ACTIVITIES = (
    ('bank_export', 'being copied to a Bank'),
    ('bank_import', 'receiving images from a Bank'),
    ('training_export', 'being frozen for training'),
)


@pytest.mark.parametrize('_name,operation', _OPERATIONS, ids=[
    name for name, _operation in _OPERATIONS
])
@pytest.mark.parametrize('activity_kind,error_text',
                         _EXCLUSIVE_DATASET_ACTIVITIES,
                         ids=[kind for kind, _message in
                              _EXCLUSIVE_DATASET_ACTIVITIES])
def test_dataset_effects_are_refused_before_they_mutate_an_export_snapshot(
        app, _name, operation, activity_kind, error_text):
    """Every effectful lane fails at its export guard, before DB/job changes."""
    with app.app_context():
        from app.extensions import db
        from app.models import FaceDatasetImage
        from app.services import dataset_activity
        from app.services import face_dataset_service as svc

        dataset = svc.create_dataset('local', f'guard {_name}', f'guard_{_name}')
        imported = FaceDatasetImage(
            dataset_id=dataset.id, source='import', status='keep',
            filename='source.png', watermark_state='detected')
        generated = FaceDatasetImage(
            dataset_id=dataset.id, source='generated', status='pending',
            filename='generated.png', variation_prompt='portrait')
        db.session.add_all((imported, generated))
        db.session.commit()
        before = [(row.id, row.status, row.watermark_state, row.job_id)
                  for row in FaceDatasetImage.query
                  .filter_by(dataset_id=dataset.id).order_by(FaceDatasetImage.id)]

        token = dataset_activity.begin_exclusive(
            dataset.id, activity_kind, total=2, detail='exclusive Dataset work')
        assert token is not None
        context = {
            'app': app,
            'svc': svc,
            'dataset_id': dataset.id,
            'imported_id': imported.id,
            'generated_id': generated.id,
        }
        try:
            with pytest.raises(RuntimeError, match=error_text):
                operation(context)
            db.session.expire_all()
            after = [(row.id, row.status, row.watermark_state, row.job_id)
                     for row in FaceDatasetImage.query
                     .filter_by(dataset_id=dataset.id)
                     .order_by(FaceDatasetImage.id)]
            assert after == before
            assert dataset_activity.get(dataset.id)['kind'] == activity_kind
        finally:
            dataset_activity.end(token)
