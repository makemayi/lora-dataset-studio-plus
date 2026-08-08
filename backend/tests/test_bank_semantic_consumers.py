"""Selected-semantic-engine service regressions.

These tests stay model-free: the selected SigLIP2 loader and text tower are
replaced at their service seams, while every consumer runs its real ranking or
coverage code.
"""
import pytest
from PIL import Image

np = pytest.importorskip('numpy')


def _emb(*coords):
    vector = np.zeros(768, dtype='float32')
    vector[:len(coords)] = coords
    vector /= np.linalg.norm(vector) + 1e-8
    return vector


def _bank(client, tmp_path, count=3):
    source = tmp_path / 'source'
    source.mkdir()
    for index in range(count):
        Image.new('RGB', (48, 48), (20 + index * 30,) * 3).save(
            source / f'{index}.jpg')
    response = client.post('/api/bank/create', json={
        'name': 'Sig consumers', 'folder': str(source)})
    assert response.status_code == 200, response.get_json()
    return response.get_json()['id']


def _uid():
    from app.config import LOCAL_USER
    return LOCAL_USER


def test_siglip2_image_worker_uses_semantic_python(
        app, client, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import ImageBank
    from app.services import bank_semantic_models as models
    from app.services import image_bank_service as banks

    bank_id = _bank(client, tmp_path, count=1)
    seen = {}

    def drive(job, python, script, payload, cache_path, progress_re, window):
        seen.update(python=python, script=script, payload=payload)
        return {'ok': True, 'ready': 1, 'computed': 1}, [], 0

    monkeypatch.setattr(models, 'semantic_python',
                        lambda: '/managed/semantic/python')
    monkeypatch.setattr(banks, '_resolve_semantic_device',
                        lambda: ('cpu', False))
    monkeypatch.setattr(banks, '_drive_infer_subprocess', drive)
    monkeypatch.setattr(banks.bank_jobs, 'progress', lambda *a, **k: None)
    monkeypatch.setattr(banks.bank_jobs, 'cancelled', lambda job: False)
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        bank.semantic_engine = 'siglip2'
        db.session.commit()
        banks._semantic_index_job(bank_id)(object())
    assert seen['python'] == '/managed/semantic/python'
    assert seen['script'] == banks._SEMANTIC_SCRIPT


def test_siglip2_selected_consumers_use_one_space_and_report_it(
        app, client, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_semantic_engine as semantic
    from app.services import bank_semantic_models as models
    from app.services import clip_text_encoder
    from app.services import image_bank_service as banks

    bank_id = _bank(client, tmp_path)
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        bank.semantic_engine = 'siglip2'
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        rows[0].framing = 'face'
        rows[1].framing = 'body'
        rows[2].framing = 'back'
        db.session.commit()
        paths = [banks.abs_image_path(bank, row) for row in rows]
        row_ids = [row.id for row in rows]

    # Deliberately leave the third row unindexed so text search must report it.
    selected = {paths[0]: _emb(1.0, 0.0), paths[1]: _emb(0.0, 1.0)}
    loads = []

    def load_selected(bank, engine):
        loads.append((bank.id, engine))
        assert bank.semantic_engine == 'siglip2'
        assert engine == 'siglip2'
        return selected

    monkeypatch.setattr(semantic, 'load_semantic_embeddings', load_selected)
    monkeypatch.setattr(
        banks, '_load_score_embeddings',
        lambda *_args, **_kwargs: pytest.fail('selected Sig consumers read CLIP'))
    text_calls = []

    def encode(text, **kwargs):
        text_calls.append((text, dict(kwargs)))
        return _emb(1.0, 0.0), False

    monkeypatch.setattr(clip_text_encoder, 'encode_query', encode)

    with app.app_context():
        diverse = banks.select_diverse(
            _uid(), bank_id, n=2, typicality=0.0)
        similar = banks.select_similar(
            _uid(), bank_id, ref_id=row_ids[0], n=2)
        balanced = banks.select_balanced(
            _uid(), bank_id, n=2, axis='framing', typicality=0.0)
        search = banks.search_by_text(_uid(), bank_id, 'portrait', n=2)
        report = banks.coverage(_uid(), bank_id)

    assert set(diverse['image_ids']) == set(row_ids[:2])
    assert diverse['engine'] == 'siglip2' and diverse['model_key'] == models.MODEL_KEY
    assert similar['image_ids'][0] == row_ids[0]
    assert set(similar['image_ids']) == set(row_ids[:2])
    assert similar['engine'] == 'siglip2' and similar['model_key'] == models.MODEL_KEY
    assert set(balanced['image_ids']) == set(row_ids[:2])
    assert balanced['engine'] == 'siglip2' and balanced['model_key'] == models.MODEL_KEY
    assert text_calls == [('portrait', {'engine': 'siglip2'})]
    assert search['engine'] == 'siglip2'
    assert search['model_key'] == models.MODEL_KEY
    assert search['unindexed'] == 1
    assert search['unscored'] == 1       # retained compatibility alias
    assert report['engine'] == 'siglip2'
    assert report['model_key'] == models.MODEL_KEY
    assert report['semantic_indexed'] == 2
    assert report['visual']['engine'] == 'siglip2'
    assert report['visual']['model_key'] == models.MODEL_KEY
    assert report['visual']['semantic_indexed'] == 2
    assert report['visual']['calibrated'] is False
    assert loads == [(bank_id, 'siglip2')] * 5


@pytest.mark.parametrize(
    'operation', ('diverse', 'balanced', 'similar', 'search', 'coverage'))
def test_semantic_consumers_refuse_result_when_engine_changes_mid_calculation(
        app, client, tmp_path, monkeypatch, operation):
    from app.extensions import db
    from app.models import BankImage, ImageBank
    from app.services import bank_semantic_engine as semantic
    from app.services import clip_text_encoder
    from app.services import image_bank_service as banks

    bank_id = _bank(client, tmp_path)
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        bank.semantic_engine = 'siglip2'
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        for row, framing in zip(rows, ('face', 'body', 'back')):
            row.framing = framing
        db.session.commit()
        paths = [banks.analysis_image_path(bank, row) for row in rows]
        row_ids = [row.id for row in rows]
    embeddings = {path: _emb(index == 0, index == 1)
                  for index, path in enumerate(paths)}
    monkeypatch.setattr(
        semantic, 'load_semantic_embeddings',
        lambda *_args, **_kwargs: embeddings)
    monkeypatch.setattr(
        clip_text_encoder, 'encode_query',
        lambda *_args, **_kwargs: (_emb(1.0, 0.0), True))

    with app.app_context():
        target = ('_coverage_embeddings' if operation == 'coverage'
                  else '_pool_embeddings')
        original = getattr(banks, target)
        changed = False

        def switch_during(*args, **kwargs):
            nonlocal changed
            result = original(*args, **kwargs)
            if not changed:
                changed = True
                db.session.get(ImageBank, bank_id).semantic_engine = 'clip'
                db.session.commit()
            return result

        monkeypatch.setattr(banks, target, switch_during)
        calls = {
            'diverse': lambda: banks.select_diverse('local', bank_id, n=2),
            'balanced': lambda: banks.select_balanced('local', bank_id, n=2),
            'similar': lambda: banks.select_similar(
                'local', bank_id, ref_id=row_ids[0], n=2),
            'search': lambda: banks.search_by_text(
                'local', bank_id, 'portrait', n=2),
            'coverage': lambda: banks.coverage('local', bank_id),
        }
        with pytest.raises(ValueError, match='semantic engine changed'):
            calls[operation]()


def test_medium_prototypes_and_job_remain_exact_legacy_clip(
        app, client, tmp_path, monkeypatch):
    from app.extensions import db
    from app.models import ImageBank
    from app.services import clip_text_encoder
    from app.services import image_bank_service as banks

    bank_id = _bank(client, tmp_path, count=1)
    with app.app_context():
        bank = db.session.get(ImageBank, bank_id)
        bank.semantic_engine = 'siglip2'
        db.session.commit()

    prototype_calls = []

    # Intentionally accepts no engine kwarg: Medium is CLIP-owned forever.
    def encode_clip_only(text):
        prototype_calls.append(text)
        return _emb(1.0), True

    monkeypatch.setattr(clip_text_encoder, 'encode_query', encode_clip_only)
    names, matrix = banks._medium_prototype_matrix()
    assert names == list(banks.MEDIUM_PROTOTYPES)
    assert len(prototype_calls) == sum(map(len, banks.MEDIUM_PROTOTYPES.values()))
    assert matrix.shape == (len(names), 768)

    score_loads = []

    def load_score(bank):
        score_loads.append((bank.id, bank.semantic_engine))
        return {}

    monkeypatch.setattr(banks, '_load_score_embeddings', load_score)
    monkeypatch.setattr(
        banks, '_load_semantic_embeddings',
        lambda *_args, **_kwargs: pytest.fail('Medium used selected semantic space'))
    monkeypatch.setattr(
        banks, '_medium_prototype_matrix',
        lambda: (['photo'], np.stack([_emb(1.0)]).astype('float32')))

    with app.app_context():
        job = {'done': 0, 'cancelled': False}
        banks._medium_job(bank_id, False)(job)
    assert score_loads == [(bank_id, 'siglip2')]
    assert 'not scored yet' in job['detail']
