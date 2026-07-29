"""Masked training that cannot produce masks must SAY SO before the GPU is paid.

The person-mask twin of test_face_mask_install_gate. Masked training is the
DEFAULT for a character run: when rembg is missing the export drops the masks and
the run trains unmasked, and the only trace was a flag on the live progress view,
which vanishes with the run. Someone pays for a full local run — or a rented pod,
since the masks are generated locally and uploaded with the images — and finds out
only by testing the LoRA (1Tomber, GitHub #24).

The arbitrage pinned here: a WARNING at the preflight gate, never a blocker.
Training unmasked is a legitimate choice, and the `masks` probe is a subprocess
import whose timeout collapses to False (a cold `import rembg` measures ~20 s), so
a hard refusal would stop launches on machines where everything works.
"""
import os

from PIL import Image

from app.extensions import db
from app.models import FaceDatasetImage
from app.services import face_dataset_service as svc
from app.config import LOCAL_USER, save_config


def _dataset(tmp_path, kind='character', n=14, trigger='pmi_act', **extra):
    save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    kw = {'kind': kind, **extra}
    if kind == 'concept':
        kw['concept_desc'] = 'balancing a spoon'
    ds = svc.create_dataset(LOCAL_USER, 'PMI', trigger, **kw)
    img_dir = svc._dataset_dir(ds.id)
    for i in range(n):
        fn = f'k{i}.png'
        Image.new('RGB', (64, 64)).save(os.path.join(img_dir, fn))
        db.session.add(FaceDatasetImage(
            dataset_id=ds.id, status='keep', filename=fn, framing='body',
            caption=f'a subject standing in a plain studio, shot number {i}'))
    db.session.commit()
    return ds


def _warn(report):
    return [w for w in report['warnings'] if 'rembg' in w]


def _row(report):
    return [c for c in report.get('checks') or [] if c['id'] == 'person_mask']


def _no_rembg(monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_masks',
                        lambda: {'ok': False, 'detail': 'import failed'})


def test_masked_launch_without_rembg_is_announced_before_the_launch(app, tmp_path,
                                                                   monkeypatch):
    from app.services import lora_training as lt
    _no_rembg(monkeypatch)
    with app.app_context():
        ds = _dataset(tmp_path)
        r = lt.training_preflight(LOCAL_USER, ds.id, masked=True)
    warn = _warn(r)
    assert warn, 'a requested person mask that cannot be produced must be announced'
    assert 'UNMASKED' in warn[0]
    row = _row(r)
    assert row and row[0]['status'] == 'warn'
    # A warning, not a blocker — see the module docstring.
    assert not [b for b in r['blockers'] if 'rembg' in b]
    assert r['verdict'] == 'warnings'


def test_the_warning_survives_the_cloud_lane(app, tmp_path, monkeypatch):
    """The masks are generated HERE at export and uploaded with the images
    (cloud_training._prepare_staging), so rembg missing on this box means the PAID
    run trains unmasked. Local origin, cloud consequence: it is not a machine-scope
    row and must not be filtered out."""
    from app.services import lora_training as lt
    _no_rembg(monkeypatch)
    with app.app_context():
        ds = _dataset(tmp_path, trigger='pmi_cloud')
        r = lt.training_preflight(LOCAL_USER, ds.id, masked=True, lane='cloud')
    assert _warn(r)
    assert _row(r)


def test_rembg_present_is_a_green_row_not_a_warning(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_masks',
                        lambda: {'ok': True, 'detail': 'rembg import OK'})
    with app.app_context():
        ds = _dataset(tmp_path, trigger='pmi_ok')
        r = lt.training_preflight(LOCAL_USER, ds.id, masked=True)
    assert not _warn(r)
    row = _row(r)
    assert row and row[0]['status'] == 'ok'


def test_unmasked_launch_says_nothing(app, tmp_path, monkeypatch):
    """Someone who unticked 🎭 Masked is not waiting for a mask — warning there is
    the noise that teaches people to click through preflights."""
    from app.services import lora_training as lt
    _no_rembg(monkeypatch)
    with app.app_context():
        ds = _dataset(tmp_path, trigger='pmi_off')
        r = lt.training_preflight(LOCAL_USER, ds.id, masked=False)
    assert not _warn(r)
    assert not _row(r)


def test_a_caller_that_states_no_intent_reads_the_dataset(app, tmp_path,
                                                         monkeypatch):
    """REWRITTEN, and the rewrite IS the feature.

    This test used to assert that a caller stating no intent got the historical
    silent payload — correct while `masked` lived in the browser's localStorage,
    because the server genuinely could not know. Now the setting is stored on the
    dataset, so the workspace readiness badge — which states no intent, by
    design — is exactly the caller that must be told: this dataset is set to
    train masked and rembg is missing, so the run would train unmasked. Staying
    silent here is the defect the stored setting was introduced to remove.

    An explicit `masked=False` is still honoured and still silent; see
    test_unmasked_launch_says_nothing."""
    from app.services import lora_training as lt
    _no_rembg(monkeypatch)
    with app.app_context():
        ds = _dataset(tmp_path, trigger='pmi_none')
        r = lt.training_preflight(LOCAL_USER, ds.id)
    assert _warn(r), 'the badge must say the dataset is set to masked'
    assert _row(r)


def test_concept_datasets_stay_silent(app, tmp_path, monkeypatch):
    """Person masks are forced OFF for a concept by design (a person mask would
    erase the very concept), so installing rembg would change nothing."""
    from app.services import lora_training as lt
    _no_rembg(monkeypatch)
    with app.app_context():
        ds = _dataset(tmp_path, kind='concept', trigger='pmi_concept')
        r = lt.training_preflight(LOCAL_USER, ds.id, masked=True)
    assert not _warn(r)
    assert not _row(r)


def test_a_probe_failure_never_blocks_the_preflight(app, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    from app import capabilities

    def boom():
        raise OSError('no such interpreter')

    monkeypatch.setattr(capabilities, 'probe_masks', boom)
    with app.app_context():
        ds = _dataset(tmp_path, trigger='pmi_boom')
        r = lt.training_preflight(LOCAL_USER, ds.id, masked=True)
    assert 'verdict' in r
    assert not _warn(r)


def test_the_route_passes_the_masked_intent_through(app, client, tmp_path, monkeypatch):
    from app.services import lora_training as lt
    seen = {}
    real = lt.training_preflight

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    from app.routes import training as training_routes
    monkeypatch.setattr(lt, 'training_preflight', spy)
    # The ai-toolkit gate is not what this test is about.
    monkeypatch.setattr(training_routes, '_require_aitoolkit', lambda *a, **k: None)
    _no_rembg(monkeypatch)
    with app.app_context():
        ds = _dataset(tmp_path, trigger='pmi_route')
        dsid = ds.id
    client.get(f'/api/dataset/{dsid}/train/preflight?masked=1')
    assert seen.get('masked') is True
    seen.clear()
    client.get(f'/api/dataset/{dsid}/train/preflight?masked=0')
    assert seen.get('masked') is False
    seen.clear()
    client.get(f'/api/dataset/{dsid}/train/preflight')
    assert seen.get('masked') is None
