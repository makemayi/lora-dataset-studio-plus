"""The reference-set self-check: does every reference photo show the SAME person?

The hole this closes is specific. Candidate scoring is best-match-of-N (`sim` is the
MAX over the references), so a photo of the WRONG person is never outvoted -- it wins
the max and RAISES every candidate's score. The corruption is therefore invisible in
the numbers it produces, and the only place it shows is the references against each
other. These tests pin that behaviour, the honest "nothing to compare" answer, and
the rule that no machine path reaches the response.
"""
import io
import json
import os

from PIL import Image


def _png(color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new('RGB', (64, 64), color).save(buf, 'PNG')
    return buf.getvalue()


def _scorer(stdout, returncode=0, stderr=''):
    """face_similarity._run_scorer's contract: (stdout, stderr_lines, rc, timed_out)."""
    lines = [ln for ln in (stderr or '').splitlines() if ln.strip()]
    return (stdout, lines, returncode, False)


def _refs_on_disk(d, n):
    out = []
    for i in range(n):
        p = os.path.join(d, 'ref%d.png' % i)
        with open(p, 'wb') as fh:
            fh.write(_png((10 * i, 0, 0)))
        out.append(p)
    return out


def _reply(refs_report, ref_ok=True):
    return json.dumps({"ref_ok": ref_ok, "results": {}, "refs": refs_report})


# --- the payload -----------------------------------------------------------

def test_check_sends_every_ref_and_no_candidate_images(app, monkeypatch, tmp_path):
    """`images` is empty on purpose: the scorer loads the model, embeds the refs and
    stops. Sending candidates here would pay for a full scoring pass to answer a
    question that only involves the references."""
    from app.services import face_similarity as fsim

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    captured = {}

    def _fake_run(*args, **kwargs):
        captured['input'] = args[1]
        return _scorer(_reply({}))

    monkeypatch.setattr('app.services.face_similarity._run_scorer', _fake_run)
    with app.app_context():
        d = str(tmp_path)
        primary, extra1, extra2 = _refs_on_disk(d, 3)
        fsim.check_reference_set(primary, extra_ref_paths=[extra1, extra2])
    payload = json.loads(captured['input'])
    assert payload['refs'] == [primary, extra1, extra2]
    assert payload['images'] == []


def test_check_drops_a_ref_whose_file_is_gone(app, monkeypatch, tmp_path):
    from app.services import face_similarity as fsim

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    captured = {}

    def _fake_run(*args, **kwargs):
        captured['input'] = args[1]
        return _scorer(_reply({}))

    monkeypatch.setattr('app.services.face_similarity._run_scorer', _fake_run)
    with app.app_context():
        d = str(tmp_path)
        primary, extra = _refs_on_disk(d, 2)
        fsim.check_reference_set(
            primary, extra_ref_paths=[extra, os.path.join(d, 'deleted.png')])
    assert json.loads(captured['input'])['refs'] == [primary, extra]


# --- the verdict -----------------------------------------------------------

def test_a_ref_below_the_floor_is_flagged(app, monkeypatch, tmp_path):
    from app.services import face_similarity as fsim

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    with app.app_context():
        d = str(tmp_path)
        primary, good, wrong = _refs_on_disk(d, 3)
        report_in = {
            primary: {"state": "scorable", "agreement": 0.61},
            good: {"state": "scorable", "agreement": 0.58},
            wrong: {"state": "scorable", "agreement": 0.04},
        }
        monkeypatch.setattr('app.services.face_similarity._run_scorer',
                            lambda *a, **k: _scorer(_reply(report_in)))
        report, error = fsim.check_reference_set(primary, extra_ref_paths=[good, wrong])
    assert error is None
    assert report['compared'] == 3
    assert report['refs'][wrong]['flagged'] is True
    assert report['refs'][primary]['flagged'] is False
    assert report['refs'][good]['flagged'] is False


def test_one_usable_ref_is_not_all_clear(app, monkeypatch, tmp_path):
    """A lone reference agrees with nothing, so it CANNOT be flagged -- and reporting
    that as `flagged: False` alone would read as "checked, fine". `compared` is 0, so
    the caller can say "nothing to compare" instead."""
    from app.services import face_similarity as fsim

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    with app.app_context():
        d = str(tmp_path)
        primary = _refs_on_disk(d, 1)[0]
        monkeypatch.setattr(
            'app.services.face_similarity._run_scorer',
            lambda *a, **k: _scorer(_reply({primary: {"state": "scorable"}})))
        report, error = fsim.check_reference_set(primary)
    assert error is None
    assert report['compared'] == 0
    assert report['refs'][primary]['agreement'] is None
    assert report['refs'][primary]['flagged'] is False


def test_no_usable_face_reports_ref_unusable_and_still_returns_the_rows(app, monkeypatch, tmp_path):
    from app.services import face_similarity as fsim

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    with app.app_context():
        d = str(tmp_path)
        primary, extra = _refs_on_disk(d, 2)
        report_in = {primary: {"state": "no_face"}, extra: {"state": "unreadable"}}
        monkeypatch.setattr(
            'app.services.face_similarity._run_scorer',
            lambda *a, **k: _scorer(_reply(report_in, ref_ok=False)))
        report, error = fsim.check_reference_set(primary, extra_ref_paths=[extra])
    assert error['kind'] == 'ref_unusable'
    assert report['refs'][primary]['state'] == 'no_face'


def test_scorer_absent_is_unavailable_not_a_crash(app, monkeypatch, tmp_path):
    from app.services import face_similarity as fsim

    monkeypatch.setattr(fsim, 'is_available', lambda: False)
    with app.app_context():
        d = str(tmp_path)
        primary = _refs_on_disk(d, 1)[0]
        report, error = fsim.check_reference_set(primary)
    assert report == {}
    assert error['kind'] == 'unavailable'


# --- the floor -------------------------------------------------------------

def test_floor_is_configurable(app, monkeypatch, tmp_path):
    from app.services import face_similarity as fsim
    from app.config import save_config

    monkeypatch.setattr(fsim, 'is_available', lambda: True)
    with app.app_context():
        save_config({'face_scoring': {'reference_agreement_floor': 0.5}})
        d = str(tmp_path)
        primary, extra = _refs_on_disk(d, 2)
        report_in = {primary: {"state": "scorable", "agreement": 0.42},
                     extra: {"state": "scorable", "agreement": 0.42}}
        monkeypatch.setattr('app.services.face_similarity._run_scorer',
                            lambda *a, **k: _scorer(_reply(report_in)))
        report, _ = fsim.check_reference_set(primary, extra_ref_paths=[extra])
    # 0.42 clears the shipped 0.20 but not a configured 0.5.
    assert report['floor'] == 0.5
    assert report['refs'][primary]['flagged'] is True


def test_an_out_of_range_floor_falls_back_instead_of_flagging_everything(app, monkeypatch):
    """A floor of 3.0 would flag every photo forever; 'nonsense' would crash the pass.
    Both fall back to the module default rather than producing a confident wrong answer."""
    from app.services import face_similarity as fsim
    from app.config import save_config

    with app.app_context():
        save_config({'face_scoring': {'reference_agreement_floor': 3.0}})
        assert fsim.reference_agreement_floor() == fsim.REFERENCE_AGREEMENT_FLOOR
        save_config({'face_scoring': {'reference_agreement_floor': 'nonsense'}})
        assert fsim.reference_agreement_floor() == fsim.REFERENCE_AGREEMENT_FLOOR


# --- the route -------------------------------------------------------------

def _dataset_with_refs(svc, LOCAL_USER, n_extras=1):
    from app.services import reference_photos_service as refs_svc
    ds = svc.create_dataset(LOCAL_USER, 'RefCheck', 'refcheck')
    d = svc._dataset_dir(ds.id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'ref.webp'), 'wb') as fh:
        fh.write(_png())
    ds.ref_filename = 'ref.webp'
    svc.db.session.commit()
    for i in range(n_extras):
        refs_svc.add_extra_ref(LOCAL_USER, ds.id, _png((0, 255 - i, 0)))
    return ds


def test_route_never_returns_a_machine_path(app, client, monkeypatch):
    """Privacy rule: diagnostics stay paste-safe. Extras are named by the same
    filename the panel already holds; the primary is named by slot alone."""
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.services import reference_photos_service as refs_svc
    from app.config import LOCAL_USER

    with app.app_context():
        ds = _dataset_with_refs(svc, LOCAL_USER, n_extras=1)
        primary = svc._ref_path(ds)
        extras = refs_svc._extra_ref_paths(ds)
        monkeypatch.setattr(fsim, 'check_reference_set', lambda *a, **k: (
            {'floor': 0.2, 'compared': 2,
             'refs': {primary: {'state': 'scorable', 'agreement': 0.55, 'flagged': False},
                      extras[0]: {'state': 'scorable', 'agreement': 0.03, 'flagged': True}}},
            None))
        resp = client.post('/api/dataset/%d/ref/check' % ds.id)
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert primary not in body
    assert extras[0] not in body
    data = json.loads(body)
    assert [r['slot'] for r in data['refs']] == ['primary', 'extra']
    assert data['refs'][0]['filename'] is None
    assert data['refs'][1]['filename'] == os.path.basename(extras[0])
    assert data['refs'][1]['flagged'] is True


def test_route_reports_the_scorer_being_absent_as_409(app, client, monkeypatch):
    from app.services import face_dataset_service as svc
    from app.services import face_similarity as fsim
    from app.config import LOCAL_USER

    with app.app_context():
        ds = _dataset_with_refs(svc, LOCAL_USER, n_extras=0)
        monkeypatch.setattr(fsim, 'check_reference_set', lambda *a, **k: (
            {}, {'kind': 'unavailable', 'detail': 'face scoring is not installed'}))
        resp = client.post('/api/dataset/%d/ref/check' % ds.id)
    assert resp.status_code == 409
    assert resp.get_json()['reason'] == 'face_scoring'


def test_route_404s_on_an_unknown_dataset(app, client):
    resp = client.post('/api/dataset/999999/ref/check')
    assert resp.status_code == 404
