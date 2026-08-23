"""The Task Center now lists ALL THREE job registries, not one and a half.

The app runs long work through three places and the page only knew the durable
one (ImageGenerationQueue + TopazJob). The other two are in-memory:

  * `dataset_activity` — 14 kinds of dataset batch (captioning, watermark passes,
    face scoring, generation fan-outs, exports, backups, subject trim). Four of
    them have a cooperative stop, and `generate` has its own.
  * `bank_jobs` — one pass per bank (scan, faces, embed, promote, dedup…), the
    longest-running work in the app, carrying its own ETA.

Both were live, several were cancellable, and neither was visible on the one page
that claims to list everything running.

These tests hold three lines that are easy to cross:
  * a Cancel button is only offered where a stop actually exists;
  * the status strip counts what the list shows;
  * the row identity is stable across polls even when two batches of the SAME
    kind run on one dataset (a local and an API fan-out do exactly that).
"""
import json

import pytest


@pytest.fixture(autouse=True)
def _clean_registries():
    from app.services import bank_jobs, dataset_activity
    dataset_activity.reset()
    bank_jobs.reset()
    yield
    dataset_activity.reset()
    bank_jobs.reset()


def _overview(client):
    resp = client.get('/api/tasks/overview')
    assert resp.status_code == 200
    return json.loads(resp.get_data(as_text=True))


def _rows(data, kind):
    return [t for t in data['tasks'] if t['kind'] == kind]


# --- dataset_activity.list_every --------------------------------------------

def test_list_every_spans_datasets_and_carries_the_owner(app):
    from app.services import dataset_activity as da
    with app.app_context():
        da.begin(1, 'caption', total=40)
        da.begin(7, 'backup', total=3)
        every = da.list_every()
    assert {e['dataset_id'] for e in every} == {1, 7}
    assert {e['kind'] for e in every} == {'caption', 'backup'}
    assert all(e['token'] for e in every), 'every entry carries its own token'


def test_list_every_agrees_with_list_all_on_one_dataset(app):
    """Two readers of one registry must never disagree about what is running."""
    from app.services import dataset_activity as da
    with app.app_context():
        da.begin(4, 'caption', total=10)
        da.begin(4, 'generate', total=5)
        every = [e['kind'] for e in da.list_every() if e['dataset_id'] == 4]
        per_dataset = [e['kind'] for e in da.list_all(4)]
    assert sorted(every) == sorted(per_dataset)


def test_an_ended_batch_leaves_the_list(app):
    from app.services import dataset_activity as da
    with app.app_context():
        token = da.begin(2, 'trim', total=8)
        assert da.list_every()
        da.end(token)
        assert da.list_every() == []


# --- the rows ---------------------------------------------------------------

def test_a_dataset_batch_appears_with_its_progress(app, client):
    from app.services import dataset_activity as da
    with app.app_context():
        token = da.begin(1, 'caption', total=40)
        da.progress(token, done=12)
        data = _overview(client)
    rows = _rows(data, 'batch')
    assert len(rows) == 1
    assert rows[0]['title'] == 'Captioning'
    assert rows[0]['progress'] == '12/40'
    assert rows[0]['status'] == 'running'


def test_a_batch_with_no_known_total_says_nothing_rather_than_zero_of_zero(app, client):
    """"0/0" reads as a stalled pass. A pass that does not know its size yet must
    show no count at all."""
    from app.services import dataset_activity as da
    with app.app_context():
        da.begin(1, 'classify')
        data = _overview(client)
    assert _rows(data, 'batch')[0]['progress'] is None


def test_cancel_is_offered_only_where_a_stop_actually_exists(app, client):
    """The scope test in the row builder mirrors dataset_activity's own arming
    scopes. Offering Cancel on a pass with no seam would be a promise the user
    only finds out is empty by clicking it."""
    from app.services import dataset_activity as da
    with app.app_context():
        for dsid, kind in enumerate(('caption', 'improve', 'watermark_detect',
                                     'trim', 'generate'), start=1):
            da.begin(dsid, kind, total=5)
        for dsid, kind in enumerate(('classify', 'backup', 'watermark_clean',
                                     'bank_export'), start=20):
            da.begin(dsid, kind, total=5)
        data = _overview(client)
    by_title = {r['title']: r for r in _rows(data, 'batch')}
    for title in ('Captioning', 'Improve & upscale', 'Finding watermarks',
                  'Subject trim', 'Generating variations'):
        assert by_title[title]['actions'] == ['cancel'], title
    for title in ('Framing classification', 'Backup', 'Cleaning watermarks',
                  'Export to bank'):
        assert by_title[title]['actions'] == [], title


def test_a_stop_already_asked_for_reads_as_stopping_and_drops_the_button(app, client):
    """The worker stops at the next item boundary, so the pass is still RUNNING.
    Chipping it as cancelled would claim something that has not happened, and
    leaving the button armed invites a second click that does nothing."""
    from app.services import dataset_activity as da
    with app.app_context():
        da.begin(1, 'caption', total=40)
        assert da.request_cancel(1, ('caption',))
        data = _overview(client)
    row = _rows(data, 'batch')[0]
    assert row['status'] == 'running'
    assert row['progress'] == 'Stopping...'
    assert row['actions'] == []


def test_two_batches_of_one_kind_on_one_dataset_keep_separate_rows(app, client):
    """A local fan-out and an API fan-out are independent jobs that share a kind.
    Keying rows on (dataset, kind) would collide them, and a duplicate key is a
    remount on every poll."""
    from app.services import dataset_activity as da
    with app.app_context():
        da.begin(1, 'generate', total=10, engine='local')
        da.begin(1, 'generate', total=4, engine='api')
        data = _overview(client)
    rows = _rows(data, 'batch')
    assert len(rows) == 2
    assert len({r['job_id'] for r in rows}) == 2


def test_every_declared_activity_kind_has_a_human_title(app):
    """The label map is the reason a row does not ship named after a variable.
    A new KIND with no entry fails HERE rather than in front of a user."""
    from app.routes.tasks import ACTIVITY_LABEL
    from app.services.dataset_activity import KINDS
    missing = [k for k in KINDS if k not in ACTIVITY_LABEL]
    assert missing == [], f'no Task Center title for: {missing}'


# --- bank rows --------------------------------------------------------------

def test_a_bank_pass_appears_with_its_count(app, client, monkeypatch):
    from app.services import bank_jobs
    from app.routes import tasks as tasks_route
    monkeypatch.setattr(tasks_route, '_bank_name', lambda bid: 'Portraits')
    monkeypatch.setattr(bank_jobs, 'list_every', lambda: [
        (3, {'kind': 'faces', 'done': 120, 'total': 8000, 'error': None,
             'cancelled': False, 'finished': False, 'detail': None,
             'started_at': 1.0, 'eta_seconds': None})])
    with app.app_context():
        data = _overview(client)
    row = _rows(data, 'bank')[0]
    assert row['title'] == 'Group by person'
    assert row['source'] == 'Portraits'
    assert row['progress'] == '120/8000'
    assert row['actions'] == ['cancel']


def test_a_bank_pass_shows_its_eta_because_the_bare_count_says_nothing(app, client, monkeypatch):
    from app.services import bank_jobs
    monkeypatch.setattr(bank_jobs, 'list_every', lambda: [
        (3, {'kind': 'scan', 'done': 120, 'total': 8000, 'error': None,
             'cancelled': False, 'finished': False, 'detail': None,
             'started_at': 1.0, 'eta_seconds': 1500})])
    with app.app_context():
        data = _overview(client)
    assert '25 min left' in _rows(data, 'bank')[0]['progress']


def test_a_finished_bank_pass_is_not_cancellable(app, client, monkeypatch):
    from app.services import bank_jobs
    monkeypatch.setattr(bank_jobs, 'list_every', lambda: [
        (3, {'kind': 'scan', 'done': 8000, 'total': 8000, 'error': None,
             'cancelled': False, 'finished': True, 'detail': None,
             'started_at': 1.0, 'eta_seconds': None})])
    with app.app_context():
        data = _overview(client)
    row = _rows(data, 'bank')[0]
    assert row['status'] == 'completed'
    assert row['actions'] == []


def test_a_bank_row_carries_no_dataset_link(app, client, monkeypatch):
    """A bank id is not a dataset id. Sending it down the dataset link opens the
    wrong page, so the row carries nothing rather than something wrong."""
    from app.services import bank_jobs
    monkeypatch.setattr(bank_jobs, 'list_every', lambda: [
        (3, {'kind': 'scan', 'done': 1, 'total': 2, 'error': None,
             'cancelled': False, 'finished': False, 'detail': None,
             'started_at': 1.0, 'eta_seconds': None})])
    with app.app_context():
        data = _overview(client)
    assert _rows(data, 'bank')[0]['resource']['dataset_id'] is None


# --- the strip must agree with the list -------------------------------------

def test_the_running_count_includes_the_in_memory_passes(app, client):
    """Counting only queue rows is how "0 running" sat above a visibly running
    caption pass."""
    from app.services import dataset_activity as da
    with app.app_context():
        da.begin(1, 'caption', total=40)
        da.begin(2, 'backup', total=2)
        data = _overview(client)
    assert data['status']['summary']['running'] == 2


# --- cancel routing ---------------------------------------------------------

def test_cancelling_a_batch_arms_the_same_stop_its_own_screen_uses(app, client):
    from app.services import dataset_activity as da
    with app.app_context():
        token = da.begin(1, 'caption', total=40)
        resp = client.post(f'/api/tasks/activity-{token}/cancel')
        assert resp.status_code == 200
        assert da.cancel_requested(1, ('caption',)) is True


def test_cancelling_a_batch_that_already_finished_is_a_404_not_a_lie(app, client):
    from app.services import dataset_activity as da
    with app.app_context():
        token = da.begin(1, 'caption', total=40)
        da.end(token)
        resp = client.post(f'/api/tasks/activity-{token}/cancel')
    assert resp.status_code == 404


def test_cancelling_an_unstoppable_batch_says_so_rather_than_pretending(app, client):
    from app.services import dataset_activity as da
    with app.app_context():
        token = da.begin(1, 'backup', total=3)
        resp = client.post(f'/api/tasks/activity-{token}/cancel')
    assert resp.status_code == 409
    assert 'cannot be stopped' in resp.get_json()['error']


def test_a_malformed_activity_id_is_a_404_not_a_crash(app, client):
    with app.app_context():
        assert client.post('/api/tasks/activity-nonsense/cancel').status_code == 404
        assert client.post('/api/tasks/activity-0:caption:1/cancel').status_code == 404


def test_cancelling_a_bank_pass_routes_to_the_bank_registry(app, client, monkeypatch):
    from app.services import bank_jobs
    seen = {}
    monkeypatch.setattr(bank_jobs, 'cancel',
                        lambda bid: seen.setdefault('bank_id', bid) or True)
    with app.app_context():
        resp = client.post('/api/tasks/bank-3/cancel')
    assert resp.status_code == 200
    assert seen['bank_id'] == 3


def test_retry_refuses_the_live_only_lanes_with_a_reason(app, client):
    """404 would read as "that job is gone" for a pass that is very much there.
    These registries hold no recipe to re-run — the owning screen does."""
    with app.app_context():
        for job_id in ('activity-1:caption:1', 'bank-3'):
            resp = client.post(f'/api/tasks/{job_id}/retry')
            assert resp.status_code == 409, job_id
            assert 'cannot be retried' in resp.get_json()['error']
