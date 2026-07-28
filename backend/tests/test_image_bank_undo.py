"""🗃️ Image bank — ↩ Undo the last bulk decision.

Marking hundreds of images in one gesture is the bank's most destructive click
and, until now, the only one without a net. These tests pin the contract of the
net itself, which is mostly a contract about HONESTY:

* the reversible bulk actions (✓/✕ on a selection, auto-reject by flag,
  duplicate/same-shot resolution) are offered an undo;
* undoing restores EXACTLY the prior state — including leaving alone the rows
  the action never touched;
* rows that vanished, or that someone changed since, are counted AND named
  rather than silently missed;
* the actions that cannot be undone cleanly (🗑 Delete rejected, ⬆ Promote)
  offer nothing — a half-working "Undo" would be worse than none;
* two bulk actions in a row never leave a stale snapshot behind.
"""
import os

from PIL import Image

from app.services import bank_undo


def _img(value=128, size=(32, 32)):
    return Image.new('RGB', size, (value, value, value))


def _mkbank(client, tmp_path, names, maker=None):
    src = tmp_path / 'src'
    for i, rel in enumerate(names):
        p = src / rel
        os.makedirs(p.parent, exist_ok=True)
        (maker(i) if maker else _img(40 + i * 7)).save(str(p))
    r = client.post('/api/bank/create', json={'name': 'B', 'folder': str(src)})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['id'], src


def _images(client, bank_id):
    return client.get(f'/api/bank/{bank_id}/images').get_json()['images']


def _by_name(client, bank_id):
    return {i['name']: i for i in _images(client, bank_id)}


def _statuses(client, bank_id):
    return {i['name']: i['status'] for i in _images(client, bank_id)}


def _offer(client, bank_id):
    """The undo offer as the workspace sees it — embedded in the bank payload,
    which is why it survives a reload."""
    return client.get(f'/api/bank/{bank_id}').get_json().get('undo')


# --- the core contract -------------------------------------------------------
def test_bulk_status_is_undoable_and_restores_the_exact_prior_state(client, tmp_path):
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg'])
    by = _by_name(client, bank_id)
    # a pre-existing manual decision the bulk action will overwrite...
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['a.jpg']['id']], 'status': 'keep'})
    # ...and one it must not touch at all.
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['d.jpg']['id']], 'status': 'keep'})

    r = client.post(f'/api/bank/{bank_id}/images/status',
                    json={'ids': [by['a.jpg']['id'], by['b.jpg']['id'],
                                  by['c.jpg']['id']], 'status': 'reject'})
    assert r.status_code == 200
    assert _statuses(client, bank_id) == {'a.jpg': 'reject', 'b.jpg': 'reject',
                                          'c.jpg': 'reject', 'd.jpg': 'keep'}

    offer = _offer(client, bank_id)
    assert offer and offer['count'] == 3, offer
    assert 'reject' in offer['label'].lower()

    r = client.post(f'/api/bank/{bank_id}/undo', json={})
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert out['restored'] == 3
    assert out['missing'] == 0 and out['conflicts'] == 0
    # a.jpg goes back to 'keep' (not to 'pending'), d.jpg was never in the lot.
    assert _statuses(client, bank_id) == {'a.jpg': 'keep', 'b.jpg': 'pending',
                                          'c.jpg': 'pending', 'd.jpg': 'keep'}
    # one shot: the offer is consumed.
    assert _offer(client, bank_id) is None
    assert client.post(f'/api/bank/{bank_id}/undo', json={}).status_code == 400


def test_undo_restores_the_reject_reason_too(client, tmp_path):
    """A restored row must carry back its reason, not just its status: the flag
    counters and the 'why was this rejected' badge read that column."""
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
    by = _by_name(client, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['a.jpg']['id']], 'status': 'reject'})
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['a.jpg']['id'], by['b.jpg']['id']],
                      'status': 'keep'})
    assert client.post(f'/api/bank/{bank_id}/undo', json={}).get_json()['restored'] == 2
    rows = {i['name']: i for i in _images(client, bank_id)}
    assert rows['a.jpg']['status'] == 'reject'
    assert rows['a.jpg']['reject_reason'] == 'manual'
    assert rows['b.jpg']['status'] == 'pending'
    assert not rows['b.jpg']['reject_reason']


def test_auto_reject_by_flag_is_undoable(client, tmp_path):
    """The bad-threshold accident: one click rejects everything a flag matches."""
    def maker(i):
        # small.jpg is under the 'small' flag's minimum side, the others are not.
        return _img(60 + i * 20, (32, 32) if i == 0 else (1024, 1024))
    bank_id, _ = _mkbank(client, tmp_path, ['small.jpg', 'big1.jpg', 'big2.jpg'],
                         maker=maker)
    client.post(f'/api/bank/{bank_id}/scan', json={})

    r = client.post(f'/api/bank/{bank_id}/apply-flags', json={'flags': ['small']})
    assert r.status_code == 200, r.get_json()
    before = _statuses(client, bank_id)
    assert before['small.jpg'] == 'reject'

    offer = _offer(client, bank_id)
    assert offer and offer['count'] >= 1
    out = client.post(f'/api/bank/{bank_id}/undo', json={}).get_json()
    assert out['restored'] == offer['count']
    assert _statuses(client, bank_id)['small.jpg'] == 'pending'


def test_duplicate_resolution_is_undoable(client, tmp_path):
    bank_id, _ = _mkbank(client, tmp_path, ['d1.jpg', 'd2.jpg'],
                         maker=lambda i: _img(128))
    client.post(f'/api/bank/{bank_id}/scan', json={})
    r = client.post(f'/api/bank/{bank_id}/dups/resolve', json={'strategy': 'best'})
    assert r.status_code == 200, r.get_json()
    rejected = sum(1 for s in _statuses(client, bank_id).values() if s == 'reject')
    assert rejected >= 1

    offer = _offer(client, bank_id)
    assert offer and offer['count'] == rejected
    assert client.post(f'/api/bank/{bank_id}/undo', json={}).get_json()['restored'] == rejected
    assert all(s == 'pending' for s in _statuses(client, bank_id).values())


def test_launch_all_auto_reject_offers_one_undo_for_the_whole_step(client, tmp_path):
    """🚀 Launch all's auto-reject is a flag pass PLUS a duplicate resolution.
    The unit the user would take back is the step, so it must publish ONE offer
    covering both — not two, of which only the second is reachable."""
    def maker(i):
        # 0: small (flagged) · 1 & 2: an identical pair (a duplicate group)
        return _img(128, (32, 32) if i == 0 else (1024, 1024))
    bank_id, _ = _mkbank(client, tmp_path, ['small.jpg', 'd1.jpg', 'd2.jpg'],
                         maker=maker)
    r = client.post(f'/api/bank/{bank_id}/pipeline',
                    json={'steps': ['scan', 'auto_reject'],
                          'reject_flags': ['small'], 'resolve_dups': True})
    assert r.status_code == 202, r.get_json()
    rejected = sum(1 for s in _statuses(client, bank_id).values() if s == 'reject')
    assert rejected >= 2, _statuses(client, bank_id)   # the small one AND a dup

    offer = _offer(client, bank_id)
    assert offer and offer['count'] == rejected
    assert 'launch all' in offer['label'].lower()
    out = client.post(f'/api/bank/{bank_id}/undo', json={}).get_json()
    assert out['restored'] == rejected
    assert all(s == 'pending' for s in _statuses(client, bank_id).values())


# --- honesty: partial success is counted AND named ---------------------------
def test_partial_undo_reports_what_it_could_not_restore(client, tmp_path, app):
    """Rows changed since the action are left alone, counted and NAMED; rows
    that left the bank are counted as missing. 'restored' never over-claims."""
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg', 'c.jpg'])
    by = _by_name(client, bank_id)
    ids = [by[n]['id'] for n in ('a.jpg', 'b.jpg', 'c.jpg')]
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': ids, 'status': 'reject'})

    # someone (the user, another tab) changes ONE of them since.
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['c.jpg']['id']], 'status': 'keep'})
    # ...which is itself a bulk action, so put the first snapshot back to test
    # the conflict path rather than the supersede path.
    from app.services import bank_undo as bu
    bu.record(bank_id, 'Reject 3 images', {
        i: {'before': ('pending', None), 'after': ('reject', 'manual')} for i in ids})

    # and one row leaves the bank entirely.
    with app.app_context():
        from app.extensions import db
        from app.models import BankImage
        BankImage.query.filter_by(id=by['b.jpg']['id']).delete()
        db.session.commit()

    out = client.post(f'/api/bank/{bank_id}/undo', json={}).get_json()
    assert out['restored'] == 1                     # only a.jpg
    assert out['missing'] == 1                      # b.jpg is gone
    assert out['conflicts'] == 1                    # c.jpg changed since
    assert 'c.jpg' in ' '.join(out['conflict_names'])
    assert _statuses(client, bank_id)['a.jpg'] == 'pending'
    assert _statuses(client, bank_id)['c.jpg'] == 'keep'   # NOT clobbered


# --- irreversible actions offer nothing --------------------------------------
def test_delete_rejected_offers_no_undo_and_invalidates_the_previous_one(client,
                                                                        tmp_path,
                                                                        monkeypatch):
    """🗑 Delete rejected drops rows and sends files to the trash: it cannot be
    undone, and it makes the PREVIOUS reject-snapshot un-restorable too — so the
    offer must disappear instead of promising a restore that would find nothing."""
    import sys
    import types
    monkeypatch.setitem(sys.modules, 'send2trash',
                        types.SimpleNamespace(send2trash=lambda p: os.remove(p)))
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
    by = _by_name(client, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['a.jpg']['id']], 'status': 'reject'})
    assert _offer(client, bank_id) is not None

    r = client.post(f'/api/bank/{bank_id}/delete-rejected', json={})
    assert r.status_code == 200, r.get_json()
    assert _offer(client, bank_id) is None
    assert client.post(f'/api/bank/{bank_id}/undo', json={}).status_code == 400


def test_promote_offers_no_undo(client, tmp_path, app):
    """⬆ Promote COPIES into a dataset through the import path — un-promoting
    would mean deleting someone else's dataset images. No offer."""
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'],
                         maker=lambda i: _img(50 + i * 40, (600, 600)))
    with app.app_context():
        from app.services import face_dataset_service as svc
        dataset_id = svc.create_dataset('local', 'From bank', 'bnk').id
    by = _by_name(client, bank_id)
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': [by['a.jpg']['id']], 'status': 'keep'})
    # consume the keep-snapshot so only the promotion could produce a new one
    client.post(f'/api/bank/{bank_id}/undo', json={})
    assert _offer(client, bank_id) is None

    r = client.post(f'/api/bank/{bank_id}/promote',
                    json={'dataset_id': dataset_id, 'image_ids': [by['a.jpg']['id']]})
    assert r.status_code == 202, r.get_json()
    assert _offer(client, bank_id) is None


# --- two actions in a row --------------------------------------------------
def test_second_bulk_action_supersedes_the_first_snapshot(client, tmp_path):
    """Depth is ONE on purpose. The second action must replace the first
    snapshot cleanly — never merge with it, never restore a two-actions-ago
    state, never leave the first one reachable."""
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
    by = _by_name(client, bank_id)
    ids = [by['a.jpg']['id'], by['b.jpg']['id']]
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': ids, 'status': 'keep'})
    client.post(f'/api/bank/{bank_id}/images/status',
                json={'ids': ids, 'status': 'reject'})

    offer = _offer(client, bank_id)
    assert offer['count'] == 2
    out = client.post(f'/api/bank/{bank_id}/undo', json={}).get_json()
    assert out['restored'] == 2
    # back to 'keep' — the state just before the LAST action, not 'pending'.
    assert set(_statuses(client, bank_id).values()) == {'keep'}
    assert _offer(client, bank_id) is None


def test_a_no_op_bulk_action_publishes_nothing(client, tmp_path):
    """Re-applying the status rows already have changes nothing, so it must not
    publish an "Undo (0 images)" offer — nor destroy the real one still standing
    from the action before it."""
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg', 'b.jpg'])
    by = _by_name(client, bank_id)
    ids = [by['a.jpg']['id'], by['b.jpg']['id']]
    client.post(f'/api/bank/{bank_id}/images/status', json={'ids': ids, 'status': 'keep'})
    assert _offer(client, bank_id)['count'] == 2
    # the same click again: two rows written, zero decisions flipped.
    client.post(f'/api/bank/{bank_id}/images/status', json={'ids': ids, 'status': 'keep'})
    offer = _offer(client, bank_id)
    assert offer['count'] == 2                     # still the REAL one
    assert client.post(f'/api/bank/{bank_id}/undo', json={}).get_json()['restored'] == 2
    assert set(_statuses(client, bank_id).values()) == {'pending'}


def test_snapshots_are_per_bank(client, tmp_path):
    """Two banks triaged side by side must not undo each other."""
    b1, _ = _mkbank(client, tmp_path / 'one', ['a.jpg'])
    b2, _ = _mkbank(client, tmp_path / 'two', ['a.jpg'])
    ids1 = [_by_name(client, b1)['a.jpg']['id']]
    client.post(f'/api/bank/{b1}/images/status', json={'ids': ids1, 'status': 'reject'})
    assert _offer(client, b1) is not None
    assert _offer(client, b2) is None
    assert client.post(f'/api/bank/{b2}/undo', json={}).status_code == 400
    assert _statuses(client, b1)['a.jpg'] == 'reject'


def test_registry_is_bounded_and_forgets_on_reset(client, tmp_path):
    bank_id, _ = _mkbank(client, tmp_path, ['a.jpg'])
    ids = [_by_name(client, bank_id)['a.jpg']['id']]
    client.post(f'/api/bank/{bank_id}/images/status', json={'ids': ids, 'status': 'reject'})
    assert bank_undo.peek(bank_id) is not None
    bank_undo.reset()
    assert bank_undo.peek(bank_id) is None
