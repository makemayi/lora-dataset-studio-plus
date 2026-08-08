"""✂ Near-duplicate shots — the advisory pass over the vectors 🔎 Search cached.

What is under test is NOT "does cosine work" but the four decisions that make
this pass safe to run on someone's bank:

  * it re-embeds NOTHING — a clip with no vector is not evaluated, and "not
    evaluated" is a third state next to clean and flagged;
  * a group keeps a REPRESENTATIVE, and the flag lands on the others — a pass
    that flagged every member would tell the user to drop all of them;
  * nothing is ever deleted or rejected: the flag is advisory, exactly like
    every other verdict in this lane;
  * a re-run fully recomputes, so lowering the threshold cannot leave last
    run's groups behind.
"""
import json

import pytest

from app.services import video_clip_dedup as dedup


def _vec(*values):
    """A vector as a plain list — the module normalises, so magnitudes here are
    free. Kept tiny (3 dims) so a reader can see the geometry."""
    return list(values)


# --- the arithmetic -------------------------------------------------------------

def test_two_shots_that_share_one_near_identical_instant_are_similar():
    """MAX over frame pairs, not mean. Two shots of the same set-up differ at
    their edges (one starts on a pan) and coincide in the middle; a mean asks
    "are these the same shot throughout" when the question is "does this bank
    hold this moment twice"."""
    a = [{'vec': _vec(1, 0, 0)}, {'vec': _vec(0, 1, 0)}, {'vec': _vec(0, 0, 1)}]
    b = [{'vec': _vec(0, 1, 0)}, {'vec': _vec(-1, 0, 0)}, {'vec': _vec(0, -1, 0)}]
    assert dedup.clip_similarity(a, b) == pytest.approx(1.0)


def test_two_unrelated_shots_score_low():
    a = [{'vec': _vec(1, 0, 0)}]
    b = [{'vec': _vec(0, 1, 0)}]
    assert dedup.clip_similarity(a, b) == pytest.approx(0.0)


def test_a_shot_with_no_vectors_has_no_similarity_to_anything():
    """None, never 0.0. Zero is a measurement ("nothing alike"); the absence of
    a vector is the absence of one, and the two lead to different sentences."""
    assert dedup.clip_similarity([], [{'vec': _vec(1, 0, 0)}]) is None


# --- grouping -------------------------------------------------------------------

def test_shots_above_the_threshold_group_and_others_stay_out():
    store = {
        1: [{'vec': _vec(1, 0, 0)}],
        2: [{'vec': _vec(0.999, 0.045, 0)}],
        3: [{'vec': _vec(0, 1, 0)}],
    }
    groups = dedup.group_clips(store, threshold=0.96)
    assert groups == [[1, 2]]


def test_grouping_is_transitive_because_a_group_is_a_pile_not_a_pair():
    """Union-find, the same shape the image lane's semantic dedup uses. Three
    frames of one slow pan chain A~B~C without A~C reaching the cut, and calling
    that two groups would ask the user to review the same pile twice."""
    store = {
        1: [{'vec': _vec(1.0, 0.0, 0)}],
        2: [{'vec': _vec(0.99, 0.141, 0)}],
        3: [{'vec': _vec(0.96, 0.279, 0)}],
    }
    groups = dedup.group_clips(store, threshold=0.96)
    assert groups == [[1, 2, 3]]


def test_the_matrix_form_groups_exactly_what_comparing_every_pair_would():
    """`group_clips` does not compare pairs in Python — it does one matmul over
    every frame of the bank and attributes the surviving pairs back to their
    shots, because the readable form is tens of minutes on a real bank. That
    optimisation is only safe if it groups the SAME thing, including the two ways
    it could silently differ: a shot pairing with itself (its own three frames are
    the most similar rows in the matrix), and multi-frame shots where only one
    pair of frames clears the cut."""
    import random
    rnd = random.Random(20260806)
    store = {}
    for cid in range(1, 26):
        n = rnd.choice((1, 2, 3))
        store[cid] = [{'vec': [rnd.gauss(0, 1) for _ in range(8)]}
                      for _ in range(n)]
    # A planted near-duplicate on ONE frame pair out of nine.
    store[7] = [{'vec': [9, 0, 0, 0, 0, 0, 0, 0]},
                {'vec': [0, 9, 0, 0, 0, 0, 0, 0]},
                {'vec': [0, 0, 9, 0, 0, 0, 0, 0]}]
    store[19] = [{'vec': [0, 0, 0, 4, 0, 0, 0, 0]},
                 {'vec': [0, 0, 4, 0, 0, 0, 0, 0]},
                 {'vec': [0, 0, 0, 0, 4, 0, 0, 0]}]

    for threshold in (0.5, 0.8, 0.99):
        assert dedup.group_clips(store, threshold) == _naive_groups(store, threshold), \
            f'the matrix form and the pairwise form disagree at {threshold}'


def _naive_groups(store, threshold):
    """The obvious O(n²) implementation, kept in the TEST rather than the module:
    it is the specification the fast one has to match."""
    ids = sorted(store)
    parent = {cid: cid for cid in ids}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sim = dedup.clip_similarity(store[a], store[b])
            if sim is not None and sim >= threshold:
                parent[find(b)] = find(a)
    piles = {}
    for cid in ids:
        piles.setdefault(find(cid), []).append(cid)
    return sorted((sorted(p) for p in piles.values() if len(p) >= 2),
                  key=lambda p: (-len(p), p[0]))


def test_a_lone_shot_is_not_a_group():
    store = {1: [{'vec': _vec(1, 0, 0)}], 2: [{'vec': _vec(0, 1, 0)}]}
    assert dedup.group_clips(store, threshold=0.96) == []


# --- who survives ---------------------------------------------------------------

def test_the_sharpest_shot_of_a_group_is_the_representative():
    info = {1: {'sharpness': 40.0}, 2: {'sharpness': 180.0}, 3: {'sharpness': 90.0}}
    assert dedup.pick_representative([1, 2, 3], info) == 2


def test_an_unmeasured_shot_never_beats_a_measured_one():
    """A clip nobody measured has no sharpness to offer. Letting it win would
    keep the one shot of the group whose quality is unknown."""
    info = {1: {'sharpness': None}, 2: {'sharpness': 12.0}}
    assert dedup.pick_representative([1, 2], info) == 2


def test_with_nothing_measured_the_first_shot_wins_so_a_re_run_is_stable():
    info = {7: {'sharpness': None}, 3: {'sharpness': None}}
    assert dedup.pick_representative([7, 3], info) == 3


# --- the pass over a bank --------------------------------------------------------

def test_the_pass_flags_every_member_but_the_representative(app, monkeypatch):
    bank_id, ids = _bank_with_clips(app, 3)
    _fake_store(monkeypatch, {
        ids[0]: [{'vec': _vec(1, 0, 0)}],
        ids[1]: [{'vec': _vec(0.999, 0.045, 0)}],
        ids[2]: [{'vec': _vec(0, 1, 0)}],
    })
    _measured(app, {ids[0]: 10.0, ids[1]: 200.0, ids[2]: 50.0})

    with app.app_context():
        out = dedup.run_dedup(bank_id, threshold=0.96)

    assert out['groups'] == 1
    assert out['flagged'] == 1
    stored = _summaries(app, bank_id)
    # ids[1] is the sharpest of the pair, so IT is kept and ids[0] carries the flag.
    assert stored[ids[1]]['duplicate_of'] is None
    assert stored[ids[0]]['duplicate_of'] == ids[1]
    assert stored[ids[1]]['duplicate_group'] == stored[ids[0]]['duplicate_group']
    # The unrelated shot is left alone entirely — not "clean", untouched.
    assert 'duplicate_group' not in stored[ids[2]]


def test_a_shot_with_no_vector_is_not_evaluated_rather_than_clean(app, monkeypatch):
    """The ternary contract every verdict in this lane keeps. A clip the embed
    pass never reached must not come back looking checked-and-fine."""
    bank_id, ids = _bank_with_clips(app, 2)
    _fake_store(monkeypatch, {ids[0]: [{'vec': _vec(1, 0, 0)}]})

    with app.app_context():
        out = dedup.run_dedup(bank_id, threshold=0.96)

    assert out['unevaluated'] == 1
    assert _summaries(app, bank_id)[ids[1]] == {}


def test_a_re_cut_shot_is_not_grouped_on_the_vectors_of_its_old_span(app, monkeypatch):
    """A trim clears ``embed_state`` and deliberately leaves the .npz alone (a
    rewrite inside an interactive gesture would cost tens of MB), so the store
    still holds three instants of a span that no longer exists. The COLUMN is the
    authority — grouping on those orphans would pair a shot with a neighbour over
    footage neither of them contains any more."""
    bank_id, ids = _bank_with_clips(app, 2)
    _fake_store(monkeypatch, {
        ids[0]: [{'vec': _vec(1, 0, 0)}],
        ids[1]: [{'vec': _vec(0.999, 0.045, 0)}],
    })
    with app.app_context():
        from app.extensions import db
        from app.models import VideoClip
        db.session.get(VideoClip, ids[1]).embed_state = None
        db.session.commit()
        out = dedup.run_dedup(bank_id, threshold=0.96)

    assert out['groups'] == 0
    assert out['unevaluated'] == 1


def test_the_pass_never_changes_a_triage_decision(app, monkeypatch):
    """Advisory means advisory: this writes verdicts, it does not reject."""
    bank_id, ids = _bank_with_clips(app, 2)
    _fake_store(monkeypatch, {
        ids[0]: [{'vec': _vec(1, 0, 0)}],
        ids[1]: [{'vec': _vec(0.999, 0.045, 0)}],
    })

    with app.app_context():
        from app.models import VideoClip
        dedup.run_dedup(bank_id, threshold=0.96)
        assert {c.status for c in VideoClip.query.filter_by(bank_id=bank_id)} == {'pending'}


def test_a_re_run_clears_the_previous_groups(app, monkeypatch):
    """Raising the cut must UNFLAG what no longer reaches it. Groups written by
    the previous run and left behind would be a verdict nothing produced."""
    bank_id, ids = _bank_with_clips(app, 2)
    _fake_store(monkeypatch, {
        ids[0]: [{'vec': _vec(1, 0, 0)}],
        ids[1]: [{'vec': _vec(0.999, 0.045, 0)}],
    })
    with app.app_context():
        dedup.run_dedup(bank_id, threshold=0.96)
    # Nothing is measured here, so the lowest id is the stable representative and
    # the flag lands on the other one.
    assert _summaries(app, bank_id)[ids[1]]['duplicate_of'] == ids[0]

    with app.app_context():
        out = dedup.run_dedup(bank_id, threshold=0.999)

    assert out['groups'] == 0
    stored = _summaries(app, bank_id)
    assert 'duplicate_of' not in stored[ids[1]]
    assert 'duplicate_group' not in stored[ids[1]]
    # …and a clip whose blob held NOTHING but the cleared verdict goes back to
    # having no summary at all, rather than to an empty JSON object that would
    # read as "measured, nothing found".
    assert stored[ids[1]] == {}


def test_the_pass_keeps_the_measurements_it_did_not_write(app, monkeypatch):
    """metrics_json is shared with the metrics pass. Overwriting the blob rather
    than merging into it would silently erase every quality score in the bank."""
    bank_id, ids = _bank_with_clips(app, 2)
    _fake_store(monkeypatch, {
        ids[0]: [{'vec': _vec(1, 0, 0)}],
        ids[1]: [{'vec': _vec(0.999, 0.045, 0)}],
    })
    _measured(app, {ids[0]: 10.0, ids[1]: 20.0})

    with app.app_context():
        dedup.run_dedup(bank_id, threshold=0.96)

    stored = _summaries(app, bank_id)
    assert stored[ids[0]]['metrics_state'] == 'ok'
    assert stored[ids[0]]['sharpness_p90'] == 10.0


def test_the_default_threshold_is_the_image_lane_s_measured_cut():
    """The number is INHERITED from a calibration this project actually ran (the
    image lane's semantic near-duplicate cut over the same CLIP space), and the
    two must not drift into two meanings of "near-identical"."""
    from app.config import DEFAULTS
    assert dedup.DEFAULT_THRESHOLD == DEFAULTS['bank']['semantic_dup_threshold']


# --- helpers ---------------------------------------------------------------------

def _bank_with_clips(app, n):
    from app.extensions import db
    from app.models import VideoBank, VideoClip, VideoSource
    with app.app_context():
        bank = VideoBank(name='b', source_path='/srv/rushes')
        db.session.add(bank)
        db.session.flush()
        src = VideoSource(bank_id=bank.id, relpath='a.mp4', duration_s=60.0,
                          fps_native=25.0, probe_state='ok')
        db.session.add(src)
        db.session.flush()
        ids = []
        for i in range(n):
            # embed_state 'ok' on every clip: the column is the AUTHORITY over
            # the store (see run_dedup), so a fixture that leaves it null is a
            # bank nothing can be compared in.
            clip = VideoClip(bank_id=bank.id, source_id=src.id,
                             start_s=float(i * 10), end_s=float(i * 10 + 5),
                             embed_state='ok')
            db.session.add(clip)
            db.session.flush()
            ids.append(clip.id)
        db.session.commit()
        return bank.id, ids


def _fake_store(monkeypatch, store):
    """The vector store, stubbed. Nothing here loads numpy-backed .npz files or
    a CLIP model: this pass READS what 🔎 Search already wrote."""
    monkeypatch.setattr(dedup, 'load_vectors', lambda bank_id: store)


def _measured(app, sharpness_by_id):
    from app.extensions import db
    from app.models import VideoClip
    with app.app_context():
        for cid, sharp in sharpness_by_id.items():
            clip = db.session.get(VideoClip, cid)
            clip.metrics_json = json.dumps({'metrics_state': 'ok',
                                            'sharpness_p90': sharp})
        db.session.commit()


def _summaries(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        rows = VideoClip.query.filter_by(bank_id=bank_id).all()
        return {r.id: (json.loads(r.metrics_json) if r.metrics_json else {})
                for r in rows}
