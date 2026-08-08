"""🗃️ Image bank — what pressing Stop COSTS, published by whoever knows.

Reported from a live 36 925-image bank, mid ✨ Score:

    Scoring pass running — 2200 / 36925 · writing 36925 score(s) to the
    database… — a few minutes on a bank this size          [ Stop ]

    "the Stop button at that point works very badly."

Measured on that instance while the pass was writing: ``POST /cancel`` answered
in 79 ms (it touches no database — the job registry is in memory) while
``GET /api/bank/<id>``, the banner's own source, took 2 745 ms. The button was
never broken, it was MUTE, and the log holds seven cancel POSTs inside 20 ms.

The front end can make the button answer its own click. What it cannot do is
say what stopping costs, because that is not a property of the pass — it is a
property of the PHASE. ✨ Score stops three different ways inside one run:

    inference   → what the child computed is salvaged, the global style
                  partition is not (it is computed over the whole bank at once)
    write-back  → the flag is read once per commit batch, so the stop lands at
                  the end of the current 200 rows
    style write → the stop does not land at all, that step is written whole or
                  not at all

Only the worker knows which of those is true this second. These tests pin that
it says so, and that a phase with nothing specific to promise stays SILENT
rather than inventing a reassurance.
"""
from unittest.mock import patch

from PIL import Image


def _bank(tmp_path, n=2):
    from app.services import image_bank_service as banks
    src = tmp_path / 'src'
    src.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new('RGB', (600, 600), (10 * i, 90, 160)).save(str(src / f'a{i}.jpg'))
    bank, _added = banks.create_bank('local', 'Dump', str(src))
    return bank.id


# ── the registry field itself ────────────────────────────────────────────────

def test_the_snapshot_carries_the_promise_and_clears_it_on_request(app):
    from app.services import bank_jobs
    bank_jobs.reset()
    job = bank_jobs.reserve(4242, 'score', total=10)
    assert bank_jobs.get(4242)['stop_cost'] is None
    assert bank_jobs.get(4242)['stop_wait'] is None

    bank_jobs.set_stop_notice(job, cost='Everything scored is kept.',
                              wait='Stopping — finishing this batch.')
    snap = bank_jobs.get(4242)
    assert snap['stop_cost'] == 'Everything scored is kept.'
    assert snap['stop_wait'] == 'Stopping — finishing this batch.'

    # A phase that has nothing specific to promise clears it. Leaving the
    # previous phase's sentence up would describe a step that is over — which is
    # worse than silence, because it reads as current.
    bank_jobs.set_stop_notice(job)
    assert bank_jobs.get(4242)['stop_cost'] is None
    assert bank_jobs.get(4242)['stop_wait'] is None
    bank_jobs.reset()


def test_a_finished_pass_promises_nothing_about_a_button_that_is_gone(app):
    from app.services import bank_jobs
    bank_jobs.reset()
    job = bank_jobs.reserve(4243, 'score', total=10)
    bank_jobs.set_stop_notice(job, cost='Everything scored is kept.',
                              wait='Stopping — finishing this batch.')
    job['finished'] = True
    snap = bank_jobs.get(4243)
    assert snap['stop_cost'] is None and snap['stop_wait'] is None
    bank_jobs.reset()


def test_a_job_mapping_from_before_this_field_still_renders(app):
    """A live upgrade leaves running jobs in memory without the new keys.

    ``get`` builds the payload the whole Bank page depends on; a KeyError here
    would blank the page for every pass already running at update time.
    """
    from app.services import bank_jobs
    bank_jobs.reset()
    job = bank_jobs.reserve(4244, 'scan', total=3)
    del job['stop_cost']
    del job['stop_wait']
    snap = bank_jobs.get(4244)
    assert snap['stop_cost'] is None and snap['stop_wait'] is None
    bank_jobs.reset()


# ── ✨ Score's three phases ──────────────────────────────────────────────────

def _score_phases(app, tmp_path):
    """Run the score job with the child stubbed; return every published
    (detail, stop_cost, stop_wait) in order."""
    from app.services import image_bank_service as banks
    with app.app_context():
        bank_id = _bank(tmp_path)
        seen = {'details': [], 'stop': []}

        def fake_drive(job, python, script, payload, cache_path, progress_re, window):
            return {'ok': True, 'results': {}, 'clusters': {}}, [], 0

        with patch.object(banks, '_drive_infer_subprocess', fake_drive), \
             patch.object(banks.bank_jobs, 'cancelled', lambda job: False), \
             patch.object(banks.bank_jobs, 'bump', lambda job, n=1: None), \
             patch.object(banks.bank_jobs, 'progress',
                          lambda job, **kw: seen['details'].append(kw.get('detail'))), \
             patch.object(banks.bank_jobs, 'set_stop_notice',
                          lambda job, **kw: seen['stop'].append(kw)), \
             patch('app.capabilities.bank_scoring_gpu_available', lambda: False):
            banks._score_job(bank_id)(object())
    return seen


def test_the_inference_phase_promises_the_scores_and_admits_the_grouping(app, tmp_path):
    seen = _score_phases(app, tmp_path)
    first = seen['stop'][0]
    assert 'saved when you stop' in first['cost']
    # The one thing a stop really loses, named rather than glossed.
    assert 'style grouping' in first['cost']
    assert 'whole bank at once' in first['cost']
    assert first['wait'].startswith('Stopping —')


def test_the_write_back_names_the_batch_the_stop_has_to_wait_out(app, tmp_path):
    from app.services.image_bank_service import _SCORE_COMMIT_EVERY
    seen = _score_phases(app, tmp_path)
    write_back = [s for s in seen['stop']
                  if s.get('wait') and 'batch' in s['wait']]
    assert write_back, 'the write-back phase published no promise'
    wait = write_back[0]['wait']
    # The number is the REAL commit budget, read from the module — a hard-coded
    # "200" in the sentence would keep promising 200 after someone retunes it.
    assert f'batch of {_SCORE_COMMIT_EVERY} rows' in wait
    cost = write_back[0]['cost']
    assert 'already written stay' in cost
    assert 'needs another full pass' in cost


def test_the_salvage_write_does_not_promise_a_batch_it_will_not_honour(app, tmp_path):
    """The stopped run's write-back is NOT interruptible, and must not pretend.

    When the child is stopped, whatever it computed is written here with the
    cancel flag already set and deliberately ignored — that write IS the rescue
    of an hour of GPU work. Offering "finishing the current batch of 200 rows"
    there would be a promise the user watches fail: the counter runs to the end.
    """
    from app.services import image_bank_service as banks
    with app.app_context():
        seen = []
        with patch.object(banks.bank_jobs, 'progress', lambda job, **kw: None), \
             patch.object(banks.bank_jobs, 'set_stop_notice',
                          lambda job, **kw: seen.append(kw)):
            banks._apply_score_results(object(), {}, {}, interruptible=False)
        assert 'runs to the end' in seen[0]['wait']
        assert 'current batch' not in seen[0]['wait']


def test_the_style_write_says_a_stop_costs_nothing_there(app, tmp_path):
    seen = _score_phases(app, tmp_path)
    grouping = [s for s in seen['stop']
                if s.get('cost') and 'finishes even if you Stop' in s['cost']]
    assert grouping, 'the style-grouping phase published no promise'
    assert 'Nothing already written is lost' in grouping[0]['cost']


def test_the_running_label_no_longer_explains_Stop_to_everyone(app, tmp_path):
    """The Stop sentence used to live INSIDE the style phase's progress detail.

    Everyone read it, at all times, whether or not they were reaching for the
    button — and at 400 px it pushed the pass name and the counter off their
    row. It belongs next to the button, which is where it now is.
    """
    seen = _score_phases(app, tmp_path)
    labels = [d for d in seen['details'] if d]
    assert any(d.startswith('writing the style grouping over') for d in labels)
    for detail in labels:
        assert 'even if you Stop' not in detail, detail


def test_the_promise_is_dropped_before_the_chained_medium_pass(app, tmp_path):
    """🎨 Medium runs inside the SAME job right after the style write.

    It stops on its own terms and has no promise of its own, so the grouping's
    "this step finishes even if you Stop" must not still be on screen under it.
    """
    seen = _score_phases(app, tmp_path)
    assert seen['stop'][-1] == {}, seen['stop'][-1]
