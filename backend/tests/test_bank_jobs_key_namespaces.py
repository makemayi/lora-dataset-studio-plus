"""The job registry is SHARED by two lanes that key it differently.

The image lane keys on a Bank's integer id. The video lane deliberately keys on
the string ``'video:<id>'`` (see ``video_bank_service.job_key``) so that video
bank 1 and image bank 1 can never occupy the same slot — two lanes, one
registry, ids that overlap by construction.

Nothing held those two facts together. When the registry grew reservations it
was written for integer ids and coerced with ``int(bank_id)``, which raises
``ValueError`` on ``'video:1'`` — every pass in the video lane answered 400
instead of 202, from the probe onwards. It surfaced in the video lane's route
tests, i.e. as a symptom, and only because that lane happens to have route tests
at all. This file pins the cause: a key the registry is given is a key it keeps.

WHY ``reserve()`` AND NOT ``start()``. ``start`` spawns a daemon thread that
keeps touching this module-level registry after the test that launched it has
returned — so a test file built on it pollutes whatever runs next, and the first
draft of this one did exactly that (it made an unrelated reservation test fail,
which cost a bisect to attribute). ``reserve`` is the same atomic slot-taking
without the thread, and slot-taking is the whole subject here.
"""
import pytest

from app.services import bank_jobs


@pytest.fixture(autouse=True)
def _clean_registry():
    """No thread is running here (see the module docstring), so clearing is
    exact rather than a race with a worker still writing back."""
    bank_jobs._jobs.clear()
    yield
    bank_jobs._jobs.clear()


def test_a_namespaced_string_key_is_accepted_and_kept():
    """`int('video:1')` raises, and the whole video lane answered 400 on it."""
    job = bank_jobs.reserve('video:1', 'detect')
    assert job['kind'] == 'detect'
    assert bank_jobs._jobs.get('video:1') is job


def test_the_two_lanes_do_not_share_a_slot_for_the_same_id():
    """The reason the video key is a string at all. Collapsed to one slot, a
    running image pass would refuse a video pass on an unrelated bank — and the
    refusal would name a pass the user cannot see."""
    bank_jobs.reserve(1, 'score')
    video = bank_jobs.reserve('video:1', 'detect')      # must NOT raise
    assert bank_jobs._jobs[1]['kind'] == 'score'
    assert bank_jobs._jobs['video:1'] is video


def test_a_numeric_id_still_collapses_to_one_slot_whatever_its_type():
    """The coercion is load-bearing and must survive the fix: an id crossing a
    JSON boundary arrives as 7 or '7', and two slots would let a second pass
    slip past the serialization this registry exists to provide."""
    bank_jobs.reserve(7, 'score')
    with pytest.raises(bank_jobs.BankJobBusy):
        bank_jobs.reserve('7', 'caption')


def test_a_busy_video_bank_still_refuses_its_own_second_pass():
    """Namespacing must not cost the serialization it exists to preserve."""
    bank_jobs.reserve('video:2', 'detect')
    with pytest.raises(bank_jobs.BankJobBusy) as caught:
        bank_jobs.reserve('video:2', 'measure')
    # The refusal names the pass that HOLDS the bank, which is what the 409 to
    # the UI is built from.
    assert caught.value.kind == 'detect'
