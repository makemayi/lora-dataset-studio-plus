"""Two pytest runs must never share one ``--basetemp``.

The whole rationale is in ``_basetemp_guard``: pytest deletes the basetemp it is
given, so a shared one makes concurrent runs wipe each other's tmp_path trees.
The damage reads as randomness — a failure over here, a handful of "ERROR at
setup" over there, everything green when re-run alone — which is how it got
filed as a flake twice instead of being fixed once.

These tests pin the guard itself, with no subprocess and no real basetemp: the
collision is decided by the claim file, so that is what is exercised.
"""
import pytest

import _basetemp_guard as guard


def test_a_free_basetemp_is_claimed_without_complaint(tmp_path):
    assert guard.claim(tmp_path / 'bt', pid=1000, now=100.0) is None
    assert guard.claim_path(tmp_path / 'bt').exists()


def test_a_basetemp_held_by_a_live_run_is_refused(tmp_path):
    """THE case. Without this the second run silently rm_rf's the first one's
    files and both suites report failures that belong to neither."""
    bt = tmp_path / 'bt'
    assert guard.claim(bt, pid=1000, now=100.0) is None
    message = guard.claim(bt, pid=2000, now=130.0)
    assert message, 'a second live run was allowed to share the basetemp'
    # The message has to be actionable on its own: which run holds it, and what
    # to do about it. A bare "in use" would just move the mystery.
    assert '1000' in message
    assert '--basetemp' in message
    assert str(guard.claim_path(bt)) in message


def test_the_same_run_may_configure_twice(tmp_path):
    bt = tmp_path / 'bt'
    assert guard.claim(bt, pid=1000, now=100.0) is None
    assert guard.claim(bt, pid=1000, now=101.0) is None


def test_an_abandoned_claim_is_taken_over(tmp_path):
    """A killed run leaves its claim behind; it must not lock the path forever."""
    bt = tmp_path / 'bt'
    assert guard.claim(bt, pid=1000, now=100.0) is None
    later = 100.0 + guard.STALE_AFTER_SECONDS + 1
    assert guard.claim(bt, pid=2000, now=later) is None


def test_release_only_drops_our_own_claim(tmp_path):
    bt = tmp_path / 'bt'
    guard.claim(bt, pid=1000, now=100.0)
    assert guard.release(bt, pid=2000) is False
    assert guard.claim_path(bt).exists()
    assert guard.release(bt, pid=1000) is True
    assert not guard.claim_path(bt).exists()


def test_the_claim_is_a_sibling_of_the_basetemp_never_a_child(tmp_path):
    """pytest rm_rf's the basetemp itself, so a marker inside it would be the
    first casualty of the very collision it exists to prevent."""
    bt = tmp_path / 'bt'
    claim = guard.claim_path(bt)
    assert claim.parent == bt.parent
    assert bt not in claim.parents


@pytest.mark.parametrize('junk', ['', 'not-a-pid', '1000'])
def test_an_unreadable_claim_is_treated_as_abandoned(tmp_path, junk):
    """A truncated or half-written file must not wedge every future run."""
    bt = tmp_path / 'bt'
    guard.claim_path(bt).write_text(junk, encoding='utf-8')
    assert guard.claim(bt, pid=2000, now=100.0) is None
