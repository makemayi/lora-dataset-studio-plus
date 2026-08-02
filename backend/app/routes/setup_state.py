"""Setup state API: has this install already been checked out, and is it still OK?

Deliberately a separate blueprint from routes/setup.py — that one owns the
wizard's detection and the one-click installers, this one owns the "we already
did this, don't make the user watch it again" memory.
"""
from flask import Blueprint, jsonify, request

from .. import capabilities
from .. import setup_state

bp = Blueprint('setup_state', __name__, url_prefix='/api/setup-state')


def _payload(state, caps, regressions):
    return {
        'verified': state['verified'],
        'verified_at': state['verified_at'],
        'checks': state['checks'],
        'regressions': regressions,
        'capabilities': caps,
    }


@bp.get('')
def get_state():
    """Whether the app may skip the onboarding redirect and re-check in the
    background instead.

    Uses the CACHED probe: this is read on every page load, right after
    /api/capabilities has already warmed it, so it costs nothing. The honest
    re-check is the POST below.
    """
    caps = capabilities.probe()
    state = setup_state.observe(caps)
    return jsonify(_payload(state, caps, setup_state.compare(caps, state)))


@bp.post('/recheck')
def recheck():
    """The background re-verification: the SAME full probe the Setup wizard runs
    (force=True, no cache), compared against what this install has proven it can
    do. `regressions` non-empty is the only thing worth interrupting the user
    for — everything else is a discreet "checked, all good"."""
    caps = capabilities.probe(force=True)
    state = setup_state.observe(caps)
    return jsonify(_payload(state, caps, setup_state.compare(caps, state)))


@bp.post('/dismiss')
def dismiss():
    """"I removed that on purpose" — stop reporting the named capabilities as
    regressions. Unknown keys are ignored rather than rejected: a stale tab
    dismissing a key this build no longer tracks should be a no-op, not a 400."""
    body = request.get_json(silent=True) or {}
    keys = body.get('keys')
    if not isinstance(keys, list):
        return jsonify({'error': 'keys must be a list'}), 400
    state = setup_state.dismiss([k for k in keys if isinstance(k, str)])
    caps = capabilities.probe()
    return jsonify(_payload(state, caps, setup_state.compare(caps, state)))
