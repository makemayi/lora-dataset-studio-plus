"""In-memory per-dataset reference-EDIT job registry + candidate file lifecycle.

Editing the reference photo is a SLOW (1-3 min) and PAID API call. Running it
inside the client's fetch made it die when a mobile tab was backgrounded ("Failed
to fetch") — and the paid result was lost. So the edit now runs in a server
background thread, exactly like the variation batch: the request returns at once,
the worker fills a CANDIDATE, and the client rediscovers it through the dataset
payload — which survives a tab sleep AND a reload.

This registry holds one pending BATCH per dataset. Its ordered engine list owns
one candidate state per engine, while one activity token covers the whole batch.

It is IN-MEMORY only (like dataset_activity) — a job dies with the process, which
is correct for a transient candidate awaiting Keep. The candidate FILE on disk is
reclaimed by a TTL sweep so a crashed process never leaks it.

TWO LANES, ONE REGISTRY. An API edit is a blocking call made from a service
thread, which fills its candidate state. A LOCAL edit (Klein / Krea 2 Edit) is a
ComfyUI job: it is enqueued on the app's existing image queue and the queue
worker calls back when it lands — the same waiting path every local generation
already uses, not a second one. `_job_id` is what lets that callback find its way
home (``find_by_job``), and `_act_token` is the dataset-activity token the
callback must close (an API worker closes its own in a ``finally``; a local one
has no thread left to do it).

Leak-proofing (what replaces the old client-held "browser GC"):
  * Keep      -> promote one named candidate, delete every sibling, clear.
  * Discard   -> delete every candidate file and clear the batch.
  * abandoned -> TTL purge: lazy on get() (deletes the file) + a disk sweep on the
                 next start() (catches a file orphaned by a process crash).
A compare-and-set on ``token`` means a SUPERSEDED worker (a second edit started
before the first returned) never overwrites the newer job — set_ready/set_failed
return False and the stale worker deletes its own candidate instead.

Disk ownership: this module owns the candidate files end to end (create is done by
the service worker, delete lives here), so the lifecycle stays in one place.
"""
from contextlib import contextmanager
import itertools
import logging
import os
import secrets
import threading
import time

logger = logging.getLogger(__name__)

# Distinct filename marker: DB-driven enumeration and the backup selector never
# pick these up (they list ref fields + image rows, never scan the dir), so a
# candidate can sit in the dataset dir without polluting the grid or a backup.
CANDIDATE_MARKER = '_datasetrefeditcand_'
_TTL_SECONDS = 30 * 60

_lock = threading.Lock()
_jobs: dict = {}                 # dataset_id -> entry dict
_revisions: dict = {}            # dataset_id -> reference mutation epoch
_counter = itertools.count(1)
_mutation_locks_guard = threading.Lock()
_mutation_locks: dict = {}        # dataset_id -> re-entrant primary-reference lock


@contextmanager
def reference_mutation(dataset_id):
    """Serialize physical reference mutations and Keep for one dataset.

    Lock order is deliberately one-way: this per-dataset lock MAY call registry
    helpers that acquire ``_lock``; code holding ``_lock`` must never wait for a
    mutation lock.  RLock permits the public mutation helpers to enforce the
    invariant themselves while callers hold it across a larger transaction.
    Locks stay process-lifetime: removing a lock while another thread is waiting
    on it would let two different locks protect the same dataset.
    """
    with _mutation_locks_guard:
        mutation_lock = _mutation_locks.setdefault(dataset_id, threading.RLock())
    with mutation_lock:
        yield


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def _candidate_public(candidate):
    return {
        'engine': candidate['engine'],
        'status': candidate['status'],
        'candidate_filename': candidate.get('candidate_filename'),
        'error': candidate.get('error'),
        'started_at': candidate['started_at'],
    }


def _public(entry):
    """Client-facing batch shape, including exact one-engine compatibility."""
    candidates = {
        engine: _candidate_public(entry['candidates'][engine])
        for engine in entry['engines']
    }
    selected = list(entry['engines'])
    states = [candidate['status'] for candidate in candidates.values()]
    status = ('running' if 'running' in states
              else 'ready' if 'ready' in states else 'failed')
    if len(selected) == 1:
        legacy = candidates[selected[0]]
        engine = legacy['engine']
        candidate_filename = legacy['candidate_filename']
        error = legacy['error']
    else:
        # Partial success is usable. Old clients never create multi-engine jobs,
        # but one unambiguous ready result is still a sensible aggregate.
        ready = [candidate for candidate in candidates.values()
                 if candidate['status'] == 'ready']
        engine = ready[0]['engine'] if len(ready) == 1 else None
        candidate_filename = (ready[0]['candidate_filename']
                              if len(ready) == 1 else None)
        errors = [candidate['error'] for candidate in candidates.values()
                  if candidate['error']]
        error = ('; '.join(errors)[:500]
                 if status == 'failed' and errors else None)
    return {
        'status': status,
        'batch_id': entry['batch_id'],
        'engine': engine,
        'engines': selected,
        'prompt': entry.get('prompt'),
        'candidate_filename': candidate_filename,
        'error': error,
        'started_at': entry['started_at'],
        'candidates': candidates,
    }


def _candidate_for_token(entry, token):
    for candidate in entry['candidates'].values():
        if candidate['token'] == token:
            return candidate
    return None


def _cancel_queue_job(job_id):
    if not job_id:
        return
    try:
        from ..job_queue import queue_manager
        queue_manager.cancel_job(job_id)
    except Exception:
        logger.warning('reference edit: could not cancel queue job %s',
                       job_id, exc_info=True)


def _cleanup_entry(entry, dsdir=None):
    """Release every resource owned by a detached batch. Never raises."""
    directory = dsdir or entry.get('_dir')
    for candidate in entry.get('candidates', {}).values():
        if directory and candidate.get('candidate_filename'):
            _unlink(os.path.join(directory, candidate['candidate_filename']))
        if candidate.get('status') == 'running':
            _cancel_queue_job(candidate.get('_job_id'))
    if entry.get('_act_token') is not None and not entry.get('_activity_closed'):
        entry['_activity_closed'] = True
        try:
            from . import dataset_activity
            dataset_activity.end(entry['_act_token'])
        except Exception:
            logger.warning('reference edit: could not close dataset activity',
                           exc_info=True)


def reference_revision(dataset_id):
    """Current mutation epoch for the dataset reference."""
    with _lock:
        return _revisions.get(dataset_id, 0)


def start_batch(dataset_id, dsdir, engines, prompt, act_token=None,
                expected_revision=None, expected_batch_id=None):
    """Register one ordered multi-engine batch and supersede the old batch.

    Returns a batch token plus one compare-and-set worker token per engine.
    Service validation owns the editable allow-list; duplicates are removed here
    defensively without changing the user's selection order.
    """
    selected = tuple(dict.fromkeys(str(engine) for engine in engines))
    if not selected:
        raise ValueError('at least one reference-edit engine is required')
    now = time.time()
    batch_token = next(_counter)
    batch_id = secrets.token_urlsafe(18)
    tokens = {engine: next(_counter) for engine in selected}
    entry = {
        'batch_id': batch_id,
        'prompt': str(prompt),
        'engines': selected,
        'candidates': {
            engine: {
                'engine': engine,
                'status': 'running',
                'candidate_filename': None,
                'error': None,
                'token': tokens[engine],
                'started_at': now,
                '_job_id': None,
                '_user_id': None,
                '_api_dispatch_claimed': False,
            }
            for engine in selected
        },
        'batch_token': batch_token,
        'started_at': now,
        '_touched': now,
        '_dir': dsdir,
        '_act_token': act_token,
        '_activity_closed': False,
        '_keep_claim': None,
    }
    with _lock:
        current_revision = _revisions.get(dataset_id, 0)
        if (expected_revision is not None
                and expected_revision != current_revision):
            raise RuntimeError(
                'the reference changed while the edit was starting; retry the edit')
        previous = _jobs.get(dataset_id)
        if expected_batch_id is not None:
            supplied_batch_id = str(expected_batch_id)
            try:
                matches_expected = bool(previous) and secrets.compare_digest(
                    previous['batch_id'], supplied_batch_id)
            except TypeError:
                matches_expected = False
            if not matches_expected:
                # Retry is a potentially paid action. Refuse it atomically if a
                # second tab replaced the batch after the first tab last polled.
                raise RuntimeError(
                    'this reference edit was replaced; refresh before retrying')
        if previous and previous.get('_keep_claim') is not None:
            raise RuntimeError(
                'the edited reference is being kept; try again in a moment')
        entry['_reference_revision'] = current_revision
        _jobs[dataset_id] = entry
    # Queue cancellation can trigger callbacks, so cleanup stays outside the lock.
    if previous:
        _cleanup_entry(previous)
    sweep(dsdir)
    return {'batch_token': batch_token, 'batch_id': batch_id, 'tokens': tokens}


def start(dataset_id, dsdir, engine, prompt):
    """Backward-compatible one-engine registration."""
    started = start_batch(dataset_id, dsdir, (engine,), prompt)
    return started['tokens'][str(engine)]


def attach_activity(dataset_id, batch_token, act_token):
    """Attach the one shared activity token to a newly registered batch."""
    with _lock:
        entry = _jobs.get(dataset_id)
        if not entry or entry['batch_token'] != batch_token:
            return False
        entry['_act_token'] = act_token
        entry['_touched'] = time.time()
        return True


def attach_job(dataset_id, token, job_id, act_token=None, user_id=None):
    """Record the ComfyUI queue job (and the dataset-activity token) a LOCAL edit
    is waiting on. Returns False when the job was already superseded between the
    enqueue and this call — the caller then cancels the job it just queued rather
    than leaving a render nobody is waiting for.

    ``user_id`` rides along only so the completion callback can name the candidate
    file the way the API worker does: it runs in the queue thread, with no request
    and no idea who owns the dataset."""
    with _lock:
        entry = _jobs.get(dataset_id)
        candidate = _candidate_for_token(entry, token) if entry else None
        if candidate is None or candidate['status'] != 'running':
            return False
        candidate['_job_id'] = str(job_id)
        candidate['_user_id'] = user_id
        if act_token is not None and entry.get('_act_token') is None:
            entry['_act_token'] = act_token
        entry['_touched'] = time.time()
        return True


def claim_api_dispatch(dataset_id, token):
    """CAS-claim the one irreversible provider dispatch for an API candidate."""
    with _lock:
        entry = _jobs.get(dataset_id)
        candidate = _candidate_for_token(entry, token) if entry else None
        if (candidate is None or candidate['status'] != 'running'
                or candidate.get('_api_dispatch_claimed')
                or entry.get('_reference_revision')
                != _revisions.get(dataset_id, 0)):
            return False
        candidate['_api_dispatch_claimed'] = True
        entry['_touched'] = time.time()
        return True


def find_by_job(job_id):
    """Resolve a queue job_id back to the edit waiting on it:
    {dataset_id, token, dir, act_token, engine}, or None when nothing is waiting
    (discarded, superseded, or purged by the TTL while ComfyUI rendered).
    Deliberately does NOT touch the TTL — a job that just landed is proof the
    user is still owed an answer, so ``get`` bumping it is the caller's job."""
    key = str(job_id)
    with _lock:
        for dataset_id, entry in _jobs.items():
            for candidate in entry['candidates'].values():
                if candidate.get('_job_id') == key:
                    entry['_touched'] = time.time()
                    return {
                        'dataset_id': dataset_id,
                        'batch_token': entry['batch_token'],
                        'token': candidate['token'],
                        'dir': entry.get('_dir'),
                        'act_token': entry.get('_act_token'),
                        'engine': candidate['engine'],
                        'user_id': candidate.get('_user_id'),
                    }
    return None


def set_ready(dataset_id, token, candidate_filename):
    """Mark one engine ready. False means stale or already terminal."""
    with _lock:
        entry = _jobs.get(dataset_id)
        candidate = _candidate_for_token(entry, token) if entry else None
        if candidate is None or candidate['status'] != 'running':
            return False
        candidate['status'] = 'ready'
        candidate['candidate_filename'] = candidate_filename
        candidate['error'] = None
        entry['_touched'] = time.time()
        return True


def set_failed(dataset_id, token, error):
    """Mark one engine failed without affecting its siblings."""
    with _lock:
        entry = _jobs.get(dataset_id)
        candidate = _candidate_for_token(entry, token) if entry else None
        if candidate is None or candidate['status'] != 'running':
            return False
        candidate['status'] = 'failed'
        candidate['error'] = str(error)[:500]
        entry['_touched'] = time.time()
        return True


def activity_update(dataset_id, token):
    """Return shared progress and atomically claim its one final end call."""
    with _lock:
        entry = _jobs.get(dataset_id)
        candidate = _candidate_for_token(entry, token) if entry else None
        if candidate is None:
            return None
        total = len(entry['candidates'])
        done = sum(item['status'] != 'running'
                   for item in entry['candidates'].values())
        act_token = entry.get('_act_token')
        managed = act_token is not None
        progress_token = None
        end_token = None
        if managed and not entry.get('_activity_closed'):
            progress_token = act_token
            if done == total:
                entry['_activity_closed'] = True
                end_token = act_token
        entry['_touched'] = time.time()
        return {
            'managed': managed,
            'activity_token': progress_token,
            'end_token': end_token,
            'done': done,
            'total': total,
        }


def get(dataset_id):
    """Public batch, lazily purging all resources after the TTL."""
    now = time.time()
    with _lock:
        entry = _jobs.get(dataset_id)
        if not entry:
            return None
        if now - entry['_touched'] <= _TTL_SECONDS:
            return _public(entry)
        _jobs.pop(dataset_id, None)
        expired = entry
    _cleanup_entry(expired)
    return None


def peek(dataset_id):
    """Like get() but WITHOUT the TTL purge — used by keep/discard which must read
    the candidate even if it just aged past the TTL boundary (the user did claim
    it). None when there is no entry."""
    with _lock:
        entry = _jobs.get(dataset_id)
        return _public(entry) if entry else None


def pending_jobs(dataset_id):
    """Every currently running local queue job in the batch."""
    with _lock:
        entry = _jobs.get(dataset_id)
        if not entry:
            return []
        return [
            {
                'job_id': candidate.get('_job_id'),
                'act_token': entry.get('_act_token'),
                'engine': candidate['engine'],
            }
            for candidate in entry['candidates'].values()
            if candidate['status'] == 'running' and candidate.get('_job_id')
        ]


def pending_job(dataset_id):
    """(queue job_id, activity token) of the edit currently registered, both None
    when there is none or it is an API edit. Read WITHOUT dropping the entry, so
    a caller about to supersede can cancel the outgoing render first."""
    jobs = pending_jobs(dataset_id)
    if not jobs:
        return None, None
    return jobs[0]['job_id'], jobs[0]['act_token']


def claim_ready(dataset_id, engine=None, batch_id=None):
    """Atomically validate and reserve one current ready candidate for Keep."""
    requested = str(engine).strip().lower() if engine is not None else None
    with _lock:
        entry = _jobs.get(dataset_id)
        if (not entry or entry.get('_keep_claim') is not None
                or entry.get('_reference_revision')
                != _revisions.get(dataset_id, 0)):
            return None
        if batch_id is None:
            # Even a one-engine stale tab must not be able to promote a newer
            # batch merely because its engine name happens to match.
            return None
        supplied_batch_id = str(batch_id)
        try:
            matches_batch = secrets.compare_digest(
                entry['batch_id'], supplied_batch_id)
        except TypeError:  # non-ASCII strings are not valid opaque IDs
            return None
        if not matches_batch:
            return None
        if requested:
            target = requested
        elif len(entry['engines']) == 1:
            target = entry['engines'][0]
        else:
            ready = [name for name, candidate in entry['candidates'].items()
                     if candidate['status'] == 'ready']
            target = ready[0] if len(ready) == 1 else None
        candidate = entry['candidates'].get(target) if target else None
        if (candidate is None or candidate['status'] != 'ready'
                or not candidate.get('candidate_filename')):
            return None
        claim_token = next(_counter)
        entry['_keep_claim'] = (
            claim_token, target, entry['_reference_revision'])
        entry['_touched'] = time.time()
        return {
            'batch_token': entry['batch_token'],
            'claim_token': claim_token,
            'batch_id': entry['batch_id'],
            'reference_revision': entry['_reference_revision'],
            'engine': target,
            'candidate_filename': candidate['candidate_filename'],
            'dir': entry.get('_dir'),
        }


def release_claim(dataset_id, batch_token, claim_token):
    """Release a Keep reservation after a failed promotion."""
    with _lock:
        entry = _jobs.get(dataset_id)
        if (not entry or entry['batch_token'] != batch_token
                or not entry.get('_keep_claim')
                or entry['_keep_claim'][0] != claim_token):
            return False
        entry['_keep_claim'] = None
        entry['_touched'] = time.time()
        return True


def clear_claimed(dataset_id, batch_token, claim_token, dsdir=None,
                  reference_mutated=False):
    """Drop a promoted batch iff the caller still owns its Keep claim."""
    with _lock:
        entry = _jobs.get(dataset_id)
        if (not entry or entry['batch_token'] != batch_token
                or not entry.get('_keep_claim')
                or entry['_keep_claim'][0] != claim_token
                or entry['_keep_claim'][2]
                != _revisions.get(dataset_id, 0)):
            return None
        if reference_mutated:
            _revisions[dataset_id] = _revisions.get(dataset_id, 0) + 1
        _jobs.pop(dataset_id, None)
    _cleanup_entry(entry, dsdir)
    return entry


def clear_batch(dataset_id, batch_token, dsdir=None):
    """CAS-clear one known batch without ever deleting a newer replacement."""
    with _lock:
        entry = _jobs.get(dataset_id)
        if (not entry or entry['batch_token'] != batch_token
                or entry.get('_keep_claim') is not None):
            return None
        _jobs.pop(dataset_id, None)
    _cleanup_entry(entry, dsdir)
    return entry


def invalidate(dataset_id, dsdir=None):
    """Atomically advance the reference epoch and detach its stale edit batch."""
    with reference_mutation(dataset_id):
        with _lock:
            _revisions[dataset_id] = _revisions.get(dataset_id, 0) + 1
            entry = _jobs.pop(dataset_id, None)
            revision = _revisions[dataset_id]
        if entry:
            _cleanup_entry(entry, dsdir)
        return revision


def clear(dataset_id, dsdir=None):
    """Drop the entry and (when ``dsdir`` is given) delete its candidate file.
    Idempotent — used by Keep (after commit), Discard, and reference-mutation
    invalidation (crop/recrop/change).

    RETURNS the dropped entry (or None). A local edit abandoned while ComfyUI is
    still rendering leaves a queue job and a dataset-activity token behind, and
    the caller is the only one that can cancel/close them — dropping the entry
    silently is how the ✦ badge used to sit there until the TTL."""
    with _lock:
        entry = _jobs.get(dataset_id)
        if not entry or entry.get('_keep_claim') is not None:
            return None
        _jobs.pop(dataset_id, None)
    _cleanup_entry(entry, dsdir)
    return entry


def sweep(dsdir):
    """Delete candidate files in ``dsdir`` older than the TTL. Catches a file left
    behind by a process that crashed mid-edit (no in-memory entry survives a
    restart, so only a disk scan can reclaim it). Never raises."""
    now = time.time()
    try:
        names = os.listdir(dsdir)
    except OSError:
        return
    for n in names:
        if CANDIDATE_MARKER in n:
            p = os.path.join(dsdir, n)
            try:
                if now - os.path.getmtime(p) > _TTL_SECONDS:
                    os.remove(p)
            except OSError:
                pass


def reset():
    """Test helper: clear the whole registry (does NOT touch files)."""
    with _lock:
        _jobs.clear()
        _revisions.clear()
