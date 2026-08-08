"""Atomic .npz writes that survive a transient Windows file lock — and that
never leave finished work stranded in a temporary nobody reads.

WHY THIS MODULE EXISTS
----------------------
Five caches in this app are written the same way: build the arrays,
``np.savez_compressed`` into ``<cache>.tmp.npz``, then ``os.replace`` the
temporary over the real file. That shape is correct on POSIX and WRONG on
Windows, in two ways that compound.

1. ``os.replace`` over an OPEN destination.
   On Linux the rename succeeds and the old inode stays alive for whoever still
   holds it. On Windows the call is REFUSED with ``WinError 5`` (access denied)
   or ``WinError 32`` (sharing violation) if anything has the destination open
   at that instant — an antivirus inspecting the freshly written megabytes, a
   backup/sync agent, or a second window of this app reading the same cache. The
   window is a few hundred milliseconds, which is exactly long enough to lose a
   nine-hour pass to it: a real ✨ Score run over a 37 000-image bank died at
   image 1849 with ``PermissionError: [WinError 5]`` on this line.

2. The temporary was the WORST place to fail.
   The work was computed, compressed and written to disk — and then thrown away,
   because nothing in the app ever looked at a ``.tmp.npz`` again. The stranded
   file held MORE entries than the cache it was meant to replace.

So this module owns both halves, once, for every writer:

  * ``save_npz_atomic`` retries the replace with a short backoff, and when it
    still cannot win it raises a message that names the likely cause (another
    program holds the file) instead of "access denied", which sends people to
    check permissions that have nothing to do with it — and says the work is
    safe;
  * ``salvage_orphan_tmp`` runs at the START of a pass and PROMOTES a temporary
    left by an earlier run, after opening and validating it. Never blindly: a
    temporary can also be the stump of a run killed mid-``savez_compressed``, so
    it is promoted only if it reads back cleanly AND holds at least as many
    entries as the cache already in place. Anything else is deleted and said out
    loud. A silently promoted corrupt cache would be far worse than the bug this
    fixes.

THE TEMPORARY NAME IS UNIQUE, ALWAYS
Two passes on the same bank used to share one ``<cache>.tmp.npz`` and could
interleave halves of two different archives into it. Every temporary written
here carries a pid + random token, so concurrent writers cannot collide. Salvage
still recognises the OLD fixed name too — the orphans already sitting on users'
disks were written under it, and those are precisely the ones worth rescuing.
"""
from __future__ import annotations

import os
import secrets
import time

# Attempt 1 is immediate; these are the waits BEFORE each following attempt.
# Eight attempts, ~9.5 s of patience in the worst case.
#
# Sized against what actually holds the file: an on-access antivirus scan of a
# freshly written cache (77 MB on a 37 000-image bank) is the common case and
# clears in a few hundred milliseconds, so the geometric start means the normal
# recovery costs one 100 ms nap and nothing more. The tail is capped at 3.2 s
# rather than doubling forever so the total stays comfortably under the 15 s a
# stopped pass is given to flush and exit — a retry loop that outlived the
# cancel budget would turn a rescue into a hard kill.
REPLACE_DELAYS = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 3.2)

# Windows error codes that mean "someone else holds this file", not "you may not
# touch this file". ERROR_ACCESS_DENIED is what a replace over an open
# destination reports; the sharing/lock violations are the same story told by a
# different layer. A missing file or a bad path is NOT retried.
_BUSY_WINERRORS = (5, 32, 33)

_sleep = time.sleep     # seam: tests replace this to prove the retry without waiting


class NpzReplaceLocked(OSError):
    """The destination stayed held by another process for the whole retry budget.

    Carries ``temporary`` — the path where the finished work is waiting — so the
    caller can say something better than "access denied", and so the next pass
    can pick it up.
    """

    def __init__(self, message, temporary):
        super().__init__(message)
        self.temporary = str(temporary)


def _is_busy(error):
    if isinstance(error, FileNotFoundError):
        return False
    if isinstance(error, PermissionError):
        return True
    return getattr(error, 'winerror', None) in _BUSY_WINERRORS


def locked_message(destination, temporary):
    """The sentence shown when the retries did not win. English, user-facing.

    It names the cause (another program is holding the file) rather than the
    symptom (access denied), because the symptom sends users into folder
    permissions, and it states that nothing was lost — which is now true.
    """
    return (f'Could not update {os.path.basename(str(destination))}: another program '
            f'is holding it open (antivirus, a backup/sync tool, or a second copy of '
            f'this app). Nothing was lost — the new data is in '
            f'{os.path.basename(str(temporary))} and the next pass picks it up '
            f'automatically.')


def temporary_name(destination):
    """A collision-proof temporary next to ``destination``.

    The ``.npz`` suffix is mandatory, not cosmetic: ``np.savez_compressed``
    appends one itself to any other name, and the file we then try to rename
    would not be the file we wrote.
    """
    token = f'{os.getpid()}-{secrets.token_hex(6)}'
    return f'{destination}.{token}.tmp.npz'


def replace_with_retry(temporary, destination):
    """``os.replace`` that survives a transient lock on ``destination``.

    Raises :class:`NpzReplaceLocked` when every attempt was refused by a holder;
    any other OSError (bad path, full disk, missing source) propagates unchanged
    — retrying those would only delay an honest error.
    """
    temporary, destination = str(temporary), str(destination)
    for delay in REPLACE_DELAYS + (None,):
        try:
            os.replace(temporary, destination)
            return
        except OSError as error:
            if not _is_busy(error) or delay is None:
                if _is_busy(error):
                    raise NpzReplaceLocked(
                        locked_message(destination, temporary), temporary) from error
                raise
        _sleep(delay)


def save_npz_atomic(destination, arrays, *, savez=None):
    """Write ``arrays`` (a ``{key: array}`` mapping) to ``destination`` atomically.

    ``arrays`` is a mapping rather than ``**kwargs`` so an array can be named
    ``destination`` or ``savez`` without silently becoming an option.

    On success the temporary is gone. On a lost race it is deliberately LEFT on
    disk — that file is the finished work, and ``salvage_orphan_tmp`` is what
    collects it next time.
    """
    if savez is None:
        import numpy as np
        savez = np.savez_compressed
    temporary = temporary_name(destination)
    savez(temporary, **arrays)
    try:
        replace_with_retry(temporary, destination)
    except NpzReplaceLocked:
        raise                       # keep the temporary: it IS the result
    except BaseException:
        try:
            os.remove(temporary)    # a failure that is not a lock leaves no litter
        except OSError:
            pass
        raise


def orphan_temporaries(destination):
    """Every leftover temporary for ``destination``, newest first.

    Matches both the unique names written here and the single fixed
    ``<cache>.tmp.npz`` older versions used — the orphans worth rescuing today
    all carry the old name.
    """
    destination = str(destination)
    folder = os.path.dirname(destination) or '.'
    prefix = os.path.basename(destination)
    found = []
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    for name in names:
        if not (name.startswith(prefix + '.') and name.endswith('.tmp.npz')):
            continue
        full = os.path.join(folder, name)
        if full == destination:
            continue
        try:
            found.append((os.stat(full).st_mtime_ns, full))
        except OSError:
            continue
    found.sort(reverse=True)
    return [path for _, path in found]


def salvage_orphan_tmp(destination, count_entries, log=None):
    """Promote a usable temporary left by an interrupted run; delete the rest.

    ``count_entries(path)`` must OPEN the archive and return how many entries it
    holds, 0 when it is unusable. Passing the cache's own loader is the point:
    the salvaged file then passes exactly the checks a real read would apply
    (keys, dtypes, row alignment, provenance), and because a loader touches every
    array it also decompresses them — which is what actually detects a file
    truncated mid-write, since an unfinished archive's arrays fail their CRC even
    when the header looked fine.

    A candidate is promoted only when it holds MORE THAN ZERO entries and AT
    LEAST as many as the cache already in place. Fewer entries means the
    temporary predates the live cache (or lost rows to the truncation): promoting
    it would trade a good cache for a worse one.

    When several orphans coexist — the writers that already used unique
    names can leave one per killed run — the NEWEST that passes the check wins
    and every other candidate is deleted. Merging them was rejected: these
    archives carry provenance and row-aligned parallel arrays, so a merge would
    have to re-validate agreement between files written by different runs,
    possibly different models, to gain rows that the next pass recomputes anyway.
    "Newest good one, sweep the rest" is bounded, explainable, and leaves no
    litter to accumulate.

    Returns the number of entries recovered, or 0 when nothing was promoted.
    """
    def say(message):
        if log:
            log(message)

    candidates = orphan_temporaries(destination)
    if not candidates:
        return 0
    try:
        current = count_entries(destination) if os.path.isfile(destination) else 0
    except Exception:                            # noqa: BLE001 — unreadable = worthless
        current = 0
    recovered = 0
    for candidate in candidates:
        count, reason = 0, None
        if recovered:
            reason = 'superseded by a newer recovered file'
        else:
            try:
                count = count_entries(candidate)
            except Exception as error:           # noqa: BLE001
                count, reason = 0, f'unreadable ({error})'
            # The discard happens OUTSIDE that except block on purpose. While it
            # is running, the live exception still holds its traceback, the
            # traceback holds the reader's frames, and those hold the numpy file
            # object open — on Windows that makes os.remove fail, silently
            # leaving behind exactly the litter this is here to clear.
            if reason is None and not count:
                reason = 'unreadable or empty'
            elif reason is None and count < current:
                reason = (f'holds {count} entries, fewer than the {current} '
                          f'already cached')
        if reason:
            _discard(candidate, say, reason)
            continue
        try:
            replace_with_retry(candidate, destination)
        except OSError as error:
            say(f'[cache] left {os.path.basename(candidate)} in place for now: {error}')
            return 0
        recovered = count
        say(f'[cache] recovered {count} entries from an interrupted run '
            f'({os.path.basename(destination)})')
    return recovered


def _discard(path, say, reason):
    try:
        os.remove(path)
    except OSError:
        return
    say(f'[cache] discarded a leftover temporary ({os.path.basename(path)}): {reason}')
