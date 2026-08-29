"""Local collectors — run a command you configured, import what it prints.

WHY THIS EXISTS
---------------
Some galleries cannot be enumerated from a server at all. They are drawn by
JavaScript behind a signed API whose signature rotates, so the image links only
exist in the clear inside a browser that is already logged in and already
looking at the page. No amount of server-side fetching reaches them.

What DOES reach them is a program on this machine — a browser you drive, a
site-specific tool, a script you wrote. This module is the socket such a program
plugs into: you configure a command, the app runs it with a URL, and whatever
images it reports are imported like any scan.

WHAT SHIPS HERE AND WHAT DOES NOT
---------------------------------
The socket ships. No collector does. That is deliberate rather than incidental:
a collector is pinned to one site's markup, it breaks when that site reskins,
and shipping one would make this repository a maintainer of scrapers for
platforms it has nothing to do with. With nothing configured the UI says the
feature is not set up and stops there — which is the honest state, not an error.

TWO SOCKETS, ONE CONTRACT SHAPE
-------------------------------
The IMAGE lane (``collectors.entries``) runs a collector and IMPORTS what it
prints: stdout must be one JSON document of image links. The VIDEO lane
(``video_collectors.entries``) runs a collector that DOWNLOADS ITS OWN FILES —
a video bank points at a live folder, so "importing" is the refresh the folder
already meant, and inventing a second ingestion path would be a second opinion
about what a bank contains (the same reasoning the local video-grab scripts
themselves carry). Its contract is therefore smaller: exit code 0 is the whole
verdict, stdout is drained but ignored, and stderr keeps exactly the same role
— ``@progress`` lines drive the job's bar, everything else is the failure
story. Both lanes substitute ``{url}``; the video lane adds ``{folder}``, the
source folder of the bank the run was started from, so a command can promise
its files land where the refresh will look.

THE CONTRACT
------------
A collector is `{name, command}`. `command` is a LIST of arguments, and the
literal token ``{url}`` is replaced by the URL. It is executed directly — never
through a shell — so a URL can never become a second command no matter what it
contains. On success it prints ONE JSON document to stdout:

    {"items": [{"url": "...", "title": "..."}], "suggested_name": "optional"}

Anything it wants to say to a human goes to stderr, which is captured and shown
when the run fails.
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time

from .. import config as cfg

logger = logging.getLogger(__name__)

#: A browser-driving collector on a large account legitimately takes minutes.
#: Long enough not to cut real work short, short enough that a wedged process
#: cannot hold a bank's job slot for an afternoon.
COLLECTOR_TIMEOUT_S = 30 * 60
#: stdout cap. 20k images of URL+title is far past anything usable and well
#: under what would strain the parser.
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
#: What one run may hand to the importer, mirroring the paste intake's own cap.
MAX_ITEMS = 2000
#: Lines on stderr starting with this carry `{done, total, detail}` for the
#: progress bar. Everything else on stderr is a human message and is kept for
#: the error report. A collector that emits none simply shows no percentage.
PROGRESS_PREFIX = '@progress'


class CollectorError(Exception):
    """A collector could not run, or did not answer the contract. `detail`
    carries the stderr tail when there is one — the collector's own words are
    almost always more useful than anything this module could infer."""

    def __init__(self, message, detail=''):
        super().__init__(message)
        self.detail = (detail or '')[-2000:]


def _configured_entries(path):
    """`[{name, command}]` from one config path, keeping entries that could run.

    A row missing a name or an argument list is dropped rather than reported:
    the settings file is hand-written, and a half-typed entry should not take
    the working ones down with it."""
    raw = cfg.get(path)
    out = []
    for entry in (raw if isinstance(raw, list) else []):
        if not isinstance(entry, dict):
            continue
        name = entry.get('name')
        name = name.strip() if isinstance(name, str) else ''
        command = entry.get('command')
        if not name or not isinstance(command, list) or not command:
            continue
        argv = [str(a) for a in command if isinstance(a, (str, int, float))]
        if len(argv) != len(command) or not argv[0].strip():
            continue
        out.append({'name': name, 'command': argv})
    return out


def configured_collectors():
    """`[{name, command}]` for the IMAGE lane.

    The config path `collectors.entries`, and both halves of it, were forced by
    a test. A top-level LIST fails every full-config save (the settings API
    requires sections to be objects, per test_settings_api), and living inside
    `bank` breaks the assertion that that section holds exactly the twelve
    thresholds the Bank panel exposes — which this is not, and must not
    become."""
    return _configured_entries('collectors.entries')


def configured_video_collectors():
    """`[{name, command}]` for the VIDEO lane (`video_collectors.entries`)."""
    return _configured_entries('video_collectors.entries')


def find_collector(name):
    """The configured IMAGE-lane collector called `name`, or None."""
    wanted = (name or '').strip()
    for c in configured_collectors():
        if c['name'] == wanted:
            return c
    return None


def find_video_collector(name):
    """The configured VIDEO-lane collector called `name`, or None."""
    wanted = (name or '').strip()
    for c in configured_video_collectors():
        if c['name'] == wanted:
            return c
    return None


def _normalise(payload):
    """The collector's JSON -> `([{url, title}], suggested_name)`."""
    if not isinstance(payload, dict):
        raise CollectorError('the collector did not print a JSON object')
    raw_items = payload.get('items')
    if not isinstance(raw_items, list):
        raise CollectorError('the collector printed no "items" list')
    items = []
    for entry in raw_items:
        if not isinstance(entry, dict):
            continue
        url = entry.get('url')
        url = url.strip() if isinstance(url, str) else ''
        if not url.lower().startswith(('http://', 'https://')):
            continue
        title = entry.get('title')
        title = title.strip()[:200] if isinstance(title, str) and title.strip() else 'Collected'
        items.append({'url': url, 'title': title})
        if len(items) >= MAX_ITEMS:
            break
    if not items:
        raise CollectorError('the collector reported no usable image links')
    name = payload.get('suggested_name')
    return items, (name.strip()[:120] if isinstance(name, str) and name.strip() else '')


def _spawn(argv, *, on_progress=None):
    """Run argv to completion and return `(returncode, stdout_bytes, stderr_tail)`.

    The one piece both lanes need IDENTICALLY, and the one that is easy to get
    wrong: stderr is drained by a THREAD while stdout is read here. Both must
    be consumed concurrently — a collector that fills one pipe's buffer while
    the reader waits on the other deadlocks, and a browser-driving collector
    writes a progress line per post for minutes. ``@progress`` lines are
    forwarded to `on_progress` and never kept; the human-readable tail is.
    The URL (and any path) is substituted into an ARGUMENT by the caller, and
    the process is started without a shell, so nothing in it can become a
    second command."""
    if on_progress:
        on_progress(0, 0, 'running …')
    try:
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                shell=False, cwd=str(cfg.BACKEND_DIR.parent))
    except FileNotFoundError as e:
        raise CollectorError(f'collector command not found: {argv[0]}') from e
    except OSError as e:
        raise CollectorError(f'the collector could not be started: {e}') from e

    tail: list[str] = []

    def _drain():
        for raw_line in proc.stderr:
            line = raw_line.decode('utf-8', 'replace').rstrip('\r\n')
            if line.startswith(PROGRESS_PREFIX):
                if on_progress:
                    try:
                        p = json.loads(line[len(PROGRESS_PREFIX):].strip())
                        on_progress(int(p.get('done') or 0), int(p.get('total') or 0),
                                    str(p.get('detail') or '')[:200])
                    except (ValueError, TypeError):
                        pass          # a malformed progress line is not a failure
                continue
            tail.append(line)
            del tail[:-40]

    reader = threading.Thread(target=_drain, daemon=True, name='collector-stderr')
    reader.start()
    try:
        raw = proc.stdout.read()
        proc.wait(timeout=COLLECTOR_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        proc.kill()
        raise CollectorError(
            f'the collector did not finish within {COLLECTOR_TIMEOUT_S // 60} minutes') from e
    finally:
        reader.join(timeout=5)
    return proc.returncode, raw, '\n'.join(tail)


def _substitute(command, url, folder=None):
    """The command list with `{url}` (and `{folder}`) swapped for real values.

    A missing `{url}` token hands the URL over as the LAST argument — a command
    that takes it positionally need not name a token it cannot put last."""
    argv = []
    for a in command:
        if a == '{url}':
            argv.append(url)
        elif a == '{folder}' and folder is not None:
            argv.append(folder)
        else:
            argv.append(a)
    if url not in argv:
        argv = argv + [url]
    return argv


def run_collector(name, url, *, on_progress=None):
    """Run the named IMAGE-lane collector against `url` and return
    `(items, suggested_name)`.

    Raises CollectorError for every failure — missing collector, non-zero exit,
    timeout, unparseable output — with the collector's stderr attached, because
    the thing that knows why it failed is the collector, not this."""
    collector = find_collector(name)
    if collector is None:
        raise CollectorError(f'no collector named {name!r} is configured')
    target = (url or '').strip()
    if not target.lower().startswith(('http://', 'https://')):
        raise CollectorError('a http(s) URL is required')

    started = time.time()
    code, raw, stderr = _spawn(_substitute(collector['command'], target),
                               on_progress=on_progress)
    if code != 0:
        raise CollectorError(f'the collector exited with code {code}', stderr)
    if len(raw) > MAX_OUTPUT_BYTES:
        raise CollectorError('the collector printed more than this can read')
    try:
        payload = json.loads(raw.decode('utf-8', 'replace'))
    except ValueError as e:
        raise CollectorError(
            'the collector did not print JSON on stdout — progress messages '
            'belong on stderr', stderr) from e

    items, suggested = _normalise(payload)
    logger.info('collector %r produced %d image(s) for %s in %.0fs',
                collector['name'], len(items), _safe(target), time.time() - started)
    return items, suggested


def run_video_collector(name, url, *, folder=None, on_progress=None):
    """Run the named VIDEO-lane collector; return when it has succeeded.

    The smaller contract, and why it is smaller: a video collector downloads
    its own files (typically into `folder`, the bank's source folder the
    command reaches through ``{folder}``), so there is nothing to parse — exit
    code 0 is the whole verdict, stdout is drained but unread, and the files
    become visible to the app through the REFRESH THE CALLER OWES, not here.
    Everything else — missing collector, bad URL, non-zero exit, timeout — is
    a CollectorError with the collector's own stderr attached, same as the
    image lane."""
    collector = find_video_collector(name)
    if collector is None:
        raise CollectorError(f'no collector named {name!r} is configured')
    target = (url or '').strip()
    if not target.lower().startswith(('http://', 'https://')):
        raise CollectorError('a http(s) URL is required')

    argv = _substitute(collector['command'], target, folder=folder)
    code, _raw, stderr = _spawn(argv, on_progress=on_progress)
    if code != 0:
        raise CollectorError(f'the collector exited with code {code}', stderr)


def _safe(url):
    """A URL shortened for the log: host + path only, never the query — a signed
    link carries its credential there (diagnostics must stay paste-safe)."""
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(url)
        return f'{parts.netloc}{parts.path}'
    except ValueError:
        return '<unparseable url>'
