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


class CollectorError(Exception):
    """A collector could not run, or did not answer the contract. `detail`
    carries the stderr tail when there is one — the collector's own words are
    almost always more useful than anything this module could infer."""

    def __init__(self, message, detail=''):
        super().__init__(message)
        self.detail = (detail or '')[-2000:]


def configured_collectors():
    """`[{name, command}]` from config, keeping only entries that could run.

    A row missing a name or an argument list is dropped rather than reported:
    the settings file is hand-written, and a half-typed entry should not take
    the working ones down with it."""
    # `bank.collectors`, not a top-level `collectors`: the settings API requires
    # every top-level section to be an object, and a bare list there fails every
    # full-config save — caught by test_settings_api the first time it was tried.
    raw = cfg.get('bank.collectors')
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


def find_collector(name):
    """The configured collector called `name`, or None."""
    wanted = (name or '').strip()
    for c in configured_collectors():
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


def run_collector(name, url, *, on_progress=None):
    """Run the named collector against `url` and return `(items, suggested_name)`.

    Raises CollectorError for every failure — missing collector, non-zero exit,
    timeout, unparseable output — with the collector's stderr attached, because
    the thing that knows why it failed is the collector, not this."""
    collector = find_collector(name)
    if collector is None:
        raise CollectorError(f'no collector named {name!r} is configured')
    target = (url or '').strip()
    if not target.lower().startswith(('http://', 'https://')):
        raise CollectorError('a http(s) URL is required')

    # The URL is substituted into an ARGUMENT, and the process is started
    # without a shell. Nothing the URL contains can become a second command.
    argv = [target if a == '{url}' else a for a in collector['command']]
    if target not in argv:
        argv = argv + [target]      # no {url} token: hand it over as the last argument

    if on_progress:
        on_progress(f'running “{collector["name"]}” …')
    started = time.time()
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=COLLECTOR_TIMEOUT_S,
                              shell=False, cwd=str(cfg.BACKEND_DIR.parent))
    except FileNotFoundError as e:
        raise CollectorError(f'collector command not found: {argv[0]}') from e
    except subprocess.TimeoutExpired as e:
        raise CollectorError(
            f'the collector did not finish within {COLLECTOR_TIMEOUT_S // 60} minutes') from e
    except OSError as e:
        raise CollectorError(f'the collector could not be started: {e}') from e

    stderr = (proc.stderr or b'').decode('utf-8', 'replace')
    if proc.returncode != 0:
        raise CollectorError(
            f'the collector exited with code {proc.returncode}', stderr)
    raw = proc.stdout or b''
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


def _safe(url):
    """A URL shortened for the log: host + path only, never the query — a signed
    link carries its credential there (diagnostics must stay paste-safe)."""
    from urllib.parse import urlsplit
    try:
        parts = urlsplit(url)
        return f'{parts.netloc}{parts.path}'
    except ValueError:
        return '<unparseable url>'
