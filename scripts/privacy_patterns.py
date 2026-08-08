"""ONE table of privacy patterns, shared by every scanner in this repo.

Two things are scanned, and they must never disagree:

* the tracked sources (`backend/tests/test_no_personal_data.py`) — what a
  reviewer sees on GitHub;
* the published archive (`scripts/check_release_artifacts.py`) — what people
  actually download.

They are not the same set. `git ls-files` shows only TRACKED files and the test
skips `frontend/dist/`, so an untracked file, an ignored directory or the
compiled bundle could ship something no test ever read — and one did: a
published release carried `backend/.pytest_cache/` because the packaging script
excluded `tests` but not the cache beside it.

The fix is scope, not new patterns. Two pattern tables that drift apart are
worse than one, so this module holds the only copy and both scanners import it.

This file lives in `scripts/` on purpose: the release archive ships
`backend/`, `frontend/dist/` and a single named script, so the pattern table
itself never travels inside the artifact it inspects.
"""

from __future__ import annotations

import re

# Suffixes worth scanning. Binaries and lockfiles are excluded; `.map` and
# `.example` are in because a source map embeds whole original files and a
# sample env file is exactly where a token gets forgotten.
SCANNED_SUFFIXES = ('.py', '.js', '.jsx', '.mjs', '.md', '.json', '.yml', '.yaml',
                    '.html', '.css', '.bat', '.ps1', '.txt', '.map', '.example',
                    '.cfg', '.ini', '.toml', '.env')

PATTERNS = {
    # `[\\/]`, not `[\/]`. Inside a character class `\/` is just `/`, so the
    # original class held ONE separator and the pattern only ever matched
    # forward-slash paths — the `…:\Users\…` shape it exists for, the one a
    # pasted Windows diagnostic actually contains, went through untouched for
    # as long as the check existed. Found by writing the counter-proof:
    # a guard nobody has watched REJECT something is a guard nobody has tested.
    'a Windows user path': re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+(?!<)[A-Za-z0-9._-]+', re.I),
    'an email address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    'an OpenAI-shaped key': re.compile(r'\bsk-[A-Za-z0-9]{20,}'),
    'a bearer token': re.compile(r'\bBearer\s+[A-Za-z0-9._-]{20,}'),
    'a Hugging Face token': re.compile(r'\bhf_[A-Za-z0-9]{20,}'),
    'a GitHub token': re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}'
                                 r'|\bgithub_pat_[A-Za-z0-9_]{30,}'),
    # CLAUDE.md forbids IPs, but nothing enforced it: two addresses from a real
    # tailnet reached main and fifteen published releases, one of them under a
    # comment that said "real tailnet IP" out loud. A tailnet address is a stable
    # node identity, which is exactly what makes it worth catching — ordinary
    # RFC1918 LAN addresses stay allowed, they identify nobody.
    'a tailnet address': re.compile(r'\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b'),
}

# The block's edges and Tailscale's own documented service address are the only
# in-block literals a scanner may name. Anything else is somebody's machine.
ALLOWED_TAILNET = ('100.64.0.0', '100.64.0.1', '100.127.255.254', '100.100.100.100')
# Windows paths are only personal when the account name is one. A documented
# placeholder is what we WANT contributors to write.
# Repairing the separator class above turned on eight matches that had never
# fired: every one is a documented elision or a redaction-test fixture, so the
# canonical stand-in names join the list rather than the fixtures being
# rewritten. The cost is named and small — a contributor genuinely called Alice
# is covered by the name list, the half built for real identities.
PLACEHOLDER_USERS = ('user', 'users', 'username', 'youruser', 'yourname',
                     'somebody', 'someone', 'me', 'you', 'public', 'default',
                     'all users', 'test', 'example', 'account', 'owner',
                     'alice', 'bob', 'carol', 'dave', 'secretuser')
# RFC 2606 / 6761 reserve these for documentation and tests; a fixture that
# uses one cannot belong to a person.
RESERVED_DOMAINS = ('.example', '.test', '.invalid', '.localhost',
                    'example.com', 'example.net', 'example.org')
ALLOWED_EMAILS = ('noreply@lora-dataset-studio.dev',)
# A secret that announces it is fake is not a secret. The redaction tests need
# token-shaped strings to prove a token gets redacted, and a UI placeholder is
# written to be copied over — flagging those teaches people to switch the
# scanner off, which costs more than it saves. The giveaway must be IN the
# matched token, so a real random secret cannot claim the exemption.
_TOKEN_LABELS = ('an OpenAI-shaped key', 'a bearer token',
                 'a Hugging Face token', 'a GitHub token')
FAKE_SECRET_HINTS = ('example', 'placeholder', 'dummy', 'fake', 'leaked',
                     'redact', 'sample', 'yourtoken', 'your_token', 'xxxx',
                     'notasecret', 'testtoken')


# Text files that carry no suffix at all. Without these, the release archive
# had exactly one member no scanner ever read.
SCANNED_BASENAMES = ('license', 'notice', 'readme', 'changelog', 'dockerfile',
                     'makefile', '.env')


def is_scannable(name: str) -> bool:
    """True when `name` is a path whose text is worth reading."""
    lowered = name.casefold()
    if lowered.endswith(SCANNED_SUFFIXES):
        return True
    return lowered.rsplit('/', 1)[-1].rsplit('\\', 1)[-1] in SCANNED_BASENAMES


def is_personal_email(found: str, before: str) -> bool:
    """`before` is the text immediately preceding the match.

    Three ways a match is NOT someone's address, each hit by a real fixture in
    this repo — a scanner is worthless if it cries wolf on all of them:
      * URL userinfo (`https://pexels.com@evil.example/`) — the domain-spoofing
        tests need exactly this shape;
      * a reserved documentation domain;
      * a stub too short to be anyone (`u@x.io`): a one-letter local part or a
        one-letter domain label is a placeholder, never a real mailbox.
    """
    if found in ALLOWED_EMAILS:
        return False
    token = re.split(r'''[\s'"(\[<]''', before)[-1]    # the run glued to the match
    if '://' in token and '/' not in token.split('://', 1)[1]:
        return False                                   # userinfo inside a URL
    local, _, domain = found.rpartition('@')
    if any(domain.endswith(d) or domain == d for d in RESERVED_DOMAINS):
        return False
    if len(local) < 3 or len(domain.split('.')[0]) < 3:
        return False
    return True


def scan_text(body: str):
    """Yield `(line_number, label, matched_text)` for every real finding.

    The per-pattern exceptions live HERE and nowhere else, so the tracked-file
    test and the archive scan can never disagree about what counts as a hit.
    """
    for label, rx in PATTERNS.items():
        for m in rx.finditer(body):
            found = m.group(0)
            if label == 'an email address' and not is_personal_email(
                    found, body[max(0, m.start() - 80):m.start()]):
                continue
            if label == 'a Windows user path':
                account = re.split(r'[\\/]+', found)[-1]
                if account.lower() in PLACEHOLDER_USERS:
                    continue
                if not account.strip('.-_'):
                    continue          # `…:\Users\...` — an elision, not an account
            if label == 'a tailnet address' and found in ALLOWED_TAILNET:
                continue
            if label in _TOKEN_LABELS and any(h in found.casefold()
                                              for h in FAKE_SECRET_HINTS):
                continue
            yield body[:m.start()].count('\n') + 1, label, found


def name_pattern(names):
    """A word-boundary regex over a list of forbidden identifiers, or None.

    The list is deliberately NOT in the repo — writing a name down to forbid it
    would publish it.
    """
    names = [n for n in (names or []) if n]
    if not names:
        return None
    return re.compile(r'\b(' + '|'.join(re.escape(n) for n in names) + r')\b', re.I)


def read_name_list(path):
    """One identifier per line, `#` comments ignored. None when absent."""
    import os
    if not path or not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return [w.strip() for w in fh if w.strip() and not w.startswith('#')]
