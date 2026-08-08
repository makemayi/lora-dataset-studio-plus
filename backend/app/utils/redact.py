"""Paste-safe redaction of local home-dir paths.

Shared by anything that emits text meant to be pasted into a PUBLIC GitHub
issue / Discord thread (the diagnostic payload, the per-run "Share
configuration" file): a raw `C:\\Users\\<realname>\\...` (or
`/home/<realname>/...`) path leaks the OS account / Unix username. Only the
drive+Users+<segment> (or /home|Users/<segment>) prefix is swapped for `~`;
the rest of the path is kept, it carries no identity.
"""
import re
from urllib.parse import urlsplit, urlunsplit

# Windows home dir, single OR double backslash (some exception reprs escape
# them): `C:\Users\<name>\...` / `C:\\Users\\<name>\\...`. Case-insensitive
# (drive letter, "users").
_WIN_HOME_RE = re.compile(r'[A-Za-z]:\\{1,2}Users\\{1,2}[^\\/:*?"<>|\r\n]+', re.IGNORECASE)
# POSIX home dir: `/home/<name>` or `/Users/<name>` (macOS).
_POSIX_HOME_RE = re.compile(r'/(?:home|Users)/[^/\r\n]+', re.IGNORECASE)

# Credentials that can end up INSIDE a training log or a stack trace: a
# Hugging Face token (`hf_…`), a bearer header, a `?token=`/`api_key=` query
# string. Training logs are pasted into public help threads verbatim, so any
# text we hand back to the UI goes through this first. Only the VALUE is
# replaced, so the line still reads ("Authorization: Bearer ***").
_TOKEN_PATTERNS = (
    re.compile(r'\bhf_[A-Za-z0-9]{8,}'),                       # Hugging Face
    re.compile(r'(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}'),          # Authorization header
    re.compile(r'(?i)\b(?:api[_-]?key|access[_-]?token|token)'  # ?token=…, api_key=…
               r'\s*[=:]\s*["\']?[A-Za-z0-9._\-]{8,}["\']?'),
)


def redact_tokens(line):
    """Strip credential-shaped substrings out of text destined for the UI or a
    pasted diagnostic. Never a security boundary (we don't know every shape) —
    a safety net over text we did not write. NULL/empty passes through."""
    if not line:
        return line
    text = str(line)
    for rx in _TOKEN_PATTERNS:
        text = rx.sub(lambda m: _mask(m.group(0)), text)
    return text


def _mask(chunk):
    """Keep the label, drop the value: `token=hf_abc…` -> `token=***`."""
    for sep in ('=', ':'):
        head, found, _tail = chunk.partition(sep)
        if found:
            return f'{head}{sep}***'
    if chunk.lower().startswith('bearer'):
        return 'Bearer ***'
    return '***'


def redact_url_secrets(url):
    """A URL that is safe to SHOW: scheme, host, port and path stay; everything
    that can carry a credential is dropped.

    The ComfyUI recovery banner names the address LDS is talking to — pointing
    at the wrong one is the very bug it exists for, so hiding the address would
    defeat it. But a ComfyUI behind a reverse proxy is legitimately configured
    as `https://user:pass@host/` or `https://host/?token=…`, and a banner is
    exactly the kind of thing that gets screenshotted into a public help thread.
    Userinfo, query and fragment go whole (a query string on a ComfyUI address
    carries no routing meaning); anything unparseable falls back to the token
    redactor rather than being shown raw. NULL/empty passes through.
    """
    if not url:
        return url
    try:
        parts = urlsplit(str(url))
        if not parts.scheme and not parts.netloc:
            return redact_tokens(url)
        host = parts.hostname or ''
        if ':' in host:                       # IPv6 literal: keep its brackets
            host = f'[{host}]'
        port = parts.port                     # ValueError on a malformed port
        return urlunsplit((parts.scheme, f'{host}:{port}' if port else host,
                           parts.path, '', ''))
    except ValueError:
        return redact_tokens(url)


def redact_user_paths(line):
    """Strip the OS account name out of an absolute home-dir path in a string.
    Only the drive+Users+<segment> (or /home|Users/<segment>) prefix becomes
    `~`; the rest of the path is preserved. NULL/empty passes through."""
    if not line:
        return line
    line = _WIN_HOME_RE.sub('~', line)
    line = _POSIX_HOME_RE.sub('~', line)
    return line
