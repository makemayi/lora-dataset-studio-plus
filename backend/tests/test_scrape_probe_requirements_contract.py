"""Pins probe_scrape_deps() against requirements-scrape.txt.

A package added to the requirements file and forgotten in the probe's module
tuple reads as "installed" the moment the ALREADY-probed modules happen to be
present — the Web images tab lights up green and fails on the first real
search instead. That is exactly the gap `ddgs` (2026-08) and `yt_dlp` shipped
with: both are imported/invoked by the scrape stack, both were declared in
requirements-scrape.txt, neither was in the probe tuple. This test makes a
repeat of that omission fail the suite instead of shipping quietly.
"""
import re
from pathlib import Path
from unittest.mock import patch

from app import capabilities

_ROOT = Path(__file__).resolve().parents[2]
_REQ = _ROOT / 'backend' / 'requirements-scrape.txt'

# requirements-scrape.txt PyPI name -> the module name it installs under.
# find_spec() probes the IMPORT name, not the PyPI name, so this mapping is
# what lets the test compare the two lists at all.
_PYPI_TO_IMPORT = {
    'gallery-dl': 'gallery_dl',
    'instaloader': 'instaloader',
    'yt-dlp': 'yt_dlp',
    'curl_cffi': 'curl_cffi',
    'brotli': 'brotli',
    'beautifulsoup4': 'bs4',
    'cloudscraper': 'cloudscraper',
    'lxml': 'lxml',
    'ddgs': 'ddgs',
}

# Declared in requirements-scrape.txt but never imported directly anywhere
# under app/scrape/** — verified by grep, not assumed. brotli is transitive
# decompression support for curl_cffi/yt-dlp; bs4/cloudscraper/lxml ride along
# unused today (picazor.py parses by regex on purpose — see its module
# docstring). Nothing in our own code raises ImportError over their absence,
# so they are not a probe gap. New entries must be justified here explicitly,
# not silently swallowed by this set.
_NOT_DIRECTLY_IMPORTED = {'brotli', 'bs4', 'cloudscraper', 'lxml'}


def _requirements_import_names():
    names = []
    for line in _REQ.read_text(encoding='utf-8').splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        pypi_name = re.split(r'[<>=!~\[]', line, maxsplit=1)[0].strip()
        assert pypi_name in _PYPI_TO_IMPORT, (
            f'{pypi_name!r} was added to requirements-scrape.txt but this test does '
            'not know its import name yet: add it to _PYPI_TO_IMPORT here, and to '
            "probe_scrape_deps()'s module tuple (or to _NOT_DIRECTLY_IMPORTED with a "
            'reason, if the scrape stack never imports it directly).'
        )
        names.append(_PYPI_TO_IMPORT[pypi_name])
    return names


def _probed_modules():
    """What probe_scrape_deps() actually asks find_spec about — behavior, not
    a regex over its source, so a rewrite of the function still gets caught."""
    seen = []
    real_find_spec = __import__('importlib.util', fromlist=['find_spec']).find_spec

    def spy(name, *a, **k):
        seen.append(name)
        return real_find_spec(name, *a, **k)

    with patch('importlib.util.find_spec', side_effect=spy):
        capabilities.probe_scrape_deps()
    return set(seen)


def test_requirements_file_names_are_all_accounted_for():
    """Fails loudly (not silently) the moment an unmapped package lands in
    requirements-scrape.txt, instead of that package quietly slipping past
    both lists below."""
    assert _requirements_import_names(), 'requirements-scrape.txt must not go empty'


def test_every_directly_imported_scrape_requirement_is_probed():
    required = _requirements_import_names()
    probed = _probed_modules()
    for module in required:
        if module in _NOT_DIRECTLY_IMPORTED:
            continue
        assert module in probed, (
            f'{module!r} is declared in requirements-scrape.txt and imported by the '
            'scrape stack, but probe_scrape_deps() does not check for it — an install '
            'missing only this package would still read "scrape deps OK".'
        )
