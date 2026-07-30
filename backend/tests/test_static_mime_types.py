"""The frontend bundle must never be served with a type the browser won't run.

Background (GitHub #12, reported and diagnosed by gessyoo): on Windows,
`mimetypes` seeds itself from the registry (HKEY_CLASSES_ROOT\\<ext>\\Content
Type), which any installed program may have overwritten. On such a machine
`.js` came back as `text/plain`, the browser refused to execute the bundle, and
the app opened on a blank page with a perfectly silent server log.

These tests NEVER touch the registry. They poison the in-process `mimetypes`
table, which is exactly what a poisoned registry produces once it has been read.
"""
import mimetypes
from pathlib import Path

import pytest

from app import FRONTEND_DIST, _STATIC_MIME_TYPES

# Types a browser will actually execute as a script module.
EXECUTABLE_JS_TYPES = {'text/javascript', 'application/javascript',
                       'application/ecmascript', 'text/ecmascript'}


@pytest.fixture()
def poisoned_mimetypes():
    """Snapshot/restore the process-wide mimetypes table around a test.

    `add_type` mutates dicts that live for the whole process; without this the
    poisoning would leak into every later test in the session.
    """
    mimetypes.guess_type('x.js')          # force lazy init so the db exists
    strict = mimetypes.types_map                       # == db.types_map[True]
    inv = getattr(mimetypes, '_db', None)
    inv = inv.types_map_inv[True] if inv is not None else {}
    saved_map, saved_inv = dict(strict), dict(inv)
    yield
    strict.clear()
    strict.update(saved_map)
    inv.clear()
    inv.update(saved_inv)


def _one_asset(suffix):
    assets = FRONTEND_DIST / 'assets'
    if not assets.is_dir():
        pytest.skip('frontend/dist not built in this checkout')
    for f in sorted(assets.iterdir()):
        if f.suffix == suffix:
            return f.name
    pytest.skip(f'no {suffix} asset in frontend/dist/assets')


# --------------------------------------------------------------------------
# 1. gessyoo's scenario, end to end
# --------------------------------------------------------------------------

def test_bundle_is_executable_even_when_js_is_mapped_to_text_plain(
        poisoned_mimetypes, monkeypatch, tmp_path):
    """A machine whose registry says `.js` is text/plain still gets a real bundle.

    This is the blank-page bug. Without the pin in create_app(), the response
    comes back as `text/plain` and the browser silently refuses to run it.
    """
    mimetypes.add_type('text/plain', '.js', strict=True)
    assert mimetypes.guess_type('x.js')[0] == 'text/plain'   # the sick machine

    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as _cfg
    monkeypatch.setattr(_cfg, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(_cfg, '_cache', None)
    from app import create_app
    application = create_app({'TESTING': True, 'WTF_CSRF_ENABLED': False,
                              'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})

    name = _one_asset('.js')
    with application.app_context():
        resp = application.test_client().get(f'/assets/{name}')
    assert resp.status_code == 200
    assert resp.mimetype in EXECUTABLE_JS_TYPES, (
        f'bundle served as {resp.mimetype!r} — the browser will not execute it')


def test_stylesheet_and_shell_survive_a_poisoned_table(
        poisoned_mimetypes, monkeypatch, tmp_path):
    """The same trap applies to the CSS and to index.html itself."""
    mimetypes.add_type('text/plain', '.css', strict=True)
    mimetypes.add_type('application/octet-stream', '.html', strict=True)

    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as _cfg
    monkeypatch.setattr(_cfg, 'ENV_PATH', tmp_path / '.env')
    monkeypatch.setattr(_cfg, '_cache', None)
    from app import create_app
    application = create_app({'TESTING': True, 'WTF_CSRF_ENABLED': False,
                              'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})
    client = application.test_client()

    name = _one_asset('.css')
    with application.app_context():
        css = client.get(f'/assets/{name}')
        shell = client.get('/')
    assert css.status_code == 200 and css.mimetype == 'text/css'
    assert shell.status_code == 200 and shell.mimetype == 'text/html'


# --------------------------------------------------------------------------
# 2. add_type really does beat an already-loaded registry entry
# --------------------------------------------------------------------------

def test_pin_overrides_an_already_loaded_entry(poisoned_mimetypes):
    """`add_type(strict=True)` wins over a value the registry already installed.

    The ordering matters: `mimetypes` is initialised (registry read included)
    long before we get a chance to speak, so the pin must be an override, not a
    default.
    """
    from app import pin_static_mime_types
    assert mimetypes.inited, 'registry (if any) already loaded at this point'
    mimetypes.add_type('text/plain', '.js', strict=True)
    mimetypes.add_type('text/plain', '.mjs', strict=True)
    mimetypes.add_type('application/octet-stream', '.webp', strict=True)
    mimetypes.add_type('application/octet-stream', '.bmp', strict=True)

    pin_static_mime_types()

    assert mimetypes.guess_type('a.js')[0] == 'text/javascript'
    assert mimetypes.guess_type('a.mjs')[0] == 'text/javascript'
    assert mimetypes.guess_type('a.webp')[0] == 'image/webp'
    assert mimetypes.guess_type('a.bmp')[0] == 'image/bmp'


# --------------------------------------------------------------------------
# 3. no-op on a healthy machine / no wrong type imposed
# --------------------------------------------------------------------------

def test_pinned_values_match_pythons_own_standard_table():
    """Every pinned type is the value Python uses when no registry interferes.

    That is what makes this a no-op on a healthy machine: we only ever restore
    the standard answer, we never impose a type of our own invention.
    """
    default = getattr(mimetypes, '_types_map_default', None)
    if default is None:                       # private, guard across versions
        pytest.skip('mimetypes._types_map_default unavailable')
    mismatches = {ext: (ours, default[ext])
                  for ext, ours in _STATIC_MIME_TYPES.items()
                  if ext in default and default[ext] != ours}
    assert not mismatches, f'diverges from the standard table: {mismatches}'


def test_pin_leaves_unrelated_extensions_alone(poisoned_mimetypes):
    from app import pin_static_mime_types
    mimetypes.add_type('application/x-custom', '.lds-not-a-real-ext', strict=True)
    pin_static_mime_types()
    assert mimetypes.guess_type('a.lds-not-a-real-ext')[0] == 'application/x-custom'


# --------------------------------------------------------------------------
# 4. the pinned set actually covers what we ship
# --------------------------------------------------------------------------

def test_every_extension_in_the_built_frontend_is_pinned():
    """Contract: a future build emitting a new asset kind must extend the table.

    Otherwise that asset silently goes back to asking the registry — the exact
    failure mode this guards against.
    """
    if not FRONTEND_DIST.is_dir():
        pytest.skip('frontend/dist not built in this checkout')
    shipped = {p.suffix.lower() for p in Path(FRONTEND_DIST).rglob('*')
               if p.is_file() and p.suffix}
    missing = shipped - set(_STATIC_MIME_TYPES)
    assert not missing, (
        f'frontend/dist ships {sorted(missing)} with no pinned content type — '
        'add them to _STATIC_MIME_TYPES')
