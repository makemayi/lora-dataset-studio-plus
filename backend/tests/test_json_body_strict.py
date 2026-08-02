"""A write whose body does not parse must SAY SO — never answer 200 and store nothing.

Reported from a settings save: a body carrying a raw Windows path
(``{"config": {"aitoolkit": {"dir": "C:\\ai-toolkit"}}}`` — ``\\a`` is not a JSON
escape, so the document is invalid) came back **200 OK**, config.json was rewritten,
and the value was simply absent. Every write route reads its body with
``get_json(silent=True) or {}``, which turns an unreadable body into an empty one:
the route then runs with no input at all and reports success.

What these tests pin:
  * the exact reported body is refused with 400, and the config file is NOT touched;
  * the error names the real parse failure AND the escaping, so the caller can fix it;
  * the same path correctly escaped still saves — the fix refuses malformed JSON,
    not backslashes;
  * the guard covers the other write routes, not just settings;
  * a body that is legitimately ABSENT stays legitimate (many POSTs carry none);
  * a multipart upload is untouched (its body is files, not JSON);
  * the two previews that deliberately degrade on junk keep degrading.
"""
import json
import os

import pytest

# The reported body, byte for byte: a lone backslash before 'a'.
MALFORMED = rb'{"config": {"aitoolkit": {"dir": "C:\ai-toolkit"}}}'


def _config_path():
    return os.environ['LDS_CONFIG']


def _stored_aitoolkit_dir():
    try:
        with open(_config_path(), encoding='utf-8') as f:
            return (json.load(f).get('aitoolkit') or {}).get('dir')
    except FileNotFoundError:
        return None


def _put_raw(client, path, raw):
    return client.put(path, data=raw, content_type='application/json')


def test_settings_put_refuses_the_unescaped_windows_path(client):
    """The report, reproduced: 400 instead of 200, and the file left alone."""
    before = _stored_aitoolkit_dir()
    r = _put_raw(client, '/api/settings', MALFORMED)

    assert r.status_code == 400, r.get_data(as_text=True)
    assert _stored_aitoolkit_dir() == before


def test_the_refusal_explains_the_escaping(client):
    """A parse error alone ("Invalid control character…") sends nobody anywhere.
    The message must name the cause the caller can act on."""
    r = _put_raw(client, '/api/settings', MALFORMED)

    error = r.get_json()['error']
    assert error.startswith('Invalid JSON body:')
    assert 'backslash' in error.lower()
    assert r'C:\\ai-toolkit' in error


def test_a_config_write_never_happens_on_a_refused_body(client, tmp_path):
    """Stronger than "the value is absent": nothing about the file changes.

    The bug rewrote config.json from an empty partial — harmless-looking, but it
    is a write on a request that failed."""
    client.put('/api/settings', json={'config': {'aitoolkit': {'dir': str(tmp_path)}}})
    stamp = os.stat(_config_path())

    r = _put_raw(client, '/api/settings', MALFORMED)

    assert r.status_code == 400
    assert os.stat(_config_path()).st_mtime_ns == stamp.st_mtime_ns
    assert _stored_aitoolkit_dir() == str(tmp_path)


def test_the_same_path_correctly_escaped_still_saves(client):
    """The counter-proof. What is refused is invalid JSON, not Windows paths."""
    r = _put_raw(
        client, '/api/settings',
        json.dumps({'config': {'aitoolkit': {'dir': r'C:\ai-toolkit'}}}).encode())

    assert r.status_code == 200, r.get_data(as_text=True)
    assert _stored_aitoolkit_dir() == r'C:\ai-toolkit'


@pytest.mark.parametrize('method, path', [
    ('post', '/api/dataset/create'),
    ('put', '/api/dataset/shot-catalog'),
    ('put', '/api/setup/ollama-deployment'),
    ('post', '/api/system/comfyui-recovery/resolve'),
])
def test_other_write_routes_refuse_an_unreadable_body(client, method, path):
    """Settings was where it was noticed; the pattern is the whole API surface.
    Each of these read their body the same way and would have run on an empty one."""
    r = getattr(client, method)(path, data=MALFORMED,
                                content_type='application/json')

    assert r.status_code == 400, r.get_data(as_text=True)
    assert 'Invalid JSON body' in r.get_json()['error']


def test_an_absent_body_stays_legitimate(client):
    """The guard refuses bodies it cannot read — not the absence of one. Plenty of
    POSTs carry none, and `or {}` is the right answer for those."""
    r = client.post('/api/settings/prompt-preview')
    assert r.status_code == 200


def test_a_deliberately_tolerant_preview_still_degrades(client):
    """/settings/prompt-preview answers while the user types, and its docstring
    argues that a default preview beats an error. Junk must still not 400 it."""
    r = client.post('/api/settings/prompt-preview', data=MALFORMED,
                    content_type='application/json')

    assert r.status_code == 200, r.get_data(as_text=True)


def test_a_form_encoded_body_is_not_accused_of_being_json(client):
    """/dataset/<id>/ref/edit/keep reads `engine`/`batch_id` from request.form when
    the body is not JSON. A form body is not malformed JSON — refusing it as such
    would be a fresh bug. (404 here: the dataset does not exist. Not 400.)"""
    r = client.post('/api/dataset/999999/ref/edit/keep',
                    data={'engine': 'flux', 'batch_id': '3'})

    assert r.status_code == 404, r.get_data(as_text=True)


def test_a_json_shaped_body_is_strict_whatever_header_carried_it(client):
    """`curl -d '{...}'` sends form-urlencoded by default — the exact way a
    hand-written body reaches this API. A body that opens with `{` was meant as
    JSON, so a broken one must still be refused rather than read as empty."""
    r = client.put('/api/settings', data=MALFORMED,
                   content_type='application/x-www-form-urlencoded')

    assert r.status_code == 400, r.get_data(as_text=True)
    assert 'Invalid JSON body' in r.get_json()['error']


def test_a_multipart_upload_is_not_read_as_json(client, tmp_path):
    """An upload's body is files. The guard must not touch it — a 400 here would
    mean the fix broke every image import."""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new('RGB', (8, 8), 'white').save(buf, 'PNG')
    buf.seek(0)
    r = client.post('/api/dataset/create',
                    json={'name': 'json-guard', 'trigger_word': 'jgtrig'})
    assert r.status_code in (200, 201), r.get_data(as_text=True)
    ds_id = r.get_json()['id']

    up = client.post(f'/api/dataset/{ds_id}/import',
                     data={'files': (buf, 'a.png')},
                     content_type='multipart/form-data')

    assert up.status_code != 400, up.get_data(as_text=True)
