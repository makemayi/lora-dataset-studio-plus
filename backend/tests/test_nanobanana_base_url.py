"""The Nano Banana engine can be pointed at a Gemini-compatible gateway.

Twin of test_chatgpt_base_url, and deliberately a SEPARATE file rather than a
parametrised merge: the two lanes look alike from the Settings page and are not
alike underneath. OpenAI's path is fixed (`/v1/images/edits`) and the model
travels in the body; Google's path CARRIES THE MODEL
(`/v1beta/models/<slug>:generateContent`). Merging the tests would invite
merging the joiners, and one of the two would then be wrong.

NOTHING here talks to a network. The same two quiet failure modes are pinned as
in the OpenAI file: the default must STAY blank (this engine uploads reference
photos of a real person), and a wrong Base URL must not read as a bad key.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from app import config
from app.services import nanobanana as nb

GATEWAY = 'https://api.gateway.example.com'
MODEL = 'gemini-3-pro-image'


def _resp(status, message=''):
    r = MagicMock(status_code=status, headers={})
    r.json.return_value = {'error': {'message': message}} if message else {}
    r.text = message
    return r


def _set_base(value):
    config.save_config({'engines': {'nanobanana_base_url': value}})


# --- the default is Google, and nothing else ---------------------------------

def test_the_shipped_default_is_google(app):
    assert config.get('engines.nanobanana_base_url') == ''
    assert nb.get_api_base() == 'https://generativelanguage.googleapis.com'
    assert nb.generate_content_url(MODEL) == (
        'https://generativelanguage.googleapis.com'
        f'/v1beta/models/{MODEL}:generateContent')
    assert nb.using_custom_api_base() is False


def test_the_setting_outranks_the_env_var_which_outranks_the_default(app, monkeypatch):
    monkeypatch.setenv('GEMINI_BASE_URL', 'https://from-env.example.com')
    assert nb.get_api_base() == 'https://from-env.example.com'
    _set_base(GATEWAY)
    assert nb.get_api_base() == GATEWAY
    monkeypatch.delenv('GEMINI_BASE_URL')
    assert nb.get_api_base() == GATEWAY


def test_a_blank_or_whitespace_setting_is_not_a_custom_base(app):
    for blank in ('', '   ', '\t'):
        _set_base(blank)
        assert nb.get_api_base() == nb.DEFAULT_API_BASE, repr(blank)
        assert nb.using_custom_api_base() is False, repr(blank)


# --- the model rides in the PATH, which is what makes this lane different ----

@pytest.mark.parametrize(('pasted', 'expected'), [
    # bare host — the shape lmuai's own docs give
    ('https://api.gateway.example.com',
     'https://api.gateway.example.com/v1beta/models/gemini-3-pro-image:generateContent'),
    ('https://api.gateway.example.com/',
     'https://api.gateway.example.com/v1beta/models/gemini-3-pro-image:generateContent'),
    # the version pasted along with it — absorbed, never doubled
    ('https://api.gateway.example.com/v1beta',
     'https://api.gateway.example.com/v1beta/models/gemini-3-pro-image:generateContent'),
    ('https://api.gateway.example.com/v1beta/',
     'https://api.gateway.example.com/v1beta/models/gemini-3-pro-image:generateContent'),
    # a full template — trusted entirely, the escape hatch for an odd mount
    ('https://gw.example.com/proxy/v1beta/models/{model}:generateContent',
     'https://gw.example.com/proxy/v1beta/models/gemini-3-pro-image:generateContent'),
])
def test_every_shape_of_base_url_lands_on_one_endpoint(app, pasted, expected):
    assert nb.generate_content_url(MODEL, pasted) == expected


def test_the_slug_travels_in_the_path_not_the_body(app):
    """The difference from the OpenAI lane, pinned: change the model, the URL
    changes. A joiner shared with `chatgpt_image.edits_url` could not do this."""
    a = nb.generate_content_url('gemini-3-pro-image', GATEWAY)
    b = nb.generate_content_url('gemini-3.1-flash-image', GATEWAY)
    assert a != b
    assert a.endswith('/models/gemini-3-pro-image:generateContent')
    assert b.endswith('/models/gemini-3.1-flash-image:generateContent')


# --- the request actually goes there -----------------------------------------

def test_the_post_is_sent_to_the_configured_gateway(app, monkeypatch):
    _set_base(GATEWAY)
    monkeypatch.setattr(nb, '_api_key', lambda: 'gw-issued-key')
    monkeypatch.setattr(nb, 'get_model', lambda: MODEL)
    sent = _resp(200)
    sent.json.return_value = {'candidates': [{'content': {'parts': [
        {'inlineData': {'data': 'aGk='}}]}}]}
    with patch('app.services.nanobanana.requests.post', return_value=sent) as post:
        nb.generate_variation(b'ref', 'a portrait')
    url = post.call_args[0][0]
    assert url == f'{GATEWAY}/v1beta/models/{MODEL}:generateContent'
    # Google-native auth header, unchanged by the gateway.
    assert post.call_args.kwargs['headers']['x-goog-api-key'] == 'gw-issued-key'


# --- wording -----------------------------------------------------------------

@pytest.mark.parametrize('status', [401, 403, 404])
def test_a_gateway_is_named_on_the_statuses_a_wrong_base_url_produces(app, status):
    _set_base(GATEWAY)
    with pytest.raises(nb.NanoBananaFatal) as e:
        nb._raise_for_status(_resp(status, 'nope'), model=MODEL)
    assert GATEWAY in str(e.value)
    assert 'Base URL' in str(e.value)


@pytest.mark.parametrize('status', [401, 403, 404])
def test_on_google_those_same_messages_are_unchanged(app, status):
    with pytest.raises(nb.NanoBananaFatal) as e:
        nb._raise_for_status(_resp(status, 'nope'), model=MODEL)
    assert 'Base URL' not in str(e.value)
    assert 'gateway' not in str(e.value)


def test_an_unreachable_gateway_does_not_blame_google(app, monkeypatch):
    _set_base(GATEWAY)
    monkeypatch.setattr(nb, '_api_key', lambda: 'gw-issued-key')
    with patch('app.services.nanobanana.requests.post',
               side_effect=requests.ConnectionError('boom')):
        with pytest.raises(nb.NanoBananaError) as e:
            nb.generate_variation(b'ref', 'a portrait')
    assert GATEWAY in str(e.value)


def test_an_unreachable_google_still_says_gemini(app, monkeypatch):
    monkeypatch.setattr(nb, '_api_key', lambda: 'key')
    with patch('app.services.nanobanana.requests.post',
               side_effect=requests.ConnectionError('boom')):
        with pytest.raises(nb.NanoBananaError) as e:
            nb.generate_variation(b'ref', 'a portrait')
    assert 'could not reach Gemini' in str(e.value)


# --- readiness ---------------------------------------------------------------

def test_the_readiness_detail_names_the_gateway(app, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities.cfg, 'secret', lambda name: 'key')
    _set_base(GATEWAY)
    assert capabilities.probe_gemini()['detail'] == f'key set, via {GATEWAY}'


def test_on_google_the_readiness_detail_is_unchanged(app, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities.cfg, 'secret', lambda name: 'key')
    assert capabilities.probe_gemini()['detail'] == 'key set'
