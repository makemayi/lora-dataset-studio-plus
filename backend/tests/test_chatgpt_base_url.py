"""The ChatGPT API-key lane can be pointed at an OpenAI-compatible gateway.

NOTHING here talks to a network. `requests.post` is patched in every test that
sends one, and no key is spent — the same honest limit as test_chatgpt_refusal:
this proves where we AIM a request and how we word the answer, not that any
particular reseller behaves.

WHY THIS FILE EXISTS
--------------------
`engines.chatgpt_base_url` is a setting whose two failure modes are both quiet:

  1. It leaks. This lane uploads the user's reference PHOTOS of a real person on
     every call, so a stray value sends those photos to a third party. The
     default therefore has to be blank and has to STAY blank — a test, not a
     comment, is what keeps that true through the next config refactor.
  2. It impersonates. A Base URL with the wrong path answers 401 or 404, exactly
     like a rejected key or an unknown model, and a user who just pasted one
     would go re-check their key instead. So the wording on those statuses is
     pinned here too — including the other direction: on OpenAI the sentences
     must be byte-identical to what they always were.
"""
from unittest.mock import MagicMock, patch

import pytest
import requests

from app import config
from app.services import chatgpt_image as ci

KEY = 'sk-proj-TESTKEYVALUE0123456789abcdefghijklmn'
GATEWAY = 'https://gateway.example.com'


def _resp(status, message=''):
    r = MagicMock(status_code=status, headers={})
    body = {'error': {'message': message}} if message else None
    if body is None:
        r.json.side_effect = ValueError('no json')
        r.text = ''
    else:
        r.json.return_value = body
        r.text = message
    return r


def _set_base(value):
    config.save_config({'engines': {'chatgpt_base_url': value}})


# --- the default is OpenAI, and nothing else ---------------------------------

def test_the_shipped_default_is_openai(app):
    """The whole point of the blank default: a fresh install uploads reference
    photos to OpenAI and to nobody else."""
    assert config.get('engines.chatgpt_base_url') == ''
    assert ci.get_api_base() == 'https://api.openai.com/v1'
    assert ci.edits_url() == 'https://api.openai.com/v1/images/edits'
    assert ci.using_custom_api_base() is False


def test_the_setting_outranks_the_env_var_which_outranks_the_default(app, monkeypatch):
    """Same order as the model slug, for the same reason: an env var that some
    install already exports must not be silently beaten by a blank setting."""
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://from-env.example.com/v1')
    assert ci.get_api_base() == 'https://from-env.example.com/v1'
    _set_base(GATEWAY + '/v1')
    assert ci.get_api_base() == GATEWAY + '/v1'
    monkeypatch.delenv('OPENAI_BASE_URL')
    assert ci.get_api_base() == GATEWAY + '/v1'


def test_a_blank_or_whitespace_setting_is_not_a_custom_base(app):
    """A field the user cleared must go back to OpenAI, not to '/images/edits'."""
    for blank in ('', '   ', '\t'):
        _set_base(blank)
        assert ci.get_api_base() == ci.DEFAULT_API_BASE, repr(blank)
        assert ci.using_custom_api_base() is False, repr(blank)


# --- the three shapes people actually paste ----------------------------------

@pytest.mark.parametrize(('pasted', 'expected'), [
    # bare host -> the OpenAI-compatible convention
    ('https://gateway.example.com', 'https://gateway.example.com/v1/images/edits'),
    ('https://gateway.example.com/', 'https://gateway.example.com/v1/images/edits'),
    ('https://gateway.example.com:8443', 'https://gateway.example.com:8443/v1/images/edits'),
    # an explicit root -> trusted as typed
    ('https://gateway.example.com/v1', 'https://gateway.example.com/v1/images/edits'),
    ('https://gateway.example.com/v1/', 'https://gateway.example.com/v1/images/edits'),
    ('https://gateway.example.com/openai/v1', 'https://gateway.example.com/openai/v1/images/edits'),
    # the whole endpoint -> used verbatim, not doubled
    ('https://gateway.example.com/v1/images/edits', 'https://gateway.example.com/v1/images/edits'),
])
def test_every_shape_of_base_url_lands_on_one_endpoint(app, pasted, expected):
    """A wrong path here is a 404 the user reads as "my key is bad", so the
    forgiving parse is the feature. A base carrying ANY path is trusted as typed:
    a gateway that mounts the API somewhere else is exactly what a guess breaks."""
    assert ci.edits_url(pasted) == expected


# --- the request actually goes there -----------------------------------------

def test_the_post_is_sent_to_the_configured_gateway(app, monkeypatch):
    """The regression this feature exists to prevent: before it, the endpoint was
    a module constant and the setting could not have changed anything."""
    _set_base(GATEWAY)
    monkeypatch.setattr(ci, '_api_key', lambda: KEY)
    with patch('app.services.chatgpt_image.requests.post') as post:
        post.return_value = _resp(200)
        post.return_value.json.side_effect = None
        post.return_value.json.return_value = {'data': [{'b64_json': ''}]}
        ci.generate_variation(b'ref', 'a portrait', force_lane='api')
    assert post.call_args[0][0] == GATEWAY + '/v1/images/edits'


def test_the_subscription_lane_never_reads_the_gateway(app, monkeypatch):
    """The Codex lane is not OpenAI-key traffic — it rides the user's ChatGPT
    session. Routing it through a reseller would hand that session's answers to
    a third party for no saving at all, since it spends no API credit."""
    _set_base(GATEWAY)
    assert ci.CODEX_RESPONSES_URL.startswith('https://chatgpt.com/')
    assert GATEWAY not in ci.CODEX_RESPONSES_URL


# --- wording: the gateway is named when it is the likely culprit -------------

@pytest.mark.parametrize('status', [401, 403, 404])
def test_a_gateway_is_named_on_the_statuses_a_wrong_base_url_produces(app, status):
    """401/403/404 are exactly what a wrong Base URL returns, and exactly what a
    bad key and an unknown model return. Naming the gateway is the only thing
    that tells the user which of the three to go and look at."""
    _set_base(GATEWAY + '/v1')
    with pytest.raises(ci.ChatGPTImageFatal) as e:
        ci._raise_for_api_status(_resp(status, 'nope'), model='gpt-image-2')
    assert GATEWAY in str(e.value)
    assert 'Base URL' in str(e.value)


@pytest.mark.parametrize('status', [401, 403, 404])
def test_on_openai_those_same_messages_are_unchanged(app, status):
    """The other half of the promise. Almost nobody sets this, so the default
    path must not pick up a clause about a gateway that is not there."""
    with pytest.raises(ci.ChatGPTImageFatal) as e:
        ci._raise_for_api_status(_resp(status, 'nope'), model='gpt-image-2')
    assert 'Base URL' not in str(e.value)
    assert 'gateway' not in str(e.value)


def test_an_unreachable_gateway_does_not_blame_openai(app, monkeypatch):
    """"could not reach OpenAI" would send the user to OpenAI's status page while
    their reseller is the thing that is down."""
    _set_base(GATEWAY)
    monkeypatch.setattr(ci, '_api_key', lambda: KEY)
    with patch('app.services.chatgpt_image.requests.post',
               side_effect=requests.ConnectionError('boom')):
        with pytest.raises(ci.ChatGPTImageError) as e:
            ci.generate_variation(b'ref', 'a portrait', force_lane='api')
    assert GATEWAY in str(e.value)


def test_an_unreachable_openai_still_says_openai(app, monkeypatch):
    monkeypatch.setattr(ci, '_api_key', lambda: KEY)
    with patch('app.services.chatgpt_image.requests.post',
               side_effect=requests.ConnectionError('boom')):
        with pytest.raises(ci.ChatGPTImageError) as e:
            ci.generate_variation(b'ref', 'a portrait', force_lane='api')
    assert 'could not reach OpenAI' in str(e.value)


# --- the readiness line says which host the key will be used against ---------

def test_the_readiness_detail_names_the_gateway(app, monkeypatch):
    """The Base URL lives two cards away from the ChatGPT readiness line. Someone
    who set it needs to see WHERE their key is going from the place they already
    check, not only from the field they typed it into."""
    from app import capabilities
    monkeypatch.setattr(capabilities.cfg, 'secret', lambda name: 'sk-test')
    _set_base(GATEWAY + '/v1')
    detail = capabilities.probe_openai()['detail']
    assert GATEWAY in detail
    assert detail.startswith('key set, via ')


def test_on_openai_the_readiness_detail_is_unchanged(app, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities.cfg, 'secret', lambda name: 'sk-test')
    assert capabilities.probe_openai()['detail'] == 'key set'
