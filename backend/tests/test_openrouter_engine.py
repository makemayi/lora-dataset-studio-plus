"""OpenRouter image engine (GitHub #13).

NOTHING here talks to OpenRouter: `requests.post` is patched in every test, so
the suite never needs a key, never spends a credit, and never depends on the
service being up. That is also the honest limit of this file — it proves the
REQUEST we build and how we read an answer we hand ourselves, not that the real
service accepts it.

The invariants worth locking, in order of how badly they hurt when broken:
  1. the API key never appears in an exception, a log record or any text a user
     can read or paste;
  2. every failure is NAMED (missing key / rejected key / no credits / unknown
     model / rate limit) and none of them silently becomes another engine;
  3. the causes that would fail every remaining row identically stop the batch;
  4. the reference images we were given are all sent, and a refusal mentions the
     count instead of the app dropping some.
"""
import base64
import logging
from unittest.mock import MagicMock, patch

import pytest

KEY = 'sk-or-v1-testkeyvalue0123456789'
PNG = b'\x89PNG\r\n\x1a\nfake-pixels'


def _resp(status=200, json_body=None, text=''):
    r = MagicMock(status_code=status)
    if json_body is None:
        r.json.side_effect = ValueError('no json')
    else:
        r.json.return_value = json_body
    r.text = text
    return r


def _ok_body(b64=None, media='image/png'):
    return {'created': 1, 'media_type': media,
            'data': [{'b64_json': b64 or base64.b64encode(PNG).decode(), 'media_type': media}],
            'usage': {'cost': 0.04}}


def _err(msg, code=400):
    return {'error': {'code': code, 'message': msg}}


# --- the request we build --------------------------------------------------

def test_sends_every_reference_as_a_data_url_with_the_configured_model(app, monkeypatch):
    """The dataset generator's whole point is reference-driven generation: all the
    references handed over must ride in input_references, principal first."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    with patch('app.services.openrouter.requests.post',
               return_value=_resp(200, _ok_body())) as post:
        out = openrouter.generate_variation([b'ref-a', b'ref-b'], 'a portrait',
                                            aspect_ratio='3:4')
    assert out == PNG
    payload = post.call_args.kwargs['json']
    assert post.call_args.args[0] == 'https://openrouter.ai/api/v1/images'
    assert payload['model'] == 'google/gemini-3-pro-image'
    assert payload['prompt'] == 'a portrait'
    assert payload['aspect_ratio'] == '3:4'
    refs = payload['input_references']
    assert len(refs) == 2
    assert refs[0]['type'] == 'image_url'
    assert refs[0]['image_url']['url'] == ('data:image/webp;base64,'
                                           + base64.b64encode(b'ref-a').decode())
    assert refs[1]['image_url']['url'].endswith(base64.b64encode(b'ref-b').decode())


def test_a_single_reference_is_accepted_like_the_other_engines(app, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    with patch('app.services.openrouter.requests.post',
               return_value=_resp(200, _ok_body())) as post:
        assert openrouter.generate_variation(b'ref', 'p') == PNG
    assert len(post.call_args.kwargs['json']['input_references']) == 1


def test_the_model_is_free_text_from_config(app, monkeypatch):
    """A model slug must never require a release: OpenRouter's catalogue moves."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app import config as cfg
    from app.services import openrouter
    with app.app_context():
        cfg.save_config({'engines': {'openrouter_model': 'bytedance-seed/seedream-4.5'}})
        assert openrouter.get_model() == 'bytedance-seed/seedream-4.5'
        with patch('app.services.openrouter.requests.post',
                   return_value=_resp(200, _ok_body())) as post:
            openrouter.generate_variation(b'r', 'p')
        assert post.call_args.kwargs['json']['model'] == 'bytedance-seed/seedream-4.5'
        # Blank falls back rather than asking OpenRouter for the empty model.
        cfg.save_config({'engines': {'openrouter_model': '   '}})
        assert openrouter.get_model() == openrouter.DEFAULT_MODEL


def test_an_endpoint_that_refuses_aspect_ratio_is_retried_without_it(app, monkeypatch):
    """Providers clamp aspect_ratio to their own subset. Losing the framing is far
    better than losing the image — but only the framing is given up, never the
    references or the prompt."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    responses = [_resp(400, _err('aspect_ratio is not supported')),
                 _resp(200, _ok_body())]
    with patch('app.services.openrouter.requests.post',
               side_effect=responses) as post:
        assert openrouter.generate_variation(b'r', 'p', aspect_ratio='9:16') == PNG
    assert post.call_count == 2
    assert 'aspect_ratio' in post.call_args_list[0].kwargs['json']
    second = post.call_args_list[1].kwargs['json']
    assert 'aspect_ratio' not in second
    assert second['prompt'] == 'p' and len(second['input_references']) == 1


# --- the secret ------------------------------------------------------------

def test_the_key_is_sent_as_a_bearer_header_and_nowhere_else(app, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    with patch('app.services.openrouter.requests.post',
               return_value=_resp(200, _ok_body())) as post:
        openrouter.generate_variation(b'r', 'p')
    call = post.call_args
    assert call.kwargs['headers']['Authorization'] == f'Bearer {KEY}'
    # The key must not have leaked into the body under any name.
    assert KEY not in str(call.kwargs['json'])


@pytest.mark.parametrize('status,body', [
    (401, _err('User not found', 401)),
    (402, _err('Insufficient credits', 402)),
    (404, _err('No endpoints found', 404)),
    (429, _err('Rate limit exceeded', 429)),
    (500, _err('upstream error', 500)),
])
def test_no_error_message_or_log_line_can_ever_carry_the_key(app, monkeypatch, caplog,
                                                             status, body):
    """The one thing that must hold for EVERY failure path, including the ones a
    user is most likely to paste into a public help channel."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    caplog.set_level(logging.DEBUG)
    with patch('app.services.openrouter.requests.post',
               return_value=_resp(status, body, text=str(body))):
        with pytest.raises(openrouter.OpenRouterError) as e:
            openrouter.generate_variation(b'r', 'p')
    assert KEY not in str(e.value)
    assert KEY[8:] not in str(e.value)          # not even a fragment
    assert KEY not in caplog.text


def test_a_missing_key_says_so_instead_of_looking_like_a_refusal(app, monkeypatch):
    """Returning None here (what the Nano Banana engine does) would surface as
    'empty response - often a content-policy refusal', sending the user to rewrite
    a prompt when the real fix is pasting a key."""
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    from app.services import openrouter
    with patch('app.services.openrouter.requests.post') as post:
        with pytest.raises(openrouter.OpenRouterFatal) as e:
            openrouter.generate_variation(b'r', 'p')
    post.assert_not_called()                    # no request without a key
    msg = str(e.value)
    assert 'OPENROUTER_API_KEY' in msg and 'Settings' in msg


# --- each class of failure, named ------------------------------------------

@pytest.mark.parametrize('status,message,expected', [
    (401, 'User not found', 'rejected the API key'),
    (403, 'Forbidden', 'rejected the API key'),
    (402, 'Insufficient credits', 'out of credits'),
    (404, 'No endpoints found for model', 'does not serve the model'),
])
def test_a_failure_that_would_repeat_on_every_row_is_fatal_and_named(app, monkeypatch,
                                                                    status, message,
                                                                    expected):
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    with patch('app.services.openrouter.requests.post',
               return_value=_resp(status, _err(message, status))):
        with pytest.raises(openrouter.OpenRouterFatal) as e:
            openrouter.generate_variation(b'r', 'p')
    assert expected in str(e.value)
    assert message in str(e.value)              # OpenRouter's own words, verbatim


@pytest.mark.parametrize('status,expected', [
    (429, 'rate-limited'),
    (502, 'HTTP 502'),
])
def test_a_transient_failure_stays_per_row(app, monkeypatch, status, expected):
    """A rate limit or a provider hiccup must NOT cancel a run that would have
    finished: it fails this image only."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    with patch('app.services.openrouter.requests.post',
               return_value=_resp(status, _err('boom', status))):
        with pytest.raises(openrouter.OpenRouterError) as e:
            openrouter.generate_variation(b'r', 'p')
    assert not isinstance(e.value, openrouter.OpenRouterFatal)
    assert expected in str(e.value)


def test_a_refused_request_reports_the_reference_count_instead_of_dropping_some(
        app, monkeypatch):
    """Models accept anywhere from 1 to 16 references and that ceiling moves with
    the catalogue, so nothing is hardcoded: we send them all and, if refused, name
    the count as a possible cause rather than truncating behind the user's back."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    refs = [b'a', b'b', b'c', b'd', b'e']
    with patch('app.services.openrouter.requests.post',
               return_value=_resp(400, _err('too many input references'))) as post:
        with pytest.raises(openrouter.OpenRouterError) as e:
            openrouter.generate_variation(refs, 'p')
    # Every reference really was sent — the count in the message is the truth.
    assert len(post.call_args_list[0].kwargs['json']['input_references']) == 5
    msg = str(e.value)
    assert '5 reference images were sent' in msg
    assert 'may accept fewer' in msg
    assert 'too many input references' in msg


def test_a_network_error_names_the_network(app, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    import requests
    from app.services import openrouter
    with patch('app.services.openrouter.requests.post',
               side_effect=requests.ConnectionError('dns failure')):
        with pytest.raises(openrouter.OpenRouterError) as e:
            openrouter.generate_variation(b'r', 'p')
    assert 'could not reach OpenRouter' in str(e.value)


def test_an_unparseable_error_body_still_yields_a_status(app, monkeypatch):
    """An edge/proxy failure answers HTML, not the documented error envelope."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    with patch('app.services.openrouter.requests.post',
               return_value=_resp(503, None, text='<html>gateway</html>')):
        with pytest.raises(openrouter.OpenRouterError) as e:
            openrouter.generate_variation(b'r', 'p')
    assert 'HTTP 503' in str(e.value)


# --- reading the answer -----------------------------------------------------

def test_a_200_with_no_image_and_no_reason_is_the_only_none(app, monkeypatch):
    """None now means one narrower thing: the provider answered without an image
    AND without saying why. That is the one case this engine cannot read, and it
    is reported as unreadable rather than being called a refusal."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    with patch('app.services.openrouter.requests.post',
               return_value=_resp(200, {'created': 1, 'data': []})):
        assert openrouter.generate_variation(b'r', 'p') is None


def test_a_moderation_block_embedded_in_a_200_is_read_out_of_the_body(app, monkeypatch):
    """OpenRouter puts moderation blocks INSIDE a 200 (error.code 403, reasons in
    error.metadata). Flattening that to None threw away a stated cause and made
    the tile read as if the app had come back empty-handed."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    from app.services.engine_errors import EngineFatal, EngineRefused
    body = {'created': 1, 'data': [],
            'error': {'code': 403, 'message': 'Blocked by moderation',
                      'metadata': {'reasons': ['sexual', 'minors']}}}
    with patch('app.services.openrouter.requests.post', return_value=_resp(200, body)):
        with pytest.raises(openrouter.OpenRouterRefused) as e:
            openrouter.generate_variation(b'r', 'p')
    msg = str(e.value)
    assert 'refused this request' in msg
    assert 'sexual, minors' in msg                      # the provider's own reasons
    assert isinstance(e.value, EngineRefused)
    assert not isinstance(e.value, EngineFatal)         # one row, not the run
    for banned in ('retry', 'try again', 'rephrase'):
        assert banned not in msg.lower()


def test_a_provider_failure_embedded_in_a_200_is_not_called_a_refusal(app, monkeypatch):
    """The mirror case: an upstream provider that died mid-response also lands in
    the body. It must keep its own words and never read as content moderation."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    from app.services.engine_errors import EngineRefused
    body = {'created': 1, 'data': [],
            'error': {'code': 502, 'message': 'Provider returned an error'}}
    with patch('app.services.openrouter.requests.post', return_value=_resp(200, body)):
        with pytest.raises(openrouter.OpenRouterError) as e:
            openrouter.generate_variation(b'r', 'p')
    assert 'Provider returned an error' in str(e.value)
    assert not isinstance(e.value, EngineRefused)
    assert 'moderation' not in str(e.value)


@pytest.mark.parametrize('body', [
    {'data': [], 'error': 'not-a-dict'}, {'data': [], 'error': None},
    {'data': [], 'error': {}}, {'data': [], 'error': {'metadata': 'not-a-dict'}},
    {'data': [], 'error': {'code': 'nonsense', 'metadata': {'reasons': 'not-a-list'}}},
])
def test_an_unreadable_embedded_error_never_crashes_the_row(app, monkeypatch, body):
    """Runs on whatever OpenRouter sends. A shape we did not anticipate must
    degrade to a sentence or to None — never to an AttributeError the user reads
    as an app bug."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    with patch('app.services.openrouter.requests.post', return_value=_resp(200, body)):
        try:
            assert openrouter.generate_variation(b'r', 'p') is None
        except openrouter.OpenRouterError as e:
            assert 'without an image' in str(e)


def test_a_vector_answer_is_refused_where_it_happens(app, monkeypatch):
    """Recraft's vector models put SVG markup in the same b64_json field. Passing
    it on would fail much later, blaming the image-saving step."""
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    body = _ok_body(b64=base64.b64encode(b'<svg/>').decode(), media='image/svg+xml')
    with patch('app.services.openrouter.requests.post', return_value=_resp(200, body)):
        with pytest.raises(openrouter.OpenRouterError) as e:
            openrouter.generate_variation(b'r', 'p')
    assert 'vector image' in str(e.value)


def test_a_non_json_success_body_is_reported_not_swallowed(app, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.services import openrouter
    with patch('app.services.openrouter.requests.post',
               return_value=_resp(200, None, text='not json')):
        with pytest.raises(openrouter.OpenRouterError) as e:
            openrouter.generate_variation(b'r', 'p')
    assert 'non-JSON' in str(e.value)


# --- wiring into the engine set --------------------------------------------

def test_the_engine_is_registered_without_disturbing_the_existing_two(app):
    """Engine ids and file tags are PERSISTED (dataset rows, filenames on disk):
    this locks that the new one was appended, never a rename or a reorder."""
    from app.services import face_dataset_service as svc
    assert svc.API_ENGINES == ('nanobanana', 'chatgpt', 'openrouter', 'qwen')
    assert svc._ENGINE_FILE_TAG['nanobanana'] == 'NBFace'
    assert svc._ENGINE_FILE_TAG['chatgpt'] == 'GPTFace'
    assert svc._ENGINE_FILE_TAG['openrouter'] == 'ORFace'
    assert svc._ENGINE_FILE_TAG['qwen'] == 'QwenFace'
    from app.services import openrouter
    assert svc._api_generate_fn('openrouter') is openrouter.generate_variation
    # ...and the other two still resolve to THEIR own module.
    from app.services import nanobanana
    assert svc._api_generate_fn('nanobanana') is nanobanana.generate_variation


def test_the_engine_lights_up_on_the_key_alone(app, monkeypatch):
    """Readiness is key-presence only: probing for real would mean a billed
    request on every capability poll."""
    from app import capabilities
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    assert capabilities.probe_openrouter() == {'ok': False, 'detail': 'key missing'}
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    probe = capabilities.probe_openrouter()
    assert probe['ok'] is True
    assert KEY not in str(probe)                # presence, never the value


def test_the_key_is_a_declared_secret_so_it_is_never_echoed_back(app, monkeypatch):
    """SECRET_KEYS membership is what makes /api/settings report presence only —
    forgetting it would have leaked the key into the settings payload."""
    from app import config as cfg
    assert 'OPENROUTER_API_KEY' in cfg.SECRET_KEYS
    monkeypatch.setenv('OPENROUTER_API_KEY', KEY)
    from app.routes.settings import _secret_presence
    with app.app_context():
        presence = _secret_presence()
    assert presence['OPENROUTER_API_KEY'] is True
    assert KEY not in str(presence)


# --- the batch stop ---------------------------------------------------------

class _SerialPool:
    """Deterministic stand-in for ThreadPoolExecutor (same reason as
    test_dataset_service: the real 3-worker pool races on the shared in-memory
    sqlite). The concurrency is not what is under test here."""
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def map(self, fn, items): return [fn(i) for i in items]


def _real_png():
    import io
    from PIL import Image
    buf = io.BytesIO(); Image.new('RGB', (64, 64), (255, 0, 0)).save(buf, 'PNG')
    return buf.getvalue()


@pytest.mark.parametrize('fatal_message,expected', [
    ('OpenRouter is out of credits (HTTP 402): Insufficient credits', 'out of credits'),
    ('OpenRouter rejected the API key (HTTP 401): User not found', 'rejected the API key'),
])
def test_a_fatal_failure_stops_the_batch_instead_of_paying_per_row(app, monkeypatch,
                                                                   fatal_message, expected):
    """Out of credits (or a rejected key) would refuse EVERY remaining row for the
    same reason. Asking the question 30 more times costs the user time and, on a
    partially-refused batch, money — so the run stops on the first one and every
    remaining row says why."""
    import concurrent.futures
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import dataset_generation_service
    from app.services.openrouter import OpenRouterFatal
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise OpenRouterFatal(fatal_message)

    monkeypatch.setattr(dataset_generation_service, '_api_generate_fn', lambda engine: boom)
    with app.app_context():
        import os
        ds = svc.create_dataset(LOCAL_USER, 'OR', 'or')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        rows = [FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model='openrouter')
                for _ in range(3)]
        svc.db.session.add_all(rows); svc.db.session.commit()
        items = [(r.id, 'p', '1:1') for r in rows]
        svc._run_nanobanana_batch(app, items, [_real_png()], engine='openrouter')
        assert len(calls) == 1                  # rows 2-3 never reached the API
        svc.db.session.expire_all()
        for r in rows:
            row = svc.db.session.get(FaceDatasetImage, r.id)
            assert row.status == 'failed'
            assert 'openrouter:' in row.fail_reason
            assert expected in row.fail_reason
            assert 'remaining rows were stopped' in row.fail_reason


def test_a_transient_failure_does_not_stop_the_batch(app, monkeypatch):
    """The mirror invariant: one rate-limited image must not cancel the rest of a
    run the user is paying for."""
    import concurrent.futures
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import dataset_generation_service
    from app.services.openrouter import OpenRouterError
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise OpenRouterError('OpenRouter rate-limited the request (HTTP 429)')
        return _real_png()

    monkeypatch.setattr(dataset_generation_service, '_api_generate_fn', lambda engine: flaky)
    with app.app_context():
        import os
        ds = svc.create_dataset(LOCAL_USER, 'OR2', 'or2')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        rows = [FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model='openrouter')
                for _ in range(3)]
        svc.db.session.add_all(rows); svc.db.session.commit()
        svc._run_nanobanana_batch(app, [(r.id, 'p', '1:1') for r in rows],
                                  [_real_png()], engine='openrouter')
        assert len(calls) == 3                  # every row was still attempted
        svc.db.session.expire_all()
        done = [svc.db.session.get(FaceDatasetImage, r.id) for r in rows]
        assert done[0].status == 'failed' and 'rate-limited' in done[0].fail_reason
        # The generated rows carry the engine's own file tag, never another's.
        assert all(d.filename and '_ORFace_' in d.filename for d in done[1:])


def test_a_generated_file_is_tagged_with_the_engine_that_made_it(app, monkeypatch):
    """The tag is baked into the filename on disk and is how a dataset folder stays
    readable after a mixed run."""
    import concurrent.futures
    from app.config import LOCAL_USER
    from app.models import FaceDatasetImage
    from app.services import face_dataset_service as svc
    from app.services import dataset_generation_service
    monkeypatch.setattr(concurrent.futures, 'ThreadPoolExecutor', _SerialPool)
    monkeypatch.setattr(dataset_generation_service, '_api_generate_fn', lambda engine: (lambda *a, **k: _real_png()))
    with app.app_context():
        import os
        ds = svc.create_dataset(LOCAL_USER, 'OR3', 'or3')
        os.makedirs(svc._dataset_dir(ds.id), exist_ok=True)
        row = FaceDatasetImage(dataset_id=ds.id, status='pending', klein_model='openrouter')
        svc.db.session.add(row); svc.db.session.commit()
        svc._run_nanobanana_batch(app, [(row.id, 'p', '1:1')], [_real_png()],
                                  engine='openrouter')
        svc.db.session.expire_all()
        saved = svc.db.session.get(FaceDatasetImage, row.id)
        assert '_ORFace_' in saved.filename
        assert os.path.isfile(os.path.join(svc._dataset_dir(ds.id), saved.filename))
