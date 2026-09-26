"""vision_llamacpp + the vision.backend dispatch seam.

llama.cpp is the SECOND local vision engine: services/vision_ollama routes its
three public seams (describe / generate_text / unload) at services/vision_llamacpp
when `vision.backend` == 'llamacpp', mirroring the Ollama contract over
llama-server's OpenAI-compatible /v1/chat/completions. Every network seam is
mocked; no server is ever contacted.
"""
import base64
import io

import pytest
import requests

from app import config as cfg


# ---------------------------------------------------------------- helpers ---

def _jpeg_bytes(color=(120, 130, 140), size=(8, 8)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, 'JPEG', quality=90)
    return buf.getvalue()


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.HTTPError(f'HTTP {self.status_code}')
            err.response = self
            raise err

    @property
    def text(self):
        return str(self._payload)


def _chat_payload(content='a caption', reasoning=''):
    return {'choices': [{'message': {'content': content,
                                     'reasoning_content': reasoning}}]}


@pytest.fixture()
def llamacpp_backend():
    cfg.save_config({'vision': {'backend': 'llamacpp'}})
    yield
    cfg.save_config({'vision': {'backend': 'ollama'}})


@pytest.fixture()
def llamacpp_down():
    cfg.save_config({'vision': {'backend': 'llamacpp'},
                     'llamacpp': {'url': 'http://127.0.0.1:59999',
                                  'vision_model': 'qwen-test'}})
    yield
    cfg.save_config({'vision': {'backend': 'ollama'}})


# ------------------------------------------------------------ dispatching ---

def test_default_backend_is_ollama(app):
    from app.services import vision_ollama
    with app.app_context():
        assert vision_ollama._use_llamacpp() is False


def test_flipping_config_routes_describe_at_llamacpp(app, monkeypatch, llamacpp_backend):
    from app.services import vision_ollama, vision_llamacpp
    hits = []

    def _no_post(*_a, **_kw):  # the ollama seam must never be touched
        raise AssertionError('ollama /api/generate called while backend=llamacpp')

    monkeypatch.setattr(vision_ollama.requests, 'post', _no_post)
    monkeypatch.setattr(vision_llamacpp.requests, 'post',
                        lambda url, json=None, timeout=None:
                        hits.append(url) or _FakeResp(_chat_payload('from llamacpp')))
    with app.app_context():
        out = vision_ollama.describe_image_ollama(_jpeg_bytes(), 'p')
    assert out == 'from llamacpp'
    assert hits and hits[0].endswith('/v1/chat/completions')


def test_flipping_config_routes_generate_and_unload(app, monkeypatch, llamacpp_backend):
    from app.services import vision_ollama, vision_llamacpp
    monkeypatch.setattr(vision_llamacpp.requests, 'post',
                        lambda *a, **kw: _FakeResp(_chat_payload('short text')))
    with app.app_context():
        assert vision_ollama.generate_text_ollama('p') == 'short text'
        # No remote unload exists for llama-server: the seam reports success so
        # batch finally-blocks stay engine-agnostic.
        assert vision_ollama.unload_vision_model() is True


def test_default_backend_still_uses_ollama_generate(app, monkeypatch):
    from app.services import vision_ollama, vision_llamacpp

    def _no_llamacpp(*_a, **_kw):
        raise AssertionError('llamacpp engine reached while backend=ollama')

    monkeypatch.setattr(vision_llamacpp.requests, 'post', _no_llamacpp)
    monkeypatch.setattr(vision_ollama.requests, 'post',
                        lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError('down')))
    with app.app_context():
        # best-effort: '' — but the ATTEMPT went at the ollama endpoint, not llamacpp
        assert vision_ollama.generate_text_ollama('p') == ''


# ------------------------------------------------- describe_image_llamacpp ---

def test_describe_success_reads_content(app, monkeypatch):
    from app.services import vision_llamacpp
    seen = {}

    def _post(url, json=None, timeout=None):
        seen['url'], seen['payload'] = url, json
        return _FakeResp(_chat_payload('a red jacket'))

    monkeypatch.setattr(vision_llamacpp.requests, 'post', _post)
    with app.app_context():
        out = vision_llamacpp.describe_image_llamacpp(_jpeg_bytes(), 'describe')
    assert out == 'a red jacket'
    assert seen['url'].endswith('/v1/chat/completions')
    msg = seen['payload']['messages'][0]['content']
    assert msg[0]['type'] == 'text' and msg[0]['text'] == 'describe'
    img_part = msg[1]['image_url']['url']
    assert img_part.startswith('data:image/jpeg;base64,')
    base64.b64decode(img_part.split(',', 1)[1])       # decodable payload
    assert seen['payload']['max_tokens'] == 800        # num_predict mapping


def test_describe_empty_content_falls_back_to_reasoning(app, monkeypatch):
    from app.services import vision_llamacpp
    monkeypatch.setattr(vision_llamacpp.requests, 'post',
                        lambda *a, **kw: _FakeResp(_chat_payload('', reasoning='the thinking')))
    with app.app_context():
        assert vision_llamacpp.describe_image_llamacpp(_jpeg_bytes(), 'p') == 'the thinking'


def test_describe_prefer_json_takes_field_holding_object(app, monkeypatch):
    from app.services import vision_llamacpp
    monkeypatch.setattr(vision_llamacpp.requests, 'post',
                        lambda *a, **kw: _FakeResp(_chat_payload('prose first',
                                                                 reasoning='{"a": 1}')))
    with app.app_context():
        out = vision_llamacpp.describe_image_llamacpp(_jpeg_bytes(), 'p', prefer_json=True)
    assert out == '{"a": 1}'


def test_describe_json_fmt_maps_to_response_format(app, monkeypatch):
    from app.services import vision_llamacpp
    seen = {}

    def _post(url, json=None, timeout=None):
        seen['payload'] = json
        return _FakeResp(_chat_payload('{}'))

    monkeypatch.setattr(vision_llamacpp.requests, 'post', _post)
    with app.app_context():
        vision_llamacpp.describe_image_llamacpp(_jpeg_bytes(), 'p', fmt='json')
    assert seen['payload']['response_format'] == {'type': 'json_object'}


def test_describe_unsafe_image_never_leaves_the_machine(app, monkeypatch):
    from app.services import vision_llamacpp

    def _no_post(*_a, **_kw):
        raise AssertionError('request sent with undecodable image')

    monkeypatch.setattr(vision_llamacpp.requests, 'post', _no_post)
    with app.app_context():
        assert vision_llamacpp.describe_image_llamacpp(b'not an image', 'p') == ''


def test_describe_http_rejection_names_the_reason(app, monkeypatch):
    from app.services import vision_llamacpp
    monkeypatch.setattr(vision_llamacpp.requests, 'post',
                        lambda *a, **kw: _FakeResp({'error': 'no multimodal capability'},
                                                   status=400))
    with app.app_context():
        assert vision_llamacpp.describe_image_llamacpp(_jpeg_bytes(), 'p') == ''
        with pytest.raises(RuntimeError, match=r'llama\.cpp rejected.*no multimodal'):
            vision_llamacpp.describe_image_llamacpp(_jpeg_bytes(), 'p', auto_start_local=True)


def test_describe_unreachable_auto_start_cannot_fix_it(app, llamacpp_down, monkeypatch):
    from app.services import vision_llamacpp

    def _refused(*_a, **_kw):
        raise requests.ConnectionError('refused')

    monkeypatch.setattr(vision_llamacpp.requests, 'post', _refused)
    with app.app_context():
        assert vision_llamacpp.describe_image_llamacpp(_jpeg_bytes(), 'p') == ''
        with pytest.raises(RuntimeError, match='unreachable.*llama-server is started by you'):
            vision_llamacpp.describe_image_llamacpp(_jpeg_bytes(), 'p', auto_start_local=True)


# ---------------------------------------------------- generate_text variants --

def test_generate_strict_raises_with_reason(app, llamacpp_down, monkeypatch):
    from app.services import vision_llamacpp
    monkeypatch.setattr(vision_llamacpp.requests, 'post',
                        lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError('down')))
    with app.app_context():
        assert vision_llamacpp.generate_text_llamacpp('p') == ''
        with pytest.raises(RuntimeError, match='down'):
            vision_llamacpp.generate_text_llamacpp('p', strict=True)


# ------------------------------------------------------- models / probing ----

def test_list_models_parses_v1_models(app, monkeypatch):
    from app.services import vision_llamacpp

    class _Resp:
        status_code = 200

        def json(self):
            return {'data': [{'id': 'qwen3-vl-8b'}, {'id': ''}, {'id': 'qwen3-vl-8b'}]}

    monkeypatch.setattr(vision_llamacpp.requests, 'get', lambda *a, **kw: _Resp())
    with app.app_context():
        out = vision_llamacpp.list_models()
    assert out == {'ok': True, 'reachable': True, 'models': ['qwen3-vl-8b']}


def test_list_models_unreachable_is_handled(app, monkeypatch):
    from app.services import vision_llamacpp

    def _down(*_a, **_kw):
        raise requests.ConnectionError('refused')

    monkeypatch.setattr(vision_llamacpp.requests, 'get', _down)
    with app.app_context():
        assert vision_llamacpp.list_models() == {'ok': False, 'reachable': False, 'models': []}


def test_probe_llamacpp_connection_end_to_end(app, monkeypatch):
    from app import capabilities

    monkeypatch.setattr(capabilities, '_http_ok', lambda url, **kw: url.endswith('/v1/models'))
    monkeypatch.setattr(capabilities, 'cfg', cfg)
    cfg.save_config({'llamacpp': {'url': 'http://127.0.0.1:59999', 'vision_model': ''}})
    with app.app_context():
        # blank model = server default: OK by design
        assert capabilities.probe_llamacpp_connection() == {'ok': True,
                                                            'detail': 'server default model'}
        cfg.save_config({'llamacpp': {'vision_model': 'qwen3-vl-8b'}})
        monkeypatch.setattr('app.services.vision_llamacpp.requests.get',
                            lambda *a, **kw: _FakeResp({'data': [{'id': 'qwen3-vl-8b'}]}))
        out = capabilities.probe_llamacpp_connection()
        assert out['ok'] is True and 'loaded' in out['detail']
        cfg.save_config({'llamacpp': {'vision_model': 'not-loaded'}})
        out = capabilities.probe_llamacpp_connection()
        assert out['ok'] is False and 'not loaded' in out['detail']


# ----------------------------------------------------------------- routes ----

def test_llamacpp_models_route(app, monkeypatch):
    from app.services import vision_llamacpp
    monkeypatch.setattr(vision_llamacpp.requests, 'get',
                        lambda *a, **kw: _FakeResp({'data': [{'id': 'm1'}]}))
    client = app.test_client()
    r = client.get('/api/llamacpp/models')
    assert r.status_code == 200
    assert r.get_json() == {'ok': True, 'reachable': True, 'models': ['m1']}


def test_vision_models_route_follows_the_backend(app, monkeypatch, llamacpp_backend):
    from app.services import vision_llamacpp
    monkeypatch.setattr(vision_llamacpp.requests, 'get',
                        lambda *a, **kw: _FakeResp({'data': [{'id': 'm1'}]}))
    client = app.test_client()
    body = client.get('/api/vision/models').get_json()
    assert body['engine'] == 'llamacpp'
    assert body['can_pull'] is False          # no pull API → the UI hides the pull box
    assert body['models'] == ['m1']

    cfg.save_config({'vision': {'backend': 'ollama'}})
    monkeypatch.setattr('app.services.ollama_control._reachable', lambda url: True)
    monkeypatch.setattr('app.capabilities._ollama_tags', lambda url: ['tag:8b'])
    body = client.get('/api/vision/models').get_json()
    assert body['engine'] == 'ollama'
    assert body['can_pull'] is True
    assert body['models'] == ['tag:8b']


# --------------------------------------------------- capabilities payload ----

def test_capabilities_payload_carries_llamacpp_and_vision(client, monkeypatch):
    from app import capabilities
    monkeypatch.setattr(capabilities, 'probe_llamacpp',
                        lambda: {'ok': True, 'detail': 'http://127.0.0.1:8080'})
    monkeypatch.setattr(capabilities, 'probe_llamacpp_model',
                        lambda reachable=None, model=None: {'ok': True, 'detail': 'x'})
    r = client.get('/api/capabilities')
    caps = r.get_json()
    assert caps['llamacpp']['reachable'] is True
    assert caps['vision']['backend'] == 'ollama'        # shipped default, unchanged
    assert caps['captioners']['llamacpp'] is True
