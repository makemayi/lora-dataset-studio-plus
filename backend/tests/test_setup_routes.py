import pytest


@pytest.fixture(autouse=True)
def _reset_runs():
    from app import setup_installer
    setup_installer._runs.clear()
    yield
    setup_installer._runs.clear()


def test_install_unknown_action_404(client):
    assert client.post('/api/setup/install/rm_rf').status_code == 404


def test_status_unknown_action_404(client):
    assert client.get('/api/setup/install/rm_rf/status').status_code == 404


def test_install_ml_extras_starts(client, monkeypatch):
    from app import setup_installer
    monkeypatch.setattr(setup_installer, 'start',
                        lambda a: {'state': 'running', 'returncode': None, 'log': []})
    r = client.post('/api/setup/install/ml_extras')
    assert r.status_code == 200 and r.get_json()['state'] == 'running'


def test_install_conflict_409(client, monkeypatch):
    from app import setup_installer
    def _raise(a): raise setup_installer.AlreadyRunning(a)
    monkeypatch.setattr(setup_installer, 'start', _raise)
    assert client.post('/api/setup/install/ml_extras').status_code == 409


def test_install_ollama_precondition_400(client, monkeypatch):
    from app import config, setup_installer
    config.save_config({'ollama': {'url': '', 'vision_model': ''}})
    # real start() runs the precondition check and raises before spawning a thread
    assert client.post('/api/setup/install/ollama_model').status_code == 400


def test_status_idle(client):
    r = client.get('/api/setup/install/ml_extras/status')
    assert r.status_code == 200 and r.get_json()['state'] == 'idle'


# --- "Install everything" orchestrator endpoints ---------------------------

def test_install_all_plan_endpoint(client, monkeypatch):
    from app import capabilities, setup_installer
    monkeypatch.setattr(capabilities, 'probe', lambda force=False: {'python': {'ml_supported': True}})
    monkeypatch.setattr(setup_installer, 'install_all_plan', lambda caps: ['face_scoring', 'masks'])
    r = client.get('/api/setup/install-all/plan')
    assert r.status_code == 200 and r.get_json()['plan'] == ['face_scoring', 'masks']


def test_install_all_starts_plan(client, monkeypatch):
    from app import capabilities, setup_installer
    started = []
    monkeypatch.setattr(capabilities, 'probe', lambda force=False: {})
    monkeypatch.setattr(setup_installer, 'start',
                        lambda a: (started.append(a) or {'state': 'running', 'returncode': None,
                                                          'log': [], 'progress': None,
                                                          'waiting_for': None,
                                                          'manual_command': ''}))
    r = client.post('/api/setup/install-all')
    body = r.get_json()
    assert r.status_code == 200
    # the {} snapshot -> the always-runnable extras (scrape stack + the three ML ones)
    assert body['plan'] == ['scrape_extras', 'face_scoring', 'masks', 'watermark_inpaint']
    assert set(body['statuses']) == set(body['plan'])
    assert started == body['plan']


def test_install_all_status_batches_requested_actions(client):
    r = client.get('/api/setup/install-all/status',
                   query_string={'actions': 'face_scoring,masks,not_real'})
    body = r.get_json()
    assert r.status_code == 200
    assert set(body['statuses']) == {'face_scoring', 'masks'}   # unknown dropped
    assert body['statuses']['face_scoring']['state'] == 'idle'


# --- ComfyUI directory validation endpoint (Setup Volet 1) -----------------

def test_validate_comfyui_dir_blank(client):
    r = client.get('/api/setup/comfyui-dir?path=')
    assert r.status_code == 200 and r.get_json()['status'] == 'empty'


def test_validate_comfyui_dir_valid(client, tmp_path):
    (tmp_path / 'main.py').touch()
    (tmp_path / 'models').mkdir()
    r = client.get('/api/setup/comfyui-dir', query_string={'path': str(tmp_path)})
    assert r.status_code == 200 and r.get_json()['status'] == 'valid'


def test_validate_comfyui_dir_nested_suggests_child(client, tmp_path):
    child = tmp_path / 'ComfyUI'
    child.mkdir()
    (child / 'main.py').touch()
    (child / 'models').mkdir()
    r = client.get('/api/setup/comfyui-dir', query_string={'path': str(tmp_path)})
    body = r.get_json()
    assert body['status'] == 'nested'
    assert body['suggestion'].endswith('ComfyUI')


def test_validate_comfyui_dir_missing(client, tmp_path):
    r = client.get('/api/setup/comfyui-dir', query_string={'path': str(tmp_path / 'nope')})
    assert r.get_json()['status'] == 'missing'


# --- Setup's cheap service-readiness poll ----------------------------------

def test_runtime_readiness_comfyui_is_always_the_users_and_never_polled(
        client, monkeypatch):
    """Nothing bundles a ComfyUI any more, so readiness never claims to own one
    and never spends a probe on it — no runtime environment variable can make
    it say otherwise."""
    from app import capabilities
    calls = []

    monkeypatch.setenv('LDS_RUNTIME', 'anything-at-all')
    monkeypatch.setattr(capabilities.cfg, 'get', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(
        capabilities,
        '_http_ok',
        lambda url, timeout=3, **_kwargs: calls.append((url, timeout)) or False,
    )

    response = client.get('/api/setup/runtime-readiness')
    body = response.get_json()

    assert body['comfyui'] == {
        'mode': 'external', 'state': 'manual',
        'ready': False, 'poll': False,
    }
    assert not [url for url, _timeout in calls if '8188' in url]
    assert all(timeout <= 1.0 for _, timeout in calls)
    assert response.headers['Cache-Control'] == 'no-store'


def test_runtime_readiness_never_runs_the_full_capability_probe(
        client, monkeypatch):
    from app import capabilities

    monkeypatch.setattr(
        capabilities.cfg, 'get',
        lambda key, default=None: 'http://127.0.0.1:11434')
    monkeypatch.setattr(capabilities, '_http_ok', lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        capabilities,
        'probe',
        lambda *_args, **_kwargs: pytest.fail('full capability probe was called'),
    )

    body = client.get('/api/setup/runtime-readiness').get_json()

    assert body['ollama'] == {
        'mode': 'local', 'state': 'ready',
        'ready': True, 'poll': False,
    }


def test_runtime_readiness_probes_the_configured_ollama_url_only(
        client, monkeypatch):
    """A launcher environment variable must not redirect or disable the probe:
    `ollama.url` in the config is the single source of truth."""
    from app import capabilities
    calls = []

    monkeypatch.setenv('LDS_OLLAMA_MODE', 'none')
    monkeypatch.setattr(
        capabilities.cfg, 'get', lambda *_args, **_kwargs: 'http://127.0.0.1:11434')
    monkeypatch.setattr(
        capabilities,
        '_http_ok',
        lambda url, **_kwargs: calls.append(url) or False,
    )

    body = client.get('/api/setup/runtime-readiness').get_json()

    assert body == {
        'comfyui': {
            'mode': 'external', 'state': 'manual',
            'ready': False, 'poll': False,
        },
        'ollama': {
            'mode': 'local', 'state': 'unreachable',
            'ready': False, 'poll': False,
        },
    }
    assert calls == ['http://127.0.0.1:11434/api/tags']


def test_runtime_readiness_http_probe_is_bounded_streamed_and_closed(monkeypatch):
    from app import capabilities
    seen = {}

    class Response:
        status_code = 204
        closed = False

        def close(self):
            self.closed = True

    response = Response()

    def fake_get(url, **kwargs):
        seen.update(url=url, **kwargs)
        return response

    monkeypatch.setattr(capabilities.requests, 'get', fake_get)

    assert capabilities._http_ok(
        'http://127.0.0.1:11434/api/tags', timeout=99, readiness=True) is True
    assert seen == {
        'url': 'http://127.0.0.1:11434/api/tags',
        'timeout': 1.0,
        'allow_redirects': False,
        'stream': True,
    }
    assert response.closed is True


def test_runtime_readiness_response_never_exposes_configured_urls_or_paths(
        client, monkeypatch):
    from app import capabilities
    secret_url = 'http://user:password@private-host.internal:11434'
    secret_path = r'C:\private\ComfyUI'

    monkeypatch.setattr(
        capabilities.cfg,
        'get',
        lambda key, *_args: secret_url if key == 'ollama.url' else secret_path,
    )
    monkeypatch.setattr(capabilities, '_http_ok', lambda *_args, **_kwargs: False)

    response = client.get('/api/setup/runtime-readiness')
    body = response.get_json()
    raw = response.get_data(as_text=True)

    assert response.status_code == 200
    assert set(body) == {'comfyui', 'ollama'}
    assert 'password' not in raw
    assert 'private-host' not in raw
    assert 'private' not in raw


def test_the_removed_ollama_deployment_route_says_so(client):
    """The Docker deployments owned this endpoint's only two answers. A stale
    SPA -- a tab left open across the update -- must get a named 410 rather than
    a generic failure, so the message can point at the setting that replaced
    it."""
    response = client.put('/api/setup/ollama-deployment', json={'mode': 'none'})

    assert response.status_code == 410
    assert 'ollama.url' in response.get_json()['error']


def test_ollama_cancel_endpoint_is_idempotent(client):
    from app import setup_installer

    setup_installer._runs['ollama_model'] = setup_installer._new_run()
    first = client.post('/api/setup/install/ollama_model/cancel')
    second = client.post('/api/setup/install/ollama_model/cancel')

    assert first.status_code == second.status_code == 200
    assert first.get_json()['cancel_requested'] is True
    assert second.get_json()['cancel_requested'] is True


def test_cancel_rejects_non_streamed_installs(client):
    response = client.post('/api/setup/install/ml_extras/cancel')
    assert response.status_code == 409
