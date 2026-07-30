from unittest.mock import patch

import pytest

from app.services import ollama_gpu_fence as fence


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


def _ps(*names):
    return _Response({'models': [{'name': name} for name in names]})


@pytest.fixture(autouse=True)
def _reset_fence():
    fence.reset_for_tests()
    yield
    fence.reset_for_tests()


def test_endpoint_scope_never_treats_a_lookalike_hostname_as_local():
    assert fence._endpoint_scope('http://127.0.0.1:11434')[0] == 'local'
    assert fence._endpoint_scope('http://[::1]:11434') == ('local', 'http://[::1]:11434')
    assert fence._endpoint_scope('http://127.evil.example:11434')[0] == 'remote'
    assert fence._endpoint_scope('http://127.0.0.1:11434/ollama')[0] == 'unknown'
    assert fence._endpoint_scope('http://127.0.0.1:11434/?tenant=lds')[0] == 'unknown'
    assert fence._endpoint_scope('http://localhost:not-a-port')[0] == 'unknown'


def test_transient_probe_failure_can_recover_on_the_next_comfy_handoff():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', side_effect=[_Response({}, status_code=500), _ps()]), \
            patch.object(fence, '_configured_local_endpoint', return_value=('local', endpoint)):
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'blocked'
        assert fence.ensure_released_for_comfy() is True

def test_mark_before_generate_refuses_a_preexisting_local_model_without_unload():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', return_value=_ps('manual-model')) as get, \
            patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'blocked'
    assert get.call_count == 1
    post.assert_not_called()


def test_comfy_handoff_rechecks_between_cells_and_preserves_manual_model():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', side_effect=[_ps(), _ps('manual-model')]) as get, \
            patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'local'
        assert fence.ensure_released_for_comfy() is False
    assert get.call_count == 2
    post.assert_not_called()


def test_comfy_handoff_releases_only_the_model_lds_admitted_then_confirms_empty():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', side_effect=[_ps(), _ps('lds-model'), _ps()]), \
            patch.object(fence.requests, 'post', return_value=_Response({})) as post:
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'local'
        assert fence.ensure_released_for_comfy() is True
    post.assert_called_once_with(
        f'{endpoint}/api/generate',
        json={'model': 'lds-model', 'keep_alive': 0},
        timeout=(10, 30), allow_redirects=False)


def test_a_remote_endpoint_is_never_probed_or_unloaded():
    remote = 'http://192.168.10.20:11434'
    with patch.object(fence.requests, 'get') as get, patch.object(fence.requests, 'post') as post:
        assert fence.mark_before_generate(remote, 'remote-model') == 'remote'
        assert fence.release_owned_models(ollama_url=remote) is None
    get.assert_not_called()
    post.assert_not_called()


def test_empty_probe_clears_a_foreign_marker_only_after_it_observes_empty():
    endpoint = 'http://127.0.0.1:11434'
    with fence._lock:
        fence._foreign_local_endpoints.add(endpoint)
    with patch.object(fence.requests, 'get', return_value=_ps()) as get, \
            patch.object(fence.requests, 'post') as post:
        assert fence.release_owned_models(ollama_url=endpoint) is True
    assert get.call_count == 1
    post.assert_not_called()
    with fence._lock:
        assert endpoint not in fence._foreign_local_endpoints


def test_malformed_or_non_successful_probe_blocks_comfy_handoff():
    endpoint = 'http://127.0.0.1:11434'
    with patch.object(fence.requests, 'get', return_value=_Response({}, status_code=500)):
        assert fence.mark_before_generate(endpoint, 'lds-model') == 'blocked'
        assert fence.ensure_released_for_comfy() is False
