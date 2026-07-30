import json, importlib

def _fresh(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    importlib.reload(config)
    return config

def test_defaults_when_no_file(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('server.port') == 5050
    assert config.get('engines.default') == 'chatgpt'
    assert config.is_configured() is False

def test_save_and_reload_deep_merge(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    config.save_config({'comfyui': {'api_url': 'http://10.0.0.2:8188'}})
    assert config.get('comfyui.api_url') == 'http://10.0.0.2:8188'
    assert config.get('server.port') == 5050          # untouched default survives
    assert config.is_configured() is True
    on_disk = json.loads((tmp_path / 'config.json').read_text(encoding='utf-8'))
    assert on_disk['comfyui']['api_url'] == 'http://10.0.0.2:8188'

def test_comfyui_dir_derivation(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    assert config.comfyui_dir('loras') is None        # unconfigured
    base = tmp_path / 'Comfy'
    (base / 'models' / 'loras').mkdir(parents=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    assert config.comfyui_dir('loras') == base / 'models' / 'loras'
    assert config.comfyui_dir('output') == base / 'output'

def test_secrets_roundtrip(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert config.secret('OPENAI_API_KEY') is None
    config.set_secrets({'OPENAI_API_KEY': 'sk-test-123'})
    assert config.secret('OPENAI_API_KEY') == 'sk-test-123'
    env_text = (config.ENV_PATH).read_text(encoding='utf-8')
    assert 'sk-test-123' in env_text

def test_secret_strips_trailing_whitespace(tmp_path, monkeypatch):
    """A pasted key with a trailing newline/space must not corrupt the Bearer header."""
    config = _fresh(monkeypatch, tmp_path)
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test-123\n')
    assert config.secret('OPENAI_API_KEY') == 'sk-test-123'
    monkeypatch.setenv('OPENAI_API_KEY', '  sk-test-456  ')
    assert config.secret('OPENAI_API_KEY') == 'sk-test-456'

def test_local_user_constant(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    assert config.LOCAL_USER == 'local'

def test_load_config_returns_defensive_copy(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    cfg = config.load_config()
    cfg['server']['port'] = 9999          # caller mutation must not corrupt the cache
    assert config.get('server.port') == 5050


def test_legacy_krea_default_pair_migrates_on_read_and_persists_on_next_save(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'krea': {'grounding_px': 1024, 'ref_boost': 4.0}}),
                    encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)

    # Reads do not rewrite user configuration, but the corrected profile applies
    # immediately to generation on an upgraded install.
    assert config.get('krea.grounding_px') == 512
    assert config.get('krea.ref_boost') == 0.25
    assert config.get('krea.steps') == 8
    untouched = json.loads(path.read_text(encoding='utf-8'))
    assert untouched['krea'] == {'grounding_px': 1024, 'ref_boost': 4.0}

    # Its next ordinary Settings save records the one-time migration too.
    config.save_config({'server': {'port': 5051}})
    saved = json.loads(path.read_text(encoding='utf-8'))['krea']
    assert saved['grounding_px'] == 512
    assert saved['ref_boost'] == 0.25
    assert saved['steps'] == 8
    assert saved['calibration_version'] == config.KREA_CALIBRATION_VERSION


def test_previous_krea_default_profile_migrates_but_a_v2_custom_profile_survives(
        tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'krea': {
        'calibration_version': 2, 'grounding_px': 512, 'ref_boost': 1.0, 'steps': 10,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 512
    assert config.get('krea.ref_boost') == 0.25
    assert config.get('krea.steps') == 8

    path.write_text(json.dumps({'krea': {
        'calibration_version': 2, 'grounding_px': 512, 'ref_boost': 0.5, 'steps': 8,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 512
    assert config.get('krea.ref_boost') == 0.5
    assert config.get('krea.steps') == 8

    path.write_text(json.dumps({'krea': {
        'calibration_version': 2, 'grounding_px': 1024, 'ref_boost': 4.0, 'steps': 10,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 1024
    assert config.get('krea.ref_boost') == 4.0
    assert config.get('krea.steps') == 10


def test_a_custom_legacy_krea_calibration_is_not_rewritten(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'krea': {'grounding_px': 768, 'ref_boost': 4.0}}),
                    encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 768
    assert config.get('krea.ref_boost') == 4.0

    # An explicit post-update choice, even the old 1024 value, receives a marker
    # and is never mistaken for the historical untouched pair on a later reload.
    config.save_config({'krea': {'grounding_px': 1024}})
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 1024
    assert config.get('krea.ref_boost') == 4.0
    saved = json.loads(path.read_text(encoding='utf-8'))['krea']
    assert saved['calibration_version'] == config.KREA_CALIBRATION_VERSION


def test_a_legacy_krea_pair_with_custom_steps_is_not_rewritten(tmp_path, monkeypatch):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'krea': {
        'grounding_px': 1024, 'ref_boost': 4.0, 'steps': 20,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 1024
    assert config.get('krea.ref_boost') == 4.0
    assert config.get('krea.steps') == 20
