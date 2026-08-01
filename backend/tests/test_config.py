import json, importlib, os
import pytest


UNSAFE_SECRET_CHARS = [
    pytest.param('\r', id='cr'),
    pytest.param('\n', id='lf'),
    pytest.param('\x00', id='nul'),
    pytest.param('\x0b', id='vertical-tab'),
    pytest.param('\x0c', id='form-feed'),
    pytest.param('\x1c', id='file-separator'),
    pytest.param('\x1d', id='group-separator'),
    pytest.param('\x1e', id='record-separator'),
    pytest.param('\x85', id='next-line'),
    pytest.param('\u2028', id='unicode-line-separator'),
    pytest.param('\u2029', id='unicode-paragraph-separator'),
    pytest.param('\t', id='other-control'),
    pytest.param('\u200b', id='unicode-format'),
]

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


def test_secret_printable_characters_roundtrip_exactly(tmp_path, monkeypatch):
    config = _fresh(monkeypatch, tmp_path)
    value = "sk printable # = ' \\\\ ü"

    config.set_secrets({'OPENAI_API_KEY': value})

    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    from dotenv import load_dotenv
    load_dotenv(config.ENV_PATH, override=True)
    assert config.secret('OPENAI_API_KEY') == value


@pytest.mark.parametrize('separator', UNSAFE_SECRET_CHARS)
def test_set_secrets_rejects_controls_without_mutating_file_or_environment(
        tmp_path, monkeypatch, separator):
    config = _fresh(monkeypatch, tmp_path)
    config.ENV_PATH.write_bytes(b"GEMINI_API_KEY='old-value'\n")
    before = config.ENV_PATH.read_bytes()
    monkeypatch.setenv('OPENAI_API_KEY', 'runtime-before')
    monkeypatch.delenv('FLASK_DEBUG', raising=False)

    with pytest.raises(ValueError, match='single line'):
        config.set_secrets({
            'OPENAI_API_KEY': f'new-value{separator}FLASK_DEBUG=1',
        })

    assert config.ENV_PATH.read_bytes() == before
    assert config.secret('OPENAI_API_KEY') == 'runtime-before'
    assert 'FLASK_DEBUG' not in os.environ


@pytest.mark.parametrize('separator', [
    pytest.param('\x0b', id='vertical-tab'),
    pytest.param('\u2028', id='unicode-line-separator'),
])
def test_set_secrets_refuses_to_normalize_poisoned_existing_env(
        tmp_path, monkeypatch, separator):
    config = _fresh(monkeypatch, tmp_path)
    poisoned = f'OPENAI_API_KEY=old{separator}FLASK_DEBUG=1\n'.encode('utf-8')
    config.ENV_PATH.write_bytes(poisoned)
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('FLASK_DEBUG', raising=False)

    with pytest.raises(ValueError, match='existing .env'):
        config.set_secrets({'GEMINI_API_KEY': 'safe-value'})

    assert config.ENV_PATH.read_bytes() == poisoned
    assert 'GEMINI_API_KEY' not in os.environ
    assert 'FLASK_DEBUG' not in os.environ

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
    assert config.get('krea.ref_boost') == 4
    assert config.get('krea.steps') == 8
    untouched = json.loads(path.read_text(encoding='utf-8'))
    assert untouched['krea'] == {'grounding_px': 1024, 'ref_boost': 4.0}

    # Its next ordinary Settings save records the one-time migration too.
    config.save_config({'server': {'port': 5051}})
    saved = json.loads(path.read_text(encoding='utf-8'))['krea']
    assert saved['grounding_px'] == 512
    assert saved['ref_boost'] == 4
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
    assert config.get('krea.ref_boost') == 4
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


def test_v3_krea_default_ref_boost_migrates_but_a_v3_custom_value_survives(
        tmp_path, monkeypatch):
    """v3 shipped ref_boost=0.25; v4 raises it to 4 (too weak for a
    character-LoRA-augmented setup) -- an untouched v3 install upgrades, a
    deliberately customised v3 value is left alone."""
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'krea': {
        'calibration_version': 3, 'grounding_px': 512, 'ref_boost': 0.25, 'steps': 8,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 512
    assert config.get('krea.ref_boost') == 4
    assert config.get('krea.steps') == 8

    path.write_text(json.dumps({'krea': {
        'calibration_version': 3, 'grounding_px': 512, 'ref_boost': 1.5, 'steps': 8,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.ref_boost') == 1.5, 'a v3 install that already tuned it keeps its choice'


def test_v3_ref_boost_migration_never_touches_a_customised_grounding_px(tmp_path, monkeypatch):
    """v3->v4 changed ONLY ref_boost's default -- grounding_px is a separate,
    independently-tuned dial and must survive this migration untouched even
    though ref_boost itself matches the untouched-v3 pattern exactly."""
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'krea': {
        'calibration_version': 3, 'grounding_px': 1280, 'ref_boost': 0.25, 'steps': 8,
    }}), encoding='utf-8')
    config = _fresh(monkeypatch, tmp_path)
    assert config.get('krea.grounding_px') == 1280
    assert config.get('krea.ref_boost') == 4
    assert config.get('krea.steps') == 8


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
