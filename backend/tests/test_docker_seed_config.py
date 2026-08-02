"""The GPU image's config seeder: fills what is empty, never what is set."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def seeder():
    """Load the seeder by path. It is a container-boot script, not part of the
    `app` package, and it deliberately imports nothing from it."""
    path = REPO_ROOT / 'packaging' / 'docker' / 'seed_comfy_config.py'
    spec = importlib.util.spec_from_file_location('seed_comfy_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seeds_the_container_paths_into_an_empty_config(seeder, tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    monkeypatch.setenv('LDS_CONFIG', str(config))
    monkeypatch.delenv('LDS_OLLAMA_URL', raising=False)

    assert seeder.main() == 0

    written = json.loads(config.read_text(encoding='utf-8'))
    assert 'base_dir' not in written['comfyui']
    assert written['comfyui']['api_url'] == 'http://127.0.0.1:8188'
    assert 'models_dir' not in written['comfyui']
    assert 'loras_dir' not in written['comfyui']
    assert 'input_dir' not in written['comfyui']
    assert 'output_dir' not in written['comfyui']
    assert 'ollama' not in written


def test_never_overwrites_a_path_the_user_chose(seeder, tmp_path, monkeypatch):
    """This runs on EVERY boot, so anything set in Settings has to survive it."""
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({
        'comfyui': {'base_dir': '/my/own/comfy'},
        'paths': {'dataset_images_root': '/keep/me'},
    }), encoding='utf-8')
    monkeypatch.setenv('LDS_CONFIG', str(config))

    assert seeder.main() == 0

    written = json.loads(config.read_text(encoding='utf-8'))
    assert written['comfyui']['base_dir'] == '/my/own/comfy'
    assert written['paths']['dataset_images_root'] == '/keep/me'


def test_base_dir_is_whichever_folder_actually_holds_models(seeder, tmp_path, monkeypatch):
    """capabilities._is_comfyui_dir accepts a folder with models/ plus main.py OR
    custom_nodes/. Under upstream's BASE_DIRECTORY layout the checkout has neither
    models/ nor custom_nodes/, so pointing at it would fail that check and the app
    would find no models at all."""
    basedir = tmp_path / 'basedir'
    (basedir / 'models').mkdir(parents=True)
    checkout = tmp_path / 'ComfyUI'
    checkout.mkdir()
    monkeypatch.setattr(seeder, 'COMFY_ROOT_CANDIDATES', (str(basedir), str(checkout)))

    assert seeder.comfy_root() == str(basedir)


def test_base_dir_stays_unset_when_no_models_folder_exists(seeder, tmp_path, monkeypatch):
    """First boot can reach the seeder before ComfyUI has created anything. A
    guess would become sticky config, so only a probed root is safe to persist."""
    monkeypatch.setattr(seeder, 'COMFY_ROOT_CANDIDATES',
                        (str(tmp_path / 'nope'), str(tmp_path / 'also-nope')))

    assert seeder.comfy_root() is None


def test_main_honours_the_probed_root(seeder, tmp_path, monkeypatch):
    """main() must actually use comfy_root(), not a hardcoded constant."""
    config = tmp_path / 'config.json'
    monkeypatch.setenv('LDS_CONFIG', str(config))
    basedir = tmp_path / 'basedir'
    (basedir / 'models').mkdir(parents=True)
    monkeypatch.setattr(seeder, 'COMFY_ROOT_CANDIDATES', (str(basedir),))

    assert seeder.main() == 0

    written = json.loads(config.read_text(encoding='utf-8'))
    assert written['comfyui']['base_dir'] == str(basedir)


def test_main_defers_base_dir_when_no_root_has_been_probed(
        seeder, tmp_path, monkeypatch, capsys):
    """A first boot can race ComfyUI's own setup. It must log that it will retry,
    without baking an unverified checkout path into persistent config."""
    config = tmp_path / 'config.json'
    monkeypatch.setenv('LDS_CONFIG', str(config))
    monkeypatch.setattr(seeder, 'COMFY_ROOT_CANDIDATES',
                        (str(tmp_path / 'nope'), str(tmp_path / 'also-nope')))

    assert seeder.main() == 0

    out = capsys.readouterr().out
    written = json.loads(config.read_text(encoding='utf-8'))
    assert 'base_dir' not in written['comfyui']
    assert '/comfy/mnt/ComfyUI' not in out
    assert 'leaving comfyui.base_dir unset' in out
    assert 'next boot' in out


def test_main_does_not_warn_when_a_real_models_folder_was_found(seeder, tmp_path, monkeypatch, capsys):
    config = tmp_path / 'config.json'
    monkeypatch.setenv('LDS_CONFIG', str(config))
    basedir = tmp_path / 'basedir'
    (basedir / 'models').mkdir(parents=True)
    monkeypatch.setattr(seeder, 'COMFY_ROOT_CANDIDATES', (str(basedir),))

    assert seeder.main() == 0

    out = capsys.readouterr().out
    assert str(basedir) in out
    assert 'fallback' not in out


def test_first_boot_fallback_is_repaired_when_models_appear_on_next_boot(
        seeder, tmp_path, monkeypatch):
    """The seeder runs before ComfyUI and can win the first-boot race. It must
    defer the root, then seed the verified parent on the following boot."""
    config = tmp_path / 'config.json'
    monkeypatch.setenv('LDS_CONFIG', str(config))
    basedir = tmp_path / 'basedir'
    checkout = tmp_path / 'ComfyUI'
    monkeypatch.setattr(seeder, 'COMFY_ROOT_CANDIDATES',
                        (str(basedir), str(checkout)))

    # Boot 1: upstream has not created either tree yet.
    assert seeder.main() == 0
    first = json.loads(config.read_text(encoding='utf-8'))
    assert 'base_dir' not in first['comfyui']

    # Boot 2: BASE_DIRECTORY has created its real models root.
    (basedir / 'models').mkdir(parents=True)
    assert seeder.main() == 0
    second = json.loads(config.read_text(encoding='utf-8'))
    assert second['comfyui']['base_dir'] == str(basedir)


def test_a_non_empty_checkout_root_is_preserved_when_another_root_exists(
        seeder, tmp_path, monkeypatch):
    """There is no provenance marker distinguishing an old automatic value
    from an explicit user choice, so every non-empty value must win."""
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({
        'comfyui': {'base_dir': '/comfy/mnt/ComfyUI'},
    }), encoding='utf-8')
    monkeypatch.setenv('LDS_CONFIG', str(config))
    basedir = tmp_path / 'basedir'
    (basedir / 'models').mkdir(parents=True)
    monkeypatch.setattr(seeder, 'COMFY_ROOT_CANDIDATES', (str(basedir),))

    assert seeder.main() == 0

    written = json.loads(config.read_text(encoding='utf-8'))
    assert written['comfyui']['base_dir'] == '/comfy/mnt/ComfyUI'


def test_next_boot_still_preserves_a_user_selected_root(seeder, tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'comfyui': {'base_dir': '/user/choice'}}),
                      encoding='utf-8')
    monkeypatch.setenv('LDS_CONFIG', str(config))
    basedir = tmp_path / 'basedir'
    (basedir / 'models').mkdir(parents=True)
    monkeypatch.setattr(seeder, 'COMFY_ROOT_CANDIDATES', (str(basedir),))

    assert seeder.main() == 0

    written = json.loads(config.read_text(encoding='utf-8'))
    assert written['comfyui']['base_dir'] == '/user/choice'


def test_seeds_ollama_only_when_a_url_is_supplied(seeder, tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    monkeypatch.setenv('LDS_CONFIG', str(config))
    monkeypatch.setenv('LDS_OLLAMA_URL', 'http://ollama.internal:11434')

    assert seeder.main() == 0

    written = json.loads(config.read_text(encoding='utf-8'))
    assert written['ollama']['url'] == 'http://ollama.internal:11434'


def test_a_corrupt_config_is_left_alone_rather_than_replaced(seeder, tmp_path,
                                                            monkeypatch, capsys):
    """A half-written config.json is the user's data. Replacing it would silently
    reset every setting; refusing and saying so loses nothing."""
    config = tmp_path / 'config.json'
    config.write_text('{not json', encoding='utf-8')
    monkeypatch.setenv('LDS_CONFIG', str(config))

    assert seeder.main() == 0
    assert config.read_text(encoding='utf-8') == '{not json'
    assert 'unreadable' in capsys.readouterr().out


@pytest.mark.parametrize('root', [[], ['keep'], 'keep', 42, None])
def test_a_non_object_json_root_is_left_byte_for_byte_untouched(
        seeder, tmp_path, monkeypatch, root):
    config = tmp_path / 'config.json'
    original = json.dumps(root)
    config.write_text(original, encoding='utf-8')
    monkeypatch.setenv('LDS_CONFIG', str(config))

    assert seeder.main() == 0

    assert config.read_text(encoding='utf-8') == original


@pytest.mark.parametrize('section', [[], ['keep'], 'keep', 42, None])
def test_a_non_object_comfyui_section_is_preserved(
        seeder, tmp_path, monkeypatch, section):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({
        'comfyui': section,
        'paths': {'dataset_images_root': '/keep/me'},
    }), encoding='utf-8')
    monkeypatch.setenv('LDS_CONFIG', str(config))

    assert seeder.main() == 0

    written = json.loads(config.read_text(encoding='utf-8'))
    assert written['comfyui'] == section
    assert written['paths'] == {'dataset_images_root': '/keep/me'}


def test_leaves_no_temp_file_behind(seeder, tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    monkeypatch.setenv('LDS_CONFIG', str(config))

    assert seeder.main() == 0

    assert [p.name for p in tmp_path.iterdir()] == ['config.json']


def test_external_comfy_values_are_authoritative_and_preserve_other_keys(
        seeder, tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({
        'comfyui': {
            'api_url': 'http://old.invalid',
            'base_dir': '/old',
            'custom_setting': 'keep',
        },
        'paths': {'dataset_images_root': '/keep/me'},
    }), encoding='utf-8')
    monkeypatch.setenv('LDS_CONFIG', str(config))
    monkeypatch.setenv('LDS_DOCKER_COMFY_MODE', 'external')

    assert seeder.main() == 0

    written = json.loads(config.read_text(encoding='utf-8'))
    assert written['comfyui'] == {
        'api_url': 'http://host.docker.internal:8188',
        'base_dir': '/external-comfyui',
        'custom_setting': 'keep',
    }
    assert written['paths'] == {'dataset_images_root': '/keep/me'}


def test_external_comfy_blank_env_values_fall_back_to_container_defaults(
        seeder, tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    monkeypatch.setenv('LDS_CONFIG', str(config))
    monkeypatch.setenv('LDS_DOCKER_COMFY_MODE', 'external')
    monkeypatch.setenv('LDS_COMFYUI_API_URL', '   ')
    monkeypatch.setenv('LDS_COMFYUI_BASE_DIR', '   ')

    assert seeder.main() == 0

    written = json.loads(config.read_text(encoding='utf-8'))
    assert written['comfyui']['api_url'] == 'http://host.docker.internal:8188'
    assert written['comfyui']['base_dir'] == '/external-comfyui'


def test_no_comfy_mode_does_not_create_a_comfyui_section(
        seeder, tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    monkeypatch.setenv('LDS_CONFIG', str(config))
    monkeypatch.setenv('LDS_DOCKER_COMFY_MODE', 'none')

    assert seeder.main() == 0

    assert not config.exists()


def test_legacy_has_comfy_false_maps_to_no_comfy_mode(
        seeder, tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    monkeypatch.setenv('LDS_CONFIG', str(config))
    monkeypatch.delenv('LDS_DOCKER_COMFY_MODE', raising=False)
    monkeypatch.setenv('LDS_DOCKER_HAS_COMFYUI', '0')

    assert seeder.main() == 0

    assert not config.exists()


@pytest.mark.parametrize(
    ('mode', 'expected_url'),
    [
        ('host', 'http://host.docker.internal:11434'),
        ('docker', 'http://ollama:11434'),
    ],
)
def test_persisted_ollama_mode_sets_its_container_reachable_url(
        seeder, tmp_path, monkeypatch, mode, expected_url):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({
        'ollama': {
            'deployment_mode': mode,
            'url': 'http://old.invalid',
            'model': 'keep-model',
        },
    }), encoding='utf-8')
    monkeypatch.setenv('LDS_CONFIG', str(config))
    monkeypatch.setenv('LDS_DOCKER_COMFY_MODE', 'none')

    assert seeder.main() == 0

    written = json.loads(config.read_text(encoding='utf-8'))
    assert written['ollama'] == {
        'deployment_mode': mode,
        'url': expected_url,
        'model': 'keep-model',
    }


def test_none_ollama_mode_leaves_a_manual_url_untouched(
        seeder, tmp_path, monkeypatch):
    config = tmp_path / 'config.json'
    original = {
        'ollama': {
            'deployment_mode': 'none',
            'url': 'http://manual.invalid',
        },
    }
    config.write_text(json.dumps(original), encoding='utf-8')
    monkeypatch.setenv('LDS_CONFIG', str(config))
    monkeypatch.setenv('LDS_DOCKER_COMFY_MODE', 'none')

    assert seeder.main() == 0

    assert json.loads(config.read_text(encoding='utf-8')) == original


def test_api_image_runs_the_seeder_before_execing_the_backend():
    dockerfile = (REPO_ROOT / 'Dockerfile').read_text(encoding='utf-8')
    entrypoint = (
        REPO_ROOT / 'packaging' / 'docker' / 'studio_api_entrypoint.sh'
    ).read_text(encoding='utf-8')

    assert 'COPY packaging/docker/seed_comfy_config.py' in dockerfile
    assert 'COPY packaging/docker/studio_api_entrypoint.sh' in dockerfile
    assert 'ENTRYPOINT ["/usr/local/bin/studio-api-entrypoint.sh"]' in dockerfile
    assert entrypoint.index('seed_comfy_config.py') < entrypoint.index(
        'exec python backend/run.py'
    )
    assert entrypoint.startswith('#!/bin/sh\nset -u\n')
    entrypoint_bytes = (REPO_ROOT / 'packaging' / 'docker' / 'studio_api_entrypoint.sh').read_bytes()
    assert b'\r\n' not in entrypoint_bytes
