"""Fizgig lane (MOSS-Image, Fizgig's Standard preset)."""
import json
import struct
from types import SimpleNamespace

import pytest


@pytest.fixture
def fizgig(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    from app import config as cfg
    from app.services import fizgig_service as fzs
    monkeypatch.setattr(cfg, '_cache', None)
    yield fzs, cfg


def _fake_install(root):
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_bytes(b'')
    (root / 'venv' / 'bin').mkdir(parents=True)
    (root / 'venv' / 'bin' / 'python').write_bytes(b'')
    (root / 'src' / 'fizgig' / 'families').mkdir(parents=True)
    (root / 'src' / 'fizgig' / 'families' / 'train.py').write_text('')


def _touch(models, rel):
    p = models / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b'')
    return p


def test_defaults_are_blank_and_unconfigured(fizgig):
    fzs, cfg = fizgig
    assert cfg.get('fizgig.dir') == ''
    assert not fzs.is_installed()


def test_installed_needs_python_and_train_script(fizgig, tmp_path):
    fzs, cfg = fizgig
    root = tmp_path / 'Fizgig'
    root.mkdir()
    cfg.save_config({'fizgig': {'dir': str(root)}})
    assert not fzs.is_installed()
    _fake_install(root)
    assert fzs.is_installed()


def test_models_prefer_int8_and_small_vae(fizgig, tmp_path):
    fzs, cfg = fizgig
    models = tmp_path / 'models'
    _touch(models, 'diffusion_models/moss_image_base_bf16.safetensors')
    int8 = _touch(models, 'diffusion_models/moss_image_base_int8_convrot.safetensors')
    _touch(models, 'clip/qwen3.5_9b_int8_convrot.safetensors')
    _touch(models, 'vae/flux2-vae.safetensors')
    small = _touch(models, 'vae/flux2-vae_small.safetensors')
    cfg.save_config({'fizgig': {'models_dir': str(models)}})
    found = fzs.assert_models_ready()
    assert found['dit'] == int8
    assert found['vae'] == small
    assert found['preview'] is None          # optional


def test_missing_models_are_named(fizgig, tmp_path):
    fzs, cfg = fizgig
    cfg.save_config({'fizgig': {'models_dir': str(tmp_path / 'empty')}})
    with pytest.raises(ValueError, match='moss_image_base_int8_convrot'):
        fzs.assert_models_ready()


def test_plan_is_the_standard_preset(fizgig, tmp_path):
    fzs, _ = fizgig
    models = {'dit': 'D.safetensors', 'te': 'T.safetensors', 'vae': 'V.safetensors',
              'preview': 'S.safetensors'}
    settings = {'rank': 32, 'alpha': 32, 'learning_rate': 1e-4, 'epochs': 70}
    plan = fzs.build_plan(fizgig_dir='E:/Fizgig', python='py', toml_path='d.toml',
                          output_dir='out', output_name='joy_fizgig', models=models,
                          settings=settings, prompts_path='p.txt')
    names = [s['name'] for s in plan['stages']]
    assert names == ['cache latents', 'cache text', 'train']
    assert plan['stages'][0]['args'][-1] == 'V.safetensors'
    assert plan['stages'][1]['args'][-1] == 'T.safetensors'
    train = plan['stages'][2]['args']

    def val(flag):
        return train[train.index(flag) + 1]
    assert val('--family') == 'moss_image'
    assert val('--dit') == 'D.safetensors'
    assert val('--network_dim') == '32' and val('--network_alpha') == '32'
    assert val('--learning_rate') == '0.0001'
    assert val('--max_train_epochs') == '70'
    assert val('--optimizer_type') == 'adamw'
    assert val('--ema_decay') == '0.98'
    assert val('--family_option') == 'lownoise_pct=30'
    assert val('--preview_checkpoint') == 'S.safetensors'
    assert val('--sample_steps') == '30' and val('--sample_cfg_scale') == '4'
    # Block 22 trained, SFT attention-only NOT used on the Base.
    assert 'lora_sft=1' not in train
    assert not any('ignore' in a or 'layers.22' in a for a in train)
    assert plan['env']['PYTHONPATH'] == 'src'


def test_plan_without_prompts_has_no_previews(fizgig):
    fzs, _ = fizgig
    plan = fzs.build_plan(fizgig_dir='F', python='py', toml_path='d.toml', output_dir='o',
                          output_name='n', models={'dit': 'D', 'te': 'T', 'vae': 'V',
                                                   'preview': 'S'},
                          settings={'rank': 16, 'alpha': 8, 'learning_rate': 2e-4,
                                    'epochs': 50})
    train = plan['stages'][2]['args']
    assert '--sample_prompts' not in train and '--preview_checkpoint' not in train


def test_dataset_toml(fizgig):
    fzs, _ = fizgig
    toml = fzs.build_dataset_toml('C:\\x\\dataset', 'C:\\x\\cache')
    assert 'resolution = [992, 992]' in toml
    assert 'image_directory = "C:/x/dataset"' in toml
    assert 'cache_directory = "C:/x/cache"' in toml
    assert 'enable_bucket = true' in toml


def test_effective_settings_fall_back_to_preset(fizgig, monkeypatch):
    fzs, _ = fizgig
    monkeypatch.setattr('app.services.lora_training._train_settings', lambda ds: {})
    assert fzs.effective_settings(None) == {'rank': 32, 'alpha': 32,
                                            'learning_rate': 1e-4, 'epochs': 70}
    monkeypatch.setattr('app.services.lora_training._train_settings',
                        lambda ds: {'rank': 16, 'learning_rate': 2e-4, 'epochs': 9999})
    s = fzs.effective_settings(None)
    assert s == {'rank': 16, 'alpha': 16, 'learning_rate': 2e-4, 'epochs': 70}


def test_progress_parses_a_real_fizgig_log_line(fizgig):
    from app.services.lora_training import _parse_training_log
    fzs, _ = fizgig
    text = ('[lds-fizgig] [3/3] train — 2026-10-20\n'
            ' 40/40 [00:10<00:00]\n'
            'steps:  99%|#########8| 2761/2800 [2:22:51<02:01,  3.10s/it, avr_loss=0.4121]'
            'steps:  99%|#########8| 2762/2800 [2:22:54<01:57,  3.10s/it, avr_loss=0.4143]')
    out = _parse_training_log(fzs.split_bar_updates(text), expected_total=2800)
    assert out['step'] == 2762 and out['total'] == 2800
    assert out['loss'] == pytest.approx(0.4143)


def test_samples_map_epochs_to_steps(fizgig, tmp_path):
    fzs, _ = fizgig
    sample = tmp_path / 'output' / 'sample'
    sample.mkdir(parents=True)
    for name in ('joy_fizgig_e000005_00_20261020094929_42.png',
                 'joy_fizgig_e000010_01_20261020100000_43.png',
                 'unrelated.png'):
        (sample / name).write_bytes(b'')
    got = fzs.list_samples(tmp_path, steps_per_epoch=40)
    assert [(s['epoch'], s['step'], s['prompt_idx']) for s in got] == [(10, 400, 1), (5, 200, 0)]


def test_fizgig_lora_is_detected_as_moss(tmp_path):
    """Fizgig saves kohya-style keys (`lora_unet_layers_*`) that the key sniff
    alone reads as SDXL; its modelspec stamp must win so deploy lands in moss."""
    from app.services import lora_training as lt
    header = {'__metadata__': {'modelspec.architecture': 'MOSS-Image/lora'},
              'lora_unet_layers_0_proj_k.lora_down.weight':
                  {'dtype': 'BF16', 'shape': [1, 1], 'data_offsets': [0, 2]}}
    blob = json.dumps(header).encode()
    p = tmp_path / 'fz.safetensors'
    p.write_bytes(struct.pack('<Q', len(blob)) + blob + b'\0\0')
    assert lt.detect_lora_arch(str(p)) == 'moss'
    assert not lt.lora_arch_conflicts('moss', 'moss')


def test_settings_map_fizgig_lane_matches_the_service(fizgig, monkeypatch):
    """Everything the map says applies on the Fizgig lane reaches the plan."""
    fzs, _ = fizgig
    from app.services import training_settings_map as tsm
    assert tsm.keys_applying_on(tsm.LANE_FIZGIG) == ['alpha', 'epochs', 'learning_rate', 'rank']
    monkeypatch.setattr('app.services.lora_training._train_settings',
                        lambda ds: {'rank': 24, 'alpha': 12, 'learning_rate': 3e-4,
                                    'epochs': 33})
    plan = fzs.build_plan(fizgig_dir='F', python='py', toml_path='d', output_dir='o',
                          output_name='n', models={'dit': 'D', 'te': 'T', 'vae': 'V'},
                          settings=fzs.effective_settings(SimpleNamespace()))
    train = plan['stages'][2]['args']
    for flag, want in (('--network_dim', '24'), ('--network_alpha', '12'),
                       ('--learning_rate', '0.0003'), ('--max_train_epochs', '33')):
        assert train[train.index(flag) + 1] == want


def test_train_route_refuses_when_unconfigured(client):
    r = client.post('/api/dataset/1/train', json={'trainer': 'fizgig'})
    assert r.status_code == 409
    assert 'Fizgig' in r.get_json()['error']


def test_runner_stops_at_first_failing_stage(tmp_path):
    import subprocess
    import sys
    from app.services import fizgig_runner
    plan = {'cwd': str(tmp_path), 'env': {},
            'stages': [{'name': 'ok', 'args': [sys.executable, '-c', 'print(1)']},
                       {'name': 'bad', 'args': [sys.executable, '-c', 'raise SystemExit(3)']},
                       {'name': 'never', 'args': [sys.executable, '-c',
                                                   'open("ran","w").write("x")']}]}
    pp = tmp_path / 'plan.json'
    pp.write_text(json.dumps(plan))
    r = subprocess.run([sys.executable, fizgig_runner.__file__, str(pp)],
                       capture_output=True, text=True)
    assert r.returncode == 3
    assert 'bad failed (exit 3)' in r.stdout
    assert not (tmp_path / 'ran').exists()
