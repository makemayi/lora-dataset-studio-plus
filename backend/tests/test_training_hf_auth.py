"""Hugging Face authentication of the LOCAL training subprocess.

Reported by SurpassHR on GitHub: training on Krea 2 (gated repo
`krea/Krea-2-Turbo`) died with a 401 GatedRepoError *from LoRA Dataset Studio*,
while downloading the very same weights straight from ai-toolkit worked. The
discriminating fact was the environment we hand the child process:

* we override `HF_HOME` so weights land on the configured disk — and
  `huggingface_hub` looks for the CLI-written token at `$HF_HOME/token`, so the
  override HID the token `hf auth login` had written in the default home. An app
  setting that relocates a CACHE must never log the user out;
* the token the app asks for in Settings ▸ API keys was never handed to the
  child EXPLICITLY. It happened to be inherited because `cfg.secret()` reads
  `os.environ` today — an accident, not a contract, and the cloud lane has always
  injected it on purpose (`cloud_training`: `env['HF_TOKEN'] = cfg.secret(...)`).

Nothing here touches the network: only the env dict is inspected.
"""
import os

from unittest.mock import patch

from app.services import lora_training as lt


def _env(hf_home, monkeypatch, environ=None):
    """Build the launch env with a controlled os.environ (no real machine state)."""
    base = {'PATH': os.environ.get('PATH', ''), **(environ or {})}
    with patch.object(os, 'environ', base):
        return lt.training_subprocess_env(hf_home=str(hf_home))


# --- the base env, unchanged ---------------------------------------------------

def test_env_still_routes_the_cache_and_forces_utf8(tmp_path, monkeypatch):
    monkeypatch.setattr(lt.cfg, 'secret', lambda name: None)
    env = _env(tmp_path / 'hf', monkeypatch)
    assert env['HF_HOME'] == str(tmp_path / 'hf')
    assert env['PYTHONIOENCODING'] == 'utf-8'
    # Belt-and-suspenders alongside PYTHONIOENCODING: UTF-8 mode also covers a
    # NESTED subprocess capture inside ai-toolkit's own dependency tree (a
    # torch elastic-launcher reader thread crashed on a non-UTF-8 byte from a
    # worker it was capturing, silently freezing the training log — the run
    # looked "stuck" but was really just unobserved).
    assert env['PYTHONUTF8'] == '1'


# --- 1. the Settings token must reach the trainer EXPLICITLY -------------------

def test_settings_token_is_injected_into_the_child_env(tmp_path, monkeypatch):
    """THE central assertion: what Settings ▸ API keys holds is what the trainer
    authenticates with — by explicit injection, exactly like the cloud lane, not
    by hoping the value happens to sit in the parent's os.environ."""
    monkeypatch.setattr(lt.cfg, 'secret',
                        lambda name: 'hf_secret_from_settings' if name == 'HF_TOKEN' else None)
    env = _env(tmp_path / 'hf', monkeypatch, environ={})   # nothing inherited
    assert env.get('HF_TOKEN') == 'hf_secret_from_settings'


def test_settings_token_wins_over_a_stale_inherited_one(tmp_path, monkeypatch):
    monkeypatch.setattr(lt.cfg, 'secret',
                        lambda name: 'hf_current' if name == 'HF_TOKEN' else None)
    env = _env(tmp_path / 'hf', monkeypatch, environ={'HF_TOKEN': 'hf_stale'})
    assert env['HF_TOKEN'] == 'hf_current'


def test_blank_settings_token_never_blanks_the_inherited_one(tmp_path, monkeypatch):
    """A token in ai-toolkit's own environment (its `.env`, loaded by run.py) is
    how this worked for the people it worked for. Saving nothing in Settings
    must not take that away."""
    monkeypatch.setattr(lt.cfg, 'secret', lambda name: None)
    env = _env(tmp_path / 'hf', monkeypatch, environ={'HF_TOKEN': 'hf_from_the_machine'})
    assert env['HF_TOKEN'] == 'hf_from_the_machine'


# --- 2. overriding HF_HOME must never de-authenticate --------------------------

def test_hf_home_override_keeps_the_cli_token_reachable(tmp_path, monkeypatch):
    """`hf auth login` wrote a token in the DEFAULT home; we point HF_HOME
    elsewhere. huggingface_hub reads `HF_TOKEN_PATH` (an env var of its own,
    defaulting to `$HF_HOME/token`) — so pinning it at the real token file keeps
    the user logged in. Verified against huggingface_hub 0.36 constants."""
    home = tmp_path / 'home'
    (home / '.cache' / 'huggingface').mkdir(parents=True)
    (home / '.cache' / 'huggingface' / 'token').write_text('hf_written_by_the_cli')
    monkeypatch.setattr(lt.cfg, 'secret', lambda name: None)
    monkeypatch.setattr(os.path, 'expanduser', lambda p: p.replace('~', str(home), 1))

    env = _env(tmp_path / 'aitoolkit-cache', monkeypatch, environ={})
    assert env.get('HF_TOKEN_PATH') == str(home / '.cache' / 'huggingface' / 'token')
    # The token itself is never copied into the env: the file stays the source.
    assert 'HF_TOKEN' not in env


def test_xdg_cache_home_is_honoured(tmp_path, monkeypatch):
    """Linux users who move their cache: huggingface_hub falls back to
    $XDG_CACHE_HOME/huggingface, so the fallback must look there too."""
    xdg = tmp_path / 'xdg'
    (xdg / 'huggingface').mkdir(parents=True)
    (xdg / 'huggingface' / 'token').write_text('hf_cli')
    monkeypatch.setattr(lt.cfg, 'secret', lambda name: None)
    env = _env(tmp_path / 'cache', monkeypatch, environ={'XDG_CACHE_HOME': str(xdg)})
    assert env.get('HF_TOKEN_PATH') == str(xdg / 'huggingface' / 'token')


def test_a_token_already_under_the_override_is_left_alone(tmp_path, monkeypatch):
    """Someone who ran `hf auth login` WITH our HF_HOME set already works today.
    Nothing may be redirected away from them."""
    ours = tmp_path / 'aitoolkit-cache'
    ours.mkdir()
    (ours / 'token').write_text('hf_in_our_cache')
    home = tmp_path / 'home'
    (home / '.cache' / 'huggingface').mkdir(parents=True)
    (home / '.cache' / 'huggingface' / 'token').write_text('hf_elsewhere')
    monkeypatch.setattr(lt.cfg, 'secret', lambda name: None)
    monkeypatch.setattr(os.path, 'expanduser', lambda p: p.replace('~', str(home), 1))

    env = _env(ours, monkeypatch, environ={})
    assert 'HF_TOKEN_PATH' not in env


def test_an_explicit_hf_token_path_is_respected(tmp_path, monkeypatch):
    pinned = tmp_path / 'pinned' / 'token'
    pinned.parent.mkdir()
    pinned.write_text('hf_pinned')
    monkeypatch.setattr(lt.cfg, 'secret', lambda name: None)
    env = _env(tmp_path / 'cache', monkeypatch, environ={'HF_TOKEN_PATH': str(pinned)})
    assert env['HF_TOKEN_PATH'] == str(pinned)


def test_no_token_anywhere_invents_nothing(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setattr(lt.cfg, 'secret', lambda name: None)
    monkeypatch.setattr(os.path, 'expanduser', lambda p: p.replace('~', str(home), 1))
    env = _env(tmp_path / 'cache', monkeypatch, environ={})
    assert 'HF_TOKEN' not in env
    assert 'HF_TOKEN_PATH' not in env


def test_a_settings_token_needs_no_file_fallback(tmp_path, monkeypatch):
    """With an explicit token, HF_TOKEN wins over any file in huggingface_hub —
    don't reach into the user's home for nothing."""
    home = tmp_path / 'home'
    (home / '.cache' / 'huggingface').mkdir(parents=True)
    (home / '.cache' / 'huggingface' / 'token').write_text('hf_cli')
    monkeypatch.setattr(lt.cfg, 'secret',
                        lambda name: 'hf_settings' if name == 'HF_TOKEN' else None)
    monkeypatch.setattr(os.path, 'expanduser', lambda p: p.replace('~', str(home), 1))
    env = _env(tmp_path / 'cache', monkeypatch, environ={})
    assert env['HF_TOKEN'] == 'hf_settings'
    assert 'HF_TOKEN_PATH' not in env


# --- 3. nothing here may leak the token ---------------------------------------

def test_the_token_is_never_written_to_a_log(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(lt.cfg, 'secret',
                        lambda name: 'hf_topsecretvalue' if name == 'HF_TOKEN' else None)
    with caplog.at_level('DEBUG'):
        _env(tmp_path / 'cache', monkeypatch, environ={})
    assert 'hf_topsecretvalue' not in caplog.text
