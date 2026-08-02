#!/usr/bin/env python3
"""Seed container-owned config values without importing the application.

The GPU image runs this before its bundled ComfyUI starts. The API image also
runs it so an external host ComfyUI can be represented by stable container paths.
Invalid user JSON is never replaced and every write is atomic.
"""

import json
import os
import sys
from pathlib import Path

COMFY_ROOT_CANDIDATES = ('/basedir', '/comfy/mnt/ComfyUI')
BUNDLED_API_URL = 'http://127.0.0.1:8188'
EXTERNAL_API_URL = 'http://host.docker.internal:8188'
EXTERNAL_BASE_DIR = '/external-comfyui'
OLLAMA_URLS = {
    'host': 'http://host.docker.internal:11434',
    'docker': 'http://ollama:11434',
}
FALSE_VALUES = {'0', 'false', 'no', 'off'}


def comfy_root(candidates=None):
    """Return the first candidate that actually contains a models directory."""
    for candidate in (candidates or COMFY_ROOT_CANDIDATES):
        if Path(candidate, 'models').is_dir():
            return candidate
    return None


def wanted(base_dir: str | None, ollama_url: str = '') -> dict:
    """Return the historical bundled-Comfy defaults fragment."""
    fragment = {'comfyui': {'api_url': BUNDLED_API_URL}}
    if base_dir:
        fragment['comfyui']['base_dir'] = base_dir
    if ollama_url:
        fragment['ollama'] = {'url': ollama_url}
    return fragment


def fill_empty(current: dict, defaults: dict) -> tuple[dict, list[str]]:
    """Fill only missing, null, or whitespace-only string values."""
    filled = []
    for section, values in defaults.items():
        if section not in current:
            node = {}
            current[section] = node
        else:
            node = current[section]
            if not isinstance(node, dict):
                continue
        for key, value in values.items():
            existing = node.get(key)
            if isinstance(existing, str):
                if existing.strip():
                    continue
            elif existing is not None:
                continue
            node[key] = value
            filled.append(f'{section}.{key}')
    return current, filled


def set_values(current: dict, section_name: str,
               values: dict) -> list[str]:
    """Set authoritative keys while preserving every other section key."""
    if section_name not in current:
        current[section_name] = {}
    section = current[section_name]
    if not isinstance(section, dict):
        return []

    changed = []
    for key, value in values.items():
        if section.get(key) != value:
            section[key] = value
            changed.append(f'{section_name}.{key}')
    return changed


def docker_comfy_mode() -> str:
    """Resolve explicit mode, with compatibility for the old boolean flag."""
    explicit = (os.environ.get('LDS_DOCKER_COMFY_MODE') or '').strip().lower()
    if explicit in {'external', 'bundled', 'none'}:
        return explicit
    has_comfy = os.environ.get('LDS_DOCKER_HAS_COMFYUI')
    if has_comfy is not None and has_comfy.strip().lower() in FALSE_VALUES:
        return 'none'
    return 'bundled'


def deployment_mode(current: dict) -> str:
    """Read the persisted LDS choice without changing malformed user data."""
    ollama = current.get('ollama')
    if ollama is None:
        return 'unset'
    if not isinstance(ollama, dict):
        return 'invalid'
    value = ollama.get('deployment_mode')
    if value is None:
        return 'unset'
    if not isinstance(value, str):
        return 'invalid'
    normalized = value.strip().lower()
    if not normalized or normalized == 'unconfigured':
        return 'unset'
    if normalized in {'none', 'host', 'docker'}:
        return normalized
    return 'invalid'


def atomic_write_json(path: Path, value: dict) -> None:
    """Write beside the destination and atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    """Seed known container values; never prevent the application from booting."""
    path = Path(os.environ.get('LDS_CONFIG', '/data/config.json'))
    try:
        current = (
            json.loads(path.read_text(encoding='utf-8'))
            if path.exists()
            else {}
        )
    except (OSError, ValueError) as exc:
        print(
            f'[studio] {path} is unreadable '
            f'({exc.__class__.__name__}); leaving it untouched.',
            flush=True,
        )
        return 0
    if not isinstance(current, dict):
        print(
            f'[studio] {path} has a non-object JSON root; '
            'leaving it untouched.',
            flush=True,
        )
        return 0

    changed: list[str] = []
    comfy_mode = docker_comfy_mode()
    legacy_ollama_url = (os.environ.get('LDS_OLLAMA_URL') or '').strip()
    explicit_ollama_mode = (
        os.environ.get('LDS_OLLAMA_MODE') or ''
    ).strip().lower()

    if comfy_mode == 'bundled':
        root = comfy_root()
        if root is not None:
            print(f'[studio] ComfyUI root: {root}', flush=True)
        else:
            print(
                f'[studio] none of {COMFY_ROOT_CANDIDATES} has a models/ '
                'directory yet; leaving comfyui.base_dir unset and trying '
                'again on the next boot.',
                flush=True,
            )
        defaults = wanted(
            root,
            legacy_ollama_url
            if not explicit_ollama_mode
            and deployment_mode(current) == 'unset'
            else '',
        )
        current, filled = fill_empty(current, defaults)
        changed.extend(filled)
    elif comfy_mode == 'external':
        external_values = {
            'api_url': (
                os.environ.get('LDS_COMFYUI_API_URL')
                or ''
            ).strip() or EXTERNAL_API_URL,
            'base_dir': (
                os.environ.get('LDS_COMFYUI_BASE_DIR')
                or ''
            ).strip() or EXTERNAL_BASE_DIR,
        }
        changed.extend(set_values(current, 'comfyui', external_values))

    mode = deployment_mode(current)
    if mode in {'host', 'docker'}:
        changed.extend(set_values(
            current,
            'ollama',
            {'url': OLLAMA_URLS[mode]},
        ))
    elif mode == 'unset':
        if explicit_ollama_mode in {'host', 'docker'}:
            requested_url = legacy_ollama_url or OLLAMA_URLS[explicit_ollama_mode]
            changed.extend(set_values(
                current,
                'ollama',
                {'url': requested_url},
            ))
        elif comfy_mode != 'bundled' and legacy_ollama_url:
            current, filled = fill_empty(
                current,
                {'ollama': {'url': legacy_ollama_url}},
            )
            changed.extend(filled)
    elif mode == 'invalid':
        print(
            '[studio] ollama.deployment_mode is invalid; '
            'leaving ollama.url untouched.',
            flush=True,
        )

    if not changed:
        if comfy_mode == 'bundled':
            print(
                '[studio] ComfyUI folders already configured; nothing seeded.',
                flush=True,
            )
        return 0

    try:
        atomic_write_json(path, current)
    except OSError as exc:
        print(
            f'[studio] could not write {path} '
            f'({exc.__class__.__name__}); leaving it unchanged.',
            flush=True,
        )
        return 0

    print('[studio] seeded: ' + ', '.join(changed), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
