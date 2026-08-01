#!/usr/bin/env python3
"""Health of BOTH halves of the GPU image, in one exit code.

Docker's healthcheck is a single boolean, and in this image a running ComfyUI with a
dead studio — or the reverse — is a broken stack either way. So both are probed and
the failing side is named. Nothing supervises across the two (ComfyUI is upstream's
foreground process, the studio a background job under its own restart loop), which
is exactly why a permanent failure needs somewhere to show up.

First boot is slow on purpose: ComfyUI creates its venv and installs torch. That is
what the Dockerfile's long --start-period covers.
"""
import os
import sys
import urllib.error
import urllib.request

TARGETS = (
    ('studio', f"http://127.0.0.1:{os.environ.get('LDS_PORT', '5050')}/api/health"),
    ('comfyui', 'http://127.0.0.1:8188/system_stats'),
)


def probe(url: str) -> str:
    """Empty string when the endpoint answers, else a short reason."""
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            if response.status >= 400:
                return f'HTTP {response.status}'
            response.read(1)
            return ''
    except urllib.error.HTTPError as exc:
        return f'HTTP {exc.code}'
    except Exception as exc:                     # noqa: BLE001 - any failure is one
        return exc.__class__.__name__


def main() -> int:
    failures = []
    for name, url in TARGETS:
        reason = probe(url)
        if reason:
            failures.append(f'{name} ({reason})')
    if failures:
        print('unhealthy: ' + ', '.join(failures), flush=True)
        return 1
    print('ok: studio + comfyui', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
