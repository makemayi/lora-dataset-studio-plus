"""Topaz Photo AI (tpai.exe) integration.

Topaz is a separate native program that owns its own GPU — it does NOT go
through ComfyUI, so it can never enter ImageGenerationQueue. This helper owns
the exe discovery, the CLI command shape and the return-code vocabulary; the
job queue (topaz_job_queue) owns scheduling and the GPU window.

Return codes (documented by `tpai.exe --help`):
    0   success         1   partial success
   -1   no valid files  -2   not logged in (open the app once)
   -3   invalid argument
Anything else is an unknown failure surfaced as-is.
"""
import logging
import os
import subprocess

from .. import config as cfg

logger = logging.getLogger(__name__)

# Probe list: the user's documented install first, then the usual layouts.
STANDARD_TPAI_PATHS = (
    r'F:\Program Files\Topaz Labs LLC\Topaz Photo AI\tpai.exe',
    r'C:\Program Files\Topaz Labs LLC\Topaz Photo AI\tpai.exe',
    r'C:\Program Files (x86)\Topaz Labs LLC\Topaz Photo AI\tpai.exe',
    r'D:\Program Files\Topaz Labs LLC\Topaz Photo AI\tpai.exe',
)

PNG = 'png'
DEFAULT_TIMEOUT_S = 600   # one image through Topaz is seconds to a couple of minutes


class TopazUnavailable(RuntimeError):
    """Deterministic refusal: exe missing or unusable (mirrors SeedVR2ModelsMissing)."""


def resolve_exe():
    """The tpai.exe to call: config override, else the first standard path that
    exists, else None."""
    override = (cfg.get('topaz.exe_path') or '').strip()
    if override:
        return override
    for p in STANDARD_TPAI_PATHS:
        if os.path.isfile(p):
            return p
    return None


def preflight():
    """Raise TopazUnavailable with a fixable message when Topaz cannot run."""
    exe = resolve_exe()
    if not exe:
        raise TopazUnavailable(
            'Topaz Photo AI was not found. Install it, or set its tpai.exe path '
            'in Settings ▸ Image engines.')
    return exe


def build_command(exe, input_path, output_dir, *, format=PNG,
                  upscale=True, denoise=True, sharpen=True,
                  lighting=False, color=False):
    """The tpai.exe argv. Enhancements are toggles; model choice and scale are
    left to Topaz Autopilot (the model API is experimental and undocumented)."""
    cmd = [exe, '-i', input_path, '-o', output_dir, '--format', format]
    for flag, enabled in (('--upscale', upscale), ('--noise', denoise),
                          ('--sharpen', sharpen), ('--lighting', lighting),
                          ('--color', color)):
        cmd += [flag, f'enabled={"true" if enabled else "false"}']
    return cmd


def run_tpai(exe, input_path, output_dir, *, timeout=DEFAULT_TIMEOUT_S, **toggles):
    """Run one image through Topaz. Returns (status, message) where status is
    one of 'ok' | 'partial' | 'no_valid_files' | 'license' | 'bad_args' |
    'timeout' | 'unknown'. Never raises for a Topaz refusal — the caller
    decides what each status means for the job."""
    cmd = build_command(exe, input_path, output_dir, **toggles)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 'timeout', f'Topaz did not finish within {timeout // 60} minutes'
    except OSError as e:
        return 'unknown', f'could not start Topaz: {e}'
    rc = proc.returncode
    mapping = {0: 'ok', 1: 'partial', -1: 'no_valid_files',
               -2: 'license', -3: 'bad_args'}
    status = mapping.get(rc, 'unknown')
    detail = (proc.stderr or proc.stdout or '').strip()[-300:]
    if status == 'license':
        return status, ('Topaz is not logged in on this machine — open Topaz '
                        'Photo AI once to complete the license sign-in, then retry.')
    return status, detail
