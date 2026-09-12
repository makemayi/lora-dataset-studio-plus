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


def build_command(exe, input_path, output_dir, *, format=PNG, upscale=True,
                  denoise=None, sharpen=None, lighting=None, color=None,
                  face_recovery=None):
    """The tpai.exe argv. The INPUT IS A POSITIONAL PATH (there is no -i flag
    — passing one makes tpai exit 127). By default ONLY the upscale toggle is
    sent and everything else is left to Topaz Autopilot, exactly like the
    desktop app: forcing denoise/sharpen on adds heavy passes Autopilot would
    skip, which is how a CLI run ends up slower than the app. The other toggles
    are an OPTIONAL channel (None = Autopilot decides, True/False = force).

    ``face_recovery`` is the undocumented but verified ``--faceRecovery`` flag:
    None = Autopilot decides (it enables it at strength 0.8 on EVERY detected
    face — the plastic-skin recipe for dataset portraits), False = force off,
    a float 0..1 = force on at that strength. Both shapes were verified against
    tpai.exe on REAL runs, not just --skipProcessing: strength uses the
    documented override key ``strength=`` — the also-accepted ``param1=``
    silently breaks the engine there ("Error running model" with exit code 0
    and NO output file), which is exactly how a run ends up 'ok' with nothing
    written."""
    cmd = [exe, input_path, '-o', output_dir, '--format', format]
    if upscale is not None:
        cmd += ['--upscale', 'enabled=true' if upscale else 'enabled=false']
    for flag, val in (('--noise', denoise), ('--sharpen', sharpen),
                      ('--lighting', lighting), ('--color', color)):
        if val is not None:
            cmd += [flag, f'enabled={"true" if val else "false"}']
    if face_recovery is False:
        cmd += ['--faceRecovery', 'enabled=false']
    elif face_recovery:
        cmd += ['--faceRecovery', 'enabled=true',
                f'strength={float(face_recovery):g}']
    return cmd


# -- smart face recovery -----------------------------------------------------
# Measured on the operator's install (2026-09): Autopilot enables Face Recovery
# at strength 0.8, hair+neck included, on EVERY face it finds. On dataset
# portraits that are already sharp that is exactly the wax/plastic look the
# community reports and Topaz's own docs warn about ("over-process faces that
# are already high resolution ... creates a plastic feeling"). Topaz's docs
# put the useful range at faces BELOW ~512px. The smart plan therefore scales
# the strength DOWN as the face gets bigger, and turns it off entirely for
# faces that need no reconstruction. Tier bounds are the largest face's SHORT
# SIDE in pixels of the SOURCE image.
#   >=512px off | 256-511px 0.30 | 128-255px 0.50 | <128px 0.70 | no face off
# ponytail: size-only heuristic — no blur/sharpness term; add one only if
# soft-but-large faces measurably need recovery too.
FACE_RECOVERY_TIERS = ((512, False), (256, 0.30), (128, 0.50), (0, 0.70))


def face_recovery_value(short_side_px):
    """Largest-face short side in px -> False (off) or a strength 0..1."""
    for limit, strength in FACE_RECOVERY_TIERS:
        if short_side_px >= limit:
            return strength
    return False


def _decodable(path):
    """True when PIL can open the file at all. A cheap in-process header check
    that keeps undecodable sources (corrupt files, fake bytes in tests) out of
    the detection subprocess — they are decided OFF locally instead."""
    try:
        from PIL import Image
        with Image.open(path):
            return True
    except Exception:                                            # noqa: BLE001
        return False


def plan_face_recovery(paths, timeout=900):
    """{path: strength | False} for a whole batch — the smart per-image mode.

    Runs the EXISTING face detector (face_mask.detect_faces, InsightFace in a
    CPU subprocess — no GPU window) ONCE over the batch, then tiers every image
    by its largest face. False is the safe answer everywhere: no face found,
    undecodable source, detector unavailable or detection failed all mean OFF,
    never a fallback to Autopilot's 0.8-on-everything — natural skin wins over
    reconstructed skin, and a failed detection must never silently upgrade
    itself into the wax default."""
    paths = [p for p in (paths or []) if p]
    plan = {p: False for p in paths}
    decodable = [p for p in paths if _decodable(p)]
    if not decodable:
        return plan
    from .face_mask import detect_faces, is_available
    if not is_available():
        logger.warning('topaz: smart face recovery off — face detection unavailable')
        return plan
    res = detect_faces(decodable, timeout=timeout)
    if not (res or {}).get('ok'):
        logger.warning('topaz: smart face recovery off — detection failed: %s',
                       (res or {}).get('error'))
        return plan
    results = res.get('results') or {}
    for p in decodable:
        boxes = (results.get(p) or {}).get('boxes') or []
        if not boxes:
            continue
        main = max(min(b[2] - b[0], b[3] - b[1]) for b in boxes)
        plan[p] = face_recovery_value(main)
    return plan


def resolve_face_recovery(mode, paths):
    """Stored enhancement mode -> what run_tpai should send per image.

    True/None -> None (Autopilot decides, no detection, one flag shape for the
    whole run); False -> forced off for all; 'smart' (the default everywhere)
    or anything else -> a per-image plan dict."""
    if mode is True or mode is None:
        return None
    if mode is False:
        return False
    return plan_face_recovery(paths)


def _clean_env():
    """Child env without Git-for-Windows dirs: when the backend is launched from
    Git Bash, tpai.exe picks up Git's perl from PATH and dies on
    'failed to load Config_git.pl' before touching an image."""
    env = os.environ.copy()
    bad = ('git', 'mingw', 'perl', 'usr\\bin', '/usr/bin')
    env['PATH'] = os.pathsep.join(
        p for p in env.get('PATH', '').split(os.pathsep)
        if p and not any(k in p.lower() for k in bad))
    for k in [k for k in env if k.upper().startswith('PERL')]:
        env.pop(k)
    return env


def cap_output_side(path, cap=None):
    """Downscale an image whose longest side exceeds ``cap`` (Lanczos, ratio
    preserved, in place). Returns the new (w, h) when a resize happened, else
    None. ``cap`` None reads the ``topaz.max_output_side`` config; 0 disables.

    WHY THIS EXISTS: tpai's CLI cannot pin the upscale factor — the numeric
    overrides are broken at the engine level (``param1=``/``scale=`` are
    accepted with an 'Overwriting...' line, then the model dies with
    'type must be number, but is string' / 'Error running model' while the
    process exits 0; measured 2026-09-12 on 4.0.1, and the same failure is an
    unresolved Topaz community report on 3.3.1). Autopilot therefore picks its
    own factor — measurably 4x+ on small sources — and the only reliable cap
    is ours, applied after the run."""
    if cap is None:
        try:
            cap = int(cfg.get('topaz.max_output_side') or 0)
        except (TypeError, ValueError):
            cap = 0
    if cap <= 0 or not os.path.isfile(path):
        return None
    from PIL import Image
    try:
        with Image.open(path) as im:
            w, h = im.size
            if max(w, h) <= cap:
                return None
            ratio = cap / max(w, h)
            size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
            resized = im.resize(size, Image.LANCZOS)
            resized.save(path)
        logger.info('topaz: output %dx%d exceeded the %dpx cap -> %dx%d',
                    w, h, cap, size[0], size[1])
        return size
    except Exception:                                            # noqa: BLE001
        # A failed cap must never lose the Topaz output itself.
        logger.exception('topaz: could not cap output side for %s', path)
        return None


def run_tpai(exe, input_path, output_dir, *, timeout=DEFAULT_TIMEOUT_S, **toggles):
    """Run one image through Topaz. Returns (status, message) where status is
    one of 'ok' | 'partial' | 'no_valid_files' | 'license' | 'bad_args' |
    'timeout' | 'unknown'. Never raises for a Topaz refusal — the caller
    decides what each status means for the job."""
    cmd = build_command(exe, input_path, output_dir, **toggles)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=_clean_env())
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
