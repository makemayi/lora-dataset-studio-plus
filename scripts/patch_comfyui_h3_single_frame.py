"""Let MiniMax H3 sample ONE frame instead of a packet of five.

WHY THIS SCRIPT EXISTS. H3 reaches a still through a video model: it samples a
packet and the app keeps exactly one frame of it. ComfyUI's own node declares
`length` as `min=5, step=17`, so the cheapest legal packet costs five frames of
sampling to produce one image — four of them are decoded and thrown away. Worse,
pulling frame 0 out of a packet is what produces grid artefacts with the
single-image VAE; a length of 1 sidesteps that as well.

The fix is four small edits to `comfy_extras/nodes_minimax_h3.py`, upstreamed as
ComfyUI issue #15644. Until that lands, every ComfyUI update silently reverts
them, and the symptom is not a bad image — it is EVERY tile failing validation
at queue time, because `minimax_h3.length = 1` is then off-grid. That failure
mode is exactly why this is a script and not a wiki page: it must be cheap to
re-run, safe to run twice, and able to answer "am I patched?" without guessing.

USAGE

    python scripts/patch_comfyui_h3_single_frame.py            # patch (idempotent)
    python scripts/patch_comfyui_h3_single_frame.py --check    # report only
    python scripts/patch_comfyui_h3_single_frame.py --revert   # restore the backup

ComfyUI is located from `comfyui.base_dir` in the app's own `config.json`, or
from `--comfy-dir`, or from `LDS_COMFYUI_DIR`. Nothing here is machine-specific:
no path is written into the repo, and diagnostics print paths as given.

EXIT CODES: 0 = patched (or already patched / reverted), 1 = could not, and with
`--check`, 2 = ComfyUI found but NOT patched. CI never runs this; it is a local
tool for a local ComfyUI.
"""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import re
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_REL = os.path.join('comfy_extras', 'nodes_minimax_h3.py')
BACKUP_SUFFIX = '.lds-preH3SingleFrame.bak'

# Each edit is (name, pattern, replacement, expected_count). Anchored on code,
# never on line numbers: upstream reformats this file freely, and a patch that
# depends on a line number fails the moment a comment above it grows a word.
EDITS = (
    (
        'align_frame_count accepts a single frame',
        re.compile(r'(def align_frame_count\(n\):\n)(    while n % 17 != 5:)'),
        r'\1    if n <= 1:\n        return 1\n\2',
        1,
    ),
    (
        'video_latent_t accepts a single frame',
        re.compile(r'(def video_latent_t\(frame_count\):\n)'
                   r'(    return 2 if frame_count <= 5)'),
        r'\1    if frame_count <= 1:\n        return 1\n\2',
        1,
    ),
    (
        'temporal_shape stops flooring the request at 5',
        re.compile(r'align_frame_count\(max\(5, length\)\)'),
        'align_frame_count(max(1, length))',
        1,
    ),
    (
        'the three `length` inputs accept 1',
        re.compile(r'(io\.Int\.Input\("length", default=124, )min=5'),
        r'\1min=1',
        3,
    ),
)

# What "already patched" looks like. Checked BEFORE editing, so a second run is
# a no-op rather than a double insertion.
MARKERS = (
    ('align_frame_count', re.compile(r'def align_frame_count\(n\):\n    if n <= 1:')),
    ('video_latent_t', re.compile(r'def video_latent_t\(frame_count\):\n    if frame_count <= 1:')),
    ('temporal_shape', re.compile(r'align_frame_count\(max\(1, length\)\)')),
    ('length inputs', re.compile(r'io\.Int\.Input\("length", default=124, min=1')),
)


def find_comfyui_dir(explicit=None):
    """Return the ComfyUI directory, or None. Explicit beats env beats config.

    An explicit `--comfy-dir` is AUTHORITATIVE: if the caller named a directory
    and the node file is not under it, the answer is None. Falling through to
    the configured install would mean patching a ComfyUI the caller did not
    name, which is the one outcome a targeting flag must never produce."""
    if explicit is not None and explicit.strip():
        return _node_root(explicit)
    for candidate in (os.environ.get('LDS_COMFYUI_DIR'), _config_base_dir()):
        if not candidate or not candidate.strip():
            continue
        root = _node_root(candidate)
        if root is not None:
            return root
    return None


def _node_root(candidate):
    """The directory holding `comfy_extras/nodes_minimax_h3.py`, or None.

    A hint may point at the portable root OR at ComfyUI itself — both spellings
    are common in the wild, so try the file under each."""
    base = os.path.abspath(os.path.expanduser(candidate.strip()))
    for root in (base, os.path.join(base, 'ComfyUI')):
        if os.path.isfile(os.path.join(root, TARGET_REL)):
            return root
    return None


def _config_base_dir():
    """`comfyui.base_dir` out of the app's config, without importing the app.

    Importing `backend.app.config` would pull the whole service in for one
    string, and this script has to run when that import is broken."""
    path = os.path.join(REPO_ROOT, 'config.json')
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return (json.load(handle).get('comfyui') or {}).get('base_dir') or ''
    except (OSError, ValueError):
        return ''


def patch_state(source):
    """(applied, missing) marker names for an already-read file."""
    applied = [name for name, rx in MARKERS if rx.search(source)]
    missing = [name for name, rx in MARKERS if not rx.search(source)]
    return applied, missing


def apply_edits(source):
    """Return (new_source, problems). Empty problems means every edit landed."""
    problems = []
    out = source
    for name, rx, repl, expected in EDITS:
        out, count = rx.subn(repl, out)
        if count != expected:
            problems.append(f'{name}: matched {count}x, expected {expected}x')
    return out, problems


def _compiles(source, label):
    """py_compile the candidate text before it is allowed near the real file."""
    handle = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False,
                                         encoding='utf-8', newline='')
    try:
        handle.write(source)
        handle.close()
        py_compile.compile(handle.name, cfile=handle.name + 'c', doraise=True)
        return True, ''
    except py_compile.PyCompileError as exc:
        return False, f'{label} does not compile: {exc}'
    finally:
        for leftover in (handle.name, handle.name + 'c'):
            try:
                os.unlink(leftover)
            except OSError:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--comfy-dir', default=None,
                        help='ComfyUI directory (overrides config.json and LDS_COMFYUI_DIR)')
    parser.add_argument('--check', action='store_true',
                        help='report whether the patch is applied; change nothing')
    parser.add_argument('--revert', action='store_true',
                        help='restore the backup this script made')
    args = parser.parse_args(argv)

    root = find_comfyui_dir(args.comfy_dir)
    if root is None:
        print('ComfyUI not found. Set comfyui.base_dir in config.json, or pass '
              '--comfy-dir, or set LDS_COMFYUI_DIR.')
        return 1

    target = os.path.join(root, TARGET_REL)
    backup = target + BACKUP_SUFFIX
    print(f'target: {target}')

    if args.revert:
        if not os.path.isfile(backup):
            print('no backup from this script to restore.')
            return 1
        shutil.copyfile(backup, target)
        os.unlink(backup)
        print('reverted; ComfyUI needs a restart. Set minimax_h3.length back to 5.')
        return 0

    with open(target, 'r', encoding='utf-8', newline='') as handle:
        raw = handle.read()

    # Match on LF only, and put the file's own endings back before writing. A
    # ComfyUI checked out with `core.autocrlf=true` has CRLF, and anchoring on
    # a bare \n there matches nothing — which reads as "this file has moved on"
    # and refuses a patch that would have applied perfectly.
    crlf = '\r\n' in raw
    source = raw.replace('\r\n', '\n') if crlf else raw

    applied, missing = patch_state(source)
    if not missing:
        print('already patched - nothing to do.')
        return 0
    if applied:
        # Half-applied is the interesting case: an upstream change landed one of
        # these edits, or a previous run died mid-write. Say so; do not guess.
        print('PARTIALLY patched - present: ' + ', '.join(applied))
        print('                    missing: ' + ', '.join(missing))
    if args.check:
        print('NOT patched. Run without --check to apply.')
        return 2

    patched, problems = apply_edits(source)
    if problems:
        print('refusing to write; the file does not look like the version this '
              'patch was written against:')
        for problem in problems:
            print('  - ' + problem)
        return 1

    ok, why = _compiles(patched, 'the patched file')
    if not ok:
        print('refusing to write: ' + why)
        return 1

    if not os.path.isfile(backup):
        shutil.copyfile(target, backup)
        print(f'backup: {backup}')
    with open(target, 'w', encoding='utf-8', newline='') as handle:
        handle.write(patched.replace('\n', '\r\n') if crlf else patched)

    still_missing = patch_state(patched)[1]
    if still_missing:
        print('wrote the file but the markers still do not match: '
              + ', '.join(still_missing))
        return 1
    print('patched. RESTART ComfyUI - the module is already imported.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
