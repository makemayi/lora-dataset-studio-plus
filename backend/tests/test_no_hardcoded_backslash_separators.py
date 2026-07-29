"""No source file may rewrite a path onto a HARDCODED backslash.

WHY THIS FILE EXISTS (2026-07-28)
---------------------------------
GitHub #21 (1Tomber): LDS on Linux generated nothing at all, because six
listers/resolvers composed their model names with a literal
`.replace("/", "\\\\")`. That produces the right string on exactly one operating
system, and every model this app ships lives in a subfolder, so on Linux the
failure was total — from the first release, not from a regression.

The behavioural tests next door prove the fix works. This one is the reason it
does not come back: the six sites were identical, they were written months apart,
and each one looked reasonable on its own on a Windows machine. A behavioural
test only covers producers someone remembered to enumerate; this one fails on the
seventh, whoever writes it and wherever they put it.

THE RULE
--------
Rewriting a path ONTO a backslash is banned in `app/`. Nothing legitimate needs
it:

  * a path this process opens  -> `os.sep` (or just `os.path.join`), which is a
    backslash here and a forward slash there — see `comfy_names.local_model_path`;
  * a value bound for a ComfyUI widget -> the separator of the ComfyUI HOST,
    which is neither ours nor a constant — see
    `comfy_names.canonical_model_widgets`, which reads it off `/object_info`;
  * a comparison between two names -> `comfy_names.normalise_model_name`, which
    flattens onto '/'.

The reverse direction (`.replace('\\\\', '/')`) is NOT banned: flattening onto
POSIX is how every comparison key in the app is built, and it is correct on both
platforms.
"""
import re
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parents[1] / 'app'

# `x.replace('/', '\\')` in either quoting style, and `'\\'.join(...)`.
_ONTO_BACKSLASH = re.compile(
    r"""replace\(\s*(['"])/\1\s*,\s*(['"])\\\\\2\s*\)"""
    r"""|(['"])\\\\\3\s*\.\s*join\(""")

# Sites where a backslash is the SUBJECT, not a separator choice. Each needs a
# reason; "it works today" is not one.
ALLOWED = {
    # Detects a Windows-Store Python by its install path. The path IS a Windows
    # path — the check is meaningless anywhere else and never builds a name.
    'services/training_diagnostics.py',
}


def _sources():
    for path in sorted(APP_DIR.rglob('*.py')):
        yield path, path.relative_to(APP_DIR).as_posix()


def test_there_is_something_to_check():
    """A glob that matched nothing would make the assertion below pass for the
    wrong reason."""
    assert sum(1 for _ in _sources()) > 50


def test_no_source_file_rewrites_a_path_onto_a_backslash():
    offenders = []
    for path, rel in _sources():
        if rel in ALLOWED:
            continue
        for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            if _ONTO_BACKSLASH.search(line):
                offenders.append(f'app/{rel}:{n}: {line.strip()}')
    assert not offenders, (
        'These lines force a Windows separator, which is correct on exactly one '
        'operating system. On Linux this is what made LoRA Dataset Studio generate '
        'NOTHING (GitHub #21, 1Tomber): every model widget was refused by ComfyUI, '
        'whose validator does an exact string match.\n  '
        + '\n  '.join(offenders)
        + '\n\nUse os.sep (or os.path.join) for a path you OPEN, '
          'comfy_names.canonical_model_widgets for a value ComfyUI validates, and '
          'comfy_names.normalise_model_name to COMPARE two names.')


@pytest.mark.parametrize('snippet,banned', [
    ("""rel = x.replace("/", "\\\\")""", True),
    ("""rel = x.replace('/', '\\\\')""", True),
    ("""rel = '\\\\'.join(parts)""", True),
    ("""key = x.replace('\\\\', '/')""", False),      # the flattening we DO want
    ("""rel = x.replace('/', os.sep)""", False),      # host-aware, fine
    ("""rel = os.path.join(folder, name)""", False),
])
def test_the_detector_itself(snippet, banned):
    """A regex that matched nothing would make the ban silently vacuous — this is
    the trap the whole file exists to avoid, so it gets its own check."""
    assert bool(_ONTO_BACKSLASH.search(snippet)) is banned
