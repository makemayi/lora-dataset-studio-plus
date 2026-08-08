"""WHICH .safetensors in a delivered dense repository IS "the model".

WHY THIS FILE EXISTS
--------------------
A dense run does not deliver one file. ai-toolkit pushes every intermediate save
(``<job>_000002750.safetensors``) and, at the end, the final one
(``<job>.safetensors``) — all of them ~26 GB, all of them in the same private
repository. So "the delivered model" is a CHOICE, and until this module there
were two of them:

* the delivery verifier sorted the matching names and took the last one. Sorted
  lexicographically ``Krea….safetensors`` comes BEFORE ``Krea…_000002750``
  (``.`` is 0x2E, ``_`` is 0x5F), so the last one was the STEP checkpoint;
* the cloud quantizer took the LARGEST sibling, first one wins on a tie — and on
  a repo whose two files are the same model saved twice, the sizes tie.

Two rules, two answers, one screen: the card named ``…_000002750.safetensors``
while the operation offered right underneath it would have taken
``….safetensors``. Nobody was lied to on purpose; there was simply no single
place saying which file wins. This is that place.

THE RULE
--------
1. Only ``.safetensors`` / ``.sft``; an ``_fp8`` export is never a master (it is
   the OUTPUT of quantizing one, and re-quantizing it is refused elsewhere).
2. The FINAL save beats every step checkpoint. It is the model the run actually
   produced; a numbered file is a snapshot on the way there.
3. Between step checkpoints, the highest step wins — later is more trained.
4. Only then, size, then the name itself, so the answer is deterministic for a
   repository that somehow holds two finals (two runs pushed to one repo).

Point 2 is the one that carries a judgement, and it is worth stating: if a run
is interrupted, NO final exists and the highest step is all there is — which is
exactly what rule 3 returns. The rule never needs to know whether a run finished.
"""
from __future__ import annotations

import os
import re

ACCEPTED_EXT = ('.safetensors', '.sft')
FP8_MARKER = '_fp8'

# ai-toolkit pads step numbers to 9 digits (`_000002750`). Four is the floor that
# still cannot swallow a legitimate name suffix like `_v2` or `_1024`.
_STEP_RE = re.compile(r'^(?P<stem>.+?)_(?P<step>\d{4,})$')


def is_fp8_export(name) -> bool:
    """True for a file this app's own quantizer produced (``*_fp8.safetensors``)."""
    return _stem(name).endswith(FP8_MARKER)


def is_weight_file(name) -> bool:
    return str(name or '').lower().endswith(ACCEPTED_EXT)


def _stem(name) -> str:
    base = os.path.basename(str(name or '').replace('\\', '/'))
    for ext in ACCEPTED_EXT:
        if base.lower().endswith(ext):
            return base[:-len(ext)]
    return base


def split_step(name) -> tuple[str, int | None]:
    """``('Krea_x', 2750)`` for a step save, ``('Krea_x', None)`` for the final."""
    match = _STEP_RE.match(_stem(name))
    if not match:
        return _stem(name), None
    return match.group('stem'), int(match.group('step'))


def sort_key(name, size=0):
    """Bigger wins. Exposed so a caller can rank a list it already holds."""
    _stem_name, step = split_step(name)
    try:
        size = int(size or 0)
    except (TypeError, ValueError):
        size = 0
    # A final save has no step; it must outrank EVERY numbered one, so it is not
    # "step = infinity" (which would be a lie about training length) but a
    # separate, higher tier.
    return (1 if step is None else 0, step or 0, size, str(name or ''))


def candidates(entries) -> list[tuple[str, int]]:
    """``entries`` = names, or ``(name, size)`` pairs → the eligible masters."""
    out = []
    for entry in entries or ():
        if isinstance(entry, (tuple, list)):
            name, size = (list(entry) + [0])[:2]
        else:
            name, size = entry, 0
        name = str(name or '')
        if not name or not is_weight_file(name) or is_fp8_export(name):
            continue
        try:
            size = int(size or 0)
        except (TypeError, ValueError):
            size = 0
        out.append((name, size))
    return out


def pick_master(entries) -> str | None:
    """THE dense master of a repository/folder listing, or None if it holds none.

    ``entries`` accepts plain names or ``(name, size)`` pairs; the name is
    returned verbatim (a Hub ``rfilename`` may carry a subfolder, and callers
    need the path they were given, not a basename).
    """
    ranked = candidates(entries)
    if not ranked:
        return None
    return max(ranked, key=lambda item: sort_key(*item))[0]


def describe_choice(entries) -> dict:
    """The pick plus what it was picked OVER — so a UI can say why."""
    ranked = candidates(entries)
    if not ranked:
        return {'name': None, 'total': 0, 'step': None, 'is_final': False, 'others': []}
    chosen = max(ranked, key=lambda item: sort_key(*item))
    _stem_name, step = split_step(chosen[0])
    return {
        'name': chosen[0],
        'total': len(ranked),
        'step': step,
        'is_final': step is None,
        'others': [name for name, _size in ranked if name != chosen[0]],
    }
