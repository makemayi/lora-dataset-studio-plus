"""Flask-side access to the ONE atomic-.npz helper (``backend/infer/npz_atomic``).

The helper lives under ``backend/infer`` because the passes that need it most —
✨ Score, faces, semantic — run in the separate ML interpreter, which imports by
bare module name from that folder and must never import anything under ``app``
(that would drag Flask into the child). The Flask side has the same two .npz
caches to write and the same Windows rename to survive, so it reaches over to
that file instead of growing a second copy: a fix duplicated across two trees is
a fix that will diverge, and this one is subtle enough (retry budget, salvage
rules) that a divergence would be silent.

Importing this module makes ``npz_atomic`` importable process-wide, and both
names resolve to the SAME module object — which is what lets a test replace its
sleep seam once and have every caller obey.
"""
from __future__ import annotations

import pathlib
import sys

_INFER_DIR = str(pathlib.Path(__file__).resolve().parents[2] / 'infer')
if _INFER_DIR not in sys.path:
    sys.path.append(_INFER_DIR)

import npz_atomic  # noqa: E402

NpzReplaceLocked = npz_atomic.NpzReplaceLocked
locked_message = npz_atomic.locked_message
orphan_temporaries = npz_atomic.orphan_temporaries
replace_with_retry = npz_atomic.replace_with_retry
salvage_orphan_tmp = npz_atomic.salvage_orphan_tmp
save_npz_atomic = npz_atomic.save_npz_atomic

__all__ = ['NpzReplaceLocked', 'locked_message', 'orphan_temporaries',
           'replace_with_retry', 'salvage_orphan_tmp', 'save_npz_atomic']
