"""The engine list exists in TWO languages — this file is the seam that holds
them together.

WHY THIS TEST EXISTS
--------------------
Python owns `svc.API_ENGINES` (what /ref/edit and the fan-out accept) and
JavaScript owns `API_ENGINES` in engineSelection.js (what the workspace offers).
Neither can import the other, so the only thing that can stop them drifting is a
test that reads both. Drift is not theoretical: OpenRouter shipped as a
generation engine while the ✦ Edit modal and the /ref/edit route each kept their
own two-engine copy, so the app offered an engine set that no longer matched what
the server accepted.

The rule the codebase now follows is DERIVE, don't copy: the edit engines are the
API engines on both sides (`EDIT_ENGINES = [...API_ENGINES]` in JS,
`svc.API_ENGINES` in the route). That kills the third and fourth copies. This
test kills the drift between the two that remain — ids AND human labels, because
the labels word user-facing refusals on both sides.

Parsing JS with a regex is crude, and deliberately so: it must fail loudly if the
declaration moves or changes shape, rather than quietly matching nothing.
"""
import re
from pathlib import Path

import pytest

# The engine lists and the message derived from them moved to
# dataset_generation_service (2026-08 split). This file MUTATES those module
# attributes to prove the message is derived, and the functions read them as
# their own module globals -- so it has to drive the owning module, not the
# re-exported references on face_dataset_service.
from app.services import dataset_generation_service as svc
from app.services import face_dataset_service as fsvc

_JS = (Path(__file__).resolve().parents[2]
       / 'frontend' / 'src' / 'components' / 'dataset' / 'engineSelection.js')


def _js_source():
    if not _JS.exists():                       # source-only checkout of the backend
        pytest.skip(f'frontend source not present ({_JS.name})')
    return _JS.read_text(encoding='utf-8')


def _js_api_engines():
    m = re.search(r'export const API_ENGINES\s*=\s*\[(.*?)\];', _js_source(), re.S)
    assert m, 'API_ENGINES declaration not found in engineSelection.js'
    return tuple(re.findall(r"'([^']+)'", m.group(1)))


def _js_edit_engines():
    """EDIT_ENGINES lives in referenceEdit.js and is spelled `[...ENGINES]`, so
    what has to match is the CANONICAL list it copies."""
    m = re.search(r'export const ENGINES\s*=\s*\[(.*?)\];', _js_source(), re.S)
    assert m, 'ENGINES declaration not found in engineSelection.js'
    return tuple(re.findall(r"'([^']+)'", m.group(1)))


def _js_edit_ref_support():
    """EDIT_REF_SUPPORT in referenceEdit.js: which references each LOCAL engine
    consumes. Lives in the OTHER file, so it gets its own reader."""
    path = _JS.parent / 'referenceEdit.js'
    if not path.exists():
        pytest.skip(f'frontend source not present ({path.name})')
    src = path.read_text(encoding='utf-8')
    m = re.search(r'export const EDIT_REF_SUPPORT\s*=\s*\{(.*?)\};', src, re.S)
    assert m, 'EDIT_REF_SUPPORT declaration not found in referenceEdit.js'
    return dict(re.findall(r"(\w+):\s*'([^']*)'", m.group(1)))


def _js_engine_labels():
    m = re.search(r'export const ENGINE_LABELS\s*=\s*\{(.*?)\};', _js_source(), re.S)
    assert m, 'ENGINE_LABELS declaration not found in engineSelection.js'
    return dict(re.findall(r"(\w+):\s*'([^']*)'", m.group(1)))


def test_the_api_engine_ids_are_identical_on_both_sides():
    """Same ids, same ORDER: the order drives the toggle order in the ✦ Edit modal
    and the batch build order on the server."""
    assert _js_api_engines() == svc.API_ENGINES


def test_the_editable_engine_ids_are_identical_on_both_sides():
    """The set the ✦ Edit modal offers and the set /ref/edit accepts. It is now
    EVERY engine — the local ones edit through the ComfyUI queue — and the ORDER
    matters twice over: it is the toggle order in the modal, and it puts the free
    engines first on a gesture that is billed per press."""
    assert _js_edit_engines() == svc.editable_engines()


def test_the_local_engines_reference_support_matches_on_both_sides():
    """The UI says at PICK time which reference photos an engine will use; the
    service is what actually forwards (or doesn't) forward them. Those two
    claiming different things is the silent drop the whole feature avoids — the
    user would be told Klein takes the extra angles and get an edit that ignored
    them, or the reverse."""
    assert _js_edit_ref_support() == fsvc.LOCAL_EDIT_REF_SUPPORT
    # And every local engine has an answer: a new one must not default to "all".
    for engine in svc.LOCAL_ENGINES:
        assert engine in fsvc.LOCAL_EDIT_REF_SUPPORT, engine


def test_the_engine_labels_are_worded_identically_on_both_sides():
    """Both sides word a refusal from these labels ('pick Klein, Krea 2 Edit, Nano
    Banana Pro, ChatGPT or OpenRouter'), so the same engine must not be called two
    different things depending on whether the client or the server said no."""
    js = _js_engine_labels()
    py = svc.engine_labels()
    for engine in svc.editable_engines():
        assert js.get(engine) == py.get(engine), engine


def test_the_refusal_message_is_derived_from_the_list_not_hardcoded():
    """The point of deriving: adding an engine to a lane rewrites the message with
    no edit anywhere else. Pinned by mutating the tuple, not by pinning the
    sentence — a pinned sentence is just the old hardcoded list again."""
    msg = svc.edit_engine_choice_message()
    labels = svc.engine_labels()
    for engine in svc.editable_engines():
        assert labels[engine] in msg

    real_engines, real_labels = svc.API_ENGINES, svc.API_ENGINE_LABELS
    real_local, real_local_labels = svc.LOCAL_ENGINES, svc.LOCAL_ENGINE_LABELS
    try:
        svc.LOCAL_ENGINES, svc.LOCAL_ENGINE_LABELS = (), {}
        svc.API_ENGINES = ('nanobanana', 'newcomer')
        svc.API_ENGINE_LABELS = dict(real_labels, newcomer='Newcomer')
        assert svc.edit_engine_choice_message() == 'pick Nano Banana Pro or Newcomer'
    finally:
        svc.API_ENGINES, svc.API_ENGINE_LABELS = real_engines, real_labels
        svc.LOCAL_ENGINES, svc.LOCAL_ENGINE_LABELS = real_local, real_local_labels
