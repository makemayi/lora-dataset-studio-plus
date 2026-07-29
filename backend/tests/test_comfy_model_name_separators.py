"""A model name must be spelled the way the TARGET ComfyUI spells it.

WHY THIS FILE EXISTS (2026-07-28)
---------------------------------
Reported by 1Tomber (GitHub #21), LDS on Linux, ComfyUI 0.27.0: nothing
generated. Every model widget carried a Windows backslash, ComfyUI validates
those by exact string match, and this app keeps essentially every model in a
subfolder — so the failure was total, and it had been there from day one. Nobody
saw it because nobody here runs Linux.

WHAT THIS FILE PINS, AND WHY IT IS PROVABLE ON WINDOWS
------------------------------------------------------
The bug reproduces on the author's machine: `os.path.join` and the hardcoded
`.replace("/", "\\")` this code used both yield a backslash here. So "compose a name,
show it carries a backslash" is a real red test on Windows — what it cannot show
is Linux, and the tests below therefore drive the LINUX case through the
separator seam (`sep=`) rather than pretending to run there.

The other half is the correction the report itself needed. It concluded "emit `/`
regardless of host OS"; ComfyUI's enums are NOT POSIX everywhere
(`folder_paths.recursive_search` uses a bare `os.path.relpath`, and a live
Windows 0.27.0 publishes `Krea\krea2_turbo_fp8.safetensors`), so emitting `/`
unconditionally would have fixed Linux by breaking every Windows install. The
separator belongs to ComfyUI's host, we read it off `/object_info`, and both
directions are tested below — including the Windows-app/WSL-ComfyUI user, who is
the mirror image of the reporter.
"""
import os

import pytest

from app.utils import comfy_names
from app.utils.comfy_names import (canonical_model_widgets, enum_separator,
                                   local_model_path, normalise_model_name,
                                   same_model_name, with_separator)

POSIX, WIN = '/', '\\'


def _listed(sep):
    """A distilled /object_info model view as a ComfyUI running on a host whose
    separator is `sep` would publish it: {class: {input: {norm: published}}}."""
    def entry(*names):
        return {normalise_model_name(n.replace('/', sep)): n.replace('/', sep)
                for n in names}
    return {
        'UNETLoader': {'unet_name': entry('Krea/krea2_turbo_fp8.safetensors',
                                          'z image/bigLove_zt3.safetensors',
                                          'klein/flux-2-klein-9b-fp8.safetensors')},
        'LoraLoaderModelOnly': {'lora_name': entry('krea/mylora.safetensors',
                                                   'sdxl/lora_x.safetensors')},
        'CheckpointLoaderSimple': {'ckpt_name': entry('Biglove/photo5.safetensors')},
        'VAELoader': {'vae_name': entry('flux2-vae.safetensors')},
    }


def _graph(**widgets):
    """A minimal API-format graph carrying one loader per widget kind."""
    cls = {'unet_name': 'UNETLoader', 'lora_name': 'LoraLoaderModelOnly',
           'ckpt_name': 'CheckpointLoaderSimple', 'vae_name': 'VAELoader'}
    return {str(i): {'class_type': cls[k], 'inputs': {k: v, 'strength_model': 1.0}}
            for i, (k, v) in enumerate(widgets.items(), start=1)}


def _values(graph):
    return [v for node in graph.values() for k, v in node['inputs'].items()
            if k in comfy_names.MODEL_FILE_INPUTS]


# --- the reported failure, end to end -------------------------------------

def test_a_windows_spelled_graph_is_respelled_for_a_linux_comfyui():
    """THE reported bug. The graph as this app composes it on Windows, sent to the
    Linux ComfyUI of #21: every name comes out exactly as that install lists it,
    and not one backslash survives."""
    graph = _graph(unet_name='Krea\\krea2_turbo_fp8.safetensors',
                   lora_name='krea\\mylora.safetensors',
                   ckpt_name='Biglove\\photo5.safetensors')
    out, changed = canonical_model_widgets(graph, _listed(POSIX))
    assert changed == 3
    assert _values(out) == ['Krea/krea2_turbo_fp8.safetensors',
                            'krea/mylora.safetensors',
                            'Biglove/photo5.safetensors']
    assert not any(WIN in v for v in _values(out))


def test_the_windows_install_that_works_today_is_left_alone():
    """The regression this fix could plausibly have caused, and the reason the
    report's own prescription ("emit / regardless of host OS") was not followed:
    a Windows ComfyUI publishes backslashes — measured on a live 0.27.0 — so
    forcing POSIX would have broken every install that works today."""
    graph = _graph(unet_name='Krea\\krea2_turbo_fp8.safetensors',
                   lora_name='krea\\mylora.safetensors')
    out, changed = canonical_model_widgets(graph, _listed(WIN))
    assert changed == 0
    assert out is graph          # copy-on-write: nothing to fix, nothing copied
    assert _values(out) == ['Krea\\krea2_turbo_fp8.safetensors',
                            'krea\\mylora.safetensors']


def test_the_reverse_direction_a_windows_app_driving_a_comfyui_in_wsl():
    """The mirror image of #21, and the reason a per-OS constant cannot be the
    fix: here the APP is on Windows (os.sep == '\\') and ComfyUI is not. The
    published list is the only authority, and it wins over the host."""
    graph = _graph(unet_name=os.path.join('z image', 'bigLove_zt3.safetensors'))
    out, _ = canonical_model_widgets(graph, _listed(POSIX))
    assert _values(out) == ['z image/bigLove_zt3.safetensors']


def test_case_only_differences_are_repaired_too():
    """Windows model folders are case-insensitive, so the app can hold a spelling
    the user's own folder does not have. Matching normalised means we hand back
    the PUBLISHED string, which is the only one the validator accepts."""
    graph = _graph(unet_name='KREA/KREA2_TURBO_FP8.SAFETENSORS')
    out, changed = canonical_model_widgets(graph, _listed(POSIX))
    assert changed == 1
    assert _values(out) == ['Krea/krea2_turbo_fp8.safetensors']


def test_a_model_this_comfyui_does_not_have_is_respelled_but_not_substituted():
    """It must stay WRONG so `unavailable_model_files` can name it. Quietly
    swapping in a neighbour would generate with a model nobody asked for."""
    graph = _graph(unet_name='Krea\\a_model_nobody_has.safetensors')
    out, _ = canonical_model_widgets(graph, _listed(POSIX))
    assert _values(out) == ['Krea/a_model_nobody_has.safetensors']


def test_with_no_object_info_the_fallback_still_fixes_linux():
    """The probe fails (timeout, ComfyUI restarting, remote worker). Falling back
    to os.sep is not a guess: the app finds these files by walking ComfyUI's tree
    on its OWN filesystem, so where that walk means anything the two hosts are the
    same one. It is also byte-for-byte the historical behaviour on Windows."""
    graph = _graph(unet_name='Krea\\krea2_turbo_fp8.safetensors')
    out, _ = canonical_model_widgets(graph, None, sep='/')
    assert _values(out) == ['Krea/krea2_turbo_fp8.safetensors']
    out_win, _ = canonical_model_widgets(graph, None, sep='\\')
    assert _values(out_win) == ['Krea\\krea2_turbo_fp8.safetensors']
    # and with no sep forced, the host's own separator — what actually runs.
    out_here, _ = canonical_model_widgets(graph, None)
    assert _values(out_here) == ['Krea' + os.sep + 'krea2_turbo_fp8.safetensors']


def test_non_model_widgets_and_links_are_never_touched():
    """A prompt that happens to contain a backslash, a link ([node, slot]), a
    number: none of them is a file name and none may be rewritten."""
    graph = {'1': {'class_type': 'CLIPTextEncode',
                   'inputs': {'text': 'a back\\slash in a prompt', 'clip': ['2', 0]}},
             '2': {'class_type': 'UNETLoader',
                   'inputs': {'unet_name': ['9', 0], 'weight_dtype': 'default'}}}
    out, changed = canonical_model_widgets(graph, _listed(POSIX))
    assert changed == 0
    assert out['1']['inputs']['text'] == 'a back\\slash in a prompt'
    assert out['2']['inputs']['unet_name'] == ['9', 0]


def test_the_caller_s_graph_is_never_mutated():
    """Copy-on-write: the retry path and several callers keep reading the graph
    they built."""
    graph = _graph(unet_name='Krea\\krea2_turbo_fp8.safetensors')
    out, _ = canonical_model_widgets(graph, _listed(POSIX))
    assert graph['1']['inputs']['unet_name'] == 'Krea\\krea2_turbo_fp8.safetensors'
    assert out is not graph


# --- the separator is READ, never assumed ---------------------------------

def test_the_separator_is_read_off_the_live_list():
    assert enum_separator(_listed(POSIX)) == '/'
    assert enum_separator(_listed(WIN)) == '\\'
    assert enum_separator(None) is None
    # A flat install carries no separator anywhere — and then none can be wrong.
    assert enum_separator({'VAELoader': {'vae_name': {'a.safetensors': 'a.safetensors'}}}) is None


# --- comparison works from both conventions, both directions --------------

@pytest.mark.parametrize('a,b', [
    ('Krea\\x.safetensors', 'Krea/x.safetensors'),
    ('Krea/x.safetensors', 'Krea\\x.safetensors'),
    ('KREA/X.SAFETENSORS', 'krea\\x.safetensors'),
    ('a/b/c.safetensors', 'a\\b\\c.safetensors'),
])
def test_two_spellings_of_one_file_compare_equal(a, b):
    assert same_model_name(a, b)


def test_different_files_still_compare_different():
    assert not same_model_name('Krea/x.safetensors', 'Krea/y.safetensors')
    assert not same_model_name('a/x.safetensors', 'b/x.safetensors')


# --- the OTHER half: local filesystem paths must NOT become POSIX ---------

def test_local_paths_keep_the_host_separator_and_stay_openable():
    """The line this fix must not cross. A widget value and a path we `open()` are
    two different strings; conflating them is how the same report would come back
    as "conversion impossible" instead of "generation impossible"."""
    assert local_model_path('z image/base.safetensors') == os.path.join(
        'z image', 'base.safetensors')
    assert local_model_path('z image\\base.safetensors') == os.path.join(
        'z image', 'base.safetensors')
    assert os.sep in local_model_path('a/b.safetensors')


def test_a_local_path_built_from_a_widget_name_still_finds_the_file(tmp_path):
    """The real check, on a real disk: whichever convention the name arrives in,
    joining it onto a directory must reach the file that exists."""
    sub = tmp_path / 'z image'
    sub.mkdir()
    (sub / 'base.safetensors').write_bytes(b'x')
    for spelling in ('z image/base.safetensors', 'z image\\base.safetensors'):
        assert os.path.isfile(os.path.join(str(tmp_path), local_model_path(spelling)))


def test_with_separator_round_trips_both_ways():
    assert with_separator('a\\b\\c', '/') == 'a/b/c'
    assert with_separator('a/b/c', '\\') == 'a\\b\\c'
    assert with_separator('flat.safetensors', '/') == 'flat.safetensors'
    assert with_separator(None, '/') == ''
