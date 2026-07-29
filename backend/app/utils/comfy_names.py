"""Model-file names as the ComfyUI on the OTHER END OF THE WIRE spells them.

WHY THIS FILE EXISTS (2026-07-28)
---------------------------------
Reported by 1Tomber (GitHub #21) running LDS on Linux: nothing generated at all.
Every subfoldered model was refused by ComfyUI before a single step ran —

    Failed to validate prompt for output 28:
    * UNETLoader 20:
      - Value not in list: unet_name: 'Krea\\krea2_turbo_fp8.safetensors'
        not in ['Krea/krea2_turbo_fp8.safetensors']

and this app puts essentially ALL of its models in subfolders (``unet/Krea``,
``unet/klein``, ``unet/z image``, ``loras/sdxl``, ``loras/krea``…), so on Linux
the answer was "nothing works", not "one model is missing".

THE PART OF THE REPORT THAT IS TRUE
-----------------------------------
ComfyUI validates a model widget by EXACT string match against the list it
publishes — ``execution.py``: ``invalid_vals = [val] if val not in combo_options
else []``. No separator tolerance, no case tolerance. Whatever list that install
serves is the only spelling it accepts. And several of our model listers spelled
their names with a HARDCODED backslash (``.replace("/", "\\\\")``), which is
wrong on any host that is not Windows — that is the bug, and it was never a
regression, it had always been there. Nobody saw it because nobody here runs
Linux.

THE PART OF THE REPORT THAT IS NOT TRUE — AND WHY IT MATTERS
------------------------------------------------------------
The report concludes "ComfyUI enums are POSIX on every platform — emit ``/``
regardless of host OS". They are NOT. ``folder_paths.recursive_search`` builds
each name with a plain ``os.path.relpath`` and normalises NOTHING:

    relative_path = os.path.relpath(os.path.join(dirpath, file_name), directory)
    result.append(relative_path)

(the same file normalises explicitly two functions further down, for input
subfolders: ``folders.append(rel_path.replace(os.sep, '/'))`` — so the absence
above is the behaviour, not an oversight). Measured against a live Windows
ComfyUI 0.27.0, ``GET /object_info/UNETLoader`` answers
``"Krea\\krea2_turbo_fp8.safetensors"`` — WITH the backslash. Emitting ``/``
unconditionally would therefore have fixed Linux by breaking every Windows
install, which today is almost all of them.

THE RULE THIS MODULE ENCODES
----------------------------
The separator that matters belongs to the ComfyUI HOST, and it is never a
constant:

  * it is not ours — a Windows LDS driving a ComfyUI in WSL or in a container
    needs ``/`` while ``os.sep`` says ``\\``;
  * it is not POSIX — a Windows ComfyUI needs ``\\``.

So we ASK. ``/object_info`` already publishes the exact strings that install
accepts (we already fetch and cache it for two other preflights), and
`canonical_model_widgets` rewrites every model widget to the published spelling
right before the graph is POSTed. Matching is done on a normalised key
(separators flattened, case folded), so it works from EITHER convention — which
is the reverse direction of the same bug, and the one a Windows-LDS/WSL-ComfyUI
user hits.

When the probe is unavailable we fall back to ``os.sep``. That is not a guess:
the app finds these files by walking the ComfyUI tree on its OWN filesystem, so
in every install where that walk means anything, ComfyUI's host IS this host.
It also makes the fallback byte-for-byte the historical behaviour on Windows.

LOCAL FILESYSTEM PATHS ARE NOT THIS
-----------------------------------
Nothing here may be used on a path this process actually opens. Those keep going
through ``os.path.join`` / ``os.sep`` and must stay in the host's convention — a
name like ``z image/base.safetensors`` handed to ``open()`` on Windows is fine,
but the mirror image (``z image\\base.safetensors`` on Linux) is a single file
whose name contains a backslash, and that is a second, separate bug (it was
real: see ``zimage_convert._resolve_z_model_path``). The split is deliberate:
`with_separator(name, os.sep)` for something we open, `canonical_model_widgets`
for something ComfyUI validates.
"""
from __future__ import annotations

import os

# The loader classes whose model-file widgets we emit, and the widget names that
# carry a file name. Lives here rather than in utils.comfyui because both the
# capability preflights and the emit-time canonicaliser key off the same two
# sets, and they must never drift apart.
MODEL_FILE_CLASSES = frozenset({
    'UNETLoader', 'CheckpointLoaderSimple', 'VAELoader', 'CLIPLoader',
    'DualCLIPLoader', 'LoraLoaderModelOnly', 'LoraLoader', 'ControlNetLoader',
    'UnetLoaderGGUF', 'UnetLoaderGGUFAdvanced', 'CLIPLoaderGGUF', 'DualCLIPLoaderGGUF',
})
MODEL_FILE_INPUTS = frozenset({
    'unet_name', 'ckpt_name', 'vae_name', 'clip_name', 'clip_name1', 'clip_name2',
    'lora_name', 'control_net_name', 'style_model_name',
})


def normalise_model_name(name) -> str:
    """Comparison key for a ComfyUI model file name: separators flattened to ``/``
    and case folded.

    Both flattenings are load-bearing. Separators, because the two ends of the
    wire can legitimately disagree (Windows app / Linux ComfyUI and the reverse).
    Case, because Windows model folders are case-insensitive, so the same file is
    reachable under spellings that differ only in case and a raw comparison would
    call a perfectly loadable model missing. A false "missing" here BLOCKS a
    working install, which is the expensive direction to get wrong."""
    return str(name or '').replace('\\', '/').casefold()


def same_model_name(a, b) -> bool:
    """True when two names denote the same model file whatever convention either
    side wrote it in. The ONLY way this app compares model names."""
    return normalise_model_name(a) == normalise_model_name(b)


def with_separator(name, sep: str) -> str:
    """`name` respelled with `sep` between its segments, from either convention.

    `sep=os.sep` is the form for a path this process will OPEN; the ComfyUI-host
    separator is the form for a widget. Never guesses which one you want."""
    flat = str(name or '').replace('\\', '/')
    return flat if sep == '/' else flat.replace('/', sep)


def local_model_path(name) -> str:
    """A model name respelled for THIS filesystem — the explicit opposite of a
    widget value. Use it everywhere a relative model name is about to be joined
    onto a real directory; `os.path.join(root, 'z image\\\\x.safetensors')` on
    Linux does not fail, it silently builds a file name nobody has."""
    return with_separator(name, os.sep)


def enum_separator(listed) -> str | None:
    """The separator the target ComfyUI actually uses in its own model lists, read
    off `listed` (the distilled ``/object_info`` view), or None when nothing in it
    carries a subfolder — a flat install tells us nothing, and rightly so: with no
    subfolder anywhere, no separator can be wrong."""
    for by_input in (listed or {}).values():
        for names in (by_input or {}).values():
            for raw in (names.values() if hasattr(names, 'values') else names):
                if '\\' in raw:
                    return '\\'
                if '/' in raw:
                    return '/'
    return None


def canonical_model_widgets(workflow, listed=None, sep=None):
    """`(workflow, changes)` — the graph with every model-file widget respelled the
    way the target ComfyUI spells it.

    `listed` is the distilled ``/object_info`` model view,
    ``{class_type: {input_name: {normalised_name: published_name}}}``; None when
    the probe failed or the target is a remote worker we cannot interrogate.
    `sep` forces the fallback separator (tests, and callers that know better);
    by default it is the separator observed in `listed`, else ``os.sep``.

    Three cases, in order:
      1. the published list knows this name (compared normalised) -> use the
         PUBLISHED string verbatim. This is the whole fix: it is correct from
         either convention, in either direction, and it also repairs a case
         difference for free;
      2. the class or input is not in the published list, or the probe failed ->
         respell with the fallback separator. Still fixes Linux (``os.sep`` is
         ``/`` there) and still a no-op on Windows;
      3. a model this ComfyUI genuinely does not have -> respelled too, then left
         alone. It must stay wrong so `unavailable_model_files` can name it; a
         silent substitution would generate with a model nobody asked for.

    COPY-ON-WRITE: the caller's dict is never mutated, and untouched nodes are
    shared, so a graph that needs no change costs one dict copy. Callers keep
    inspecting the graph they built (several tests and the retry path do).
    """
    if not isinstance(workflow, dict):
        return workflow, 0
    if sep is None:
        sep = enum_separator(listed) or os.sep
    out, changes = workflow, 0
    for node_id, node in workflow.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get('inputs')
        if not isinstance(inputs, dict):
            continue
        by_input = (listed or {}).get(node.get('class_type')) or {}
        fixed = None
        for name, value in inputs.items():
            if name not in MODEL_FILE_INPUTS or not isinstance(value, str) or not value:
                continue
            published = by_input.get(name)
            new = None
            if published:
                new = published.get(normalise_model_name(value))
            if new is None:
                new = with_separator(value, sep)
            if new == value:
                continue
            if fixed is None:
                fixed = dict(inputs)
            fixed[name] = new
            changes += 1
        if fixed is not None:
            if out is workflow:
                out = dict(workflow)
            out[node_id] = dict(node, inputs=fixed)
    return out, changes
