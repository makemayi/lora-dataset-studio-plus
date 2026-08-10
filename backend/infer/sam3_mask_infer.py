"""Auto-mask generator — SAM 3, run by the app's OWN automask interpreter.

    stdin  : {"images": [path…], "prompt": "head", "out_dir": path,
              "checkpoint": path, "threshold": 0.5, "device": "cuda"|"cpu"|null,
              "cancel_file": path|null}
    stdout : last line = JSON {"ok", "written", "cancelled",
                               "results": {path: {"state", "coverage", "boxes"}}}
    Logs   -> stderr.

WHY THIS RUNS OUTSIDE THE FLASK VENV, LIKE EVERY OTHER MODEL IN THIS APP
------------------------------------------------------------------------
SAM 3 wants torch ≥ 2.7 and CUDA ≥ 12.6 and documents no CPU path, which is not
something to force into the app's own interpreter. Same subprocess contract as
mask_infer.py (rembg) and face_mask_infer.py (InsightFace): a JSON payload in, a
JSON line out, progress on stderr, and the parent owns everything else.

THE WEIGHTS ARE BORROWED, NEVER DOWNLOADED HERE
-----------------------------------------------
`checkpoint` is Meta's own `sam3.pt`, which a ComfyUI install already has (the
segmentation packs fetch it from facebook/sam3). The parent resolves it through
the same model-path machinery every engine uses, so an extra_model_paths.yaml
root works identically. This script never reaches the network: a missing file is
an error with a name, not a 3.4 GB surprise download mid-click.

WHAT COMES BACK
---------------
One 8-bit PNG per image, white where the phrase matched, saved under the SAME
base name — the convention the app's other mask writers already use. `coverage`
(the white fraction) rides along in the JSON because the caller needs it to
refuse two useless answers without opening the file: nothing matched (0), and
"everything matched", which is what a too-generic phrase returns.

PROGRESS PROTOCOL (stderr, parsed by services/auto_mask.py)
    `[sam3] phase=<starting|loading|masking>` then `[sam3] i/N …`
Phases exist because the per-image counter lies about the wait: importing torch
and loading a 3.4 GB checkpoint costs tens of seconds before image 1, and a bar
frozen at 0/N that long is indistinguishable from a hang.
"""
import json
import os
import sys

PHASES = ('starting', 'loading', 'masking')


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _phase(name):
    _log(f'[sam3] phase={name}')


def cancel_requested(cancel_file) -> bool:
    """The parent asks for a stop by CREATING that file. A file rather than a
    signal because this is a plain stdin/stdout child on Windows too."""
    return bool(cancel_file) and os.path.exists(cancel_file)


def _fail(message):
    print(json.dumps({'ok': False, 'error': message}))
    return 1


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        images = [p for p in (payload.get('images') or []) if p]
        prompt = (payload.get('prompt') or '').strip()
        out_dir = payload['out_dir']
        checkpoint = payload['checkpoint']
        threshold = float(payload.get('threshold') or 0.5)
        device = payload.get('device') or None
        cancel_file = payload.get('cancel_file') or None
    except Exception as e:                              # noqa: BLE001
        return _fail(f'payload: {e}')
    if not prompt:
        return _fail('a phrase naming the region is required')
    if not os.path.isfile(checkpoint):
        return _fail(f'checkpoint not found: {checkpoint}')
    os.makedirs(out_dir, exist_ok=True)

    # First line out of the child: everything above is microseconds, everything
    # below is tens of seconds.
    _phase('starting')
    try:
        import numpy as np
        import torch
        from PIL import Image
    except Exception as e:                              # noqa: BLE001
        return _fail(f'import: {type(e).__name__}: {e}')
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as e:                              # noqa: BLE001
        return _fail(
            f'the sam3 package is not importable in this interpreter '
            f'({type(e).__name__}: {e}). Rebuild the automask environment from '
            f'Setup.')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _phase('loading')
    try:
        # load_from_HF=False is the whole borrowing contract: the weights are the
        # ones already on disk, and this process never reaches the network.
        model = build_sam3_image_model(
            checkpoint_path=checkpoint, load_from_HF=False, device=device)
        # The processor applies `confidence_threshold` itself and returns masks
        # already filtered and resized to the ORIGINAL image — so nothing here
        # re-thresholds, and nothing rescales.
        processor = Sam3Processor(model, device=device,
                                  confidence_threshold=threshold)
    except Exception as e:                              # noqa: BLE001
        return _fail(f'could not load SAM 3: {type(e).__name__}: {e}')

    _phase('masking')
    results, written, cancelled = {}, 0, False
    total = len(images)
    for i, path in enumerate(images, 1):
        if cancel_requested(cancel_file):
            cancelled = True
            break
        try:
            with Image.open(path) as im:
                image = im.convert('RGB')
                width, height = image.size
                # bf16 autocast is REQUIRED, not an optimisation: the model's own
                # predictors run under it (sam3_base_predictor, multiplex_base)
                # and the image processor does not, so calling it bare dies on
                # "mat1 and mat2 must have the same dtype, but got BFloat16 and
                # Float" the moment the first linear layer is reached.
                with _autocast(torch, device):
                    state = processor.set_image(image)
                    state = processor.set_text_prompt(prompt, state)
            masks = _as_numpy(state)
            if masks is None or not len(masks):
                results[path] = {'state': 'no_match', 'coverage': 0.0}
                _log(f'[sam3] {i}/{total} no_match')
                continue
            # ONE mask covering every match. A caller that wanted the instances
            # apart would have to compose them again, and none does yet.
            union = np.any(masks, axis=0).astype('uint8') * 255
            coverage = float((union > 127).sum()) / float(max(1, width * height))
            name = os.path.splitext(os.path.basename(path))[0] + '.png'
            Image.fromarray(union, mode='L').save(os.path.join(out_dir, name), 'PNG')
            results[path] = {'state': 'ok', 'coverage': round(coverage, 4),
                             'instances': int(len(masks))}
            written += 1
            _log(f'[sam3] {i}/{total} ok coverage={coverage:.3f}')
        except Exception as e:                          # noqa: BLE001
            results[path] = {'state': f'error: {type(e).__name__}: {e}',
                             'coverage': 0.0}
            _log(f'[sam3] {i}/{total} {results[path]["state"]}')

    print(json.dumps({'ok': True, 'written': written, 'cancelled': cancelled,
                      'results': results}))
    return 0


def _autocast(torch, device):
    """The bf16 context the model was built for, or a no-op on CPU (where bf16
    autocast is either unsupported or catastrophically slow)."""
    import contextlib
    if str(device).startswith('cuda'):
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _as_numpy(state):
    """`state['masks']` as a [N, H, W] boolean array, or None.

    The processor hands back `[N, 1, H, W]` booleans at the ORIGINAL resolution,
    already filtered by its own confidence threshold — so this only drops the
    channel axis. Written defensively anyway: the state keys are the one thing a
    package update can rename, and a mask service that dies on a KeyError is
    worse than one that reports finding nothing."""
    import numpy as np

    masks = state.get('masks') if isinstance(state, dict) else getattr(state, 'masks', None)
    if masks is None:
        return None
    masks = _to_np(masks)
    if masks is None or masks.size == 0:
        return None
    while masks.ndim > 3:
        masks = masks.squeeze(1) if masks.shape[1] == 1 else masks.reshape(
            -1, *masks.shape[-2:])
    if masks.ndim == 2:
        masks = masks[None, ...]
    return masks > 0.5


def _to_np(value):
    import numpy as np
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach().to('cpu')
    try:
        return np.asarray(value, dtype='float32')
    except Exception:                                   # noqa: BLE001
        return None


if __name__ == '__main__':
    sys.exit(main())
