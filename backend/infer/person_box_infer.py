"""Person boxes for the Bank's Crop-to-person pass, via Grounding DINO tiny.

stdin  : {"images": [<absolute paths>], "models_root": "<dir>|null"}
stdout : ONE JSON line {"ok": true, "results": {<path>: {"boxes": [[x1,y1,x2,y2], ...]}}}
         or {"ok": false, "error": "..."} — everything human goes to stderr.

Runs in the Bank scoring interpreter (the only one guaranteed to carry
transformers + torch), one MODEL LOAD for a whole batch: a bank holds thousands
of pictures, and reloading a detector per picture would multiply the pass by the
load time. Boxes come back RAW and UNMERGED — merging, picking the largest and
the aspect-ratio window are geometry, and geometry lives server-side where it
can be tested (`services/person_crop.py`).

The prompt reads like two words for one thing on purpose: zero-shot detectors
recall better when the prompt names the class twice ("person. human body."),
which is what the operator's own batch script measured and shipped with.
"""
from __future__ import annotations

import json
import os
import sys

# The model is ALWAYS local (cache_dir = data/models/grounding_dino, downloaded
# once at first use), so this MUST run offline. huggingface_hub still sends a
# HEAD request per file to verify it even when the file is cached, and a machine
# that cannot reach huggingface.co answers that HEAD with a connection timeout
# (WinError 10060) — the subprocess retries forever and the pass hangs on
# "loading the person detector" (measured: 90 s with no output, then 11.6 s when
# HF_HUB_OFFLINE=1 is set, model loads in 2.1 s). Offline is the correct mode for
# a model that is provably on disk, and a genuinely missing file fails loudly.
os.environ.setdefault('HF_HUB_OFFLINE', '1')
# huggingface_hub may already be imported by a host that ran this before us, so
# also force the transformers/diffusers flags that read the env at import time.
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')

import torch

torch.set_num_threads(os.cpu_count() or 1)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "person. human body."
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.30
DETECT_SIZE = {"shortest_edge": 480, "longest_edge": 800}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _load_model(models_root):
    """Local-first model load for Grounding DINO.

    The policy, asked for directly: load from the LOCAL cache first and NEVER
    touch the network for a model that is already on disk (a machine that cannot
    reach huggingface.co answers every HEAD request with a connection timeout —
    WinError 10060 — and the default from_pretrained retries it forever, hanging
    the pass on "loading the person detector"). Only a model that is genuinely
    ABSENT locally falls back to the network, under a short download timeout, and
    a network failure there is a clear error — never a silent unbounded retry.
    """
    from transformers import (AutoModelForZeroShotObjectDetection,
                              AutoProcessor)

    def _local_first(fn):
        return _from_pretrained_local_first(
            fn, MODEL_ID, models_root,
            name=fn.__module__.split('.')[-1] if hasattr(fn, '__module__') else fn)

    return (_local_first(AutoProcessor.from_pretrained),
            _local_first(AutoModelForZeroShotObjectDetection.from_pretrained))


def _from_pretrained_local_first(from_pretrained, model_id, models_root, *, name):
    """One `from_pretrained` call, local-first. Returns the loaded object, or a
    clear RuntimeError when neither the local cache nor the network has it.
    """
    # Pass 1 — local only. A cached model NEVER touches the network.
    try:
        return from_pretrained(model_id, cache_dir=models_root,
                               local_files_only=True)
    except (OSError, EnvironmentError, ValueError, RuntimeError):
        pass  # not here yet — try the network, bounded

    # Pass 2 — the model is not local; try the network, bounded. Offline flags
    # off, a short download timeout on, and a failure is reported, not retried.
    _log(f"[person-crop] {model_id} not in the local cache ({models_root or '(default)'}); "
         "trying a bounded download")
    for _flag in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
        os.environ.pop(_flag, None)
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '30')
    try:
        return from_pretrained(model_id, cache_dir=models_root)
    except Exception as net_err:  # noqa: BLE001 — reported, never retried below
        raise RuntimeError(
            f'model {model_id!r} ({name}) is not in the local cache '
            f'{models_root or "(default)"} and could not be downloaded: '
            f'{type(net_err).__name__}: {net_err}') from net_err


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        images = [str(p) for p in (payload.get("images") or [])]
        models_root = payload.get("models_root") or None
        if not images:
            print(json.dumps({"ok": False, "error": "no images were given"}))
            return 1

        from PIL import Image, ImageOps
        from transformers import (AutoModelForZeroShotObjectDetection,
                                  AutoProcessor)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _log(f"[person-crop] {len(images)} image(s), device={device}")
        t0 = __import__("time").time()
        processor, model = _load_model(models_root)
        model = model.to(device).eval()
        _log(f"[person-crop] model loaded in {__import__('time').time() - t0:.1f}s")

        results = {}
        for i, path in enumerate(images, 1):
            try:
                with Image.open(path) as im:
                    img = ImageOps.exif_transpose(im).convert("RGB")
                w, h = img.size
                inputs = processor(images=img, text=PROMPT, size=DETECT_SIZE,
                                   return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                res = processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"], threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD, target_sizes=[(h, w)])[0]
                boxes = [[round(float(v), 1) for v in b.tolist()]
                         for b in res["boxes"]]
                results[path] = {"boxes": boxes}
            except Exception as e:  # noqa: BLE001 — one file must not sink the batch
                results[path] = {"boxes": [], "error": f"{type(e).__name__}: {e}"}
                _log(f"[person-crop] {i}/{len(images)} ERROR {e}")
            if i % 25 == 0 or i == len(images):
                _log(f"[person-crop] {i}/{len(images)}")

        print(json.dumps({"ok": True, "results": results}))
        return 0
    except Exception as e:  # noqa: BLE001 — the caller reads this and reports it
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
