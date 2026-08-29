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

import torch

torch.set_num_threads(os.cpu_count() or 1)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
PROMPT = "person. human body."
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.30
DETECT_SIZE = {"shortest_edge": 480, "longest_edge": 800}


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


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
        processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=models_root)
        model = (AutoModelForZeroShotObjectDetection.from_pretrained(
            MODEL_ID, cache_dir=models_root).to(device).eval())
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
