"""Face boxes for the Bank's promote-to-dataset framings, via InsightFace.

stdin  : {"images": [<absolute paths>], "models_root": "<dir>|null"}
stdout : ONE JSON line {"ok": true, "results": {<path>: {
             "nx0","ny0","nx1","ny1",   -- the BIGGEST face, normalised against
             "n_faces",                 -- the EXIF-oriented picture
             "yaw"}}}
         or {"ok": false, "error": "..."} — everything human goes to stderr.

WHY A SECOND DETECTOR PROCESS WHEN THE FACES PASS ALREADY RUNS ONE. The subject
pass stores a VERDICT and a cluster id, never a box — and boxes are bulky,
rot-sensitive data the grid has no use for. Framing crops are the first reader
that needs coordinates, and they need them at PROMOTION time, on the exact bytes
being promoted. So this file shares the subject pass's model (antelopev2), its
padding rescue, and its validated-read discipline, and stores nothing: the crop
geometry is computed server-side from these numbers, per run.

The boxes are NORMALISED against the EXIF-ORIENTED image on purpose — the same
orientation `analysis_image_path` serves and the same one the crop will be cut
from, so a window computed from these numbers lands on the right pixels.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bank_image_guard import read_validated_bank_image  # noqa: E402
from face_score_infer import _repair_nested_antelopev2  # noqa: E402


def _log(m):
    print(m, file=sys.stderr, flush=True)


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        images = [str(p) for p in (payload.get("images") or [])]
        models_root = payload.get("models_root") or None
        if not images:
            print(json.dumps({"ok": False, "error": "no images were given"}))
            return 1
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return 1

    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        print(json.dumps({"ok": False,
                          "error": f"insightface unavailable: {type(e).__name__}: {e}"}))
        return 1

    try:
        _repair_nested_antelopev2(models_root)
        import onnxruntime
        avail = onnxruntime.get_available_providers()
        device = 'cuda' if 'CUDAExecutionProvider' in avail else 'cpu'
        used_gpu = device == 'cuda' and 'CUDAExecutionProvider' in avail
        kwargs = {'name': 'antelopev2', 'providers': (
            ['CUDAExecutionProvider', 'CPUExecutionProvider'] if used_gpu
            else ['CPUExecutionProvider'])}
        if models_root:
            kwargs['root'] = models_root
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=0 if used_gpu else -1, det_size=(640, 640))
        import cv2
        import numpy as np
        _log(f"[face-box] {len(images)} image(s), used_gpu={used_gpu}")
    except Exception as e:  # noqa: BLE001 — must exit as clean JSON, not a traceback
        print(json.dumps({"ok": False,
                          "error": f"model load failed: {type(e).__name__}: {e}"}))
        return 1

    def biggest(faces):
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) \
            if faces else None

    results = {}
    for i, p in enumerate(images, 1):
        try:
            raw = read_validated_bank_image(p)
            img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                results[p] = {"error": "unreadable"}
                continue
            h, w = img.shape[:2]
            faces = app.get(img)
            f = biggest(faces)
            if f is None:   # padding rescue: SCRFD misses full-frame closeups
                pad = int(0.25 * max(h, w))
                f = biggest(app.get(cv2.copyMakeBorder(
                    img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))))
            if f is None:
                results[p] = {"n_faces": 0}
                continue
            x0, y0, x1, y1 = [float(v) for v in f.bbox]
            pose = getattr(f, 'pose', None)
            yaw = (float(pose[1]) if pose is not None else None)
            results[p] = {
                "nx0": round(max(0.0, x0) / w, 4), "ny0": round(max(0.0, y0) / h, 4),
                "nx1": round(min(w, x1) / w, 4), "ny1": round(min(h, y1) / h, 4),
                "n_faces": int(len(faces)),
                "yaw": round(yaw, 1) if yaw is not None and yaw == yaw else None,
            }
        except Exception as e:  # noqa: BLE001 — one file must not sink the batch
            results[p] = {"error": f"{type(e).__name__}: {e}"}
            _log(f"[face-box] {i}/{len(images)} ERROR {e}")
        if i % 25 == 0 or i == len(images):
            _log(f"[face-box] {i}/{len(images)}")

    print(json.dumps({"ok": True, "results": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
