"""Face similarity scorer — InsightFace antelopev2, lance dans un interprete DEDIE
(insightface y est installe, PAS dans le venv Flask). CPU force (provider CPU + ctx_id=-1)
-> pas de GPU, ne touche pas ComfyUI.
Protocole stdin: {"refs": [path, ...], "images": [paths], "models_root": path|null} -> stdout
UNE ligne JSON {"ref_ok": bool, "results": {path: {state, sim?, det, bbox_frac, yaw}}}.
`refs` est une liste non-vide (le primaire est refs[0] par convention, mais rien dans
l'algo ne privilegie sa position) : chaque candidat est compare a CHAQUE ref utilisable,
`sim` est le MAX — « ressemble a AU MOINS une des photos de confiance ».
Logs -> stderr.
Gating 3-etats + padding rescue (valide empiriquement sur test3).
YAW_MAX porte a 70° (2026-08-15): a 40° un profil 3/4 etait rejete alors qu'antelopev2
extrait encore une embedding discriminante jusqu'a ~70° (la valeur absolue baisse, mais
« ressemble a la ref » reste separable). Les profils restent HORS auto-triage (le
front ne trie que face_state == 'scorable') mais recoivent un score affichable.

BBOX_MIN 0.06 -> 0.02 (2026-08-16, mesure dataset 4) : 0.06 classait « too_small » des
visages de 2-5% de la photo — des plans pleine-corps, exactement ce qu'un set LoRA
contient — et 22/25 de ces images donnaient en fait un score utile (sim 0.48-0.89, la
meme distribution que les scorable). En dessous de 0.02 le visage est vraiment trop
petit pour une embedding fiable. det_size RESTE a 640 : a 1024, SCRFD rate les gros
plans pleine cadre (mesure dataset 20 : 0 direct / 5), et la reference EST un gros
plan — ne pas y retoucher."""
from __future__ import annotations
import json, sys

DET_MIN, BBOX_MIN, YAW_MAX = 0.50, 0.02, 70.0


def lap_var_bgr(img):
    """Laplacian variance of a (possibly BGR) image — the SAME metric the bank's
    per-frame sharpness uses, so 'face is blurry' means the same thing as
    'frame is blurry' and the adaptive gate in video_frame_select can treat the
    two scales together. img may be 2D gray or 3D BGR; converted internally.
    A crop too small for the 3x3 kernel is unmeasurable, not an error."""
    import numpy as np
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.ndim != 2 or a.shape[0] < 3 or a.shape[1] < 3:
        return 0.0
    lap = (a[1:-1, :-2] + a[1:-1, 2:] + a[:-2, 1:-1]
           + a[2:, 1:-1] - 4.0 * a[1:-1, 1:-1])
    return float(lap.var())


def _log(m): print(m, file=sys.stderr, flush=True)


def _repair_nested_antelopev2(models_root=None):
    """L'antelopev2.zip d'insightface 0.7.3 contient un DOSSIER RACINE (contrairement
    a buffalo_l) : l'auto-extract pose les .onnx dans .../models/antelopev2/antelopev2/,
    or FaceAnalysis globbe NON-recursivement -> 0 modele charge -> AssertionError
    (`'detection' in self.models`). CHAQUE install fraiche en auto-download est
    touchee, et ca ne s'auto-repare jamais (le dossier externe existe, insightface
    ne re-telecharge pas). On aplatit une fois pour toutes ici."""
    import glob, os, shutil
    root = models_root or os.path.join(os.path.expanduser('~'), '.insightface')
    outer = os.path.join(root, 'models', 'antelopev2')
    inner = os.path.join(outer, 'antelopev2')
    if not os.path.isdir(inner) or glob.glob(os.path.join(outer, '*.onnx')):
        return
    moved = 0
    for f in glob.glob(os.path.join(inner, '*.onnx')):
        shutil.move(f, outer)
        moved += 1
    try:
        os.rmdir(inner)
    except OSError:
        pass  # reliquats (zip...) — sans consequence
    if moved:
        _log(f"[face] repaired nested antelopev2 layout ({moved} model(s) moved up)")


def main() -> int:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as e:
        print(json.dumps({"ref_ok": False, "results": {}, "error": f"bad json: {e}"})); return 1
    refs = [str(p) for p in (req.get("refs") or [])]
    images = [str(p) for p in (req.get("images") or [])]
    models_root = req.get("models_root") or None
    if not refs or not images:
        print(json.dumps({"ref_ok": False, "results": {}, "error": "missing refs/images"})); return 1

    import numpy as np, cv2
    from insightface.app import FaceAnalysis
    kwargs = {'name': 'antelopev2', 'providers': ['CPUExecutionProvider']}
    if models_root:
        kwargs['root'] = models_root

    def _load():
        app = FaceAnalysis(**kwargs)
        app.prepare(ctx_id=-1, det_size=(640, 640))
        return app

    # DEUX reparations, et la seconde est celle qui compte sur une install
    # NEUVE. Le telechargement d'antelopev2 se fait DANS FaceAnalysis(), donc au
    # premier lancement la reparation d'avant tourne sur un dossier vide, ne fait
    # rien, et l'auto-download recree exactement la disposition imbriquee que
    # cette fonction existe pour aplatir -> AssertionError. Reparer APRES l'echec
    # puis reessayer une fois est le seul ordre qui couvre le premier lancement.
    _repair_nested_antelopev2(models_root)
    try:
        app = _load()
    except Exception as first:
        _repair_nested_antelopev2(models_root)
        try:
            app = _load()
        except Exception as e:
            # Un crash de chargement (modeles absents/corrompus) doit sortir en
            # JSON propre — pas en traceback muet que le parent resume en « pas
            # de JSON ». On rapporte la PREMIERE erreur quand la seconde est le
            # meme symptome, sinon les deux.
            detail = f"{type(e).__name__}: {e}"
            if type(e) is not type(first) or str(e) != str(first):
                detail += f" (first attempt: {type(first).__name__}: {first})"
            print(json.dumps({"ref_ok": False, "results": {},
                              "error": f"model load failed: {detail}"}))
            return 1
    import onnxruntime as ort
    _log(f"[face] providers: {ort.get_available_providers()}")

    def biggest(faces):
        return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) if faces else None

    def detect(img):
        f = biggest(app.get(img))
        if f is None:  # padding rescue : SCRFD rate les gros plans plein cadre
            h, w = img.shape[:2]; pad = int(0.25 * max(h, w))
            f2 = biggest(app.get(cv2.copyMakeBorder(img, pad, pad, pad, pad,
                                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))))
            if f2 is not None:
                f2._padded = True
                return f2
        return f

    def analyze(path):
        img = cv2.imread(path)
        if img is None: return {"state": "unreadable"}
        h, w = img.shape[:2]
        f = detect(img)
        if f is None: return {"state": "no_face"}
        scale = 1.0
        if getattr(f, "_padded", False):
            pad = int(0.25 * max(h, w)); scale = (w + 2*pad) * (h + 2*pad) / (w * h)
        area = (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1])
        bbox_frac = float(area / (w * h) / scale)
        # Face-region sharpness: global sharpness cannot see a face that moved
        # during the exposure while the body stayed still. Crop the face bbox
        # (adjusting for the padding rescue) and measure IT.
        x0 = int(f.bbox[0]); y0 = int(f.bbox[1])
        x1 = int(f.bbox[2]); y1 = int(f.bbox[3])
        if getattr(f, "_padded", False):
            pad = int(0.25 * max(h, w))
            x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
            x1 = min(w, x1 - pad); y1 = min(h, y1 - pad)
        if x1 > x0 and y1 > y0:
            face_sharp = lap_var_bgr(img[y0:y1, x0:x1])
        else:
            face_sharp = 0.0
        det = float(f.det_score)
        yaw = float(f.pose[1]) if getattr(f, "pose", None) is not None else 0.0
        state = "scorable"
        if det < DET_MIN: state = "low_det"
        elif bbox_frac < BBOX_MIN: state = "too_small"
        elif abs(yaw) > YAW_MAX: state = "extreme_pose"
        return {"state": state, "det": round(det, 3), "bbox_frac": round(bbox_frac, 4),
                "yaw": round(yaw, 1), "face_sharp": round(face_sharp, 3),
                "_emb": f.normed_embedding}

    ref_embs = []
    for i, r in enumerate(refs, 1):
        ref_res = analyze(r)
        ref_emb = ref_res.pop("_emb", None)
        if ref_emb is None:
            _log(f"[face] ref {i}/{len(refs)} unusable: {ref_res.get('state')}")
            continue
        ref_embs.append(ref_emb)
    if not ref_embs:
        print(json.dumps({"ref_ok": False, "results": {},
                          "error": f"no usable face in any of {len(refs)} reference photo(s)"}))
        return 1

    results = {}
    for i, p in enumerate(images, 1):
        try:
            r = analyze(p); emb = r.pop("_emb", None)
            if r["state"] in ("scorable", "extreme_pose") and emb is not None:
                sims = [float(np.dot(ref_emb, emb)) for ref_emb in ref_embs]
                r["sim"] = round(max(sims), 4)
            results[p] = r
            _log(f"[face] {i}/{len(images)} {r['state']} sim={r.get('sim')}")
        except Exception as e:
            results[p] = {"state": "error", "error": str(e)}
            _log(f"[face] {i}/{len(images)} ERROR {e}")
    print(json.dumps({"ref_ok": True, "results": results}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
