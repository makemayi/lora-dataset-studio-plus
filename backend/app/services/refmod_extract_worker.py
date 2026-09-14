"""MiniMax-H3 RefMod extraction worker.

Executed by the COMFYUI interpreter (it owns torch + the comfy-adjacent deps),
never by the LDS backend process. Reads one JSON manifest on stdin:

    {"node_dir":   ".../custom_nodes/ComfyUI-MiniMaxH3Mod",
     "images":     ["abs/path.png", ...],
     "vae":        ".../models/vae/<h3 video vae>.safetensors",
     "output_dir": ".../models/refmods",
     "name":       "minimaxh3_<dataset>_v1_refmod",
     "description": "..."}

and writes the RefMod, then prints ONE json line on stdout:
    {"ok": true, "tokens": N, "frames": N, "path": "...", "mb": 1.5}

Any failure prints {"ok": false, "error": "..."} and exits 1. The encode flow
mirrors the node pack's own extract_mod.py (encode mode, 1024px short edge,
fp16 latents, max_tokens budget from the manifest) — one implementation per runtime, by design:
this file runs where torch lives, extract_mod.py is the pack's CLI.
"""
from __future__ import annotations

import json
import os
import sys

import torch


def _fail(message: str) -> None:
    print(json.dumps({"ok": False, "error": message}, ensure_ascii=True))
    sys.exit(1)


def _load_face_mask(path: str, resolution: int, canvas):
    """LDS face mask PNG (face=BLACK, background=WHITE) -> RefMod keep-mask
    [1, th, tw] (face=1): the two conventions are inverted, so this flips the
    values, then mirrors the image chain's resize (short edge -> canvas
    center-crop) so mask and encoded pixels share one geometry."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L")).astype("float32") / 255.0
    m = (1.0 - torch.from_numpy(arr)).clamp(0.0, 1.0)   # invert: face becomes 1
    h, w = m.shape
    scale = min(1.0, resolution / min(h, w))
    if canvas is not None:
        tw, th = canvas
    else:
        tw = max(32, round(w * scale / 32) * 32)
        th = max(32, round(h * scale / 32) * 32)
    mp = m.unsqueeze(0).unsqueeze(1)                    # [1, 1, h, w]
    import comfy.utils
    mp = comfy.utils.common_upscale(mp, tw, th, "bilinear", "center")
    return mp.squeeze(1).clamp(0.0, 1.0)                # [1, th, tw]


def main() -> None:
    try:
        manifest = json.loads(sys.stdin.read() or "{}")
        node_dir = manifest["node_dir"]
        images = [p for p in manifest.get("images", []) if os.path.isfile(p)]
        vae_path = manifest["vae"]
        output_dir = manifest["output_dir"]
        name = manifest["name"]
        description = manifest.get("description", "")
    except Exception as e:  # noqa: BLE001 - the only exit path for a bad manifest
        _fail(f"bad manifest: {e}")
        return

    if not images:
        _fail("no readable images in the manifest")
        return
    for required in (node_dir, vae_path, output_dir):
        if not required or not os.path.isdir(required) and not os.path.isfile(required):
            _fail(f"missing path in manifest: {required}")
            return

    # Order matters: node_dir must WIN over the ComfyUI root — both ship a
    # ``nodes`` module, and ComfyUI's own one has no _mask_latent (measured:
    # the mask pass silently degraded to unmasked with the root first).
    comfy_root = os.path.abspath(os.path.join(node_dir, "..", ".."))
    if comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)
    if node_dir not in sys.path:
        sys.path.insert(0, node_dir)

    try:
        import torch

        from common import load_image_file, resize_ref as _resize_ref, ensure_min_size
        from core import H3RefMod, fit_token_budget, pool_latent, optimize_latent, aspect_grid

        import comfy.model_management
        import comfy.sd
        import comfy.utils

        # Face-mask suppression reuses the pack's own math (zero drift).
        # nodes.py uses PACKAGE-RELATIVE imports (from .common, from .core...),
        # so it must be loaded as a package — a bare ``from nodes import ...``
        # dies with 'attempted relative import' (measured). If the load fails
        # (ComfyUI API moved), masks degrade to unmasked.
        _pack_mask_latent = None
        try:
            import importlib.util
            _pkg_name = "lds_h3mod_pack"
            _spec = importlib.util.spec_from_file_location(
                _pkg_name, os.path.join(node_dir, "__init__.py"),
                submodule_search_locations=[node_dir])
            _pkg = importlib.util.module_from_spec(_spec)
            sys.modules[_pkg_name] = _pkg
            _spec.loader.exec_module(_pkg)
            _pack_mask_latent = importlib.import_module(
                f"{_pkg_name}.nodes")._mask_latent
        except Exception as mask_import_error:  # noqa: BLE001
            print(f"[extract] note: mask support unavailable ({mask_import_error})")
    except Exception as e:  # noqa: BLE001
        _fail(f"cannot import the RefMod node pack or ComfyUI: {e}")
        return

    try:
        device = comfy.model_management.get_torch_device()
        sd, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata, device=device)
        vae.throw_exception_if_invalid()

        resolution = 1024
        # POOLED mode (2026-09-13): each frame's full latent is average-pooled
        # to a pool×pool latent grid (+ optional gradient refinement toward the
        # full encode). Cuts the reference tokens the DiT attends per block by
        # ~4x (20 frames: 20480 → ~5-8k), which is the workflow speed the user
        # was losing; identity stays because refine pulls the pooled latent
        # back toward the full encode (pack default pool 32 = its identity
        # floor — below that pooling averages away the face).
        pool = int(manifest.get('pool') or 48)
        refine = int(manifest.get('refine') or 1500)
        max_tokens = int(manifest.get('max_tokens') or 8192)
        masks = manifest.get('masks') or []
        background_retention = float(manifest.get('background_retention') or 0.0)
        first = load_image_file(images[0], max_edge=resolution * 2)
        h, w = first.shape[1], first.shape[2]
        # 1024px LONG side (2026-09-12, user call): short-side-1024 blew portrait
        # refs up to ~1376 tokens/frame, which pushed the 20-picture pick over
        # the token budget so fit_token_budget dropped 4 of them; a long-side
        # cap keeps every pick under budget AND cuts the ref tokens the DiT
        # attends (the generation-time speed cost the user was feeling).
        scale = min(1.0, resolution / max(h, w))
        canvas = (max(32, round(w * scale / 32) * 32),
                  max(32, round(h * scale / 32) * 32))

        frames, shapes = [], []
        masked_n = 0
        for i, path in enumerate(images):
            src = ensure_min_size(_resize_ref(load_image_file(path, max_edge=resolution * 2),
                                              resolution, canvas))
            with torch.no_grad():
                z = vae.encode(src.to(device)).float()
            mask_path = masks[i] if i < len(masks) else None
            if mask_path and os.path.isfile(mask_path) and _pack_mask_latent is not None:
                # Suppress everything outside the face toward a blurred copy of
                # itself (background_retention=0 → clothing/skyline structure is
                # erased at the latent, the documented "keep identity only" lever).
                z = _pack_mask_latent(z, _load_face_mask(mask_path, resolution, canvas),
                                      background_retention, name)
                masked_n += 1
            elif mask_path:
                print(f"[extract] note: no mask support — {os.path.basename(path)} "
                      f"encoded unmasked")
            if pool >= 16:
                gh, gw = aspect_grid(pool, pool, src.shape[1] / src.shape[2])
                zp = pool_latent(z, 1, gh, gw)
                if refine > 0:
                    zp = optimize_latent(zp, z, steps=refine, device=device)
                z = zp
            frames.append(z.float().cpu().to(torch.float16))
            shapes.append(f"{z.shape[2]}x{z.shape[3]}x{z.shape[4]}")

        latent = torch.cat(frames, dim=2)
        latent = fit_token_budget(latent, max_tokens, name)
        total_t = latent.shape[2]
        px_w, px_h = latent.shape[4] * 16, latent.shape[3] * 16
        mod = H3RefMod(
            name=name, kind="video" if total_t > 1 else "image", latent=latent,
            latent_h=latent.shape[3], latent_w=latent.shape[4], latent_t=total_t,
            mode="training" if pool >= 16 else "encode",
            source="stack" if len(frames) > 1 else "image",
            source_shape=" +".join(shapes),
            pool=(f"pooled {pool}x{pool} refine {refine}" if pool >= 16 else
                  f"full-res {px_w}x{px_h}px (short-edge cap {resolution}px)"),
            optimize_steps=0, tags=[f"{len(frames)} img"],
            description=description, concept_type="identity",
        )
        mod.path = os.path.join(output_dir, name)
        saved = mod.save(mod.path)
        mb = latent.numel() * latent.element_size() / 1024 / 1024
        print(json.dumps({"ok": True, "tokens": mod.token_count, "frames": total_t,
                          "path": saved, "mb": round(mb, 2), "masked": masked_n},
                         ensure_ascii=True))
    except Exception as e:  # noqa: BLE001
        _fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 - pre-3.7 or exotic stdout; JSON stays ASCII
        pass
    main()
