"""Resolve the Z-Image text encoder and VAE against what is ACTUALLY on disk.

WHY THIS EXISTS
---------------
``workflows/ZImage_bigLove_ZT3_optimal.json`` was captured on a developer's own
ComfyUI, so its loader nodes carry that machine's filenames verbatim:

    node 2 (CLIPLoader) : "Z image\\qwen_3_4b.safetensors"
    node 3 (VAELoader)  : "z ae.safetensors"

Nothing rewrote those before enqueue, so the app effectively demanded that every
user reproduce one person's folder layout — and the Studio preflight, honest but
one-way, only NAMED the file it wanted. Reported by bobba84 (GitHub #18), who had
to rename ``z_ae.safetensors`` (underscore) to ``z ae.safetensors`` (space) and
re-case ``Z Image/`` to ``Z image/`` to make the app work. That is backwards: the
app has real resolvers for Klein and Krea (``klein_edit_helper``,
``krea_edit_helper``) which scan every registered root and adapt. Z-Image had
none. This module is that resolver.

Two legitimate layouts must BOTH work, and they disagree on every axis:

  * ComfyUI's own Z-Image documentation puts the pieces at
    ``models/text_encoders/qwen_3_4b.safetensors`` and ``models/vae/ae.safetensors``
    (flat, no sub-folder, and the VAE is named ``ae``).
  * This app's shipped workflow (and its README) describes
    ``models/text_encoders/Z image/qwen_3_4b.safetensors`` and
    ``models/vae/z ae.safetensors``.

Neither is "the" layout, so matching is by NORMALISED PHRASE, not by string
equality: word characters are split on any separator (space, ``_``, ``-``,
folder separators) and lower-cased, so ``z ae`` / ``z_ae`` / ``z-ae`` / ``Z_AE``
all become the phrase ``z ae``, and ``Z Image\\qwen_3_4b`` becomes
``z image qwen 3 4b``. Matching is then whole-word phrase containment, NOT bare
substring: ``xyz_ae.safetensors`` normalises to ``xyz ae`` and does NOT match the
phrase ``z ae`` — a plain ``'zae' in name`` test would have accepted it.

NARROWNESS IS THE POINT. models/text_encoders/ on a shared ComfyUI holds several
families' encoders and a wrong one is WORSE than a missing one: missing produces
an actionable message, wrong dies at sample time on a shape mismatch or renders
noise. The phrases below deliberately do not match ``qwen3vl_4b_*`` (Krea 2) nor
``qwen_3_8b_*`` (Klein), which sit in the very same folder.

THE ONE HONEST COMPROMISE (read before "simplifying" it)
--------------------------------------------------------
The last VAE tier accepts a file literally named ``ae.safetensors``. That IS the
name ComfyUI's own Z-Image docs tell people to save, so refusing it would break
the single most common fresh install. It is also, verbatim, the FLUX.1 VAE
filename. On a ComfyUI shared with FLUX and carrying no z-qualified VAE, this
tier can therefore hand the sampler the wrong autoencoder. It is ranked LAST (any
z-qualified candidate anywhere wins over it), it is logged when it fires, and the
escape hatch is the ``zimage.vae`` setting. Do not promote it, and do not delete
it either.

EXPLICIT CHOICE ALWAYS WINS. ``selected`` (a caller-supplied name) then the
``zimage.vae`` / ``zimage.text_encoder`` settings are honoured before any scan,
and honoured AS-IS when the named file isn't on disk — if a user names a file,
the failure must be about THEIR file, not silently about another one (same rule
as ``klein_edit_helper.resolve_klein_unet``).

DETERMINISTIC TIE-BREAK. Several candidates can match on a real install (a
``z_ae.safetensors`` at the vae root and another under ``vae/z image/``). The
winner is NOT "whatever os.walk yielded first" — the candidates are sorted by an
explicit key, in this order:

    1. tier of the phrase that matched (canonical/z-qualified before the bare
       ``ae`` fall-back);
    2. exact canonical relative name (case-insensitive) first;
    3. exact canonical BASENAME first;
    4. sitting under a z-image-named sub-folder first;
    5. ComfyUI's own root priority (extra_model_paths ``is_default`` roots, then
       ``<base>/models``, then the rest — i.e. the order comfy_model_paths yields);
    6. shallowest path;
    7. lower-cased relative name, lexicographically.

Only LOADABLE files are ever auto-picked (``comfy_model_paths.is_loadable_model``):
a ``.gguf`` is listed by this app but core ComfyUI's CLIPLoader/VAELoader cannot
open one, and a resolver that picked by name alone has already shipped that bug
once (see comfy_model_paths.LOADABLE_MODEL_EXTENSIONS).
"""
from __future__ import annotations

import logging
import os
import re

from .. import config as cfg
from . import comfy_model_paths

logger = logging.getLogger(__name__)

# The refs the shipped workflow carries — kept as the canonical target so an
# install that DOES match the app's documented layout resolves at tier 0.
CANONICAL_VAE = 'z ae.safetensors'
CANONICAL_TEXT_ENCODER = os.path.join('Z image', 'qwen_3_4b.safetensors')

# Config keys that PIN a file (blank = auto-resolve), mirroring `krea.base_model`.
CFG_KEY_VAE = 'zimage.vae'
CFG_KEY_TEXT_ENCODER = 'zimage.text_encoder'

_EXT_RE = re.compile(r'\.(safetensors|sft|gguf|ckpt|pt|pth|bin)$', re.IGNORECASE)
_WORD_RE = re.compile(r'[a-z0-9]+')

# Match tiers, most specific first. Each tier is ``(mode, phrases)``; the tier INDEX
# is the first sort key, so a lower tier always beats a higher one no matter where on
# disk it lives. Two modes, because the two questions are genuinely different:
#   'path'  — the phrase appears as whole words anywhere in the ROOT-RELATIVE name
#             (folder included), so `vae/z image/ae.safetensors` matches 'z image ae'.
#   'exact' — the BASENAME is exactly that phrase and nothing else. Reserved for names
#             so generic that containment would be a trap.
_VAE_TIERS = (
    # 0 — unambiguously the Z-Image autoencoder, whatever the separator/case.
    ('path', ('z ae', 'zae', 'z image ae', 'zimage ae', 'z image vae', 'zimage vae')),
    # 1 — the FLUX-colliding bare name from ComfyUI's own Z-Image docs. See the module
    #     docstring: accepted on purpose, ranked last on purpose, and 'exact' on purpose
    #     — a 'path' match here would swallow `xyz_ae.safetensors` and every other
    #     unrelated autoencoder whose name happens to end in "ae".
    ('exact', ('ae',)),
)
_TEXT_ENCODER_TIERS = (
    # Qwen3-4B, the Z-Image text encoder. Never qwen3vl_4b (Krea 2) and never
    # qwen_3_8b (Klein) — both normalise to phrases none of these match.
    ('path', ('qwen 3 4b', 'qwen3 4b', 'qwen 34b', 'qwen34b')),
)

# Folder names that mark a Z-Image-dedicated sub-folder (sort key 4).
_ZIMAGE_FOLDER_PHRASES = ('z image', 'zimage')


def _phrase(text: str) -> str:
    """``'Z Image\\qwen_3_4b.safetensors'`` -> ``' z image qwen 3 4b '``.

    Extension dropped, every run of word characters kept, everything else (space,
    ``_``, ``-``, ``.``, folder separators) collapsed to a single space, and the
    result padded with spaces so a caller can test WHOLE-PHRASE containment with a
    plain ``in``. The padding is what makes ``xyz_ae`` fail the ``z ae`` test."""
    stem = _EXT_RE.sub('', str(text or ''))
    return ' ' + ' '.join(_WORD_RE.findall(stem.lower())) + ' '


def _matches(phrase_key: str, phrase: str) -> bool:
    return f' {phrase} ' in phrase_key


def _tier_of(rel_name: str, tiers) -> int | None:
    """Index of the first tier whose phrase the (path-aware) name matches, or None.

    A ``'path'`` tier is evaluated against the FULL relative name, so a bare
    ``ae.safetensors`` dropped inside ``vae/z image/`` matches the z-qualified tier
    through its folder — which is exactly the intent, and why it outranks a bare
    root-level ``ae``."""
    rel = rel_name.replace('\\', os.sep)
    path_key = _phrase(rel)
    base_key = _phrase(os.path.basename(rel))
    for i, (mode, phrases) in enumerate(tiers):
        if mode == 'exact':
            if any(base_key == f' {p} ' for p in phrases):
                return i
        elif any(_matches(path_key, p) for p in phrases):
            return i
    return None


def _in_zimage_folder(rel_name: str) -> bool:
    head = os.path.dirname(rel_name.replace('\\', os.sep))
    if not head:
        return False
    key = _phrase(head)
    return any(_matches(key, p) for p in _ZIMAGE_FOLDER_PHRASES)


def _pinned(selected, cfg_key):
    """The explicit choice for this asset: the caller's `selected`, else the config
    key (blank/absent = none). Normalised to the OS separator so a value typed with
    forward slashes still resolves."""
    for raw in (selected, cfg.get(cfg_key)):
        value = str(raw or '').strip()
        if value:
            return value.replace('/', os.sep).replace('\\', os.sep)
    return None


def _listing(folder_type):
    """``[(rank, rel_name, abs_path)]`` for every model file across the folder type's
    search roots, ``rank`` being ComfyUI's own root priority (0 = highest). Wraps
    comfy_model_paths.list_models, which already dedups by rel name and walks
    sub-folders — the two things Klein's flat ``os.listdir`` helper does not do, and
    the reason a text encoder inside ``text_encoders/Z image/`` was invisible."""
    out, rank_of = [], {}
    for i, root in enumerate(comfy_model_paths.search_roots(folder_type)):
        rank_of[os.path.normpath(root)] = i
    for rel, ab in comfy_model_paths.list_models(folder_type):
        root = os.path.normpath(ab[: len(ab) - len(rel)].rstrip(os.sep)) if rel else ''
        out.append((rank_of.get(root, len(rank_of)), rel, ab))
    return out


def _resolve(folder_type, canonical, tiers, selected=None, cfg_key=None):
    """The shared resolver. Returns the value a loader node should carry (relative to
    its root, with the REAL on-disk casing and sub-folder), or None when nothing
    matches. See the module docstring for the tie-break contract."""
    pinned = _pinned(selected, cfg_key)
    listing = _listing(folder_type)
    if pinned:
        for _rank, rel, _ab in listing:
            if rel.lower() == pinned.lower() or os.path.basename(rel).lower() == pinned.lower():
                return rel
        # Named but absent -> hand it back untouched: the resulting preflight/ComfyUI
        # failure must be about the file the user asked for, never about another one
        # this resolver silently substituted.
        return pinned

    canon_lower = canonical.lower()
    canon_base = os.path.basename(canonical).lower()
    scored = []
    for rank, rel, _ab in listing:
        if not comfy_model_paths.is_loadable_model(rel):
            continue          # .gguf is listed but no core loader can open it
        tier = _tier_of(rel, tiers)
        if tier is None:
            continue
        rel_l = rel.lower()
        scored.append(((tier,
                        0 if rel_l == canon_lower else 1,
                        0 if os.path.basename(rel_l) == canon_base else 1,
                        0 if _in_zimage_folder(rel) else 1,
                        rank,
                        rel.count(os.sep),
                        rel_l), rel))
    if not scored:
        return None
    scored.sort(key=lambda e: e[0])
    key, best = scored[0]
    if key[0] >= len(tiers) - 1 and len(tiers) > 1:
        logger.warning(
            "Z-Image %s resolved to %r through the ambiguous fall-back name — if your "
            "images look wrong, pin the right file with the %s setting.",
            folder_type, best, cfg_key)
    return best


def resolve_zimage_vae(selected=None):
    """``vae_name`` for node 3 of the Z-Image workflow, or None when no plausible
    Z-Image autoencoder is on disk."""
    return _resolve('vae', CANONICAL_VAE, _VAE_TIERS,
                    selected=selected, cfg_key=CFG_KEY_VAE)


def resolve_zimage_text_encoder(selected=None):
    """``clip_name`` for node 2 of the Z-Image workflow, or None when no Qwen3-4B
    text encoder is on disk. Never returns Krea's qwen3vl_4b nor Klein's qwen_3_8b."""
    return _resolve('text_encoders', CANONICAL_TEXT_ENCODER, _TEXT_ENCODER_TIERS,
                    selected=selected, cfg_key=CFG_KEY_TEXT_ENCODER)


def _roots_hint(folder_type):
    """The folder names (not machine paths — this text is user-facing and must stay
    paste-safe) the resolver scanned, so a 'not found' message can say WHERE it
    looked without leaking anyone's disk layout."""
    n = len(comfy_model_paths.search_roots(folder_type))
    extra = max(0, n - 1)
    return (f'models/{folder_type}/'
            + (f' (+{extra} extra_model_paths.yaml root(s))' if extra else ''))


def vae_missing_hint() -> str:
    return ('Searched every models/vae root (and sub-folders) for '
            'z_ae / z ae / z-ae / ae.safetensors, any capitalisation — scanned '
            + _roots_hint('vae'))


def text_encoder_missing_hint() -> str:
    return ('Searched every models/text_encoders root (and sub-folders) for a '
            'Qwen3-4B encoder (qwen_3_4b / qwen3_4b, any capitalisation, in any '
            'sub-folder) — scanned ' + _roots_hint('text_encoders'))
