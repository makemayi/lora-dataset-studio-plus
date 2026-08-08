"""Captioning for a face dataset: the long/short caption passes, the
concept-omission guarantee, Caption Lab's per-candidate preview, and the
vocabulary instructions that steer them.

The backend (JoyCaption vs Qwen3-VL) is chosen in Settings and resolved per
call; both are imported inside the functions that use them, never at module
level, so importing this module costs nothing.

Split out of face_dataset_service.py (2026-08, Phase 3 of a multi-phase file
split) -- pure move, no behavior change. The caption-INSTRUCTION helpers it
reads but does not own (`caption_options`, `_with_caption_instructions`,
`_combined_caption_instructions`, `_cap_caption`, `_CAPTION_BACKENDS`,
`_CAPTION_VOCABULARIES`, `_VOCABULARY_INSTRUCTION`, `_CAPTION_INSTRUCTIONS_MAX`)
stay in face_dataset_service.py: they sit with the per-dataset caption OPTIONS
in another part of that file, and moving them would widen a pure-move diff.
"""
import json
import os
import re
import time

from PIL import Image

from ..extensions import db
from ..models import FaceDatasetImage
from .. import config as cfg
from . import dataset_activity
from .ollama_control import normalize_ollama_model_ref
# Prompt/text helpers straight from face_variations: it has no dependency on
# this module, so there is no cycle to schedule around.
from .face_variations import (
    DESCRIPTIVE_CAPTION_PROMPT, CAPTION_REFINE_CONCEPT_PROMPT,
    CAPTION_LEAK_FIX_PROMPT, EXPAND_CONCEPT_TERMS_PROMPT,
    caption_prompt_for, caption_prompt_for_style, caption_prompt_for_concept,
    concept_lexical_field, drop_identity_sentences, drop_identity_tags,
)


# --- Captioning (JoyCaption / Qwen3-VL, backend picked in Settings) --------
# --- Concept-omission guarantee (ban-list + verify + corrective rewrite) -----
# Negative prompting ALONE leaks (~35% measured e2e on 3 unseen concepts): the
# robustness comes from a deterministic OUTPUT check + targeted correction. Pipeline
# per caption: regex detection (ban-list) -> if leak, Qwen rewrite naming the leaked
# words (<=2 tries) -> mechanical safety net (drop the offending clause). The Qwen
# calls are threaded in via `describe` (our vision seam is a local import inside the
# caption batch); `describe=None` degrades to mechanical scrub only (backend 'joycaption').

# The abliterated Qwen3-VL SOMETIMES emits its reasoning trace ("the task says... we
# need to remove...") or an infinite loop instead of the refined caption - seen ~1/4
# of images. We detect these unusable outputs to fall back on a DIRECT Qwen caption.
# Matches the reasoning/meta phrasings the abliterated Qwen leaks INSTEAD of a caption.
# Widened after real leaks slipped through ("Yes, this describes…", "The original caption
# says…", "Now, check for…", "I think this works"): allow words between "the task/caption"
# and its verb, and add the yes/now/check/i-think markers. Descriptive prose essentially
# never contains these, so a false reject just falls back to a direct caption - cheap.
_REFINE_REASONING_RE = re.compile(
    r'(?:'
    r'\bthe (?:problem|instruction|task|draft|original|caption)(?:\s+\w+){0,4}\s+'
    r'(?:says?|said|mentions?|has|reads?|describes?|is)\b'
    r'|\bwe (?:need|can|should) to (?:remove|rephrase|avoid|describe|keep)'
    r'|\bso we (?:need|can|should)\b'
    r'|\blet me\b|\brephrase\b|\bwait,|\bnow,\s|\bcheck for\b'
    r'|\bi think\b|\bi need to\b|\byes,\s+(?:this|that|the|we|it|but)'
    r')', re.I)

# A concept caption is scene-exhaustive prose; anything this short is a degenerate
# output (e.g. "taking a picture") that just names the concept - never a real caption.
_MIN_CONCEPT_CAPTION_CHARS = 40


def _refine_output_ok(text, prior) -> bool:
    """True if `text` looks like a CLEAN caption - not the Qwen reasoning trace, not a
    degenerate one-liner, not a loop/rambling (bounded to ~2x the source caption `prior`)."""
    t = (text or '').strip()
    if len(t) < _MIN_CONCEPT_CAPTION_CHARS or _REFINE_REASONING_RE.search(t):
        return False
    return len(t) <= 2 * len(prior or '') + 400


def _usable_caption(text) -> bool:
    """A committable concept caption: non-empty prose that is NOT a reasoning trace.
    Length is deliberately NOT gated here - a legitimately terse caption left after the
    clause-scrub must still commit; only the refine-vs-fallback choice (_refine_output_ok)
    weighs length. A degenerate "taking a picture" is handled upstream: the ban-list
    scrubs the concept out, leaving an empty string this rejects."""
    t = (text or '').strip()
    return bool(t) and not _REFINE_REASONING_RE.search(t)


# Words from concept_desc that are never discriminating (articles + generic adjectives
# a legit caption uses elsewhere: "bare shoulders", "full-body"...).
_TERMS_STOP = frozenset((
    'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'by', 'with', 'to', 'from',
    'that', 'this', 'as', 'is', 'are', 'his', 'her', 'their', 'its', 'it', 'one',
    'act', 'shown', 'worn', 'being', 'person', 'subject', 'focal', 'point', 'visible',
    'bare', 'exposed', 'full', 'close', 'closeup', 'close-up', 'wearing', 'showing'))


# A concept training caption must describe the SUBJECT, never the act of image capture.
# The abliterated Qwen reliably leaks capture-language ("holding a phone to frame the
# shot", "point-of-view mirror", "capturing her reflection") that the LLM ban-list
# expansion never fully enumerates - for "a candid mirror selfie" it returned only
# mirror/self-* variants, so phone/smartphone/camera/reflection leaked into ~45/54
# captions. This DETERMINISTIC lexicon is unioned into the ban-list whenever the concept
# is photographic (selfie/mirror/photo/portrait/pov/camera/phone), so those words are
# ALWAYS scrubbed regardless of the LLM. Reproducible from a fresh clone - no reliance on
# the flaky expansion for words we already know.
_CAPTURE_TRIGGERS = ('selfie', 'mirror', 'photo', 'picture', 'portrait', 'camera',
                     'phone', 'pov', 'point of view', 'snapshot', 'webcam', 'pic ')
_CAPTURE_LEXICON = frozenset((
    'selfie', 'self-portrait', 'self-portraiture', 'self-photograph', 'self-shot',
    'mirror', 'reflection', 'reflected', 'reflective surface',
    'phone', 'smartphone', 'cellphone', 'cell phone', 'mobile phone', 'iphone',
    'camera', 'webcam', 'front-facing', 'pov', 'point of view', 'point-of-view'))


def _fallback_concept_terms(desc) -> list:
    """Minimal ban-list WITHOUT the LLM: the meaningful words of concept_desc itself
    (always included, even when the LLM expansion succeeds - the user's words are the
    ground truth), PLUS the capture lexicon when the concept is photographic, PLUS the
    derived body/pose lexical field (so a POSE concept's periphrases - "knees lifted",
    "feet raised", "thighs" for "leg behind head position" - are scrubbed even though the
    description never spells them, and the LLM expansion is FORBIDDEN from listing pose
    words). Deterministic, reproducible from a fresh clone - the leg_behind fix."""
    d = (desc or '').lower()
    words = re.split(r'[^a-zA-Z-]+', d)
    terms = {w.strip('-') for w in words
             if len(w.strip('-')) >= 3 and w.strip('-') not in _TERMS_STOP}
    if any(k in d for k in _CAPTURE_TRIGGERS):
        terms |= _CAPTURE_LEXICON
    terms |= set(concept_lexical_field(desc))
    return sorted(terms)


def _concept_terms_re(terms):
    """Leak-detection regex: word boundaries, space/hyphen interchangeable ("two-piece"
    <-> "two piece"), plurals/-s/-es/-ing/-ed tolerated. None if the list is empty."""
    pats = []
    for t in terms or []:
        t = (t or '').strip().lower()
        if len(t) < 3:
            continue
        p = re.escape(t).replace(r'\ ', r'[\s-]+').replace(r'\-', r'[\s-]+')
        pats.append(p)
    if not pats:
        return None
    return re.compile(r'\b(?:' + '|'.join(pats) + r')(?:e?s|ing|ed)?\b', re.I)


def _scrub_concept_clauses(caption, leak_re):
    """MECHANICAL net: drop the clauses (segments between , ; .) containing a forbidden
    term - the whole clause, not just the word, to keep grammatical prose. If it destroys
    too much (<30 chars), remove only the words."""
    parts = re.split(r'([.;,])', caption or '')
    kept = []
    for i in range(0, len(parts), 2):
        seg = parts[i]
        punc = parts[i + 1] if i + 1 < len(parts) else ''
        if seg.strip() and leak_re.search(seg):
            continue
        kept.append(seg + punc)
    out = re.sub(r'\s{2,}', ' ', ''.join(kept)).strip(' ,;')
    if len(out) >= 30:
        return out
    out = re.sub(r'\s{2,}', ' ', leak_re.sub('', caption or '')).strip(' ,;')
    return out


def _parse_terms_json(raw) -> list:
    """Extract the term list from an LLM blocklist reply. Tolerates noise around the
    object AND — critically for the abliterated Qwen, which frequently LOOPS and never
    closes the JSON array (so json.loads fails) — salvages the quoted strings directly,
    KEEPING their order: the model emits the good, concept-specific terms first, then
    combinatorial padding ("mirror selfie shot", "self-portrait photograph"…). Ordered
    de-dup (the loop repeats), stopwords dropped, capped so the padding can't dominate."""
    raw = raw or ''
    terms = None
    start, end = raw.find('{'), raw.rfind('}')
    if 0 <= start < end:
        try:
            data = json.loads(raw[start:end + 1])
            if isinstance(data, dict) and isinstance(data.get('terms'), list):
                terms = data['terms']
        except ValueError:
            terms = None
    if terms is None:
        # Unclosed/looping array → pull the quoted strings after "terms" in order.
        m = re.search(r'"terms"\s*:\s*\[(.*)', raw, re.S)
        terms = re.findall(r'"([^"\\]{1,60})"', m.group(1) if m else raw)
    out, seen = [], set()
    for t in terms:
        if not isinstance(t, str):
            continue
        t = t.strip().lower()
        if 3 <= len(t) <= 40 and t not in _TERMS_STOP and t not in seen:
            seen.add(t)
            out.append(t)
            if len(out) >= 25:
                break
    return out


def _get_concept_terms(ds, image_path=None, describe=None) -> list:
    """Dataset ban-list: union of (LLM expansion cached in ds.concept_terms) and (words
    of concept_desc). The expansion runs ONCE (vision model already warm in the GPU
    window, the image is just a vehicle - the prompt ignores it) and is cached ONLY if it
    succeeds (a failure retries next batch). `describe` is our describe_image_ollama seam;
    None -> fallback words only (no LLM call)."""
    base = _fallback_concept_terms(ds.concept_desc)
    stored = []
    if getattr(ds, 'concept_terms', None):
        try:
            stored = [t for t in json.loads(ds.concept_terms) if isinstance(t, str)]
        except ValueError:
            stored = []
    if stored:
        return sorted(set(stored) | set(base))
    if image_path and describe is not None:
        try:
            with open(image_path, 'rb') as fh:
                raw = describe(
                    fh.read(),
                    EXPAND_CONCEPT_TERMS_PROMPT.format(concept=(ds.concept_desc or '').strip()),
                    # 1200 is ample for a 6-15 term list; keeping it tight bounds the
                    # abliterated model's combinatorial loop so the salvage in
                    # _parse_terms_json keeps the good leading terms.
                    num_predict=1200, prefer_json=True, fmt='json',
                    keep_alive=_VISION_BATCH_KEEPALIVE)
        except OSError:
            raw = ''
        expanded = _parse_terms_json(raw)
        if expanded:
            ds.concept_terms = json.dumps(expanded)
            db.session.commit()
            logger.info('concept terms: %d terms generated for ds%s', len(expanded), ds.id)
            return sorted(set(expanded) | set(base))
        logger.info('concept terms: empty LLM expansion for ds%s -> desc fallback', ds.id)
    return base


def _enforce_concept_omission(caption, leak_re, image_bytes, concept_desc, describe=None):
    """Guarantee omission: detect forbidden terms in `caption`, ask Qwen for a rewrite
    that NAMES the offending words (<=2 tries, kept by _refine_output_ok), then a
    mechanical net (clause drop). Returns the caption (unchanged if no leak). `describe`
    is the vision seam; None -> skip the LLM fix, go straight to the mechanical scrub."""
    if not leak_re or not (caption or '').strip():
        return caption
    if describe is not None:
        for _ in range(2):
            leaked = sorted({m.group(0).lower() for m in leak_re.finditer(caption)})
            if not leaked:
                return caption
            fixed = ''
            try:
                fixed = describe(
                    image_bytes,
                    CAPTION_LEAK_FIX_PROMPT.format(existing=caption, concept=concept_desc,
                                                   leaked=', '.join(leaked)),
                    num_predict=5000, keep_alive=_VISION_BATCH_KEEPALIVE)
            except Exception:  # noqa: BLE001 - best-effort correction
                fixed = ''
            fixed = (fixed or '').strip().strip('"').strip()
            if _refine_output_ok(fixed, caption):
                caption = fixed
    if leak_re.search(caption):
        caption = _scrub_concept_clauses(caption, leak_re)
    return caption


def _caption_concept(ds, force, backend, token=None, image_ids=None,
                     ollama_model=None, extra_instructions=''):
    """Concept caption pipeline (INVERTED logic): describe everything INCLUDING identity
    but OMIT the recurring act so it binds to the trigger. JoyCaption is literal (it NAMES
    the act/fluids/watermark) -> its drafts are REFINED by Qwen, then every caption passes
    the ban-list omission guarantee. Backend gating is honored:
      - 'joycaption' -> Joy drafts only + mechanical scrub (no Qwen calls);
      - 'ollama'     -> Joy skipped, every image direct-Qwen + enforcement;
      - 'auto'       -> Joy drafts refined by Qwen, no-Joy images direct-Qwen, all enforced."""
    concept_desc = (ds.concept_desc or '').strip()
    # Dynamic omission clause: for a POSE concept the generic "describe their pose and
    # body position" line would instruct the VLM to describe the very concept - the
    # builder folds in a concept-specific negative ("do NOT describe the position of the
    # legs/knees/feet…") that overrides it. Byte-identical to the old prompt for non-body
    # concepts. This is the generation-side half of the leg_behind fix.
    cap_prompt = caption_prompt_for_concept(concept_desc)
    # Extra user instructions apply to the DIRECT-caption prompt (the Qwen refine of a Joy
    # draft is a structured transform left untouched). The concept omission still fronts
    # the prompt and the ban-list enforcement still post-filters every caption.
    cap_prompt = _with_caption_instructions(cap_prompt, extra_instructions)
    q = FaceDatasetImage.query.filter_by(dataset_id=ds.id, status='keep')
    if image_ids is not None:
        q = q.filter(FaceDatasetImage.id.in_(image_ids))
    if not force:
        q = q.filter((FaceDatasetImage.caption.is_(None)) | (FaceDatasetImage.caption == ''))
    todo = [(img, _img_path(img)) for img in q.all() if img.filename]
    todo = [(img, p) for img, p in todo if p and os.path.exists(p)]
    if not todo:
        return 0
    # Total for the persistent progress indicator (token owned by the caller).
    dataset_activity.progress(token, total=len(todo),
                              detail=f'Preparing {len(todo)} concept caption(s)…')
    n = 0
    remaining = list(todo)
    refine_targets = []  # (img, p, joycap) -> Joy draft refined by Qwen
    # 1) JoyCaption batch (draft) when the backend allows it.
    if backend in ('auto', 'joycaption'):
        jc = {}
        try:
            from .joycaption import caption_images_joycaption, is_available
            if is_available():
                dataset_activity.progress(
                    token, detail=f'Loading JoyCaption model and captioning {len(todo)} images…')
                jc = caption_images_joycaption(
                    [p for _, p in todo], prompt=cap_prompt, activity_token=token,
                    should_cancel=lambda: dataset_activity.cancel_requested(ds.id))
            elif backend == 'joycaption':
                raise RuntimeError('JoyCaption backend is not available - check the ai-toolkit folder in Settings')
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning('caption concept: JoyCaption indisponible (%s)', e)
        still = []
        for img, p in remaining:
            cap = (jc.get(p) or '').strip().strip('"').strip()
            if cap:
                refine_targets.append((img, p, cap))
            else:
                still.append((img, p))
        remaining = still
    # 2a) Backend 'joycaption' forced: no Qwen. Store Joy drafts scrubbed mechanically
    #     (leak_re from the desc words only) - respects "no Ollama fallback".
    if backend == 'joycaption':
        leak_re = _concept_terms_re(_fallback_concept_terms(concept_desc))
        for img, p, joycap in refine_targets:
            if dataset_activity.cancel_requested(ds.id):
                break   # graceful stop at an image boundary (see caption_images)
            dataset_activity.bump(token)
            try:
                with open(p, 'rb') as fh:
                    data = fh.read()
            except OSError:
                data = b''
            final = _enforce_concept_omission(joycap, leak_re, data, concept_desc) or joycap
            img.caption = _cap_caption(final)
            db.session.commit()
            n += 1
        return n
    # 2b) Qwen passes ('auto'/'ollama'): refine Joy drafts, direct-caption the rest, all
    #     enforced. One model load -> unload once at the end.
    if refine_targets or remaining:
        try:
            from .vision_ollama import describe_image_ollama, unload_vision_model
        except ImportError:
            raise RuntimeError('vision (Ollama) service not configured/available yet')
        # Bind the per-dataset model once for EVERY Concept inference pass. Without
        # this, only the main caption/refine used the override while blocklist
        # expansion and omission rewrites silently loaded the global model.
        def describe(image_bytes, prompt, **kwargs):
            if ollama_model:
                kwargs['model'] = ollama_model
            return describe_image_ollama(image_bytes, prompt, **kwargs)
        # Ban-list (LLM expansion cached + desc words) -> leak regex, compiled ONCE per
        # batch, AFTER the Joy subprocess finished (never two models in VRAM at once).
        sample = refine_targets[0][1] if refine_targets else remaining[0][1]
        leak_re = _concept_terms_re(_get_concept_terms(ds, image_path=sample,
                                                       describe=describe))
        try:
            for img, p, joycap in refine_targets:
                if dataset_activity.cancel_requested(ds.id):
                    break   # graceful stop at an image boundary (see caption_images)
                dataset_activity.bump(token)
                with open(p, 'rb') as fh:
                    data = fh.read()
                refined = ''
                # The refine prompt is where the concept-omitting caption is actually
                # PRODUCED when JoyCaption is available (the dominant path), so the
                # per-dataset extra instructions — including the NSFW vocabulary preset —
                # must ride here too. Applied ONLY to cap_prompt before, they never reached
                # the refine, so an 'explicit' preset silently produced a neutral caption:
                # the (abliterated) refiner rewrote the crude Joy draft "as a clean caption"
                # with no register directive. Empty extras keep the prompt byte-identical.
                refine_prompt = _with_caption_instructions(
                    CAPTION_REFINE_CONCEPT_PROMPT.format(existing=joycap,
                                                         concept=concept_desc),
                    extra_instructions)
                try:
                    refined = describe(
                        data, refine_prompt,
                        num_predict=5000,
                        keep_alive=_VISION_BATCH_KEEPALIVE,
                        timeout=(10, 300))
                except Exception as e:  # noqa: BLE001 - refine best-effort
                    logger.warning('caption concept: Qwen refine failed (%s)', e)
                refined = (refined or '').strip().strip('"').strip()
                if _refine_output_ok(refined, joycap):
                    final = refined
                else:
                    # Unusable refine (reasoning trace / loop) -> direct Qwen caption
                    # (natively omits the concept), else keep the Joy draft.
                    logger.info('caption concept: refine rejected -> direct Qwen (image %s)', img.id)
                    alt = ''
                    try:
                        alt = describe(data, cap_prompt, num_predict=2000,
                                       keep_alive=_VISION_BATCH_KEEPALIVE,
                                       timeout=(10, 300))
                    except Exception:  # noqa: BLE001
                        alt = ''
                    alt = (alt or '').strip().strip('"').strip()
                    final = alt or joycap
                final = _enforce_concept_omission(final, leak_re, data, concept_desc,
                                                  describe=describe) or final
                if not _usable_caption(final):
                    # Refine AND direct both unusable → fall back to the Joy draft (clean
                    # prose), scrubbed of any leak; leave blank if even that fails.
                    final = _enforce_concept_omission(joycap, leak_re, data, concept_desc,
                                                      describe=describe) or joycap
                    if not _usable_caption(final):
                        # force=re-do-all: overwrite any stale pre-fix caption with blank
                        # (trigger-only is valid for a concept LoRA) rather than retain it.
                        if force and (img.caption or ''):
                            img.caption = ''
                            db.session.commit()
                        logger.info('caption concept: no usable caption for image %s -> left blank', img.id)
                        continue
                img.caption = _cap_caption(final)
                db.session.commit()
                n += 1
            for img, p in remaining:
                if dataset_activity.cancel_requested(ds.id):
                    break   # graceful stop at an image boundary (see caption_images)
                dataset_activity.bump(token)
                with open(p, 'rb') as fh:
                    data = fh.read()
                cap = describe(
                    data, cap_prompt, num_predict=2000,
                    keep_alive=_VISION_BATCH_KEEPALIVE,
                    auto_start_local=True, timeout=(10, 300))
                cap = (cap or '').strip().strip('"').strip()
                if cap:
                    cap = _enforce_concept_omission(cap, leak_re, data, concept_desc,
                                                    describe=describe) or cap
                if _usable_caption(cap):
                    img.caption = _cap_caption(cap)
                    db.session.commit()
                    n += 1
                else:
                    if force and (img.caption or ''):
                        img.caption = ''
                        db.session.commit()
                    logger.info('caption concept: no usable direct caption for image %s -> left blank', img.id)
        finally:
            if ollama_model:
                unload_vision_model(model=ollama_model)
            else:
                unload_vision_model()  # libère la VRAM pour ComfyUI en fin de batch
    return n


def caption_images(user_id, dataset_id, force=False, mode=None, image_ids=None,
                   report=None):
    """Caption les images gardees. Defaut: seulement celles SANS caption ; force=True
    re-capte TOUTES les gardees (ecrase) - pour rejouer apres un changement de prompt.
    Chaque caption passe par drop_identity_sentences (retire une eventuelle phrase
    d'identite isolee).

    `image_ids` (optionnel) restreint la passe a ce sous-ensemble d'images gardees —
    utilise par le bouton 🔄 Re-caption cible du panneau Identity-leak (une seule image
    ou « toutes les fuyantes ») ; None -> tout le dataset (comportement batch). Meme
    moteur, meme mode, meme contexte kind et memes regles de nettoyage que le lot complet.

    `captioning.backend` (réglages) pilote qui capte quoi :
      - 'none'       -> désactivé, RuntimeError (mappée 409 par la route).
      - 'joycaption' -> JoyCaption seul, PAS de repli Ollama.
      - 'ollama'     -> Ollama (Qwen3-VL) seul, JoyCaption jamais tenté.
      - 'auto'       -> comportement historique : JoyCaption en priorité,
                        fallback Ollama pour les images qu'il n'a pas captées."""
    _guard_not_bank_export(dataset_id)
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        return 0
    # Per-dataset method overrides (Captions ⚙️ Options): the chosen engine, an extra
    # instruction appended to the prompt, and the Ollama vision model to run. Each falls
    # back to the global default when the dataset never set it.
    opts = caption_options(ds)
    backend = (opts.get('backend') or cfg.get('captioning.backend') or 'auto').lower()
    if backend == 'none':
        raise RuntimeError('No captioning backend configured')
    # Vocabulary preset (NSFW register) + free-text steer, combined into the one block that
    # rides at the end of every prompt this run builds.
    extra_instructions = _combined_caption_instructions(opts)
    ollama_model = (opts.get('ollama_model') or '').strip() or None
    # A targeted subset (Identity-leak panel): normalize to ints once, drop non-numeric.
    # `None` = whole dataset; an EMPTY subset (nothing to re-caption) short-circuits to 0
    # rather than silently captioning everything.
    ids = None
    if image_ids is not None:
        ids = [int(i) for i in image_ids
               if isinstance(i, (int, float, str)) and str(i).lstrip('-').isdigit()]
        if not ids:
            return 0
    # Dataset CONCEPT : logique INVERSÉE (décrire tout SAUF l'acte récurrent → il se lie
    # au trigger). Pipeline dédié Joy→Qwen + garantie d'omission (ban-list) : entièrement
    # à part du chemin character ci-dessous. Respecte le backend gating.
    # The persistent indicator is owned HERE (begin/finally) so the concept body stays
    # unindented; it only feeds progress via the passed token.
    if is_concept(ds):
        token = dataset_activity.begin(
            dataset_id, 'recaption' if force else 'caption',
            detail='Preparing concept captioning…')
        started = time.monotonic()
        logger.info('captioning started: dataset=%s backend=%s force=%s kind=concept',
                    dataset_id, backend, force)
        try:
            n = _caption_concept(ds, force, backend, token=token, image_ids=ids,
                                 ollama_model=ollama_model,
                                 extra_instructions=extra_instructions)
            logger.info('captioning finished: dataset=%s backend=%s captioned=%s elapsed=%.1fs',
                        dataset_id, backend, n, time.monotonic() - started)
            return n
        except Exception:
            logger.exception('captioning failed: dataset=%s backend=%s kind=concept elapsed=%.1fs',
                             dataset_id, backend, time.monotonic() - started)
            raise
        finally:
            dataset_activity.end(token)
    # Style de caption : prose (Z-Image) vs tags booru (SDXL booru-native type bigLove).
    # Défaut AUTO selon le type entraîné ; un mode explicite (UI) l'emporte.
    ttype = (getattr(ds, 'train_type', None) or 'zimage').lower()
    mode = (mode or ('booru' if ttype == 'sdxl' else 'prose')).lower()
    style = is_style(ds)
    if style:
        # Dataset STYLE : captions de CONTENU pur — le rendu n'est jamais décrit (le
        # prompt porte la règle) pour qu'il soit absorbé par le LoRA. AUCUN nettoyage
        # d'identité : les sujets varient, leur description EST le contenu contrôlable.
        cap_prompt = caption_prompt_for_style(mode)
        def cleaner(text):
            return text
    else:
        # Fidélité corps : le prompt bannit EN PLUS les marques corporelles permanentes
        # (tatouages/cicatrices/piercings…) et le post-filtre les retire — elles doivent
        # se lier au trigger, pas aux mots (même principe que le visage).
        body = is_body_fidelity(ds)
        cap_prompt = caption_prompt_for(mode, body=body)
        base_cleaner = drop_identity_tags if mode == 'booru' else drop_identity_sentences
        def cleaner(text):
            return base_cleaner(text, body=body)
    # Extra user instructions ride at the END of the prompt (both engines) — the kind
    # omission rules stay first, and the cleaner above still post-filters the output.
    cap_prompt = _with_caption_instructions(cap_prompt, extra_instructions)
    q = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
    if ids is not None:
        q = q.filter(FaceDatasetImage.id.in_(ids))
    if not force:
        q = q.filter((FaceDatasetImage.caption.is_(None)) | (FaceDatasetImage.caption == ''))
    rows = q.all()
    todo = [(img, _img_path(img)) for img in rows if img.filename]
    todo = [(img, p) for img, p in todo if p and os.path.exists(p)]
    if not todo:
        return 0
    # Persistent progress indicator (survives a page reload): 'recaption' when force
    # overwrites existing captions, else 'caption'. try/finally guarantees end() runs
    # even if the vision pass raises → no phantom "Captioning…" spinner after a crash.
    token = dataset_activity.begin(
        dataset_id, 'recaption' if force else 'caption', total=len(todo),
        detail=f'Preparing to caption {len(todo)} image(s)…')
    started = time.monotonic()
    logger.info('captioning started: dataset=%s backend=%s mode=%s force=%s images=%s',
                dataset_id, backend, mode, force, len(todo))
    try:
        n = 0
        remaining = todo
        # In 'auto', why JoyCaption didn't contribute (deps missing / crash). Kept so a
        # LATER Ollama failure reports BOTH reasons instead of only the Ollama one —
        # otherwise a user whose JoyCaption is silently unavailable debugs blind (issue #6).
        joycaption_note = ''
        # 1) JoyCaption en BATCH (un seul chargement du 8B NF4, via le venv ai-toolkit) -
        # sauté entièrement quand le backend force 'ollama'.
        if backend in ('auto', 'joycaption'):
            jc = {}
            try:
                from .joycaption import availability, caption_images_joycaption, is_available
                if is_available():
                    dataset_activity.progress(
                        token,
                        detail=f'Loading JoyCaption model and captioning {len(todo)} images…')
                    # Consigne « ne décris pas le visage » → les traits se lient au trigger,
                    # pas aux mots de la caption (deep-research 2026-06-14).
                    jc = caption_images_joycaption(
                        [p for _, p in todo], prompt=cap_prompt, activity_token=token,
                        should_cancel=lambda: dataset_activity.cancel_requested(dataset_id))
                elif backend == 'joycaption':
                    # Explicit choice, explicit failure: a user who forced 'joycaption' in
                    # Settings must be told WHY (the exact missing deps + pip command),
                    # not get a silent 0 (only 'auto' is allowed to fall back to Ollama).
                    raise RuntimeError(
                        'JoyCaption backend is not available — '
                        + (availability().get('detail') or 'check the ai-toolkit folder in Settings'))
                else:  # auto: JoyCaption unavailable -> remember the reason, fall back to Ollama
                    joycaption_note = availability().get('detail') or 'JoyCaption unavailable'
            except RuntimeError:
                raise
            except Exception as e:
                joycaption_note = str(e)
                logger.warning('caption_images: JoyCaption indisponible (%s)', e)
            still = []
            for img, p in remaining:
                cap = (jc.get(p) or '').strip().strip('"').strip()
                if cap:
                    cleaned = cleaner(cap) or cap
                    img.caption = _cap_caption(cleaned)
                    db.session.commit()
                    n += 1
                    _writer(report, CAPTION_WRITER_JOYCAPTION)
                    dataset_activity.bump(token)   # this image is captioned (done)
                else:
                    still.append((img, p))
            remaining = still
            dataset_activity.progress(
                token, detail=f'JoyCaption finished; {len(remaining)} image(s) remaining…')
            if backend == 'joycaption':  # backend forcé JoyCaption -> pas de repli Ollama
                logger.info('captioning finished: dataset=%s backend=%s captioned=%s elapsed=%.1fs',
                            dataset_id, backend, n, time.monotonic() - started)
                return n
        # 2) Ollama (Qwen3-VL) pour les images non couvertes par JoyCaption ('auto'),
        # ou pour TOUT le lot si le backend force 'ollama'.
        if remaining:
            try:
                from .vision_ollama import describe_image_ollama, unload_vision_model
            except ImportError:
                raise RuntimeError('vision (Ollama) service not configured/available yet')
            try:
                for index, (img, p) in enumerate(remaining, 1):
                    # Graceful stop: the user asked to stop and we're at an image
                    # boundary (nothing decoding) — leave the rest uncaptioned and let
                    # the finally below free the model, exactly like a normal finish.
                    if dataset_activity.cancel_requested(dataset_id):
                        break
                    dataset_activity.progress(
                        token,
                        detail=f'Captioning with Ollama — image {index}/{len(remaining)}…')
                    with open(p, 'rb') as fh:
                        cap = describe_image_ollama(
                            fh.read(), cap_prompt, num_predict=2000, model=ollama_model,
                            keep_alive=_VISION_BATCH_KEEPALIVE,
                            auto_start_local=(index == 1), timeout=(10, 300))
                    cap = (cap or '').strip().strip('"').strip()
                    if cap:
                        cleaned = cleaner(cap) or cap
                        img.caption = _cap_caption(cleaned)
                        db.session.commit()
                        n += 1
                        _writer(report, CAPTION_WRITER_OLLAMA)
                    dataset_activity.bump(token)   # image handled (captioned or not)
            except RuntimeError as e:
                # 'auto' tried JoyCaption first and it was unavailable, then Ollama
                # failed too — report BOTH so the user isn't repairing blind (they'd
                # otherwise see only the Ollama error and never learn JoyCaption's deps
                # are missing, issue #6). backend='ollama' has no note -> re-raise as-is.
                if joycaption_note:
                    raise RuntimeError(f'JoyCaption unavailable: {joycaption_note} · Ollama: {e}') from e
                raise
            finally:
                unload_vision_model()  # libère la VRAM pour ComfyUI en fin de batch
        logger.info('captioning finished: dataset=%s backend=%s captioned=%s elapsed=%.1fs',
                    dataset_id, backend, n, time.monotonic() - started)
        return n
    except Exception:
        logger.exception('captioning failed: dataset=%s backend=%s elapsed=%.1fs',
                         dataset_id, backend, time.monotonic() - started)
        raise
    finally:
        dataset_activity.end(token)


def caption_paths(paths, *, prompt=None, backend=None, ollama_model=None,
                  extra_instructions=None, should_cancel=None, on_caption=None,
                  progress=None) -> dict:
    """Caption a list of image FILE PATHS with the app's configured engines, returning
    {path: caption}. Dataset-free, purely DESCRIPTIVE captioning (no trigger word, no
    identity/concept/style omission) for the image bank and the future launch-all
    pipeline — a bank caption is a plain description that doubles as search text.

    Reuses the SAME inference bricks as the dataset caption pass (`caption_images`):
    JoyCaption in one batch load, then Ollama (Qwen3-VL) per image for whatever it
    didn't cover, gated by `captioning.backend`. What it deliberately SKIPS is all the
    per-dataset kind logic (prompt building, leak cleaners, dual shorts).

    prompt          : override the default neutral descriptive prompt.
    backend         : override captioning.backend ('auto'|'joycaption'|'ollama'|'none').
    ollama_model    : override the Ollama vision model (None = global default).
    extra_instructions : appended to the prompt (both engines), like the dataset options.
    should_cancel() : polled at each image boundary in the Ollama phase for a graceful
                      stop (JoyCaption runs as one batch and isn't interruptible mid-load,
                      same as the dataset pass). The Ollama phase overlaps several calls
                      (see vision_pool), so a stop drains what is in flight — a couple of
                      seconds — and every drained answer is still handed to on_caption.
    on_caption(path, caption) : fired as each caption lands, for incremental persistence.
                      ALWAYS called on the caller's own thread, never on a worker, so it
                      is free to use the database session.
    progress(done, total)     : progress callback (every handled image, captioned or not).

    Best-effort: a totally unavailable engine raises RuntimeError (so the caller can
    surface WHY); an individual empty caption is simply skipped. Unloads the Ollama model
    at the end (VRAM back to ComfyUI). Holding the GPU-exclusive vision window is the
    CALLER's job, so launch-all can keep ONE window across several steps."""
    paths = [p for p in (paths or []) if p and os.path.isfile(p)]
    total = len(paths)
    out: dict = {}
    if progress:
        progress(0, total)
    if not paths:
        return out
    backend = (backend or cfg.get('captioning.backend') or 'auto').lower()
    if backend == 'none':
        raise RuntimeError('No captioning backend configured')
    cap_prompt = prompt or DESCRIPTIVE_CAPTION_PROMPT
    if extra_instructions:
        cap_prompt = _with_caption_instructions(cap_prompt, (extra_instructions or '').strip())
    ollama_model = (ollama_model or '').strip() or None
    done = 0

    def _emit(p, cap):
        nonlocal done
        out[p] = cap
        if on_caption:
            on_caption(p, cap)
        done += 1
        if progress:
            progress(done, total)

    remaining = list(paths)
    # 1) JoyCaption batch (single 8B NF4 load via the ai-toolkit venv) — skipped when
    # the backend forces 'ollama'.
    joycaption_note = ''
    if backend in ('auto', 'joycaption'):
        jc = {}
        try:
            from .joycaption import availability, caption_images_joycaption, is_available
            if is_available():
                jc = caption_images_joycaption(remaining, prompt=cap_prompt,
                                               should_cancel=should_cancel)
            elif backend == 'joycaption':
                raise RuntimeError(
                    'JoyCaption backend is not available — '
                    + (availability().get('detail') or 'check the ai-toolkit folder in Settings'))
            else:  # auto: unavailable → remember why, fall back to Ollama
                joycaption_note = availability().get('detail') or 'JoyCaption unavailable'
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001 — any JoyCaption crash falls back to Ollama in auto
            joycaption_note = str(e)
            logger.warning('caption_paths: JoyCaption unavailable (%s)', e)
        still = []
        for p in remaining:
            cap = (jc.get(p) or '').strip().strip('"').strip()
            if cap:
                _emit(p, _cap_caption(cap))
            else:
                still.append(p)
        remaining = still
        if backend == 'joycaption':
            return out
    # 2) Ollama (Qwen3-VL) for whatever JoyCaption didn't cover, or the whole set when
    # the backend forces 'ollama'.
    if remaining:
        try:
            from .vision_ollama import describe_image_ollama, unload_vision_model
        except ImportError:
            raise RuntimeError('vision (Ollama) service not configured/available yet')
        from .vision_pool import map_vision

        def _describe(path, *, auto_start=False):
            """One caption call. Runs on a WORKER thread under map_vision, so it
            touches nothing but the file and the network."""
            with open(path, 'rb') as fh:
                return describe_image_ollama(
                    fh.read(), cap_prompt, num_predict=2000, model=ollama_model,
                    keep_alive=_VISION_BATCH_KEEPALIVE,
                    auto_start_local=auto_start, timeout=(10, 300))

        def _land(path, cap):
            """Persist one answer. Always on the CALLING thread — `on_caption` is
            what writes to the database, and that session isn't thread-safe."""
            nonlocal done
            cap = (cap or '').strip().strip('"').strip()
            if cap:
                _emit(path, _cap_caption(cap))
            else:
                done += 1  # handled-but-empty still advances the bar
                if progress:
                    progress(done, total)

        try:
            # The first image runs ALONE, and is the only one allowed to start a
            # stopped local Ollama: a cold server must be woken (and diagnosed)
            # once, not by several callers racing into the same restart. It also
            # warms the model, so the calls that follow overlap real inference
            # instead of queueing behind a model load.
            first, rest = remaining[0], remaining[1:]
            if not (should_cancel and should_cancel()):
                _land(first, _describe(first, auto_start=True))
                # The rest overlap: most of a caption call is round-trip waiting,
                # not GPU work (services/vision_pool.py has the measurements).
                # should_cancel is still polled per image, so the graceful stop
                # keeps its meaning — it just drains the calls in flight first.
                for path, cap, error in map_vision(rest, _describe,
                                                   should_cancel=should_cancel):
                    if error is not None:
                        # A file that vanished mid-pass, a permission error: one
                        # image is skipped and counted, the batch goes on.
                        logger.warning('caption_paths: %s skipped: %s',
                                       os.path.basename(path), error)
                        done += 1
                        if progress:
                            progress(done, total)
                        continue
                    _land(path, cap)
        except RuntimeError as e:
            # 'auto' tried JoyCaption first and it was unavailable, then Ollama failed too
            # — report BOTH so the caller isn't debugging blind (issue #6 reasoning).
            if joycaption_note:
                raise RuntimeError(f'JoyCaption unavailable: {joycaption_note} · Ollama: {e}') from e
            raise
        finally:
            unload_vision_model()  # hand the VRAM back to ComfyUI
    return out


# --- Caption Lab: per-candidate preview (no persistence) ---------------------
# The 🧪 Caption Lab lets the user try a caption CONFIG (engine × Ollama model ×
# vocabulary register) on ONE image and read the result WITHOUT writing anything to
# the row. It rides on caption_paths() — the dataset-free by-path brick — so it runs
# purely DESCRIPTIVE captioning (no kind omission, no dual short): the point is to
# compare raw model output side by side and pick the config, not to produce the final
# stored caption (that still goes through the normal caption pass with its kind rules).

def _compose_preview_instructions(vocabulary, instructions, length=None) -> str | None:
    """Combine the presets (the SAME appended register and length text the dataset pass
    uses) with the user's free extra instructions into the single ``extra_instructions``
    string caption_paths appends to the prompt. Same order as the dataset pass — presets
    first, free text last. None when nothing is set (byte-identical to a plain descriptive
    pass)."""
    parts = _caption_preset_parts(vocabulary, length)
    extra = (instructions or '').strip()[:_CAPTION_INSTRUCTIONS_MAX]
    if extra:
        parts.append(extra)
    return '\n'.join(parts) if parts else None


def vocabulary_instruction(vocabulary) -> str | None:
    """The caption instruction appended for a vocabulary register (one of
    CAPTION_VOCABULARIES: 'explicit' | 'clinical' | 'safe'), or None for '' / an unknown
    value. Shared with the image bank so its NSFW lane reuses the dataset's exact register
    text — 'explicit' only spells acts out when paired with an abliterated vision model,
    and the output cleaners still run, so it changes wording, never what binds."""
    return _VOCABULARY_INSTRUCTION.get((vocabulary or '').strip().lower())



# Public so the image bank's caption lane validates against — and appends — the
# SAME backends and preset texts as the dataset pass, rather than duplicating
# the tuples. Upstream keeps these next to CAPTION_VOCABULARIES; this fork's
# split moved the pass here, so the public names follow the code.


def caption_preset_instructions(vocabulary=None, length=None) -> str | None:
    """The combined preset block (vocabulary register, then length) for a run that
    has no per-dataset options to read — the image bank's per-run lane. None when
    neither is set, so a call without presets appends nothing at all."""
    parts = _caption_preset_parts(vocabulary, length)
    return '\n\n'.join(parts) if parts else None


def preview_caption(user_id, dataset_id, image_id, *, backend=None, ollama_model='',
                    vocabulary=None, length=None, instructions=None,
                    should_cancel=None) -> dict:
    """Caption ONE dataset image with a candidate config and return the text WITHOUT
    persisting it — the Caption Lab's ephemeral A/B probe. Reuses caption_paths(), so the
    engine/model/GPU serialization contract is identical to the batch pass.

    backend      : '' / None → global default; else one of _CAPTION_BACKENDS ('none' is
                   rejected here — a preview with captioning disabled makes no sense).
    vocabulary   : '' / None → the model's own wording; else an _CAPTION_VOCABULARIES
                   preset, appended as an instruction exactly like the dataset options.
    instructions : free extra instructions, appended after the vocabulary preset.
    should_cancel: polled by caption_paths at the image boundary (Ollama phase) so the
                   existing Stop path can abort a preview cleanly.

    Returns {caption, chars, duration_ms, cancelled}. Raises ValueError (bad image/config)
    → 400, RuntimeError (engine unavailable) → 409, GpuBusyError → 503 (via the route's
    vision window). Never writes to the DB or the filesystem."""
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    img = db.session.get(FaceDatasetImage, image_id)
    if not img or img.dataset_id != ds.id or not img.filename:
        raise ValueError('image not found')
    path = _img_path(img)
    if not os.path.isfile(path):
        raise ValueError('image file missing on disk')
    backend = (backend or '').strip().lower() or None
    if backend and backend not in _CAPTION_BACKENDS:
        raise ValueError(f'invalid captioning backend: {backend}')
    if backend == 'none':
        raise ValueError('captioning is disabled for this candidate')
    vocab = (vocabulary or '').strip().lower() or None
    if vocab and vocab not in _CAPTION_VOCABULARIES:
        raise ValueError(f'invalid caption vocabulary: {vocab}')
    size = (length or '').strip().lower() or None
    if size and size not in _CAPTION_LENGTHS:
        raise ValueError(f'invalid caption length: {size}')
    extra = _compose_preview_instructions(vocab, instructions, size)
    ollama_model = normalize_ollama_model_ref(
        ollama_model, allow_empty=True) or None
    started = time.perf_counter()
    out = caption_paths([path], backend=backend, ollama_model=ollama_model,
                        extra_instructions=extra, should_cancel=should_cancel)
    duration_ms = int((time.perf_counter() - started) * 1000)
    caption = (out.get(path) or '').strip()
    # A stop consumed before the (single) image ran leaves no caption — surface it so the
    # Lab card reads "cancelled" rather than a misleading empty result.
    cancelled = bool(not caption and should_cancel and should_cancel())
    return {'caption': caption, 'chars': len(caption),
            'duration_ms': duration_ms, 'cancelled': cancelled}


# --- Short-caption derivation (ai-toolkit dual long+short captioning) --------
# When a dataset opts into dual captions, ai-toolkit trains each image with BOTH the long
# and the short caption in the same step (short_and_long_captions doubles the batch — see
# BaseSDTrainProcess.process_general_training_batch in the installed toolkit). The short is
# DERIVED from the already-stored long via a text-only Ollama pass (no vision decode, no
# second model, no GPU-heavy image work), then run through the SAME kind omission the long
# went through so shortening can never reintroduce a banned identity/concept/aesthetic term.

_SHORTEN_BASE = (
    'Rewrite the following image caption as a much SHORTER caption: one concise sentence, '
    'or a few key comma-separated phrases, naming only the most salient clearly-visible '
    'elements. Do NOT add any detail that is not already present. Do NOT explain yourself '
    'or add commentary. Reply with ONLY the short caption.\n')


def _shorten_prompt(ds, long_caption) -> str:
    """Text-only shortening prompt whose kind rule MIRRORS the long-caption omission:
    character omits identity, concept omits the recurring element, style omits the look."""
    if is_style(ds):
        rule = ('Describe visible CONTENT only (subject, action, setting). Never name any '
                'aesthetic, medium, art style, or artist.\n')
    elif is_concept(ds):
        rule = (f'Never mention or describe this recurring element: '
                f'{(ds.concept_desc or "").strip()}. Keep it fully omitted.\n')
    else:
        rule = ("Never mention or describe the person's identity, face, or facial "
                'features.\n')
    return f'{_SHORTEN_BASE}{rule}\nCAPTION:\n{(long_caption or "").strip()}\n'


def _scrub_short_like_long(ds, text, mode) -> str:
    """Apply the SAME deterministic kind omission a long caption gets — reusing the
    existing scrubbers, none of which touch the GPU: style content-only strip, concept
    ban-list clause-scrub (describe=None → mechanical net only), character identity drop."""
    t = (text or '').strip().strip('"').strip()
    if not t:
        return ''
    if is_style(ds):
        return style_content_caption(ds, t)
    if is_concept(ds):
        leak_re = _concept_terms_re(_get_concept_terms(ds, describe=None))
        return _enforce_concept_omission(t, leak_re, b'', (ds.concept_desc or '').strip(),
                                         describe=None) or ''
    cleaner = drop_identity_tags if mode == 'booru' else drop_identity_sentences
    return cleaner(t, body=is_body_fidelity(ds)) or ''


def derive_short_captions(user_id, dataset_id, image_ids=None, force=False, mode=None,
                          token=None, generate=None) -> int:
    """Derive caption_short from each kept image's stored long caption (text-only Ollama,
    kind omission preserved). No-op unless the dataset has dual captions enabled.

    `force=False` fills only images that still lack a short; `force=True` overwrites (the
    re-caption path — a fresh long implies a fresh short). `mode` matches the long pass
    (booru for SDXL, else prose). `generate` is the text seam (injected in tests); None →
    the real generate_text_ollama with the batch keep-alive + one unload at the end.

    Best-effort per image: an empty/failed generation (or one scrubbed down to nothing)
    leaves the short as-is — a still-missing short degrades to the long caption at export.
    Returns the number of shorts written."""
    ds = get_dataset(user_id, dataset_id)
    if not ds or not dual_captions_enabled(ds):
        return 0
    ttype = (getattr(ds, 'train_type', None) or 'zimage').lower()
    mode = (mode or ('booru' if ttype == 'sdxl' else 'prose')).lower()
    q = FaceDatasetImage.query.filter_by(dataset_id=dataset_id, status='keep')
    if image_ids is not None:
        ids = [int(i) for i in image_ids
               if isinstance(i, (int, float, str)) and str(i).lstrip('-').isdigit()]
        if not ids:
            return 0
        q = q.filter(FaceDatasetImage.id.in_(ids))
    rows = [i for i in q.all() if (i.caption or '').strip()]
    if not force:
        rows = [i for i in rows if not (i.caption_short or '').strip()]
    if not rows:
        return 0
    if generate is None:
        from .vision_ollama import generate_text_ollama, unload_vision_model
        # Same model override as the long-caption pass so the short is derived by (and the
        # VRAM freed for) the model the dataset actually captions with.
        omodel = caption_options(ds).get('ollama_model') or None
        def gen(p):
            return generate_text_ollama(p, num_predict=400, model=omodel,
                                        keep_alive=_VISION_BATCH_KEEPALIVE)
        def _unload():
            return unload_vision_model(model=omodel)
    else:
        gen = generate
        def _unload():
            return None
    # When no caller owns an indicator (the /caption route runs shorts as a follow-up
    # pass), own one here so this loop is visible AND Stop-able like the long pass: the
    # kind matches (caption/recaption) so request_cancel finds it and the amber banner
    # names it. A caller-supplied token means the long pass still owns the indicator.
    own_token = None
    if token is None:
        own_token = dataset_activity.begin(dataset_id, 'recaption' if force else 'caption',
                                           total=len(rows),
                                           detail=f'Deriving {len(rows)} short caption(s)…')
        token = own_token
    n = 0
    try:
        for img in rows:
            if dataset_activity.cancel_requested(dataset_id):
                break   # graceful stop at an image boundary (see caption_images)
            dataset_activity.bump(token)
            short = _scrub_short_like_long(ds, gen(_shorten_prompt(ds, img.caption)), mode)
            if not short:
                continue
            img.caption_short = _cap_caption(short) or None
            db.session.commit()
            n += 1
    finally:
        _unload()
        if own_token is not None:
            dataset_activity.end(own_token)
    return n

# --- Borrow: face_dataset_service.py primitives -----------------------------
# MUST stay at the bottom of this file, same reason as in
# reference_photos_service.py and watermark_service.py: this module and
# face_dataset_service.py import names from each other, and whichever side is
# imported first must find the other fully defined by the time the reach-back
# import resolves.
from .face_dataset_service import (
    _guard_not_bank_export,
    get_dataset, caption_options, dual_captions_enabled, is_concept, is_style,
    is_body_fidelity, style_content_caption, _img_path, _cap_caption,
    _with_caption_instructions, _combined_caption_instructions,
    _CAPTION_BACKENDS, _CAPTION_VOCABULARIES, _VOCABULARY_INSTRUCTION,
    _CAPTION_LENGTHS, _caption_preset_parts,
    _writer, CAPTION_WRITER_JOYCAPTION, CAPTION_WRITER_OLLAMA,
    _CAPTION_INSTRUCTIONS_MAX, _VISION_BATCH_KEEPALIVE, CAPTION_VOCABULARIES,
    logger,
)


# Public alias, deliberately AFTER the borrow-back import: `_CAPTION_BACKENDS`
# is owned by face_dataset_service and only reaches this module at the bottom of
# the file, so a module-level alias any earlier would run before it is bound.
# Same trap as _backup_extra_ref_names' call-time default.
CAPTION_BACKENDS = _CAPTION_BACKENDS
