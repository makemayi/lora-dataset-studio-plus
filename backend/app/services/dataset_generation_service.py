"""Everything that produces NEW images for a dataset: the Klein and Krea 2
edit fan-outs, the API-engine fan-out (Nano Banana / ChatGPT / OpenRouter),
regenerate and face swap, and the whole "Upscale & improve" pass -- its
per-pass profile, its engine dispatch, the single-image and re-improve lanes,
and the bulk server job that drains them.

Split out of face_dataset_service.py (2026-08, Phase 5 of a multi-phase file
split) -- pure move, no behavior change.
"""
import os
import re
import threading
import time
import uuid

from PIL import Image

from ..extensions import db
from ..models import FaceDataset, FaceDatasetImage
from .. import config as cfg
from . import dataset_activity, trash
# Prompt/label helpers straight from face_variations: it has no dependency on
# this module, so there is no cycle to schedule around.
from .face_variations import (
    aspect_for_label, is_nsfw_label, prompt_by_label, wrap_variation,
    wrap_variation_klein, wrap_variation_krea, wrap_variation_minimax_h3,
    get_identity_prompt,
)


# --- Fan-out generation (Klein edit) ---------------------------------------
def _sync_generate_activity(dataset_id):
    """Reconcile the Klein 'generate' indicator with the dataset's live count of
    in-flight Klein jobs (pending rows that still carry a job_id and have no file
    yet). Klein completions arrive one-by-one on the job-queue monitor thread with
    only a job_id — no batch handle — so we track the honest pending COUNT rather
    than a per-batch job set (duplicated/cancelled completions would corrupt one).
    Called on enqueue, on each completion, and on cancel; the registry TTL is the
    last-resort net. API rows (job_id is NULL) are excluded — those batches own a
    separate begin()/end() 'generate' entry from _run_nanobanana_batch."""
    local = (FaceDatasetImage.query
             .filter_by(dataset_id=dataset_id, status='pending')
             .filter(FaceDatasetImage.filename.is_(None))
             .filter(FaceDatasetImage.job_id.isnot(None)))
    pending = local.count()
    # There are THREE local engines now. They queue on the same single GPU and
    # complete the same way, so the COUNT is shared; the label says what is
    # actually on it.
    #
    # This used to be a two-way test that claimed 'krea' only when every row was
    # Krea and said **'klein' for everything else** — so a MiniMax H3 run was
    # badged Klein. Reported 2026-08-09 as "the task is stuck": a face swap had
    # finished, two H3 jobs were still queued behind it (H3 takes minutes), and
    # a progress bar naming the engine that had already finished reads as a
    # hung batch rather than a running one. The old comment even said a wrong
    # badge is worse than a vague one; H3 shipped and made it wrong.
    #
    # `klein_model` carries the ENGINE ID for the non-Klein lanes and a model
    # filename (or NULL) for Klein, so map first and count distinct after. A
    # genuinely mixed queue gets the vague answer, on purpose.
    engine = 'klein'
    if pending:
        markers = {m for (m,) in local.with_entities(FaceDatasetImage.klein_model)}
        engines = {m if m in (KREA_ENGINE, MINIMAX_H3_ENGINE) else 'klein'
                   for m in markers}
        engine = engines.pop() if len(engines) == 1 else 'local'
    dataset_activity.sync_pending(dataset_id, 'generate', pending, engine=engine)


def generate_variations(user_id, dataset_id, variations, multiplier, klein_model=None,
                        lora_strength=None, generation_lora_preset=None):
    """For each (variation x multiplier), enqueue a Klein edit of the reference
    and create a pending FaceDatasetImage. Returns the created image ids.

    The row is committed BEFORE enqueuing (so an enqueue/commit failure can never
    leave an untracked orphan job); on enqueue failure the row is marked 'failed'
    and the error re-raised (already-enqueued variations keep their rows).

    `generation_lora_preset`: NAME of the generation-LoRA preset picked for
    this run (optional generation LoRAs, Idea by @waltm) — resolved from the
    CONFIG only (fail-closed: the request can't define files/strengths/order;
    an unknown name degrades to no extra LoRAs with a log). The preset's chain
    applies to EVERY variation of the run — picking the preset IS the intent,
    there is no automatic per-variation gating."""
    _guard_not_bank_export(dataset_id)
    try:
        from .klein_edit_helper import enqueue_klein_edit
    except ImportError:
        raise RuntimeError('ComfyUI is not configured')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('reference image required')
    # No model named by the caller → the DATASET's pick (None = auto, i.e. exactly
    # what generation resolved before the setting existed). An explicit request
    # value still wins: it is what the workspace picker sends, and a legacy browser
    # that still holds editPage_flux2KleinModel_v1 must keep generating with it.
    if not klein_model:
        klein_model = dataset_klein_model(ds)
    # Preflight the Klein model files BEFORE creating any rows: a missing model
    # then surfaces as one actionable "downloading, retry" 409 (route handler) —
    # not a dataset full of failed tiles, each doomed by a ComfyUI validation
    # error on a file that isn't there.
    from .klein_edit_helper import klein_missing_assets, KLEIN_REQUIRED, KleinModelsMissing
    _missing = klein_missing_assets()
    if any(a in _missing for a in KLEIN_REQUIRED):
        raise KleinModelsMissing(_missing)
    mult = max(1, int(multiplier))
    total = len(variations) * mult
    if total > MAX_FANOUT:
        raise ValueError(f'fan-out too large ({total} > {MAX_FANOUT})')
    # Anti-DoS: the fan-out is free (never debited) → cap pending in-flight
    # generations per dataset so one user can't monopolize the single GPU.
    in_flight = (FaceDatasetImage.query
                 .filter_by(dataset_id=dataset_id, status='pending')
                 .filter(FaceDatasetImage.filename.is_(None)).count())
    if in_flight + total > MAX_FANOUT:
        raise ValueError(f'too many generations in flight ({in_flight}), wait or cancel')
    # Extra identity refs (multi-references) : chaînées en ReferenceLatent natifs
    # côté Klein — mêmes fichiers que le chemin Nano Banana multi-réfs.
    extra_paths = [os.path.join(_dataset_dir(ds.id), fn) for fn in extra_ref_filenames(ds)]
    # Optional generation LoRAs: resolve the picked preset from the config ONCE
    # (fail-closed — unknown name -> [] with a log). Same chain for every job.
    from .klein_edit_helper import resolve_generation_lora_preset
    run_loras = resolve_generation_lora_preset(generation_lora_preset)
    ids = []
    # try/finally: advertise the live 'generate' indicator even if an enqueue
    # fails partway (the already-queued rows are still in flight). Each Klein job
    # completes asynchronously; _sync_generate_activity keeps the count honest and
    # link_completed_dataset_image clears it when the last one lands.
    try:
        for v in variations:
            for _ in range(mult):
                img = FaceDatasetImage(dataset_id=dataset_id, source='generated', status='pending',
                                       variation_label=v.get('label'), framing=v.get('framing'),
                                       variation_prompt=v['prompt'], klein_model=klein_model)
                db.session.add(img)
                db.session.commit()
                # Captured NOW, while the row certainly exists: ⏹ Stop
                # deletes exactly this shape (pending + no filename) and it
                # can land while the enqueue below is in flight.
                image_id = img.id
                # NSFW (flag explicite OU label du catalogue NSFW) : wrapper sans le
                # clamp SFW — chemin Klein local uniquement, les moteurs API sont
                # refusés en amont (route + generate_variations_nanobanana).
                nsfw = bool(v.get('nsfw')) or is_nsfw_label(v.get('label'))
                try:
                    job_id = enqueue_klein_edit(
                        user_id=str(user_id), source_filename=ds.ref_filename,
                        source_path=_ref_path(ds),
                        # Dataset suffix applied AT WRAP — the row above keeps the
                        # raw catalog prompt, so regenerate re-applies the CURRENT
                        # suffix exactly once (never a double application).
                        edit_prompt=wrap_variation_klein(
                            v['prompt'], nsfw=nsfw, framing=v.get('framing'),
                            suffix=dataset_prompt_suffix(ds, v.get('framing')),
                            subject_type=subject_type_of(ds),
                            # Picks this shot's concrete garment, like the Krea
                            # path — deterministic, so a regenerate reproduces it.
                            label=v.get('label') or ''),
                        klein_model=klein_model,
                        lora_strength=lora_strength, extra_ref_paths=extra_paths,
                        generation_loras=run_loras, sampler_steps=_generation_steps(),
                        base_lora_strength=_generation_base_lora_strength(),
                        extra_metadata={'is_dataset': True, 'dataset_id': dataset_id,
                                        'variation_label': v.get('label')})
                except Exception:
                    row = _live_image_row(image_id)
                    if row is not None:
                        row.status = 'failed'
                        db.session.commit()
                    raise
                row = _live_image_row(image_id)
                if row is None:
                    continue     # Stop removed it mid-enqueue; nothing to report
                row.job_id = job_id
                db.session.commit()
                ids.append(image_id)
    finally:
        _sync_generate_activity(dataset_id)
    return ids


def generate_variations_krea(user_id, dataset_id, variations, multiplier,
                            generation_lora_preset=None):
    """Krea 2 Identity Edit fan-out — the second LOCAL engine, same contract as
    `generate_variations` (Klein): one pending row committed BEFORE its job is
    enqueued, the whole batch preflighted up front, the created ids returned.

    Fewer knobs than the Klein path, but no longer none: Krea has no consistency
    LoRA, and its identity LoRA IS the pipeline. Stacking untested LoRAs on an
    edit model still degrades it — that caution was this lane's reason for having
    no LoRA input at all, and it is now the USER's call per run instead of ours:
    Krea 2 cannot render some registers a dataset needs, and a LoRA is the only
    lever that reaches them. `generation_lora_preset` NAMES a preset from config
    (absent = none, which is still the default and still the byte-identical
    graph). The other dial, `grounding_px`, remains a SETTING rather than a
    per-run argument, because it changes the meaning of every shot in the batch
    identically.

    The row stores the ENGINE ID in `klein_model`, like the API rows do, so the
    grid badge can say "Krea 2 Edit"; the base model itself is re-resolved
    deterministically at enqueue and at regenerate."""
    _guard_not_bank_export(dataset_id)
    from . import krea_edit_helper as keh
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('reference image required')
    # Assets AND custom nodes, before any row exists: a missing piece then
    # surfaces as one actionable 409 instead of a grid of silently-failing tiles.
    keh.preflight()
    # Resolved ONCE per run, not per variation: every cell of a run gets the same
    # always-on stack (that is what "always-on" means), and one config read is
    # enough. Unknown/blank name -> [] (fail-closed, see the resolver).
    run_loras = keh.resolve_generation_lora_preset(generation_lora_preset)
    mult = max(1, int(multiplier))
    total = len(variations) * mult
    if total > MAX_FANOUT:
        raise ValueError(f'fan-out too large ({total} > {MAX_FANOUT})')
    in_flight = (FaceDatasetImage.query
                 .filter_by(dataset_id=dataset_id, status='pending')
                 .filter(FaceDatasetImage.filename.is_(None)).count())
    if in_flight + total > MAX_FANOUT:
        raise ValueError(f'too many generations in flight ({in_flight}), wait or cancel')
    ids = []
    try:
        for v in variations:
            source_path = _krea_pose_source_path(ds, v.get('prompt') or '')
            for _ in range(mult):
                img = FaceDatasetImage(dataset_id=dataset_id, source='generated',
                                       status='pending', variation_label=v.get('label'),
                                       framing=v.get('framing'),
                                       variation_prompt=v['prompt'],
                                       klein_model=KREA_ENGINE)
                db.session.add(img)
                db.session.commit()
                # Captured NOW, while the row certainly exists: ⏹ Stop
                # deletes exactly this shape (pending + no filename) and it
                # can land while the enqueue below is in flight.
                image_id = img.id
                nsfw = bool(v.get('nsfw')) or is_nsfw_label(v.get('label'))
                try:
                    job_id = keh.enqueue_krea_edit(
                        user_id=str(user_id), source_filename=os.path.basename(source_path),
                        source_path=source_path, framing=v.get('framing'),
                        # Suffix applied AT WRAP, like Klein: the row keeps the raw
                        # catalog prompt so a regenerate re-applies the CURRENT
                        # suffix exactly once. The label rides along because it
                        # picks this shot's outfit deterministically.
                        edit_prompt=wrap_variation_krea(
                            v['prompt'], nsfw=nsfw, framing=v.get('framing'),
                            suffix=dataset_prompt_suffix(ds, v.get('framing')),
                            subject_type=subject_type_of(ds),
                            label=v.get('label') or ''),
                        # Krea v1.2 fit geometry accepts the catalog canvas even
                        # when it differs from the dataset reference.
                        aspect_ratio=aspect_for_label(v.get('label'), v.get('framing')),
                        generation_loras=run_loras,
                        extra_metadata={'is_dataset': True, 'dataset_id': dataset_id,
                                        'variation_label': v.get('label')})
                except Exception:
                    row = _live_image_row(image_id)
                    if row is not None:
                        row.status = 'failed'
                        db.session.commit()
                    raise
                row = _live_image_row(image_id)
                if row is None:
                    continue     # Stop removed it mid-enqueue; nothing to report
                row.job_id = job_id
                db.session.commit()
                ids.append(image_id)
    finally:
        _sync_generate_activity(dataset_id)
    return ids


def generate_variations_minimax_h3(user_id, dataset_id, variations, multiplier):
    """MiniMax H3 fan-out — the third LOCAL engine, same contract as the Klein and
    Krea lanes: one pending row committed BEFORE its job is enqueued, the whole
    batch preflighted up front, the created ids returned.

    No LoRA input at all, unlike Krea: H3 is a video model driven by a reference
    photo, and nothing has been measured about stacking LoRAs on it. An
    unmeasured knob is worse than a missing one.

    THE LOOP NESTING IS LOAD-BEARING — do not "improve" it by interleaving for
    variety. `for v in variations: for _ in range(mult)` puts every copy of one
    catalog card on the queue CONSECUTIVELY, and H3's text encode depends only on
    the prompt (the seed reaches the sampler through RandomNoise alone), so
    ComfyUI serves the ~40 s encode from cache for every copy after the first.
    Measured on an RTX 3090: 78 s for a new card, 37 s for another copy of it.
    Interleaving would make every image pay the encode — 130 minutes instead of
    75 for 100 images over 20 cards, with nothing visible in the UI to explain
    it. `test_minimax_h3_lane` pins the ordering for that reason.

    The row stores the ENGINE ID in `klein_model`, like the other lanes, so the
    grid badge can say "MiniMax H3"; the models themselves are re-resolved
    deterministically at enqueue and at regenerate."""
    _guard_not_bank_export(dataset_id)
    from . import minimax_h3_helper as mh
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('reference image required')
    # Assets AND the frame-selector pack, before any row exists: a missing piece
    # then surfaces as one actionable 409 instead of a grid of failing tiles.
    mh.preflight()
    mult = max(1, int(multiplier))
    total = len(variations) * mult
    if total > MAX_FANOUT:
        raise ValueError(f'fan-out too large ({total} > {MAX_FANOUT})')
    in_flight = (FaceDatasetImage.query
                 .filter_by(dataset_id=dataset_id, status='pending')
                 .filter(FaceDatasetImage.filename.is_(None)).count())
    if in_flight + total > MAX_FANOUT:
        raise ValueError(f'too many generations in flight ({in_flight}), wait or cancel')
    ids = []
    try:
        for v in variations:
            # The angle reference slots are engine-agnostic: a shot whose prompt
            # asks for a side view is fed the matching pose photo instead of the
            # front one, exactly as the Krea lane does it.
            source_path = _krea_pose_source_path(ds, v.get('prompt') or '')
            for _ in range(mult):
                img = FaceDatasetImage(dataset_id=dataset_id, source='generated',
                                       status='pending', variation_label=v.get('label'),
                                       framing=v.get('framing'),
                                       variation_prompt=v['prompt'],
                                       klein_model=MINIMAX_H3_ENGINE)
                db.session.add(img)
                db.session.commit()
                # Captured NOW, while the row certainly exists: ⏹ Stop
                # deletes exactly this shape (pending + no filename) and it
                # can land while the enqueue below is in flight.
                image_id = img.id
                nsfw = bool(v.get('nsfw')) or is_nsfw_label(v.get('label'))
                try:
                    job_id = mh.enqueue_minimax_h3(
                        user_id=str(user_id), source_filename=os.path.basename(source_path),
                        source_path=source_path, framing=v.get('framing'),
                        # Suffix applied AT WRAP, like the other two lanes: the
                        # row keeps the raw catalog prompt so a regenerate
                        # re-applies the CURRENT suffix exactly once.
                        edit_prompt=wrap_variation_minimax_h3(
                            v['prompt'], nsfw=nsfw, framing=v.get('framing'),
                            suffix=dataset_prompt_suffix(ds, v.get('framing')),
                            subject_type=subject_type_of(ds),
                            label=v.get('label') or ''),
                        aspect_ratio=aspect_for_label(v.get('label'), v.get('framing')),
                        extra_metadata={'is_dataset': True, 'dataset_id': dataset_id,
                                        'variation_label': v.get('label')})
                except Exception:
                    row = _live_image_row(image_id)
                    if row is not None:
                        row.status = 'failed'
                        db.session.commit()
                    raise
                row = _live_image_row(image_id)
                if row is None:
                    continue     # Stop removed it mid-enqueue; nothing to report
                row.job_id = job_id
                db.session.commit()
                ids.append(image_id)
    finally:
        _sync_generate_activity(dataset_id)
    return ids


# --- The ✨ Upscale & improve profile, read ONCE per pass ----------------------
# Every knob below lives in config and is user-editable, so it must be read at
# ENQUEUE time, never frozen into the candidate row: re-running the pass (🔄 on
# an improved tile) has to pick up whatever the user has since changed. These
# four helpers are the single source of truth shared by the first pass
# (improve_existing_image) and the re-run (reimprove_image) — two copies would
# drift, and a re-run that used yesterday's settings is exactly the bug.
def _improve_prompt() -> str:
    """The improvement instruction, editable in Settings ▸ identity_prompts.
    klein_improve, and switchable OFF entirely — disabled applies NO prompt
    (pure upscale)."""
    if cfg.get('identity_prompts.klein_improve_enabled', True):
        return get_identity_prompt('klein_improve')
    return ''


_IMPROVE_LABELS = {
    'klein': 'Klein upscale & improve',   # NEVER change: stored in user databases
    'seedvr2': 'SeedVR2 upscale',
}


def _improve_candidate_label(source, engine='klein') -> str:
    """Label of the candidate produced from ``source`` (its parent image).

    Names the engine that ACTUALLY ran: a SeedVR2 result labelled "Klein upscale
    & improve" tells the user the one thing they chose this pass to avoid."""
    base_label = _IMPROVE_LABELS.get(engine, _IMPROVE_LABELS['klein'])
    source_label = (source.variation_label or '').strip()
    return (f'{base_label} · {source_label}' if source_label else base_label)[:120]


def _improve_extra_metadata(source, label, engine='klein') -> dict:
    return {
        'is_dataset': True,
        'dataset_id': source.dataset_id,
        'variation_label': label,
        'derivation_kind': KLEIN_IMAGE_IMPROVE,
        'parent_image_id': source.id,
        'source_image_id': source.id,
        'action': 'upscale_improve',
        # WHICH engine produced this candidate. Additive: `derivation_kind` and
        # `action` keep the values every existing row and every reader already
        # carries (they are stored, and stored ids are never renamed here), so a
        # SeedVR2 result curates, undoes and re-improves exactly like a Klein one.
        'improve_engine': engine,
    }


# --- Which engine runs the ✨ improve pass ------------------------------------
# Two passes, one lane. Klein REWRITES (a diffusion edit that re-renders skin
# and micro-detail from a prompt: it fixes a soft photo and it changes it);
# SeedVR2 RESTORES (one-step diffusion super-resolution that leaves the content
# where it was). Which one you want depends on whether the frame's exact look is
# the thing you are training on — so it is a choice, not a default we can pick
# for everyone. Requested in issue #32 by SurpassHR, whose complaint was exactly
# that Klein "tends to change the detail and color of the original image".
IMPROVE_ENGINES = ('klein', 'seedvr2')


def resolve_improve_engine(requested=None):
    """The engine an improve request will run on: the explicit pick when it names
    a known engine, else the `improve.engine` setting, else Klein.

    Fail-SAFE rather than fail-closed: an unknown name falls back instead of
    raising, because a stale tab must degrade to the historical behaviour rather
    than refuse a batch. Klein is the fallback because it is what every improve
    did before this setting existed."""
    for candidate in (requested, cfg.get('improve.engine')):
        name = str(candidate or '').strip().lower()
        if name in IMPROVE_ENGINES:
            return name
        if name:
            logger.warning('unknown improve engine %r — falling back to klein', candidate)
    return 'klein'


def _improve_preflight(engine):
    """Raise the engine's structured missing-assets exception when it cannot run.

    Called BEFORE any candidate row is created, so a missing model surfaces once
    per batch instead of once per image. Each engine keeps its own exception type
    (KleinModelsMissing / KleinNodesMissing / SeedVR2ModelsMissing) — the routes
    already turn each into its own actionable 409 body, and collapsing them into
    one would lose the "install the node pack" vs "place the weights" distinction
    that makes those bodies useful."""
    if engine == 'seedvr2':
        from . import seedvr2_helper
        seedvr2_helper.preflight()
        return
    # 'klein' engine id kept as-is (stored in derivation_kind/improve_engine on
    # existing rows — never renamed, see CLAUDE.md), but as of the Krea2 Ostris
    # Edit + SeedVR2 workflow swap it no longer runs Flux.2 Klein 9B for THIS
    # lane; klein_edit_helper stays the engine for every other Klein call site
    # (regenerate, variation restaging, small-image rescue).
    from . import krea_hq_helper as khh
    khh.preflight()


def _enqueue_improve(engine, *, user_id, source, source_path, prompt, label,
                     dataset=None, extra_metadata=None):
    """Hand ONE improve off to the chosen engine and return its job id.

    SeedVR2 needs no prompt (there is no prompt in a restoration); the
    'klein' engine's Krea2+SeedVR2 pipeline reuses the SAME editable
    klein_improve `prompt` for its Krea2 edit stage.

    `extra_metadata` overrides what the finished job is linked back TO. Default
    (None) keeps the dataset-image contract every existing caller relies on. The
    ◉ Canvas improve passes its own because its source is a `LoraTestImage`, not
    a `FaceDatasetImage`: the two live in different tables with independent id
    spaces, and the completion callback is chosen by this metadata. The engine
    dispatch below stays the single place that knows Krea HQ from SeedVR2 — that
    is the whole point of routing the second lane through here rather than
    growing a parallel copy of it.

    `dataset` is accepted for that same caller symmetry; this fork's Krea2-HQ
    improve resolves its own profile from the shipped workflow, so nothing here
    reads it (upstream's Klein improve derived per-dataset LoRA strength and
    steps from it)."""
    meta = (dict(extra_metadata) if extra_metadata is not None
            else _improve_extra_metadata(source, label, engine=engine))
    if engine == 'seedvr2':
        from . import seedvr2_helper
        return seedvr2_helper.enqueue_seedvr2_upscale(
            user_id=str(user_id), source_filename=source.filename,
            source_path=source_path, extra_metadata=meta)
    from . import krea_hq_helper as khh
    return khh.enqueue_krea_hq_improve(
        user_id=str(user_id), source_filename=source.filename,
        source_path=source_path, edit_prompt=prompt, extra_metadata=meta)


def improve_existing_image(user_id, image_id, engine=None):
    """Serialize one source's improve request, including the queue hand-off."""
    image = _owned_image(user_id, image_id)
    if image is None:
        return None
    _guard_not_bank_export(image.dataset_id)
    lock = _IMAGE_IMPROVE_LOCKS[hash((str(user_id), image_id))
                                % len(_IMAGE_IMPROVE_LOCKS)]
    with lock:
        return _improve_existing_image_locked(user_id, image_id, engine=engine)


def _improve_existing_image_locked(user_id, image_id, engine=None):
    """Queue one non-destructive upscale/improvement of an existing image.

    The source row and file are deliberately never modified.  The result is a
    regular generated dataset image linked back to the source only for
    provenance; unlike the small-scrape review pair it remains compatible with
    the ordinary keep/reject/delete actions.

    Returns ``{'candidate_id', 'job_id'}``, ``None`` for an image not owned by
    ``user_id``, and returns the already-active candidate idempotently when the
    same source is clicked twice.
    """
    img = _owned_image(user_id, image_id)
    if not img:
        return None
    _guard_not_bank_export(img.dataset_id)
    if img.derivation_kind in _SMALL_IMAGE_DERIVATIONS:
        raise ValueError(
            'resolve the small-image rescue pair before improving either image')
    if img.derivation_kind == KLEIN_IMAGE_IMPROVE:
        raise ValueError('an upscale & improve candidate cannot be improved again')
    if not img.filename:
        raise ValueError('image file required')
    source_path = _img_path(img)
    if not os.path.isfile(source_path):
        raise ValueError('image file missing')

    # A completed Klein job remains status=pending until the user curates it, so
    # both an in-flight candidate (no filename yet) and an unreviewed result are
    # active.  Repeated clicks return that same job instead of consuming the GPU
    # or producing visually indistinguishable duplicates.
    active = (FaceDatasetImage.query
              .filter_by(dataset_id=img.dataset_id, parent_image_id=img.id,
                         derivation_kind=KLEIN_IMAGE_IMPROVE, status='pending')
              .order_by(FaceDatasetImage.id.desc()).first())
    if active:
        if active.job_id:
            return {'candidate_id': active.id, 'job_id': active.job_id}
        # This tiny state exists only between the row commit and queue enqueue.
        # Refuse a concurrent click rather than creating a second candidate.
        raise RuntimeError('this image improvement is already being queued')

    engine = resolve_improve_engine(engine)
    _improve_preflight(engine)

    in_flight = (FaceDatasetImage.query
                 .filter_by(dataset_id=img.dataset_id, status='pending')
                 .filter(FaceDatasetImage.filename.is_(None)).count())
    if in_flight + 1 > MAX_FANOUT:
        raise ValueError(
            f'too many generations in flight ({in_flight}), wait or cancel')

    prompt = _improve_prompt()
    # What the tile SHOWS as the prompt behind this candidate. A SeedVR2 run has
    # no prompt at all — it is a restoration — so storing the Klein improve
    # prompt on one would put a sentence on screen that had no effect on the
    # image. The honest value is the pass that ran.
    stored_prompt = (prompt[:500] if engine == 'klein'
                     else 'SeedVR2 upscale (no prompt — restoration pass)')
    label = _improve_candidate_label(img, engine)
    candidate = FaceDatasetImage(
        dataset_id=img.dataset_id, source='generated', status='pending',
        parent_image_id=img.id, derivation_kind=KLEIN_IMAGE_IMPROVE,
        # The stamp travels with the sentence it describes: a candidate that
        # inherits a hand-written caption inherits the protection on it, or the
        # first forced pass would rewrite the words on the copy while sparing them
        # on the original.
        framing=img.framing, caption=img.caption,
        caption_origin=img.caption_origin,
        variation_label=label, variation_prompt=stored_prompt,
        # The generated candidate remains derived from the credited source.
        # Revalidate before copying so a malformed legacy row cannot surface.
        source_metadata=_source_metadata_storage(img.source_metadata),
    )
    db.session.add(candidate)
    db.session.commit()
    # Captured while both rows certainly exist: the commit above expires them,
    # and ⏹ Stop deletes exactly this candidate's shape (pending, no filename)
    # — the enqueue below is the window, same as the variation paths.
    candidate_id = candidate.id
    dataset_id_of_source = img.dataset_id

    try:
        job_id = _enqueue_improve(
            engine, user_id=user_id, source=img, source_path=source_path,
            prompt=prompt, label=label,
            dataset=get_dataset(user_id, dataset_id_of_source))
    except Exception:
        # No broken tile: the original is still untouched and the user can retry
        # as soon as the queue/ComfyUI issue is fixed. Nothing to remove if Stop
        # already removed it — and trying would raise from inside this `except`,
        # replacing the real enqueue error with a database one.
        row = _live_image_row(candidate_id)
        if row is not None:
            db.session.delete(row)
            db.session.commit()
        raise

    row = _live_image_row(candidate_id)
    if row is None:
        # Stop removed the candidate mid-enqueue. Reporting its id would have the
        # tile poll for a generation that can never arrive.
        _sync_generate_activity(dataset_id_of_source)
        return None
    row.job_id = job_id
    db.session.commit()
    _sync_generate_activity(dataset_id_of_source)
    return {'candidate_id': candidate_id, 'job_id': job_id}


# The three ways a re-run can be impossible, worded as the user reads them. The
# tile mirrors them (frontend/src/components/dataset/improveRerun.js) so the
# button explains itself BEFORE the click rather than through a 400 after it.
REIMPROVE_PARENT_GONE = ('the source image this improvement came from was deleted '
                         '— nothing left to re-improve from')
REIMPROVE_SOURCE_FILE_GONE = ('the source image file is missing on disk '
                              '— nothing left to re-improve from')
REIMPROVE_IN_FLIGHT = 'this improvement is still generating'
REIMPROVE_STATE_CHANGED = ('this improvement changed while it was being re-queued '
                           '— review it and try again')


def reimprove_image(user_id, image_id):
    """Re-run the ✨ Upscale & improve pass that produced ``image_id``.

    The generic regenerate route is deliberately CLOSED to these rows: it starts
    from the dataset's reference photo and the catalog prompt, so on an improved
    tile it would quietly produce an unrelated variation. The right gesture is
    this one — run the improve pass again, from the SAME parent image, with the
    settings as they are TODAY (klein.improve_* + the klein_improve instruction
    are user-editable, and tuning them is the whole reason to re-run).

    Replaces IN PLACE, exactly like regenerate_image: same row id, same
    parent/derivation links, the previous result goes to the Trash once the new
    job is safely queued. A second candidate next to the first would break the
    one-live-improvement-per-source invariant that improve_existing_image and
    bulk_improve_eligible_ids already enforce.

    Returns ``{'candidate_id', 'job_id'}``, or None when the image is not owned
    by ``user_id``. Raises ValueError (-> 400) when the row is not an improvement
    or its parent is gone, RuntimeError (-> 409) while the pass is still running.
    """
    img = _owned_image(user_id, image_id)
    if not img:
        return None
    _guard_not_bank_export(img.dataset_id)
    if img.derivation_kind != KLEIN_IMAGE_IMPROVE:
        raise ValueError('only an upscale & improve result can be re-improved')
    # Take the same stripe as a first-pass improve OF THE PARENT: the two paths
    # compete for the same "one live candidate per source" slot.
    lock = _IMAGE_IMPROVE_LOCKS[hash((str(user_id), img.parent_image_id))
                                % len(_IMAGE_IMPROVE_LOCKS)]
    with lock:
        return _reimprove_image_locked(user_id, image_id)


def _reimprove_image_locked(user_id, image_id):
    img = _owned_image(user_id, image_id)
    if not img:
        return None
    if img.derivation_kind != KLEIN_IMAGE_IMPROVE:
        raise ValueError('only an upscale & improve result can be re-improved')
    if img.status == 'pending' and not img.filename:
        raise RuntimeError(REIMPROVE_IN_FLIGHT)

    # The parent is what this pass runs on. It carries no ForeignKey (legacy
    # databases), so a deleted source leaves a dangling id — check the row, not
    # just the column.
    parent = (FaceDatasetImage.query
              .filter_by(id=img.parent_image_id, dataset_id=img.dataset_id).first()
              if img.parent_image_id else None)
    if not parent or not parent.filename:
        raise ValueError(REIMPROVE_PARENT_GONE)
    source_path = _img_path(parent)
    if not os.path.isfile(source_path):
        raise ValueError(REIMPROVE_SOURCE_FILE_GONE)

    # A re-run uses the CURRENTLY selected engine, not the one that produced the
    # row: "re-improve" means "try again with what I have set now", and someone
    # who switched to SeedVR2 precisely because the Klein result changed too much
    # would otherwise get the same Klein result back.
    engine = resolve_improve_engine()
    _improve_preflight(engine)

    in_flight = (FaceDatasetImage.query
                 .filter_by(dataset_id=img.dataset_id, status='pending')
                 .filter(FaceDatasetImage.filename.is_(None)).count())
    if in_flight + 1 > MAX_FANOUT:
        raise ValueError(
            f'too many generations in flight ({in_flight}), wait or cancel')

    prompt = _improve_prompt()
    label = _improve_candidate_label(parent)

    # Enqueue BEFORE touching the row (regenerate_image's ordering): a ComfyUI
    # refusal must leave the current result on screen, not a broken tile.
    from ..job_queue import queue_manager
    old_state = {field: getattr(img, field) for field in (
        'filename', 'caption', 'status', 'fail_reason', 'fail_kind', 'job_id',
        'variation_label', 'variation_prompt', 'framing',
        'watermark_state', 'watermark_bbox', 'watermark_regions')}
    expected_transition_caption = (old_state['caption']
                                   if old_state['caption'] else parent.caption)
    old_path = _img_path(img) if img.filename else None
    job_id = _enqueue_improve(
        engine, user_id=user_id, source=parent, source_path=source_path,
        prompt=prompt, label=label)

    try:
        # Do this while the candidate is still Keep.  The CAS observes both
        # rows in the database, so an intervening parent reject/failed decision
        # is never overwritten by this re-run's temporary fallback.
        parent_rekept = _rekeep_pending_parent_for_reimprove(img)
        if not _transition_reimprove_candidate(
                img, old_state, parent, label, prompt, job_id,
                expected_transition_caption):
            # The status/file/job snapshot changed after enqueue. Rolling back
            # also undoes a just-applied parent fallback; the except path below
            # cancels this unlinked job and maps the race to a 409.
            raise RuntimeError(REIMPROVE_STATE_CHANGED)
        db.session.commit()
    except Exception:
        db.session.rollback()
        try:
            queue_manager.cancel_job(job_id, str(user_id), 'image')
        except Exception:
            logger.exception('reimprove: failed to cancel unlinked job %s', job_id)
        raise

    # The DB no longer references the old file. If Trash itself fails, restore
    # the exact previous row state and cancel the job we just queued.
    try:
        if old_path and os.path.exists(old_path):
            trash.send_to_trash(
                old_path, context=f'dataset-{img.dataset_id}-reimprove-{img.id}')
    except Exception:
        try:
            # Do not restore the old row over an unresolved new prompt: its
            # eventual callback still owns `job_id` and would otherwise write
            # into the restored candidate.
            if not queue_manager.cancel_job(job_id, str(user_id), 'image', commit=False):
                raise RuntimeError(
                    'The replacement generation still has unconfirmed ComfyUI work.')
            restored_candidate = _restore_reimprove_candidate_after_trash_failure(
                img, old_state, job_id, expected_transition_caption)
            if parent_rekept and restored_candidate:
                # Candidate first: if a user changed it during the Trash call,
                # its CAS fails and the fallback parent remains Keep instead of
                # overwriting that newer decision.
                _undo_rekeep_parent_after_reimprove_trash_failure(img, old_state)
            db.session.commit()
        except Exception:
            db.session.rollback()
            logger.exception('reimprove: failed to restore row %s after Trash error',
                             image_id)
        raise

    _sync_generate_activity(img.dataset_id)
    return {'candidate_id': img.id, 'job_id': job_id}


# --- Bulk Klein upscale & improve: a SERVER job --------------------------------
# The ✨ Improve button used to loop in the BROWSER, one request per image. On a
# 250-image selection that produced two bugs with a single root cause — the batch
# only existed in the tab:
#   * everything past MAX_FANOUT was REFUSED. That cap is a CONCURRENCY limit
#     ("how many generations may be in flight at once"), and a client loop that
#     keeps pushing simply walks into it: 60 queued, 190 counted as failures.
#   * ⏹ Stop was powerless. cancel_pending did its job (rows cancelled, ComfyUI
#     prompts interrupted) and the tab immediately re-queued the next 60. Closing
#     the tab killed whatever was left.
# So the batch runs server-side now: one background thread per dataset, advertised
# through dataset_activity (kind 'improve') so the progress SURVIVES a reload, and
# draining the selection in WAVES — it waits for a slot to free instead of hitting
# the wall — with a cooperative stop checked at every image boundary.
IMPROVE_SLOT_POLL_SECONDS = 2.0
# Give up (and say so) if no slot frees for this long. A ComfyUI that died mid-batch
# would otherwise leave the thread polling a count that never drops, and the dataset
# stuck behind an "in progress" indicator until the registry TTL expires.
IMPROVE_SLOT_TIMEOUT_SECONDS = 15 * 60
# Chunk the id lookup: a selection is user-sized and SQLite caps bound parameters.
_IMPROVE_ID_CHUNK = 400


def _improve_in_flight(dataset_id):
    """Live count of generations in flight on ``dataset_id`` — the very number
    improve_existing_image checks against MAX_FANOUT. Ends the worker thread's read
    transaction first (a rollback on a clean session is a no-op) so each poll sees
    the rows COMMITTED by the job-queue monitor thread rather than a stale snapshot;
    without it the count would never drop and the batch would stall forever."""
    db.session.rollback()
    return (FaceDatasetImage.query
            .filter_by(dataset_id=dataset_id, status='pending')
            .filter(FaceDatasetImage.filename.is_(None)).count())


def bulk_improve_eligible_ids(user_id, dataset_id, image_ids):
    """The subset of ``image_ids`` this dataset can actually improve, in selection
    order and de-duplicated. Mirrors the client-side partition
    (frontend/src/utils/kleinBulkImprove.js) so the total the job advertises is the
    number it will really work on — a batch that announced 250 and refused 40 of
    them one by one is exactly the dishonesty this rewrite removes."""
    wanted, seen = [], set()
    for raw in image_ids or []:
        try:
            image_id = int(raw)
        except (TypeError, ValueError):
            continue
        if image_id not in seen:
            seen.add(image_id)
            wanted.append(image_id)
    if not wanted:
        return []
    rows = {}
    for start in range(0, len(wanted), _IMPROVE_ID_CHUNK):
        chunk = wanted[start:start + _IMPROVE_ID_CHUNK]
        for row in (FaceDatasetImage.query
                    .filter(FaceDatasetImage.dataset_id == dataset_id,
                            FaceDatasetImage.id.in_(chunk)).all()):
            rows[row.id] = row
    # Sources whose improvement is already pending review (or still generating):
    # re-improving them would just make an indistinguishable duplicate.
    busy_parents = {row.parent_image_id for row in (
        FaceDatasetImage.query
        .filter_by(dataset_id=dataset_id, derivation_kind=KLEIN_IMAGE_IMPROVE,
                   status='pending').all())}
    eligible = []
    for image_id in wanted:
        img = rows.get(image_id)
        if not img or not img.filename:
            continue
        if img.derivation_kind in _SMALL_IMAGE_DERIVATIONS:
            continue
        if img.derivation_kind == KLEIN_IMAGE_IMPROVE:
            continue
        if image_id in busy_parents:
            continue
        eligible.append(image_id)
    return eligible


def start_bulk_improve(app, user_id, dataset_id, image_ids, engine=None):
    """Start the server-side ✨ Upscale & improve batch over ``image_ids``.

    Returns ``{'queued', 'skipped', 'engine'}`` — how many images the job will
    process, how many of the selection were not eligible, and which engine ran
    (the caller echoes it so the toast can name it). Raises ValueError (-> 400) on
    an unknown dataset / an empty eligible set, RuntimeError (-> 409) when a batch
    is already running, and the engine's missing-assets exceptions (-> structured
    409) so a missing model surfaces ONCE instead of once per image."""
    _guard_not_bank_export(dataset_id)
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if dataset_activity.running(dataset_id, dataset_activity.IMPROVE_KINDS):
        raise RuntimeError('an improvement batch is already running on this dataset')
    engine = resolve_improve_engine(engine)
    _improve_preflight(engine)
    eligible = bulk_improve_eligible_ids(user_id, dataset_id, image_ids)
    if not eligible:
        raise ValueError('no selected image is eligible for improvement')
    skipped = max(0, len(set(image_ids or [])) - len(eligible))
    total = len(eligible)
    token = dataset_activity.begin(dataset_id, 'improve', total=total,
                                   detail=f'Queuing improvements… 0/{total}',
                                   engine=engine)

    def _run():
        try:
            with app.app_context():
                _drain_improve_queue(user_id, dataset_id, eligible, token,
                                     engine=engine)
        except Exception:   # noqa: BLE001 — a background crash must not strand the indicator
            logger.exception('bulk improve batch failed on dataset %s', dataset_id)
        finally:
            dataset_activity.end(token)
            dataset_activity.clear_cancel(dataset_id, dataset_activity.IMPROVE_KINDS)

    # Under TESTING the job runs INLINE (same rule as bank_jobs): the suite uses a
    # per-connection in-memory sqlite, so a real worker thread would open a fresh,
    # EMPTY database.
    if app.config.get('TESTING'):
        _run()
    else:
        threading.Thread(target=_run, daemon=True,
                         name=f'ds-{dataset_id}-improve').start()
    return {'queued': total, 'skipped': skipped, 'engine': engine}


def _drain_improve_queue(user_id, dataset_id, image_ids, token, sleep=time.sleep,
                         engine=None):
    """Queue one improvement per id, in WAVES that respect the MAX_FANOUT
    concurrency cap: when the dataset already has that many generations in flight
    the worker WAITS for a slot (the count drops as ComfyUI writes the files) rather
    than firing a request doomed to be refused. Stops at the next image boundary
    when ⏹ Stop arms the flag. Returns a summary dict (also used by the tests)."""
    total = len(image_ids)
    queued = failed = 0
    waited = 0.0
    stopped = stalled = False

    def _stop_requested():
        return dataset_activity.cancel_requested(dataset_id,
                                                 dataset_activity.IMPROVE_KINDS)

    for index, image_id in enumerate(image_ids):
        while not stopped and not stalled and _improve_in_flight(dataset_id) + 1 > MAX_FANOUT:
            if _stop_requested():
                stopped = True
            elif waited >= IMPROVE_SLOT_TIMEOUT_SECONDS:
                stalled = True
            else:
                dataset_activity.progress(
                    token,
                    detail=f'Queuing improvements… {queued}/{total} — waiting for a '
                           f'free generation slot ({total - index} left)')
                sleep(IMPROVE_SLOT_POLL_SECONDS)
                waited += IMPROVE_SLOT_POLL_SECONDS
        if stalled:
            break
        if stopped or _stop_requested():
            stopped = True
            break
        waited = 0.0
        try:
            improve_existing_image(user_id, image_id, engine=engine)
            queued += 1
        except Exception as exc:   # noqa: BLE001 — one refusal never sinks the batch
            failed += 1
            logger.warning('bulk improve: image %s could not be queued (%s)',
                           image_id, exc)
        dataset_activity.bump(token)
        dataset_activity.progress(
            token, detail=f'Queuing improvements… {queued}/{total}')
    return {'total': total, 'queued': queued, 'failed': failed,
            'stopped': stopped, 'stalled': stalled,
            'remaining': total - queued - failed}


def regenerate_image(user_id, image_id, lora_strength=None, prompt=None, app=None,
                     engine=None, klein_model=None, generation_lora_preset=None):
    """Re-enqueue a single generated variation IN PLACE (same row id): cancel any
    in-flight job, drop the old file, reset the row to pending with the new
    job_id. Returns the new job_id, or None if the image is not owned / not a
    generated variation. Raises ValueError if the dataset has no reference or
    the variation prompt can't be recovered.

    `prompt` (optional) is the user-EDITED core creative prompt from the tile's
    ✏️ bubble. When given it REPLACES and is PERSISTED into `variation_prompt`
    (so a later plain regenerate / reject-regenerate reuses the edit), then feeds
    the identity-guard wrapper like any catalog prompt — the face lock is still
    applied on top, the user only steers the creative half. Empty/None = the
    current behaviour (recover the prompt from the row or the label).

    `engine` (optional, one of ``KNOWN_ENGINES``) is an EXPLICIT caller
    override. The ordinary workspace Retry omits it, so it reuses the engine
    recorded on the row; callers that deliberately pass one can still move a
    tile to another lane. Exception:
    an NSFW-labelled tile always stays on the local Klein path (fail-closed —
    NSFW never goes to third-party APIs, mirroring the batch generate rule).
    `klein_model` (optional) is the workspace's Klein model pick, used when a
    row born on an API engine switches to Klein (its klein_model column holds
    an engine TAG, not a real model file).
    `generation_lora_preset` (optional): NAME of the generation-LoRA preset
    picked in the workspace (Idea by @waltm). Both local engines resolve it —
    Klein and Krea each from their OWN config list (`klein.generation_lora_presets`
    / `krea.generation_lora_presets`), so the same name can mean two different
    chains depending on which engine `target` resolves to below — resolved from
    the CONFIG only (fail-closed; unknown name degrades to no extra LoRAs)."""
    img = _owned_image(user_id, image_id)
    if not img or img.source != 'generated':
        return None
    _guard_not_bank_export(img.dataset_id)
    if img.derivation_kind == KLEIN_SMALL_IMAGE:
        raise ValueError('small-image rescue candidates cannot be regenerated; re-import the source')
    if img.derivation_kind == KLEIN_IMAGE_IMPROVE:
        raise ValueError('upscale & improve candidates cannot be regenerated from the dataset reference')
    ds = db.session.get(FaceDataset, img.dataset_id)
    if not ds.ref_filename:
        raise ValueError('reference image required')
    edited = (prompt or '').strip()
    stored_prompt = edited[:500] if edited else img.variation_prompt
    prompt = stored_prompt or prompt_by_label(img.variation_label or '')
    if prompt is None:
        raise ValueError('variation prompt unknown')
    requested = (engine or '').strip() or None
    if requested is not None and requested not in KNOWN_ENGINES:
        raise ValueError(f'unknown engine: {requested}')
    # A row remembers its origin through `klein_model`: an engine TAG for the API
    # rows and for Krea, a real model FILE for Klein. Anything that isn't a known
    # tag is therefore a Klein row.
    origin = img.klein_model if img.klein_model in KNOWN_ENGINES else 'klein'
    target = requested or origin
    # `engines.enabled` deliberately does NOT gate a retry. It answers "what may
    # a NEW batch use"; this row already ran on `origin`, and rewriting the
    # target to the default engine is the silent engine swap the frontend refuses
    # to cause (see useDataset.js: it sends no engine on 🔄 for exactly that
    # reason). It was also a real trap: a selection saved before an engine
    # existed — MiniMax H3 on any install narrowed to Krea — sent every H3 retry
    # to Krea 2, and a Klein row on an install with only API engines ticked would
    # quietly BILL one. An engine that is genuinely unusable now fails in its own
    # preflight below, naming what is missing, which is the honest answer.
    if is_nsfw_label(img.variation_label) and target in API_ENGINES:
        # Fail-closed: NSFW never reaches a third-party API. It stays on whatever
        # LOCAL engine the row came from (a Krea row keeps Krea) — forcing Klein
        # here would silently change engine behind the user's back.
        target = origin if origin in LOCAL_ENGINES else 'klein'
    # Complete every fallible target-specific preflight before changing either
    # the row or its current file. Klein enqueue is itself part of preparation:
    # if the later DB transition fails, that exact new job is cancelled below.
    from ..job_queue import queue_manager
    old_state = {
        field: getattr(img, field) for field in (
            'filename', 'caption', 'status', 'fail_reason', 'fail_kind', 'job_id',
            'klein_model', 'variation_prompt', 'watermark_state',
            'watermark_bbox', 'watermark_regions', 'face_score', 'face_state',
            'content_sig', 'content_sig_stat')
    }
    old_path = (os.path.join(_dataset_path(img.dataset_id), img.filename)
                if img.filename else None)
    new_job_id = None
    api_generate = None
    aspect = None
    ref_bytes = None
    model = None
    if target in API_ENGINES:
        engine = target
        api_generate = _api_generate_fn(engine)
        ref_path = os.path.join(_dataset_path(ds.id), ds.ref_filename)
        if not os.path.exists(ref_path):
            raise ValueError('reference image file missing')
        aspect = aspect_for_label(img.variation_label, img.framing)
        ref_bytes = _all_ref_bytes(ds)  # principale + extras (multi-références)
    elif target == KREA_ENGINE:
        # Krea 2 Identity Edit: same shape as the Klein branch below, minus the
        # knobs it doesn't have. Its preflight raises KreaModelsMissing HERE,
        # before the row transition — so the tile keeps its current image.
        engine = KREA_ENGINE
        from . import krea_edit_helper as _keh
        ref_path = os.path.join(_dataset_path(ds.id), ds.ref_filename)
        new_job_id = _keh.enqueue_krea_edit(
            user_id=str(user_id), source_filename=ds.ref_filename,
            source_path=ref_path, framing=img.framing,
            edit_prompt=wrap_variation_krea(
                prompt, nsfw=is_nsfw_label(img.variation_label),
                framing=img.framing,
                suffix=dataset_prompt_suffix(ds, img.framing),
                subject_type=subject_type_of(ds),
                label=img.variation_label or ''),
            aspect_ratio=aspect_for_label(img.variation_label, img.framing),
            generation_loras=_keh.resolve_generation_lora_preset(generation_lora_preset),
            extra_metadata={'is_dataset': True, 'dataset_id': img.dataset_id,
                            'variation_label': img.variation_label})
    elif target == MINIMAX_H3_ENGINE:
        # MiniMax H3: same shape as the Krea branch above. It needs its OWN
        # branch and not the Klein fallthrough below — the `else` was written
        # when Klein was the only local engine, so without this a 🔄 on an H3
        # tile silently re-rendered it on Klein, i.e. the retry answered with a
        # different engine than the one the tile was made with.
        engine = MINIMAX_H3_ENGINE
        from . import minimax_h3_helper as _mh
        ref_path = os.path.join(_dataset_path(ds.id), ds.ref_filename)
        new_job_id = _mh.enqueue_minimax_h3(
            user_id=str(user_id), source_filename=ds.ref_filename,
            source_path=ref_path, framing=img.framing,
            edit_prompt=wrap_variation_minimax_h3(
                prompt, nsfw=is_nsfw_label(img.variation_label),
                framing=img.framing,
                suffix=dataset_prompt_suffix(ds, img.framing),
                subject_type=subject_type_of(ds),
                label=img.variation_label or ''),
            aspect_ratio=aspect_for_label(img.variation_label, img.framing),
            extra_metadata={'is_dataset': True, 'dataset_id': img.dataset_id,
                            'variation_label': img.variation_label})
    else:
        try:
            from .klein_edit_helper import enqueue_klein_edit, resolve_generation_lora_preset
        except ImportError:
            raise RuntimeError('ComfyUI is not configured')
        # The route cannot know the effective engine when this is an ordinary
        # retry: it may be a Krea/API row, or a Klein row with a model filename.
        # Check Klein-only nodes only after the row's target has been resolved,
        # before modifying its current file/state.
        from . import klein_edit_helper as _kleh
        missing_nodes = _kleh.klein_missing_nodes()
        if missing_nodes:
            raise KleinNodesMissing(_kleh.klein_missing_assets(), missing_nodes)
        # Klein target: keep the row's real model file when it has one; a row born
        # on an API engine holds an engine TAG here, not a model — use the
        # workspace's Klein pick instead (None = enqueue's default model).
        # …and when the workspace sent none either, the DATASET's pick (None = auto).
        model = (img.klein_model if img.klein_model not in API_ENGINES
                 else ((klein_model or '').strip() or dataset_klein_model(ds)))
        ref_path = os.path.join(_dataset_path(ds.id), ds.ref_filename)
        extra_paths = [os.path.join(_dataset_path(ds.id), fn)
                       for fn in extra_ref_filenames(ds)]
        new_job_id = enqueue_klein_edit(
            user_id=str(user_id), source_filename=ds.ref_filename,
            source_path=ref_path,
            edit_prompt=wrap_variation_klein(
                prompt, nsfw=is_nsfw_label(img.variation_label),
                framing=img.framing,
                # CURRENT dataset suffix, applied at wrap: `prompt` is the raw
                # stored/edited creative prompt, so this is the ONLY application.
                suffix=dataset_prompt_suffix(ds, img.framing),
                subject_type=subject_type_of(ds),
                label=img.variation_label or ''),
            klein_model=model,
            lora_strength=lora_strength, extra_ref_paths=extra_paths,
            generation_loras=resolve_generation_lora_preset(generation_lora_preset),
            sampler_steps=_generation_steps(),
            base_lora_strength=_generation_base_lora_strength(),
            extra_metadata={'is_dataset': True, 'dataset_id': img.dataset_id,
                            'variation_label': img.variation_label})

    # Persist the replacement state first. The old file remains in place until
    # this commit succeeds, eliminating rows that reference an already-moved file.
    try:
        if old_state['status'] == 'pending' and not old_state['filename'] \
                and old_state['job_id']:
            if not queue_manager.cancel_job(
                    old_state['job_id'], str(user_id), 'image', commit=False):
                raise RuntimeError(
                    'The previous generation still has unconfirmed ComfyUI work; cancel it safely first.')
        if edited:
            img.variation_prompt = stored_prompt
        _clear_watermark_metadata(img)
        img.face_score = None
        img.content_sig = None
        img.content_sig_stat = None
        img.face_state = None
        # Engine TAG for the API engines and for Krea (each resolves its own
        # model); the real model FILE for Klein.
        img.klein_model = (engine if target in API_ENGINES or target == KREA_ENGINE
                           else model)
        img.filename = None
        img.caption = None
        img.status = 'pending'
        img.job_id = new_job_id
        img.fail_reason = None
        img.fail_kind = None
        db.session.commit()
    except Exception:
        db.session.rollback()
        if new_job_id:
            try:
                queue_manager.cancel_job(new_job_id, str(user_id), 'image')
            except Exception:
                logger.exception('regenerate: failed to cancel unlinked job %s',
                                 new_job_id)
        raise

    # The DB no longer references the old filename. If Trash itself fails, put
    # the exact previous row state back and cancel the prepared Klein job.
    try:
        if old_path and os.path.exists(old_path):
            trash.send_to_trash(
                old_path, context=f'dataset-{img.dataset_id}-regenerate-{img.id}')
    except Exception:
        try:
            # The row must keep the replacement job identity unless its exact
            # cancellation is committed in the same restoration transaction.
            if new_job_id and not queue_manager.cancel_job(
                    new_job_id, str(user_id), 'image', commit=False):
                raise RuntimeError(
                    'The replacement generation still has unconfirmed ComfyUI work.')
            for field, value in old_state.items():
                setattr(img, field, value)
            db.session.commit()
        except Exception:
            db.session.rollback()
            logger.exception('regenerate: failed to restore row %s after Trash error',
                             image_id)
        raise

    # API target ('nanobanana'/'chatgpt' — requested, or the row's origin when
    # no engine was given): the row's klein_model column carries the engine tag.
    # With an `app` handle the call runs in a background thread (the row flips
    # to in-flight IMMEDIATELY so the tile shows "…" and the polling/banner UI
    # reacts at once); without it the call is synchronous (test path / legacy
    # callers).
    if target in API_ENGINES:
        if app is not None:
            # Threaded path: _run_nanobanana_batch owns the 'generate' indicator
            # (begin/bump/end) so a single API regenerate takes the same lock as a
            # batch — every concurrent action stays disabled until it finishes.
            try:
                threading.Thread(target=_run_nanobanana_batch,
                                 args=(app, [(img.id, prompt, aspect,
                                              dataset_prompt_suffix(ds, img.framing))],
                                       ref_bytes, engine, img.dataset_id),
                                 daemon=True).start()
            except Exception as e:
                img.status = 'failed'
                img.fail_reason = f'{engine}: failed to start generation: {e}'[:500]
                db.session.commit()
                raise
            return engine
        # Synchronous path (legacy / no-app callers): guard the same 'generate'
        # indicator directly so the payload advertises the regenerate too, and a
        # raise never leaks the entry (finally end()).
        token = None
        try:
            token = dataset_activity.begin(
                img.dataset_id, 'generate', total=1, engine=engine)
            gen_kwargs = {'aspect_ratio': aspect}
            if engine == 'chatgpt':
                from .chatgpt_image import _use_subscription
                gen_kwargs['force_lane'] = 'subscription' if _use_subscription() else 'api'
            try:
                out = api_generate(
                    ref_bytes,
                    wrap_variation(prompt, ref_count=len(ref_bytes),
                                   suffix=dataset_prompt_suffix(ds, img.framing),
                                   subject_type=subject_type_of(ds)),
                    **gen_kwargs)
            except EngineRefused as e:
                # Même règle que dans le lot : un refus se nomme, il ne se
                # déguise pas en panne (et il n'invente pas de contournement).
                out = None
                img.status = 'failed'
                img.fail_reason = f'{engine}: {str(e)[:400]}'
                img.fail_kind = 'refused'
                db.session.commit()
                return engine
            except SubscriptionQuotaExceeded:
                out = None
                img.status = 'failed'
                img.fail_reason = _QUOTA_MSG
                img.fail_kind = 'error'
                db.session.commit()
                return engine
            except SubscriptionUnavailable as e:
                out = None
                img.status = 'failed'
                img.fail_reason = f'chatgpt: {e}'
                img.fail_kind = 'error'
                db.session.commit()
                return engine
            except Exception as gen_err:
                # Catch engine-specific errors (Qwen/OpenRouter/etc) that are fatal
                # (no key, bad config) vs retryable (API rate limit, transient).
                # EngineFatal marks it as non-retryable; others stay as failed/retryable.
                from .qwen_image import QwenImageFatal
                from .openrouter import OpenRouterFatal
                from .nanobanana import NanoBananaFatal
                if isinstance(gen_err, (QwenImageFatal, OpenRouterFatal, NanoBananaFatal)):
                    out = None
                    img.status = 'failed'
                    img.fail_reason = f'{engine}: {gen_err}'
                    db.session.commit()
                    return engine
                # For other exceptions, let them bubble up so the outer handler records them
                raise
            if out:
                fn = f"{user_id}_{_ENGINE_FILE_TAG[engine]}_{uuid.uuid4().hex[:8]}.webp"
                write_image_atomic(os.path.join(_dataset_dir(img.dataset_id), fn),
                                   normalize_to_webp(out))
                img.filename = fn
                img.unseen = True
            else:
                img.status = 'failed'
                img.fail_reason = f'{engine}: {_EMPTY_MSG}'
                img.fail_kind = 'empty'
            db.session.commit()
            return engine
        except Exception as e:
            db.session.rollback()
            current = db.session.get(FaceDatasetImage, image_id)
            if current and current.filename is None:
                current.status = 'failed'
                current.fail_reason = f'{engine}: {e}'[:500]
                current.fail_kind = 'error'
                db.session.commit()
            raise
        finally:
            if token is not None:
                dataset_activity.end(token)

    # Advertise the in-flight Klein job so a single regenerate takes the same lock
    # as a batch; link_completed_dataset_image clears it on completion.
    _sync_generate_activity(img.dataset_id)
    return new_job_id


# --- Which engine runs the 🎭↔ swap ------------------------------------------
# Two engines, one button. Klein REPAINTS the head with a swap LoRA on a Flux.2
# graph; MiniMax H3 masks the head out and lets a VIDEO model re-stage the
# identity into the hole, then stitches the crop back. They need different
# weights (Klein 9B + the swap LoRA vs ~40 GB of H3), take different times, and
# fail differently — so it is a choice, not a default anyone can pick for
# everyone. APPEND-ONLY: the ids are stored in config, never renamed.
FACE_SWAP_ENGINES = ('klein', 'minimax_h3')


def resolve_face_swap_engine(requested=None):
    """The engine a 🎭↔ swap will run on: the explicit pick when it names a known
    engine, else the `face_swap.engine` setting, else Klein.

    Fail-SAFE rather than fail-closed, exactly like resolve_improve_engine: an
    unknown name falls back instead of raising, because a stale tab must degrade
    to the historical behaviour rather than refuse the click. Klein is the
    fallback because it is what every swap did before this setting existed."""
    for candidate in (requested, cfg.get('face_swap.engine')):
        name = str(candidate or '').strip().lower()
        if name in FACE_SWAP_ENGINES:
            return name
        if name:
            logger.warning('unknown face swap engine %r — falling back to klein',
                           candidate)
    return 'klein'


def face_swap_image(user_id, image_id, engine=None):
    """Face-swap this tile in place: its CURRENT image becomes the target
    (image1), the dataset's reference photo becomes the identity source
    (image2), and the chosen engine's fixed swap workflow overwrites the tile
    with the result. Same cancel/trash/pending-transition shape as
    regenerate_image, EXCEPT it does not touch variation_prompt /
    variation_label / klein_model — a face swap is an identity post-process
    on an already-generated tile, not a re-generation from the catalog
    prompt, so the row keeps remembering what it originally was (and a later
    🔄 Regenerate on this tile still falls back to its ORIGINAL engine/prompt).

    Returns the new job_id, or None if the image is not owned / has no
    current file to use as the target. Raises ValueError if the dataset has
    no reference image (there is nothing to use as image2).

    `engine` picks the swap engine for THIS call (see resolve_face_swap_engine);
    None uses the `face_swap.engine` setting. Both engines take the identical
    (user_id, target_path, ref_path, extra_metadata) contract, so the choice is
    one lookup — and the row bookkeeping below is engine-agnostic on purpose: a
    swap is a post-process on an already-generated tile either way."""
    img = _owned_image(user_id, image_id)
    if not img or not img.filename:
        return None
    ds = db.session.get(FaceDataset, img.dataset_id)
    if not ds.ref_filename:
        raise ValueError('reference image required')
    if resolve_face_swap_engine(engine) == 'minimax_h3':
        from .minimax_h3_swap_helper import enqueue_h3_swap as enqueue_face_swap
    else:
        from .face_swap_helper import enqueue_face_swap
    from ..job_queue import queue_manager
    target_path = os.path.join(_dataset_path(img.dataset_id), img.filename)
    ref_path = os.path.join(_dataset_path(ds.id), ds.ref_filename)
    old_state = {
        field: getattr(img, field) for field in (
            'filename', 'caption', 'status', 'fail_reason', 'fail_kind', 'job_id',
            'watermark_state', 'watermark_bbox', 'watermark_regions',
            'face_score', 'face_state', 'content_sig', 'content_sig_stat')
    }
    old_path = target_path
    new_job_id = enqueue_face_swap(
        user_id=str(user_id), target_path=target_path, ref_path=ref_path,
        extra_metadata={'is_dataset': True, 'dataset_id': img.dataset_id,
                        'variation_label': img.variation_label})

    # Persist the replacement state first — the old file stays on disk until
    # this commit succeeds, same ordering rationale as regenerate_image.
    try:
        _clear_watermark_metadata(img)
        img.face_score = None
        img.content_sig = None
        img.content_sig_stat = None
        img.face_state = None
        img.filename = None
        img.caption = None
        img.status = 'pending'
        img.job_id = new_job_id
        img.fail_reason = None
        img.fail_kind = None
        db.session.commit()
    except Exception:
        db.session.rollback()
        try:
            queue_manager.cancel_job(new_job_id, str(user_id), 'image')
        except Exception:
            logger.exception('face_swap: failed to cancel unlinked job %s', new_job_id)
        raise

    # The DB no longer references the old filename. If Trash itself fails,
    # put the exact previous row state back and cancel the prepared job.
    try:
        if old_path and os.path.exists(old_path):
            trash.send_to_trash(
                old_path, context=f'dataset-{img.dataset_id}-faceswap-{img.id}')
    except Exception:
        try:
            if new_job_id and not queue_manager.cancel_job(
                    new_job_id, str(user_id), 'image', commit=False):
                raise RuntimeError(
                    'The replacement generation still has unconfirmed ComfyUI work.')
            for field, value in old_state.items():
                setattr(img, field, value)
            db.session.commit()
        except Exception:
            db.session.rollback()
            logger.exception('face_swap: failed to restore row %s after Trash error',
                             image_id)
        raise
    return new_job_id


# --- Fan-out generation (API engines: Nano Banana / ChatGPT) ---------------
# Both engines share the exact generate_variation contract (refs + prompt +
# aspect -> bytes|None), so the whole fan-out below is engine-parametric. The
# filename tag keeps the provenance readable in the dataset folder.
# APPEND-ONLY: both the engine ids and the file tags are persisted (dataset rows
# record the engine, and the tag is baked into the filename on disk), so a value
# here is never renamed or reordered.
API_ENGINES = ('nanobanana', 'chatgpt', 'openrouter', 'qwen')
_ENGINE_FILE_TAG = {'nanobanana': 'NBFace', 'chatgpt': 'GPTFace', 'openrouter': 'ORFace', 'qwen': 'QwenFace'}

# The LOCAL engines — they render on the user's own GPU through ComfyUI, cost
# nothing, and are the only ones allowed to receive NSFW shots. Klein is the
# historical one; Krea 2 Identity Edit is the second (krea_edit_helper); MiniMax
# H3 is the third (minimax_h3_helper).
# APPEND-ONLY for the same reason as API_ENGINES: each id is persisted in
# FaceDatasetImage.klein_model as that row's engine tag.
LOCAL_ENGINES = ('klein', 'krea', 'minimax_h3')
KREA_ENGINE = 'krea'
# Stored on every row this lane creates (in `klein_model`, like the others) and
# read back by the grid badge and by regenerate. NEVER rename: it is in user
# databases the moment the first image is generated.
MINIMAX_H3_ENGINE = 'minimax_h3'
LOCAL_ENGINE_LABELS = {'klein': 'Klein', 'krea': 'Krea 2 Edit',
                       'minimax_h3': 'MiniMax H3'}
# Engines that GENERATE dataset images but do not edit a reference photo. Not a
# permanent property of the engine — a deliberate scope line: MiniMax H3 ships as
# a generation lane only, and `_enqueue_local_reference_edit` has no branch for
# it. Offering it in the ✦ Edit modal would fail at enqueue instead of at pick
# time, which is the exact failure `editable_engines()` exists to prevent.
# Mirrored by GENERATE_ONLY_ENGINES in engineSelection.js; a contract test pins
# the pair.
GENERATE_ONLY_ENGINES = ('minimax_h3',)
# Every engine a generate/regenerate request may name.
KNOWN_ENGINES = LOCAL_ENGINES + API_ENGINES

# Human names for the engines, in the SAME wording as the frontend's ENGINE_LABELS
# (frontend/src/components/dataset/engineSelection.js). Only used to word messages
# — the ids above are the persisted values. A contract test pins ids AND labels
# across the two languages, so neither side can grow an engine on its own.
API_ENGINE_LABELS = {'nanobanana': 'Nano Banana Pro', 'chatgpt': 'ChatGPT',
                     'openrouter': 'OpenRouter', 'qwen': 'Qwen Image'}


def engine_labels():
    """Every engine id -> its human label, both lanes. Merged rather than kept as a
    third dict: the two halves are already the source of truth for their side."""
    return dict(LOCAL_ENGINE_LABELS, **API_ENGINE_LABELS)


def editable_engines():
    """Engines /ref/edit accepts, LOCAL ones first (free, on the user's own GPU),
    then the paid API ones — the canonical order the workspace cards use too.

    A FUNCTION, not a constant: it is derived at call time from the two tuples
    above, so a lane growing an engine reaches the edit path with no second edit
    here. Klein and Krea used to be excluded because the edit ran as a blocking
    provider call and they have no blocking call to make; they now ride the same
    ComfyUI queue as every other local render, so the exclusion had outlived its
    reason — and it was the reason the app's only FREE edit lane was invisible.

    GENERATE_ONLY_ENGINES is the one thing this does not derive. An engine can be
    a generation lane without being an edit lane, and offering one the route has
    no branch for would fail at enqueue instead of at pick time."""
    return (tuple(e for e in LOCAL_ENGINES if e not in GENERATE_ONLY_ENGINES)
            + tuple(e for e in API_ENGINES if e not in GENERATE_ONLY_ENGINES))


def edit_engine_choice_message():
    """The refusal for a non-editable engine, DERIVED from editable_engines():
    "pick Klein, Krea 2 Edit, Nano Banana Pro, ChatGPT or OpenRouter". Hardcoding
    the sentence is how the previous one ("pick ChatGPT or Nano Banana") kept
    naming two engines after a third became editable — a message that lies is
    worse than no message."""
    labels = engine_labels()
    names = [labels.get(e, e) for e in editable_engines()]
    if not names:
        return 'no image engine can edit the reference'
    head, last = names[:-1], names[-1]
    return 'pick ' + (f"{', '.join(head)} or {last}" if head else last)

from .chatgpt_image import SubscriptionQuotaExceeded, SubscriptionUnavailable
from .engine_errors import EngineError, EngineFatal, EngineRefused

# La phrase des moteurs qui ne SAVENT pas pourquoi ils rentrent bredouilles
# (ChatGPT, OpenRouter : 200 sans image, aucune métadonnée de refus lisible).
# Nano Banana ne passe plus par là — il lève NanoBananaRefused avec la vraie
# cause. Ce qui reste ici doit donc dire l'ambiguïté au lieu de la trancher :
# l'ancienne version promettait « retry usually works », ce qui était faux la
# moitié du temps et envoyait réessayer un refus définitif. On nomme l'ignorance.
_EMPTY_MSG = ('empty response — no image and no reason given (a content-policy '
              'refusal and a transient API error look identical here)')

_QUOTA_MSG = ('chatgpt: subscription image quota reached — remaining rows were '
              'stopped; rerun in API-key mode or wait for your plan quota to reset')
_LOST_MSG = ('chatgpt: subscription connection lost — remaining rows stopped; '
             'reconnect in Settings, then regenerate')


def _api_generate_fn(engine):
    if engine == 'chatgpt':
        from .chatgpt_image import generate_variation
    elif engine == 'openrouter':
        from .openrouter import generate_variation
    elif engine == 'qwen':
        from .qwen_image import generate_variation
    else:
        from .nanobanana import generate_variation
    return generate_variation


def _run_nanobanana_batch(app, items, ref_bytes, engine='nanobanana', dataset_id=None):
    """Worker body: generate each (image_id, prompt) via the selected API engine
    and link the result. Runs in a background thread (factored out so tests can
    call it synchronously). Each row commits independently; an API failure marks
    that row 'failed' (visible + regenerable) without stopping the batch.

    ``dataset_id`` (when known) drives the 'generate' activity indicator: one
    begin() with total=len(items), a bump() per item handled (success OR fail),
    and end() in a finally — so the ⚡ Generate button (and every concurrent
    action) stays disabled for the WHOLE batch, and the indicator can never leak
    even if a row raises. Also used for single-image API regenerate (items=1),
    which therefore takes the same lock. ``None`` = no indicator (legacy callers)."""
    api_generate = _api_generate_fn(engine)
    from concurrent.futures import ThreadPoolExecutor
    # Guard d'identité adapté au nombre de références (multi = « use EVERY ref »).
    n_refs = len(ref_bytes) if isinstance(ref_bytes, (list, tuple)) else 1
    tag = _ENGINE_FILE_TAG.get(engine, 'NBFace')
    # Pin the ChatGPT auth lane ONCE for the whole batch. Without this, a
    # mid-batch token refresh failure (auth.openai.com non-200 -> logout())
    # would make every later row's OWN _use_subscription() call see
    # connected=False and silently reroute onto the paid API key — breaking
    # the feature's headline invariant. Pinning + stopping the batch instead
    # (via SubscriptionUnavailable below) closes that hole.
    force_lane = None
    if engine == 'chatgpt':
        from .chatgpt_image import resolve_lane
        force_lane = resolve_lane()
    # Set the moment ANY row hits the plan quota (or the pinned subscription
    # lane loses its token) — every later row would fail too, so the rest of
    # the batch fails fast instead of burning one call each.
    quota_exhausted = threading.Event()
    stop_msg = {'text': _QUOTA_MSG}   # set to the actual stop reason when it fires
    # Refus fournisseur du lot. Une liste (append est atomique sous le GIL) pour
    # que le log de fin rende un DÉCOMPTE au lieu de laisser 12 trous sur 40 se
    # découvrir tuile par tuile. L'UI, elle, recompte depuis fail_kind.
    refused = []
    token = dataset_activity.begin(dataset_id, 'generate', total=len(items), engine=engine) \
        if dataset_id is not None else None

    def _run_one(item):
        # item = (image_id, prompt, aspect, suffix) ; aspect optionnel (rétro-compat
        # → '1:1'), suffix optionnel (direction créative du dataset, déjà composée
        # par cadrage au call-site — rétro-compat → '').
        image_id, prompt = item[0], item[1]
        aspect = item[2] if len(item) > 2 else '1:1'
        suffix = item[3] if len(item) > 3 else ''
        # subject_type optionnel (retro-compat -> 'human' = lock historique).
        subject_type = item[4] if len(item) > 4 else 'human'
        # Stop AVANT l'appel API : cancel_pending supprime les lignes en vol — si
        # celle-ci a disparu, ne pas payer une génération qui sera jetée (le bouton
        # Stop doit économiser le RESTE du batch, pas seulement masquer les tuiles).
        with app.app_context():
            row = db.session.get(FaceDatasetImage, image_id)
            if row is None or row.status != 'pending':
                logger.info(f"{engine} batch: row {image_id} cancelled - API call skipped")
                return
        if quota_exhausted.is_set():
            # A previous row hit the plan quota: skip the API for every row not
            # yet started (later calls would 429 too). Up to max_workers rows may
            # already be in flight past this check when the event trips — each is
            # still failed via the dedicated except below, so the batch wastes at
            # most ~max_workers calls, not all.
            with app.app_context():
                img = db.session.get(FaceDatasetImage, image_id)
                if img is not None:
                    img.status = 'failed'
                    img.fail_reason = stop_msg['text']
                    img.fail_kind = 'error'
                    db.session.commit()
            return
        out = None
        fail_reason = None
        fail_kind = 'error'
        gen_kwargs = {'aspect_ratio': aspect}
        if engine == 'chatgpt':
            gen_kwargs['force_lane'] = force_lane
        try:
            out = api_generate(ref_bytes,
                               wrap_variation(prompt, ref_count=n_refs, suffix=suffix,
                                              subject_type=subject_type),
                               **gen_kwargs)
            if not out:
                # api_generate signale certains refus/vides par un retour falsy
                # sans lever — sans raison, la tuile "failed" resterait muette.
                fail_reason = f'{engine}: {_EMPTY_MSG}'
                fail_kind = 'empty'
        except EngineRefused as e:
            # Le fournisseur a répondu, et a refusé CETTE image. Ce n'est pas une
            # panne : on ne coupe pas le lot (les lignes suivantes ont leurs
            # chances — le filtre n'est pas déterministe), on nomme la cause et
            # on la compte à part pour pouvoir en rendre un décompte honnête.
            refused.append(image_id)
            logger.info(f"{engine} batch: refused at row {image_id}: {e}")
            fail_reason = f'{engine}: {str(e)[:400]}'
            fail_kind = 'refused'
        except SubscriptionQuotaExceeded as e:
            quota_exhausted.set(); stop_msg['text'] = _QUOTA_MSG
            logger.warning(f"{engine} batch: quota exhausted at row {image_id}: {e}")
            fail_reason = _QUOTA_MSG
        except SubscriptionUnavailable as e:
            quota_exhausted.set(); stop_msg['text'] = _LOST_MSG
            logger.warning(f"{engine} batch: subscription lost at row {image_id}: {e}")
            fail_reason = _LOST_MSG
        except EngineFatal as e:
            # No key, key rejected, no credits, unknown or unusable model: every
            # remaining row would fail on the exact same cause, so stop the batch
            # instead of asking the provider the same refused question once per
            # image. Reuses the ChatGPT quota machinery — same need, same shape.
            # Raised by all three API engines (see services/engine_errors.py):
            # the model of each is user-editable, and a wrong slug must fail once,
            # loudly, not N times in a row.
            msg = f'{engine}: {str(e)[:400]} — remaining rows were stopped'
            quota_exhausted.set(); stop_msg['text'] = msg
            logger.warning(f"{engine} batch: fatal at row {image_id}: {e}")
            fail_reason = msg
        except Exception as e:
            logger.warning(f"{engine} batch: generation error for row {image_id}: {e}")
            fail_reason = f'{engine}: {str(e)[:400]}'
        with app.app_context():
            img = db.session.get(FaceDatasetImage, image_id)
            if img is None:
                return
            if out:
                ds = db.session.get(FaceDataset, img.dataset_id)
                if ds is None:
                    # The whole DATASET was deleted while this batch ran. The row
                    # above survived only because its cascade has not landed yet;
                    # reading ds.user_id here raised AttributeError, which escaped
                    # _run_one and abandoned every REMAINING item of the batch.
                    # Nothing to write this result to, so drop it and let the rest
                    # of the batch finish.
                    logger.info('%s batch: dataset gone for row %s, dropping the '
                                'result', engine, image_id)
                    return
                fn = f"{ds.user_id}_{tag}_{uuid.uuid4().hex[:8]}.webp"
                try:
                    # Conserve le ratio demandé (pas de letterbox carré sur les corps).
                    write_image_atomic(os.path.join(_dataset_dir(img.dataset_id), fn),
                                       normalize_to_webp(out))
                    img.filename = fn
                    img.unseen = True
                except Exception as e:
                    logger.warning(f"{engine} batch: save failed for row {image_id}: {e}")
                    img.status = 'failed'
                    img.fail_reason = f'saving the image failed: {str(e)[:400]}'
                    img.fail_kind = 'error'
            else:
                img.status = 'failed'
                img.fail_reason = fail_reason
                img.fail_kind = fail_kind
            db.session.commit()

    def _one(item):
        # Progress-tracking wrapper: bump the indicator once per item handled,
        # whatever the outcome (a raised _run_one still counts as one handled and
        # never strands the counter). No-op when token is None (bump(None)).
        try:
            return _run_one(item)
        finally:
            dataset_activity.bump(token)

    logger.info(f"{engine} batch: start ({len(items)} variation(s))")
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(_one, items))
    finally:
        dataset_activity.end(token)   # idempotent; end(None) is a no-op
    tally = f" — {len(refused)} refused by the provider" if refused else ''
    logger.info(f"{engine} batch: done ({len(items)} variation(s)){tally}")


def generate_variations_nanobanana(app, user_id, dataset_id, variations, multiplier,
                                   engine='nanobanana'):
    """API fan-out (Nano Banana or ChatGPT, per `engine`): pre-create pending
    rows (job_id stays None - that is the marker for API-generated rows), then
    fill them from a background thread. The existing polling/banner/cancel UI
    works unchanged (pending + no file = in flight). Returns the created ids."""
    _guard_not_bank_export(dataset_id)
    if engine not in API_ENGINES:
        raise ValueError(f'unknown API engine: {engine}')
    # Fail-closed : les variations NSFW ne partent JAMAIS vers un moteur API
    # (comptes/API tiers) — elles n'existent que sur le chemin Klein local.
    if any(v.get('nsfw') or is_nsfw_label(v.get('label')) for v in variations):
        raise ValueError('NSFW variations run on the local Klein engine only')
    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    if not ds.ref_filename:
        raise ValueError('reference image required')
    ref_path = _ref_path(ds)
    if not os.path.exists(ref_path):
        raise ValueError('reference image file missing')
    mult = max(1, int(multiplier))
    total = len(variations) * mult
    if total == 0:
        raise ValueError('no variations selected')
    if total > MAX_FANOUT:
        raise ValueError(f'fan-out too large ({total} > {MAX_FANOUT})')
    # Principale + refs additionnelles : Nano Banana s'appuie sur toutes les
    # images pour la cohérence d'identité (une seule = comportement historique).
    ref_bytes = _all_ref_bytes(ds)
    subject_type = subject_type_of(ds)   # steers the identity lock at wrap time

    ids, items = [], []
    for v in variations:
        for _ in range(mult):
            # klein_model=<engine> marks API-generated rows (the regenerate
            # path dispatches on it; never collides with real .safetensors names).
            img = FaceDatasetImage(dataset_id=dataset_id, source='generated', status='pending',
                                   variation_label=v.get('label'), framing=v.get('framing'),
                                   variation_prompt=v['prompt'], klein_model=engine, job_id=None)
            db.session.add(img)
            db.session.commit()
            ids.append(img.id)
            # Suffix composed HERE (per-framing) and carried by the work item: the
            # row keeps the raw prompt, the batch worker applies it at wrap time.
            items.append((img.id, v['prompt'],
                          aspect_for_label(v.get('label'), v.get('framing')),
                          dataset_prompt_suffix(ds, v.get('framing')),
                          subject_type))

    threading.Thread(target=_run_nanobanana_batch,
                     args=(app, items, ref_bytes, engine, dataset_id),
                     daemon=True).start()
    return ids

# --- Borrow: face_dataset_service.py primitives -----------------------------
# MUST stay at the bottom of this file, same reason as in the sibling split
# modules: this module and face_dataset_service.py import names from each
# other, and whichever side loads first must find the other fully defined by
# the time the reach-back import resolves.
from .face_dataset_service import (
    _live_image_row,
    _guard_not_bank_export,
    get_dataset, dataset_klein_model, dataset_prompt_suffix, subject_type_of,
    normalize_to_webp, write_image_atomic, cancel_pending,
    link_completed_dataset_image, KleinNodesMissing,
    _img_path, _ref_path, _dataset_dir, _dataset_path, _owned_image, _krea_pose_source_path,
    _clear_watermark_metadata, _source_metadata_storage,
    _generation_steps, _generation_base_lora_strength,
    _rekeep_pending_parent_for_reimprove, _transition_reimprove_candidate,
    _restore_reimprove_candidate_after_trash_failure,
    _undo_rekeep_parent_after_reimprove_trash_failure,
    _IMAGE_IMPROVE_LOCKS, _SMALL_IMAGE_DERIVATIONS,
    KLEIN_IMAGE_IMPROVE, KLEIN_SMALL_IMAGE, MAX_FANOUT, logger,
)
# Straight from the module that OWNS them, not via face_dataset_service's
# re-export: a cross-module borrow routed through the parent would make the
# ORDER of the parent's re-export blocks load-bearing. Pointed at the owner,
# each module still has every name of its own defined before its bottom import
# runs, so the cycle resolves whichever module is imported first.
from .reference_photos_service import _all_ref_bytes, extra_ref_filenames
