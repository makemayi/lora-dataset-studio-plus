"""🗃️ Image bank API — triage a big unsorted folder before it becomes datasets.

All heavy passes (quality scan, face clustering, promotion) return 202 and run
in ONE background thread per bank; the UI polls GET /bank/<id> whose payload
embeds the live job. 409 = a job is already running on this bank.
"""
import logging
import os

from flask import Blueprint, current_app, jsonify, request, send_file

from ..config import LOCAL_USER
from ..models import BankImage
from ..services import bank_jobs
from ..services import image_bank_service as banks

logger = logging.getLogger(__name__)

bp = Blueprint('bank', __name__, url_prefix='/api')


def _app():
    return current_app._get_current_object()


@bp.get('/banks')
def banks_list():
    """Every bank + its card previews. ?dataset_id=<id> additionally embeds each
    bank's promotable count for that dataset, so the dataset-side bank chooser
    opens on ONE request instead of one per bank. An unknown/junk dataset_id
    simply omits the field (never a 400: the list itself is still valid).

    Every bank's source folder is re-walked first (see refresh_bank): a bank
    points at a LIVE folder, so images dropped in it after the bank was created
    show up here instead of needing a rebuild. Strictly additive, ~5 ms a bank,
    and the per-bank outcome rides back in ``folder_sync`` so the UI can say why
    the counters moved."""
    sync = banks.refresh_banks(LOCAL_USER, force=True)
    rows = banks.list_banks(
        LOCAL_USER, dataset_id=request.args.get('dataset_id') or None)
    for row in rows:
        row['folder_sync'] = sync.get(row['id'])
    return jsonify({'banks': rows})


@bp.post('/bank/create')
def bank_create():
    data = request.get_json(silent=True) or {}
    try:
        bank, added = banks.create_bank(LOCAL_USER, data.get('name'),
                                        data.get('folder'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    # Nested folders mean two banks over the same files: harmless while triaging
    # (statuses are per bank), but 🗑 Delete rejected in one amputates the other.
    # Say it now, once, rather than at the destructive click only.
    return jsonify({'ok': True, 'id': bank.id, 'added': added,
                    'overlaps': banks.overlapping_banks(LOCAL_USER, bank.id)})


@bp.post('/bank/from-dataset')
def bank_from_dataset():
    """Reverse of promote: build a NEW bank from a dataset's kept images, under a
    name the user chooses. Copies the files so the two never share (curating one
    would otherwise mutate the other). 202 + background job, like the other passes."""
    data = request.get_json(silent=True) or {}
    try:
        bank_id = banks.start_dataset_import(_app(), LOCAL_USER,
                                             data.get('dataset_id'), data.get('name'))
    except bank_jobs.BankJobBusy as e:
        return jsonify({'error': str(e)}), 409
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    # the id rides back so the UI can jump straight to the bank being filled
    return jsonify({'ok': True, 'id': bank_id}), 202


@bp.get('/bank/<int:bank_id>')
def bank_get(bank_id):
    """The workspace payload. The source folder is re-walked first so images
    added to it show up while the bank is open — cooldown-limited, because this
    route is ALSO the 2 s job poll. ?refresh=1 forces the walk (the workspace
    sends it when it opens the bank). The outcome rides in ``folder_sync``."""
    sync = banks.refresh_bank(LOCAL_USER, bank_id,
                              force=request.args.get('refresh') == '1')
    payload = banks.bank_payload(LOCAL_USER, bank_id)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    payload['folder_sync'] = sync
    return jsonify(payload)


@bp.post('/bank/<int:bank_id>/flag-preview')
def bank_flag_preview(bank_id):
    """How many images each flag WOULD hold at the thresholds in the body —
    the live effect readout under the Bank's 🎚 threshold controls.

    Read-only and cheap: verdicts are recomputed from persisted raw scores, so
    this is one COUNT per flag, the same queries the workspace payload already
    runs. Nothing is saved — the user still has to press Save.

    POST (not GET) because it carries the UNSAVED candidate thresholds, exactly
    like /settings/prompt-preview carries unsaved prompt text. A junk body
    degrades to "the saved thresholds" instead of 400: a preview that answers
    'error' while you are mid-keystroke is worth less than one that answers with
    what is currently in effect."""
    body = request.get_json(silent=True) or {}
    overrides = body.get('thresholds')
    out = banks.flag_preview(LOCAL_USER, bank_id,
                             overrides if isinstance(overrides, dict) else None)
    if out is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(out)


@bp.delete('/bank/<int:bank_id>')
def bank_delete(bank_id):
    if not banks.delete_bank(LOCAL_USER, bank_id):
        return jsonify({'error': 'not found'}), 404
    return jsonify({'ok': True})


@bp.post('/bank/<int:bank_id>/relocate')
def bank_relocate(bank_id):
    """Point a bank at a new folder after the user moved it (another disk, a
    rename). Two-step ON PURPOSE: {folder} alone only REPORTS how many of the
    bank's files are in there, {folder, confirm: true} applies it. 400 when the
    folder holds none of them (that is a different folder, not a moved one),
    409 while a pass is running. Nothing is ever deleted — a partial match keeps
    every row and its analysis."""
    data = request.get_json(silent=True) or {}
    try:
        out = banks.relocate_bank(LOCAL_USER, bank_id, data.get('folder'),
                                  confirm=bool(data.get('confirm')))
    except bank_jobs.BankJobBusy as e:
        return jsonify({'error': str(e)}), 409
    except banks.BankRelocateMismatch as e:
        return jsonify({'error': str(e), **e.preview}), 400
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.get('/bank/<int:bank_id>/images')
def bank_images(bank_id):
    args = request.args

    def _int(name):
        v = args.get(name)
        try:
            return int(v) if v not in (None, '') else None
        except ValueError:
            return None

    # subfolder is a STRING facet ('' = bank root), distinct from the int filters.
    subfolder = args.get('subfolder')
    # ids: the "show selected" VIEW — a comma-separated ordered id list that
    # overrides the facets. Present-but-empty means "an empty selection" (0 rows),
    # so distinguish None (no id view) from '' (empty view).
    ids_arg = args.get('ids')
    ids = None
    if ids_arg is not None:
        ids = []
        for tok in ids_arg.split(','):
            tok = tok.strip()
            if tok:
                try:
                    ids.append(int(tok))
                except ValueError:
                    pass
    payload = banks.list_images(
        LOCAL_USER, bank_id,
        status=args.get('status') or None,
        flag=args.get('flag') or None,
        cluster=_int('cluster'), group=_int('group'), style=_int('style'),
        semantic_group=_int('semantic_group'),
        subfolder=subfolder if subfolder is not None else None,
        search=args.get('search') or None,
        sort=args.get('sort') or None,
        res_bucket=args.get('res_bucket') or None,
        framing=args.get('framing') or None,
        origin=args.get('origin') or None,
        ids=ids,
        offset=_int('offset') or 0, limit=_int('limit') or 200)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.get('/bank/<int:bank_id>/subfolders')
def bank_subfolders(bank_id):
    payload = banks.subfolders_payload(LOCAL_USER, bank_id)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


def _start(fn, *args, **kwargs):
    """Shared start-a-job envelope: 202 on launch, 409 when busy, 400/503 on
    validation errors."""
    try:
        fn(*args, **kwargs)
    except bank_jobs.BankJobBusy as e:
        return jsonify({'error': str(e)}), 409
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 503
    return jsonify({'ok': True}), 202


@bp.post('/bank/<int:bank_id>/scan')
def bank_scan(bank_id):
    data = request.get_json(silent=True) or {}
    return _start(banks.start_scan, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')))


@bp.post('/bank/<int:bank_id>/faces')
def bank_faces(bank_id):
    return _start(banks.start_faces, _app(), LOCAL_USER, bank_id)


@bp.post('/bank/<int:bank_id>/score')
def bank_score(bank_id):
    """Aesthetic + NSFW + style scoring pass (bank-scoring extra). 202/409/503."""
    return _start(banks.start_score, _app(), LOCAL_USER, bank_id)


@bp.post('/bank/<int:bank_id>/semantic-dedup')
def bank_semantic_dedup(bank_id):
    """Stage-2 semantic near-duplicate pass (crops/variants) over the ✨ Score
    embeddings — CPU, no GPU. {threshold: 0.95} overrides the config for an ad-hoc
    re-tri without a re-scan. 202/409; 400 with a "run Score first" hint when no
    embeddings exist yet."""
    data = request.get_json(silent=True) or {}
    threshold = data.get('threshold')
    try:
        threshold = float(threshold) if threshold not in (None, '') else None
    except (TypeError, ValueError):
        threshold = None
    return _start(banks.start_semantic_dedup, _app(), LOCAL_USER, bank_id,
                  threshold=threshold)


@bp.post('/bank/<int:bank_id>/watermark')
def bank_watermark(bank_id):
    """Overlaid-watermark scan (Qwen3-VL). {rescan:true} re-checks scanned rows."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_watermark, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')))


@bp.get('/bank/<int:bank_id>/watermark/levels')
def bank_watermark_levels(bank_id):
    """Where each cleaning level stands: flagged / croppable / left to inpaint /
    already cropped / already inpainted / dismissed / needing a re-scan."""
    payload = banks.watermark_levels(LOCAL_USER, bank_id)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.post('/bank/<int:bank_id>/watermark/crop')
def bank_watermark_crop(bank_id):
    """Level 1 — crop away the border-band watermarks (CPU/PIL, invents no pixel).
    The source folder is never written to: the crop lands in the bank's own
    working copy. 202/409/400."""
    return _start(banks.start_watermark_crop, _app(), LOCAL_USER, bank_id)


@bp.post('/bank/<int:bank_id>/watermark/inpaint')
def bank_watermark_inpaint(bank_id):
    """Level 2 — repaint what is still flagged. {method:'auto'|'lama'|'klein'}.
    202/409/400/503 (503 carries the actionable reason: engine missing, GPU busy)."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_watermark_inpaint, _app(), LOCAL_USER, bank_id,
                  method=data.get('method') or 'auto')


@bp.post('/bank/<int:bank_id>/watermark/undo')
def bank_watermark_undo(bank_id):
    """Drop the cleaned versions of {image_ids} (empty = all) and re-flag them.
    Synchronous — it only deletes blobs we made."""
    data = request.get_json(silent=True) or {}
    try:
        n = banks.undo_watermark_clean(LOCAL_USER, bank_id,
                                       data.get('image_ids') or None)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'restored': n})


@bp.post('/bank/<int:bank_id>/watermark/dismiss')
def bank_watermark_dismiss(bank_id):
    """Rule {image_ids} NOT watermarked — they leave both cleaning levels and are
    never re-flagged by a later scan."""
    data = request.get_json(silent=True) or {}
    try:
        n = banks.dismiss_watermarks(LOCAL_USER, bank_id, data.get('image_ids'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'dismissed': n})


@bp.post('/bank/<int:bank_id>/framing')
def bank_framing(bank_id):
    """Classify every non-rejected image by shot type (face/bust/body/back),
    reusing the dataset Qwen3-VL classifier. {rescan:true} re-classifies scanned
    rows. 202/409/503."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_framing, _app(), LOCAL_USER, bank_id,
                  rescan=bool(data.get('rescan')))


@bp.get('/bank/<int:bank_id>/coverage')
def bank_coverage(bank_id):
    """Read-only coverage advice (idea by @antonp): what the kept set leans on and
    what's thin for a good LoRA, from data the passes already computed. 404 when
    the bank is gone."""
    payload = banks.coverage(LOCAL_USER, bank_id)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.post('/bank/<int:bank_id>/caption')
def bank_caption(bank_id):
    """Caption a selection (image_ids) or every non-rejected image, reusing the
    dataset caption engines. {force:true} re-captions already-captioned rows.
    {vocabulary} picks the register ('explicit'|'clinical'|'safe') — same lane as
    the dataset caption; invalid → 400. 202/409/503/400."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_caption, _app(), LOCAL_USER, bank_id,
                  ids=data.get('image_ids') or None, force=bool(data.get('force')),
                  vocabulary=data.get('vocabulary') or None)


@bp.post('/bank/<int:bank_id>/pipeline')
def bank_pipeline(bank_id):
    """Launch the chained "Launch all" triage pipeline. Body: {steps:[...],
    reject_flags:[...], resolve_dups:bool}. 202/409/400 — every step's own
    prerequisite is checked INSIDE the job (a missing extra skips that step,
    it never fails the launch)."""
    data = request.get_json(silent=True) or {}
    return _start(banks.start_pipeline, _app(), LOCAL_USER, bank_id,
                  steps=data.get('steps') or None,
                  reject_flags=data.get('reject_flags') or None,
                  resolve_dups=bool(data.get('resolve_dups')))


@bp.post('/bank/<int:bank_id>/promote')
def bank_promote(bank_id):
    data = request.get_json(silent=True) or {}
    try:
        dataset_id = int(data.get('dataset_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'dataset_id is required'}), 400
    return _start(banks.start_promote, _app(), LOCAL_USER, bank_id,
                  data.get('image_ids') or [], dataset_id)


@bp.post('/bank/<int:bank_id>/promote-to-bank')
def bank_promote_to_bank(bank_id):
    """⬆ Promote's SECOND destination: copy the selection into a brand-new bank
    instead of a dataset — isolating candidates out of a big dump without
    committing them to a training container. Same shape as /bank/from-dataset:
    202 + background job, and the new bank's id back so the UI can jump to the
    bank being filled. Empty image_ids = every kept image.

    The files are COPIED: banks never share theirs, and the app rewrites images
    in place, so anything cheaper would make the two banks one at the first
    re-crop. 409 while another pass runs on the SOURCE bank."""
    data = request.get_json(silent=True) or {}
    try:
        new_id = banks.start_bank_promote(_app(), LOCAL_USER, bank_id,
                                          data.get('image_ids') or [],
                                          data.get('name'))
    except bank_jobs.BankJobBusy as e:
        return jsonify({'error': str(e)}), 409
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'id': new_id}), 202


@bp.get('/bank/<int:bank_id>/selection-size')
def bank_selection_size(bank_id):
    """How many images a promotion would copy and what they WEIGH — the number
    the confirmation shows before the click. ?ids=1,2,3 for a selection, absent
    for every kept image. Images are ~300 KB apiece so this is usually a
    footnote; video is three orders of magnitude above, which is exactly why the
    dialog states the measured figure instead of assuming one."""
    raw = request.args.get('ids')
    ids = [int(p) for p in raw.split(',') if p.strip().isdigit()] if raw else []
    out = banks.selection_size(LOCAL_USER, bank_id, ids)
    if out is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(out)


@bp.get('/bank/<int:bank_id>/promotable')
def bank_promotable(bank_id):
    """How many kept images 'promote all' would copy into ?dataset_id right now
    — the honest count for the promote modal (per-target: images already on
    OTHER datasets still count)."""
    try:
        dataset_id = int(request.args.get('dataset_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'dataset_id is required'}), 400
    n = banks.promotable_count(LOCAL_USER, bank_id, dataset_id)
    if n is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify({'count': n})


@bp.post('/bank/<int:bank_id>/cancel')
def bank_cancel(bank_id):
    return jsonify({'ok': bank_jobs.cancel(bank_id)})


@bp.get('/bank/<int:bank_id>/dup-groups')
def bank_dup_groups(bank_id):
    try:
        offset = int(request.args.get('offset') or 0)
        limit = int(request.args.get('limit') or 50)
    except ValueError:
        offset, limit = 0, 50
    payload = banks.dup_groups_payload(LOCAL_USER, bank_id,
                                       offset=offset, limit=limit)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.post('/bank/<int:bank_id>/dups/resolve')
def bank_dups_resolve(bank_id):
    data = request.get_json(silent=True) or {}
    strategy = data.get('strategy') or 'best'
    if strategy not in ('best', 'first'):
        return jsonify({'error': 'strategy must be best or first'}), 400
    try:
        out = banks.resolve_dups(LOCAL_USER, bank_id, strategy=strategy,
                                 group=data.get('group'),
                                 keep_ids=data.get('keep_ids'),
                                 respect_existing_keep=False)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.get('/bank/<int:bank_id>/semantic-dup-groups')
def bank_semantic_dup_groups(bank_id):
    try:
        offset = int(request.args.get('offset') or 0)
        limit = int(request.args.get('limit') or 50)
    except ValueError:
        offset, limit = 0, 50
    payload = banks.semantic_dup_groups_payload(LOCAL_USER, bank_id,
                                                offset=offset, limit=limit)
    if payload is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(payload)


@bp.post('/bank/<int:bank_id>/semantic-dups/resolve')
def bank_semantic_dups_resolve(bank_id):
    data = request.get_json(silent=True) or {}
    strategy = data.get('strategy') or 'best'
    if strategy not in ('best', 'first'):
        return jsonify({'error': 'strategy must be best or first'}), 400
    try:
        out = banks.resolve_semantic_dups(LOCAL_USER, bank_id, strategy=strategy,
                                          group=data.get('group'),
                                          keep_ids=data.get('keep_ids'),
                                          respect_existing_keep=False)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/images/status')
def bank_images_status(bank_id):
    data = request.get_json(silent=True) or {}
    try:
        n = banks.set_status(LOCAL_USER, bank_id, data.get('ids') or [],
                             data.get('status'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'changed': n})


@bp.post('/bank/<int:bank_id>/rotate')
def bank_rotate(bank_id):
    """Turn {ids} by {degrees} CLOCKWISE (90/180/270, negative = left).

    Idea by 1Tomber (GitHub #17). Synchronous and cheap: it writes ONE integer
    per row — the user's own files are never touched, the turned copy is built
    lazily by the resolver on the next read."""
    data = request.get_json(silent=True) or {}
    try:
        result = banks.rotate_images(LOCAL_USER, bank_id, data.get('ids'),
                                     data.get('degrees'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **result})


@bp.post('/bank/<int:bank_id>/apply-flags')
def bank_apply_flags(bank_id):
    data = request.get_json(silent=True) or {}
    try:
        out = banks.apply_flags(LOCAL_USER, bank_id, data.get('flags') or [])
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, 'rejected': out})


def _curation_filters(data):
    """The shared candidate-pool filters for the curation selectors — the same
    facets as the grid (status ∩ flag ∩ cluster ∩ style ∩ subfolder ∩ search),
    read out of a JSON body. Unknown keys (e.g. the grid's ``sort``) are ignored."""
    def _int(name):
        v = data.get(name)
        try:
            return int(v) if v not in (None, '') else None
        except (TypeError, ValueError):
            return None

    subfolder = data.get('subfolder')
    return {
        'status': data.get('status') or None,
        'flag': data.get('flag') or None,
        'cluster': _int('cluster'),
        'style': _int('style'),
        # '' is a meaningful subfolder (bank root); '__all__'/None mean "no scope".
        'subfolder': subfolder if subfolder not in (None, '__all__') else None,
        'search': data.get('search') or None,
    }


@bp.post('/bank/<int:bank_id>/select-diverse')
def bank_select_diverse(bank_id):
    """Farthest-point selection of the N most VARIED images in the current filter,
    reusing the ✨ Score embeddings (no GPU). Returns the chosen ids for the UI to
    check — never mutates. 400 with a "run Score first" hint when unscored.

    {typicality} (0–1) tempers the sampling so isolated aberrations stop winning
    on isolation alone; omitted ⇒ the service default, an explicit 0 ⇒ the
    historical pure farthest-point behaviour."""
    data = request.get_json(silent=True) or {}
    try:
        n = int(data.get('n') or 60)
    except (TypeError, ValueError):
        n = 60
    typ = data.get('typicality')
    try:
        typ = banks._TYPICALITY_DEFAULT if typ in (None, '') else float(typ)
    except (TypeError, ValueError):
        typ = banks._TYPICALITY_DEFAULT
    try:
        out = banks.select_diverse(LOCAL_USER, bank_id, n=n, typicality=typ,
                                   filters=_curation_filters(data))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/select-similar')
def bank_select_similar(bank_id):
    """Rank the current filter by CLIP similarity to a reference bank image
    ({ref_id}); returns the top-N ids (or everything ≥ {min_score}) for the UI to
    check. Reuses the ✨ Score embeddings (no GPU). 400 when unscored / bad ref."""
    data = request.get_json(silent=True) or {}
    try:
        ref_id = int(data.get('ref_id'))
    except (TypeError, ValueError):
        return jsonify({'error': 'ref_id is required'}), 400
    try:
        n = int(data.get('n') or 60)
    except (TypeError, ValueError):
        n = 60
    min_score = data.get('min_score')
    try:
        min_score = float(min_score) if min_score not in (None, '') else None
    except (TypeError, ValueError):
        min_score = None
    try:
        out = banks.select_similar(LOCAL_USER, bank_id, ref_id, n=n,
                                   min_score=min_score,
                                   filters=_curation_filters(data))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.post('/bank/<int:bank_id>/search-text')
def bank_search_text(bank_id):
    """Rank the current filter by CLIP similarity to a written QUERY. Reuses the
    ✨ Score embeddings; only the phrase is encoded, in the ML interpreter.

    Top-N only, and deliberately NO min_score — unlike select-similar. On a real
    bank the correct-hit and unrelated-pair score distributions overlap (correct
    0.177-0.233, unrelated up to 0.197), so no threshold separates them and a
    knob here would be a control over a boundary that does not exist. See
    ``banks.search_by_text``.

    400 = the request cannot be answered (no query, bank never scored).
    503 = the FEATURE is unavailable here (no torch/open_clip, encoder failed) —
    a different thing, and the UI says so differently: one is "do this first",
    the other is "this install cannot do this at all"."""
    from ..services.clip_text_encoder import TextEncodeError
    data = request.get_json(silent=True) or {}
    try:
        n = int(data.get('n') or 60)
    except (TypeError, ValueError):
        n = 60
    try:
        out = banks.search_by_text(LOCAL_USER, bank_id, data.get('query'), n=n,
                                   filters=_curation_filters(data))
    except TextEncodeError as e:
        return jsonify({'error': str(e), 'reason': 'encoder_unavailable'}), 503
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    return jsonify({'ok': True, **out})


@bp.get('/bank/text-search/status')
def bank_text_search_status():
    """Is text search available, is the model already warm, how many phrases are
    cached, would a download be needed — everything the UI needs to set
    expectations BEFORE the click rather than after an unexplained wait."""
    from ..services import clip_text_encoder
    return jsonify({'ok': True, **clip_text_encoder.status()})


@bp.post('/bank/text-search/release')
def bank_text_search_release():
    """Reap the warm text encoder now (~2.4 GB back). Called when the search
    panel closes; the idle timer is the backstop for a tab that just went away."""
    from ..services import clip_text_encoder
    return jsonify({'ok': True, 'released': clip_text_encoder.release()})


@bp.get('/bank/<int:bank_id>/delete-rejected/preview')
def bank_delete_rejected_preview(bank_id):
    """What 🗑 Delete rejected would really do, for the confirmation dialog: how
    many files, where they would go, and which OTHER banks share them (nested
    source folders make one bank's cleanup another bank's amputation)."""
    out = banks.rejected_delete_preview(LOCAL_USER, bank_id)
    if out is None:
        return jsonify({'error': 'not found'}), 404
    return jsonify(out)


@bp.post('/bank/<int:bank_id>/delete-rejected')
def bank_delete_rejected(bank_id):
    """Destructive: delete the SOURCE files of every rejected image from disk
    (OS trash, else the app's own trash, else a permanent delete) and drop their
    rows. The ONLY bank action that writes to the source folder — the front-end
    gates it behind a type-DELETE confirmation fed by the preview above."""
    try:
        out = banks.delete_rejected(LOCAL_USER, bank_id)
    except ValueError:
        return jsonify({'error': 'not found'}), 404
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 409
    return jsonify({'ok': True, **out})


def _row_or_404(bank_id, image_id):
    bank = banks.get_bank(LOCAL_USER, bank_id)
    if not bank:
        return None, None
    row = BankImage.query.filter_by(id=image_id, bank_id=bank_id).first()
    return bank, row


@bp.get('/bank/<int:bank_id>/thumb/<int:image_id>')
def bank_thumb(bank_id, image_id):
    bank, row = _row_or_404(bank_id, image_id)
    if not bank or not row:
        return jsonify({'error': 'not found'}), 404
    tpath = banks.ensure_thumb(bank, row)
    if not tpath:
        return jsonify({'error': 'unreadable'}), 404
    return send_file(tpath, mimetype='image/webp', max_age=3600)


@bp.get('/bank/<int:bank_id>/file/<int:image_id>')
def bank_file(bank_id, image_id):
    """The full-size image. Serves the watermark-cleaned version when one exists;
    ?original=1 serves the untouched source instead — that pair IS the before/after
    comparison (no third lightbox needed)."""
    bank, row = _row_or_404(bank_id, image_id)
    if not bank or not row:
        return jsonify({'error': 'not found'}), 404
    path = (banks.abs_image_path(bank, row)
            if request.args.get('original') in ('1', 'true')
            else banks.resolved_image_path(bank, row))
    if not path or not os.path.isfile(path):
        return jsonify({'error': 'file missing'}), 404
    return send_file(path, max_age=0)
