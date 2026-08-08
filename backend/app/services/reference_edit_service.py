"""The reference-photo EDIT lane: cropping and auto-recropping the dataset's
primary reference, and the engine-driven edit that replaces it -- preflight,
enqueue, the ComfyUI output resolution that reads the result back, and the
keep / discard / commit endings.

Distinct from reference_photos_service.py, which owns the reference photos
themselves (extra refs and pose slots) as plain CRUD. This module owns the
asynchronous job around ONE of them.

Split out of face_dataset_service.py (2026-08, Phase 6 of a multi-phase file
split) -- pure move, no behavior change.
"""
import ntpath
import os
import posixpath
import re
import stat
import threading
import time
import uuid

from ..extensions import db
from ..models import FaceDatasetImage
from . import dataset_activity, reference_edit_jobs
from .engine_errors import EngineError
from .chatgpt_image import SubscriptionQuotaExceeded, SubscriptionUnavailable


def _ref_crop_source_path(ds) -> str:
    """The image a manual/auto re-crop reads from: the full-frame ORIGINAL when we
    kept one, else the cropped ref (legacy datasets uploaded before we stored the
    original — they can still be re-cropped, only not wider than the existing crop)."""
    name = ds.ref_original_filename or ds.ref_filename
    return os.path.join(_dataset_dir(ds.id), name)


def crop_reference(user_id, dataset_id, x, y, w, h):
    """Manually crop the dataset reference to (x,y,w,h), long side capped at 1024
    (never enlarged). The box is
    in the ORIGINAL's pixel space (the editor shows the original), and we write the
    derived square to ref_filename WITHOUT touching the original — so the user can
    re-crop wider or tighter any number of times."""
    with reference_mutation(dataset_id):
        ds = get_dataset(user_id, dataset_id)
        if not ds or not ds.ref_filename:
            return False
        ok, _scale = _crop_resize_file(
            _ref_crop_source_path(ds), x, y, w, h, dst=_ref_path(ds))
        if ok:
            invalidate_reference_edit(dataset_id)   # a pending Before/After is now stale
        return ok


def recrop_reference_auto(user_id, dataset_id):
    """Re-run the automatic head-crop on the ORIGINAL, overwriting ref_filename.
    Returns (ok, head_detected). CALLER holds the GPU vision window. Lets the user
    reset to the auto framing after manual edits, without re-uploading the photo."""
    with reference_mutation(dataset_id):
        ds = get_dataset(user_id, dataset_id)
        if not ds or not ds.ref_filename:
            return False, False
        try:
            with open(_ref_crop_source_path(ds), 'rb') as fh:
                raw = fh.read()
        except OSError:
            return False, False
        webp, detected = face_crop_to_square_webp(
            raw, pad=REF_CROP_PAD, return_detected=True)
        with open(_ref_path(ds), 'wb') as fh:
            fh.write(webp)
        invalidate_reference_edit(dataset_id)    # a pending Before/After is now stale
        return True, detected


def _edit_engine_call(engine, refs, prompt):
    """Dispatch an edit to an API engine: reference list + prompt -> edited bytes
    (or None). PURE dispatch — no file reading (the caller snapshots the refs in
    the REQUEST thread, never the worker). The reference is a face crop, so the
    edit keeps its framing (1:1) — an in-place change, not a re-composition. The
    ChatGPT auth lane is pinned once, like the batch, so it never silently
    reroutes onto the paid API key mid-call."""
    if engine not in API_ENGINES:
        raise ValueError(f'unknown edit engine: {engine}')
    generate = _api_generate_fn(engine)
    gen_kwargs = {'aspect_ratio': '1:1'}
    if engine == 'chatgpt':
        from .chatgpt_image import _use_subscription
        gen_kwargs['force_lane'] = 'subscription' if _use_subscription() else 'api'
    return generate(refs, prompt, **gen_kwargs)


def normalize_edit_engines(engines):
    """Canonical ordered engine selection for both legacy and batch requests."""
    raw_values = [engines] if isinstance(engines, str) else list(engines or ())
    selected = []
    for raw in raw_values:
        engine = str(raw or '').strip().lower()
        if not engine:
            raise ValueError('select at least one engine for the reference edit')
        if engine not in selected:
            selected.append(engine)
    allowed = editable_engines()
    if not selected:
        raise ValueError('select at least one engine for the reference edit')
    # Apply the finite bound after normalization/deduplication. Membership below
    # makes it structurally unreachable today, but keeps the contract explicit.
    if len(selected) > len(allowed):
        raise ValueError(f'select at most {len(allowed)} edit engines')
    if any(engine not in allowed for engine in selected):
        raise ValueError(edit_engine_choice_message())
    return tuple(selected)


def _preflight_local_reference_edit(ds, engine):
    """Run local preflights available without enqueuing a render."""
    if engine == KREA_ENGINE:
        from . import krea_edit_helper as helper
        helper.preflight()
    # Klein's complete admission check intentionally lives in enqueue_klein_edit;
    # all local enqueues finish before any paid API thread is started below.


def start_reference_edit(app, user_id, dataset_id, engine, prompt,
                         extra_edit_ref_bytes=None, retry_batch_id=None):
    """Start a background reference-edit job and RETURN AT ONCE (the request no
    longer blocks 1-3 min, so a backgrounded mobile tab can't kill it). Snapshots
    the reference + extras + modal images HERE, in the request thread (never
    re-read in the worker), registers the candidate job and a 'edit_reference'
    activity so the existing payload poll tracks it, then spawns the worker.

    Ref list sent to the engine: the primary reference FIRST (ChatGPT's
    /images/edits treats image[0] as the edit base), then the dataset's extra refs
    and any transient edit-reference images the user added in the modal — all ride
    along as identity anchors so the edit keeps the same face. Every API engine
    forwards the WHOLE list (OpenRouter as `input_references`); a model that takes
    fewer says so in its own error rather than having some silently dropped here.

    LOCAL engines (Klein, Krea 2 Edit) take a different route entirely — see
    _start_local_reference_edit: no blocking call, a ComfyUI queue job instead.
    They also take FEWER references (Klein: the dataset's extras, by path; Krea:
    the primary only), which is a fact of their graphs and is stated in the UI at
    pick time rather than discovered as a silent drop here.

    Raises ValueError for a bad engine / empty prompt / missing reference (the
    route maps it to 400/404)."""
    _guard_not_bank_export(dataset_id)
    engines = normalize_edit_engines(engine)
    prompt = (prompt or '').strip()
    if not prompt:
        raise ValueError('describe the edit first')
    # Capture the mutation epoch before *any* dataset/reference-state read. If a
    # mutation commits after this point, start_batch's CAS rejects the snapshot;
    # if it committed before this point, the reads below see the new state.
    reference_revision = reference_edit_jobs.reference_revision(dataset_id)
    ds = get_dataset(user_id, dataset_id)
    if not ds or not ds.ref_filename:
        raise ValueError('reference image required')
    if not os.path.exists(_ref_path(ds)):
        raise ValueError('reference image file missing')
    api_engines = tuple(item for item in engines if item in API_ENGINES)
    local_engines = tuple(item for item in engines if item in LOCAL_ENGINES)
    transient_refs = tuple(extra_edit_ref_bytes or ())
    if len(transient_refs) > MAX_EDIT_REFERENCE_UPLOADS:
        raise ValueError(
            f'add at most {MAX_EDIT_REFERENCE_UPLOADS} extra edit references')

    # One immutable primary+persistent-reference snapshot feeds every lane. API
    # siblings consume the bytes directly; local siblings below consume temporary
    # files written once from these exact bytes, never a later read of the master.
    dataset_ref_bytes = tuple(_all_ref_bytes(ds))
    refs = None
    if api_engines:
        snapshotted = list(dataset_ref_bytes)
        for index, raw in enumerate(transient_refs, 1):
            if raw:
                snapshotted.append(sanitize_external_reference(
                    raw, label=f'extra edit reference {index}'))
        refs = tuple(snapshotted)
    elif transient_refs:
        # Preserve the historical one-local-engine refusal. In a mixed batch the
        # uploads are valid API-only inputs and are not silently discarded.
        local = local_engines[0]
        raise ValueError(
            f'{engine_labels().get(local, local)} renders on your own GPU and cannot take '
            'the extra reference images added here — remove them, or pick an API engine')

    # Validate every selected local lane before replacing the current results.
    # Full enqueue happens before API threads below, closing remaining admission
    # gaps without ever billing a paid sibling first.
    for local in local_engines:
        _preflight_local_reference_edit(ds, local)
    dsdir = _dataset_dir(dataset_id)
    started = reference_edit_jobs.start_batch(
        dataset_id, dsdir, engines, prompt,
        expected_revision=reference_revision,
        expected_batch_id=retry_batch_id)
    batch_token, tokens = started['batch_token'], started['tokens']
    act_token = dataset_activity.begin(
        dataset_id, 'edit_reference', total=len(engines),
        engine=engines[0] if len(engines) == 1 else None)
    if not reference_edit_jobs.attach_activity(dataset_id, batch_token, act_token):
        dataset_activity.end(act_token)
        raise RuntimeError('reference edit was superseded while it was starting')

    # Prove admission for every local sibling before starting a paid API thread.
    # If the second local enqueue fails, clear cancels the first queue job and
    # closes the shared activity exactly once.
    local_snapshot_paths = []
    try:
        if local_engines:
            snapshot_tag = uuid.uuid4().hex[:8]
            for index, raw in enumerate(dataset_ref_bytes):
                filename = (
                    f'{user_id}{reference_edit_jobs.CANDIDATE_MARKER}'
                    f'snapshot_{snapshot_tag}_{index}.webp')
                path = os.path.join(dsdir, filename)
                local_snapshot_paths.append(path)
                write_image_atomic(path, raw)
        for local in local_engines:
            _enqueue_local_reference_edit(
                user_id, dataset_id, ds, local, prompt, tokens[local],
                local_snapshot_paths[0], local_snapshot_paths[1:])
    except Exception:
        reference_edit_jobs.clear_batch(dataset_id, batch_token, dsdir)
        raise
    finally:
        for path in local_snapshot_paths:
            reference_edit_jobs._unlink(path)

    for api_engine in api_engines:
        token = tokens[api_engine]
        try:
            threading.Thread(
                target=_run_reference_edit,
                args=(app, user_id, dataset_id, token, act_token,
                      api_engine, refs, prompt, True),
                daemon=True).start()
        except Exception as exc:
            logger.exception(
                'reference edit worker could not start (dataset %s, engine %s)',
                dataset_id, api_engine)
            reference_edit_jobs.set_failed(
                dataset_id, token, f'{api_engine}: failed to start edit: {exc}')
            _finish_reference_edit_activity(
                dataset_id, token, act_token, shared_activity=True)

    # The route returns this exact opaque id to the browser. Reading the registry
    # after this function returns would race a second tab starting another batch.
    return started['batch_id']


#: Reference images each LOCAL engine actually consumes, so the UI can say it at
#: pick time. Klein chains the dataset's extra refs as native ReferenceLatent
#: nodes; Krea's Krea2EditModelPatch takes ONE source (a second slot exists but
#: what it does to identity has not been measured — see enqueue_krea_edit).
#: Neither takes the modal's transient uploads: both engines want file PATHS and
#: the transient images are request-scoped bytes. Refused loudly by the route.
#: LOAD-BEARING, not documentation: the enqueue below reads it, so a third local
#: engine cannot be added without deciding what it does with the extra refs. The
#: values are mirrored in frontend EDIT_REF_SUPPORT (contract-tested), because
#: the UI has to say this at pick time, not discover it as a silent drop.
LOCAL_EDIT_REF_SUPPORT = {'klein': 'dataset_only', 'krea': 'modal_one',
                          # H3 is GENERATE-ONLY (see GENERATE_ONLY_ENGINES), so
                          # this never reaches the modal. Declared anyway,
                          # because every local engine must say what it consumes
                          # instead of falling through to a default — and 'none'
                          # is in neither limit map below, so it takes nothing.
                          'minimax_h3': 'none'}

#: How many DATASET extras each support value forwards. None = no ceiling beyond
#: the dataset's own MAX_EXTRA_REFS. A support value absent from this map takes
#: none — which is what a newly added engine should do until someone decides.
LOCAL_EDIT_REF_LIMITS = {'dataset_only': None}

#: How many of the MODAL's own uploads each support value forwards. They reach a
#: local engine as temporary FILES written from the request bytes — the same
#: hand-off the primary reference already used, which is why "local engines
#: cannot take the images added here" was always a routing decision rather than
#: a limitation of the graphs.
MODAL_EDIT_REF_LIMITS = {'modal_one': 1}


def local_edit_extra_refs(engine, extra_ref_paths):
    """The DATASET extras THIS engine consumes, in order (Klein's angles).

    One place decides, so the enqueue below and what the modal claims can never
    disagree — the failure this prevents is a UI promising angles to an engine
    whose graph was never going to read them."""
    limit = LOCAL_EDIT_REF_LIMITS.get(LOCAL_EDIT_REF_SUPPORT.get(engine), 0)
    paths = list(extra_ref_paths or [])
    return paths if limit is None else paths[:limit]


def local_edit_modal_refs(engine, modal_ref_paths):
    """The MODAL's uploads THIS engine consumes, in order (Krea's second subject)."""
    limit = MODAL_EDIT_REF_LIMITS.get(LOCAL_EDIT_REF_SUPPORT.get(engine), 0)
    return list(modal_ref_paths or [])[:limit]


def local_engines_taking_dataset_refs(engines):
    """The selected local engines that read the dataset's extra angles. Empty
    means nothing will open them, which is what lets the caller skip writing
    temporary copies no consumer exists for."""
    return [e for e in (engines or [])
            if LOCAL_EDIT_REF_LIMITS.get(LOCAL_EDIT_REF_SUPPORT.get(e), 0) != 0]


def local_engines_taking_modal_refs(engines):
    """Selected local engines that read the dialog's own uploads. An empty result
    with uploads present is what turns them into a loud refusal instead of a
    silent drop."""
    return [e for e in (engines or [])
            if MODAL_EDIT_REF_LIMITS.get(LOCAL_EDIT_REF_SUPPORT.get(e), 0)]


def _enqueue_local_reference_edit(user_id, dataset_id, ds, engine, prompt, token,
                                  ref_path, extra_ref_paths, modal_ref_paths=()):
    """Reference edit on the user's OWN GPU: free, private, no key, no bill — and
    therefore the lane that makes "try five prompts until it's right" reasonable.

    It does NOT get its own waiting machinery. The edit is enqueued on the app's
    existing ComfyUI image queue exactly like a generated variation, and the queue
    worker's completion dispatch calls link_completed_reference_edit when it
    lands. The registry entry is what the modal polls either way, so the client
    sees one contract for both lanes (running -> ready|failed).

    Every local enqueue completes before a selected API thread starts. A missing
    weight or node pack therefore surfaces as the same actionable 409 as generate,
    and any already-enqueued local sibling is cancelled without billing an API."""
    meta = {'is_reference_edit': True, 'dataset_id': dataset_id}
    try:
        if engine == KREA_ENGINE:
            from . import krea_edit_helper as helper
            job_id = helper.enqueue_krea_edit(
                user_id=str(user_id), source_filename=os.path.basename(ref_path),
                source_path=ref_path, edit_prompt=prompt, extra_metadata=meta,
                # From the DIALOG, never from the dataset's angles: the `_b` slot
                # wants a different subject, and the dataset pool holds only more
                # views of the same one. One image — the slot has room for one.
                extra_ref_paths=local_edit_modal_refs(engine, modal_ref_paths))
        else:
            from .klein_edit_helper import enqueue_klein_edit
            # The dataset's extra refs DO reach Klein (native ReferenceLatent
            # chaining) — the same anchors the API lane sends as bytes. Gated on
            # the table above rather than on the engine name, so the two can't
            # disagree.
            extras = local_edit_extra_refs(engine, extra_ref_paths)
            job_id = enqueue_klein_edit(
                user_id=str(user_id), source_filename=os.path.basename(ref_path),
                source_path=ref_path, edit_prompt=prompt, extra_ref_paths=extras,
                # The dataset's model, like every other Klein lane: this edit
                # produces the REFERENCE the whole dataset is built from, so it is
                # the last place that should run on a different model than the
                # images it anchors. None (never chose) = the historical auto pick.
                klein_model=dataset_klein_model(ds),
                sampler_steps=_generation_steps(),
                # An EDIT must obey the instruction, not a style LoRA nobody
                # picked: node 139 is pinned at 0.8 in the workflow file and the
                # setting (default 0) is what decides it here.
                base_lora_strength=_generation_base_lora_strength(),
                extra_metadata=meta)
    except Exception as exc:
        from .klein_edit_helper import KleinModelsMissing
        from .krea_edit_helper import KreaModelsMissing
        if isinstance(exc, (KleinModelsMissing, KreaModelsMissing)):
            # Typed on purpose: the route turns these into the SAME auto-download
            # 409 the generate path returns. Flattening them to a ValueError would
            # downgrade "I've started fetching the weight" to a bare 400.
            raise
        logger.exception('local reference edit could not be queued (dataset %s)', dataset_id)
        raise ValueError(f'{engine_labels().get(engine, engine)}: {exc}') from exc
    if not reference_edit_jobs.attach_job(
            dataset_id, token, job_id, user_id=str(user_id)):
        # Superseded between the enqueue and here: cancel the render nobody awaits.
        _cancel_local_edit_job(job_id)
        raise RuntimeError('reference edit was superseded while it was starting')
    return job_id


def _cancel_local_edit_job(job_id):
    """Best-effort cancel of one just-enqueued job not owned by a live batch."""
    if not job_id:
        return
    try:
        from ..job_queue import queue_manager
        queue_manager.cancel_job(job_id)
    except Exception:
        logger.warning('reference edit: could not cancel queue job %s', job_id, exc_info=True)


def _finish_reference_edit_activity(dataset_id, token, fallback_act_token=None,
                                    shared_activity=False):
    """Advance one candidate and close the batch activity exactly once."""
    update = reference_edit_jobs.activity_update(dataset_id, token)
    if update is not None:
        progress_token = update.get('activity_token')
        if progress_token is not None:
            dataset_activity.progress(
                progress_token, done=update['done'], total=update['total'])
        if update.get('end_token') is not None:
            dataset_activity.end(update['end_token'])
        elif not update.get('managed') and not shared_activity:
            dataset_activity.end(fallback_act_token)
        return
    # Legacy direct worker calls register no shared token. A managed batch that
    # disappeared was already ended by its supersede/clear/TTL cleanup.
    if not shared_activity:
        dataset_activity.end(fallback_act_token)


def link_completed_reference_edit(job_id, filename, failed=False, reason=None):
    """Queue-worker callback for a LOCAL reference edit: turn the finished ComfyUI
    output into the candidate the modal is waiting on.

    Symmetric with link_completed_dataset_image, minus the DB row — a reference
    edit has no FaceDatasetImage; its whole state is the in-memory registry entry.
    No entry = the user discarded or superseded meanwhile: the output is deleted
    rather than left in ComfyUI's folder."""
    entry = reference_edit_jobs.find_by_job(job_id)
    dataset_id = entry['dataset_id'] if entry else None
    try:
        if entry is None:
            _drop_comfy_output(filename)
            return
        if failed:
            reference_edit_jobs.set_failed(
                dataset_id, entry['token'],
                f"{entry['engine']}: {reason or 'the render failed — see 🪵 Server log in Settings'}")
            return
        data = _read_comfy_output(filename)
        if not data:
            reference_edit_jobs.set_failed(
                dataset_id, entry['token'],
                f"{entry['engine']}: the finished image could not be retrieved from ComfyUI "
                '(not on disk, and the /view API fetch failed)')
            return
        # Same naming as the API worker's candidate — the marker is what keeps it
        # out of the grid and the backups; the owner prefix keeps it recognisable.
        cand_fn = (f"{entry.get('user_id') or 'local'}"
                   f'{reference_edit_jobs.CANDIDATE_MARKER}{uuid.uuid4().hex[:8]}.webp')
        cand_path = os.path.join(entry['dir'], cand_fn)
        write_image_atomic(cand_path, normalize_to_webp(data))
        _drop_comfy_output(filename)
        if not reference_edit_jobs.set_ready(dataset_id, entry['token'], cand_fn):
            reference_edit_jobs._unlink(cand_path)     # superseded: drop our orphan
    except Exception as exc:
        logger.exception('reference edit link failed (job %s)', job_id)
        if entry is not None:
            reference_edit_jobs.set_failed(dataset_id, entry['token'],
                                           f"{entry['engine']}: {exc}")
    finally:
        if entry is not None:
            _finish_reference_edit_activity(
                dataset_id, entry['token'], entry.get('act_token'),
                shared_activity=True)


def _validated_comfy_output_name(filename):
    """Return an opaque Comfy output basename, or ``None`` when untrusted.

    Comfy completion payloads cross a trust boundary.  In particular, a stale
    completion is still cleaned up, so accepting a path here would turn that
    cleanup into an arbitrary-file delete.  Keep the accepted shape deliberately
    narrower than a generic relative path: Comfy reference-edit outputs are
    always direct children of its output directory.
    """
    if not isinstance(filename, str) or not filename or '\x00' in filename:
        return None
    if filename in {'.', '..'} or filename != filename.strip():
        return None
    if '/' in filename or '\\' in filename or ':' in filename:
        return None
    if (os.path.isabs(filename) or ntpath.isabs(filename)
            or posixpath.isabs(filename) or ntpath.splitdrive(filename)[0]):
        return None
    if (os.path.basename(filename) != filename
            or ntpath.basename(filename) != filename
            or posixpath.basename(filename) != filename):
        return None
    return filename


def _resolve_comfy_output(filename):
    """Resolve a trusted direct-child output without traversing a reparse path.

    The boolean is distinct from ``path is not None``: a valid basename with no
    configured local output directory may still be fetched through Comfy's
    ``/view`` endpoint, while any rejected path must fail closed everywhere.
    """
    name = _validated_comfy_output_name(filename)
    d = _comfy_output_dir()
    if name is None:
        return None, None, False
    if not d:
        return name, None, True
    try:
        root = os.path.abspath(os.fspath(d))
        candidate = os.path.abspath(os.path.join(root, name))
        if os.path.normcase(os.path.commonpath((root, candidate))) != os.path.normcase(root):
            return name, None, False
    except (OSError, TypeError, ValueError):
        return name, None, False

    # Check the candidate and every existing ancestor using lstat, before a
    # content read/delete can follow a symlink or Windows junction/reparse point.
    current = candidate
    while True:
        _st, blocked = _safe_lstat(current)
        if blocked:
            return name, None, False
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    try:
        canonical_root = os.path.realpath(root)
        canonical_candidate = os.path.realpath(candidate)
        contained = os.path.commonpath((canonical_root, canonical_candidate))
        if os.path.normcase(contained) != os.path.normcase(canonical_root):
            return name, None, False
    except (OSError, ValueError):
        return name, None, False
    return name, candidate, True


def _comfy_output_path(filename):
    _name, candidate, allowed = _resolve_comfy_output(filename)
    return candidate if allowed else None


def _is_reparse_stat(st):
    attrs = getattr(st, 'st_file_attributes', 0)
    reparse_flag = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
    return stat.S_ISLNK(st.st_mode) or bool(attrs & reparse_flag)


def _safe_lstat(path):
    """Return (stat, blocked): metadata errors and reparse points fail closed."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return None, False
    except OSError:
        return None, True
    return st, _is_reparse_stat(st)


def _read_comfy_output(filename):
    """Bytes of a finished ComfyUI output, from disk when we can see its folder,
    else over the /view API (a custom or unconfigured output path). None when
    neither works."""
    name, p, allowed = _resolve_comfy_output(filename)
    if not allowed:
        return None
    if p:
        root_st, root_blocked = _safe_lstat(os.path.dirname(p))
        if root_blocked:
            return None
        file_st, file_blocked = _safe_lstat(p)
        if file_blocked or (file_st is not None and not stat.S_ISREG(file_st.st_mode)):
            return None
        if root_st is not None and not stat.S_ISDIR(root_st.st_mode):
            return None
        if file_st is not None:
            fd = None
            try:
                flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0)
                flags |= getattr(os, 'O_NOFOLLOW', 0)
                fd = os.open(p, flags)
                opened_st = os.fstat(fd)
                if (not stat.S_ISREG(opened_st.st_mode)
                        or not os.path.samestat(file_st, opened_st)):
                    return None
                with os.fdopen(fd, 'rb') as fh:
                    fd = None
                    return fh.read()
            except OSError:
                # A file that changed after lstat is treated as hostile rather
                # than retried through Comfy's /view endpoint.
                return None
            finally:
                if fd is not None:
                    os.close(fd)
    from ..utils.comfyui import fetch_output_image_bytes
    return fetch_output_image_bytes(name)


def _drop_comfy_output(filename):
    """Remove an unlinked ComfyUI output. Only ever called on a file this app just
    produced for a transient candidate — never on user data."""
    _name, p, allowed = _resolve_comfy_output(filename)
    if not allowed:
        return
    if not p:
        return
    root_st, root_blocked = _safe_lstat(os.path.dirname(p))
    file_st, file_blocked = _safe_lstat(p)
    if (root_blocked or file_blocked or root_st is None or file_st is None
            or not stat.S_ISDIR(root_st.st_mode)
            or not stat.S_ISREG(file_st.st_mode)):
        return
    try:
        os.remove(p)
    except OSError:
        pass


def _run_reference_edit(app, user_id, dataset_id, token, act_token, engine, refs,
                        prompt, shared_activity=False):
    """Worker body: call the engine, write the candidate, mark the job ready — all
    in a background thread (factored out so tests can call it synchronously).

    ORDERING (load-bearing): set_ready() runs BEFORE dataset_activity.end(). The
    payload poll stops the moment activity clears, with ONE final refresh — which
    must already see the ready candidate, or the modal stays on the spinner. A
    superseded worker (a newer edit started meanwhile) gets False from
    set_ready/set_failed and deletes its own orphan candidate."""
    with app.app_context():
        try:
            # Last reversible boundary. A delayed worker whose batch was already
            # discarded/superseded must not send (and bill) a provider request.
            if not reference_edit_jobs.claim_api_dispatch(dataset_id, token):
                return
            try:
                out = _edit_engine_call(engine, refs, prompt)
            except SubscriptionQuotaExceeded as e:
                reference_edit_jobs.set_failed(dataset_id, token, str(e))
                return
            except SubscriptionUnavailable as e:
                reference_edit_jobs.set_failed(dataset_id, token, f'chatgpt: {e}')
                return
            except EngineError as e:
                # A NAMED engine failure (no key, rejected key, no credits, unknown
                # model, a model that won't take reference images). The message is
                # already user-facing and actionable, so it is surfaced verbatim —
                # loudly, with the engine named — instead of a stack trace. Matters
                # most for OpenRouter, whose model is free text in Settings: a slug
                # that can't edit must say so, not look like a refused prompt.
                reference_edit_jobs.set_failed(dataset_id, token, f'{engine}: {e}')
                return
            if out is None:
                # Un refus NOMMÉ est déjà parti par `except EngineError` au-dessus
                # (EngineRefused en hérite) : ici il ne reste que le vide muet des
                # moteurs qui ne savent pas distinguer refus et hoquet.
                reference_edit_jobs.set_failed(dataset_id, token,
                                               f'{engine}: {_EMPTY_MSG}')
                return
            cand_fn = f'{user_id}{reference_edit_jobs.CANDIDATE_MARKER}{uuid.uuid4().hex[:8]}.webp'
            cand_path = os.path.join(_dataset_dir(dataset_id), cand_fn)
            write_image_atomic(cand_path, normalize_to_webp(out))
            # set_ready BEFORE the finally end() — see ORDERING above.
            if not reference_edit_jobs.set_ready(dataset_id, token, cand_fn):
                reference_edit_jobs._unlink(cand_path)      # superseded: drop our orphan
        except Exception as e:
            logger.exception('reference edit worker failed (dataset %s)', dataset_id)
            reference_edit_jobs.set_failed(dataset_id, token, f'{engine}: {e}')
        finally:
            _finish_reference_edit_activity(
                dataset_id, token, act_token,
                shared_activity=shared_activity)


def keep_reference_edit(user_id, dataset_id, engine=None, batch_id=None):
    """Promote the READY candidate to be the reference (reuses the atomic,
    fail-safe commit_edited_reference), then delete the candidate file + clear the
    job. Returns the new ref_filename, or None when there is no ready candidate
    (route -> 409) — including a candidate file that vanished under us."""
    # The mutation lock closes the claim -> file/DB promotion TOCTOU. Without it,
    # an upload/crop could commit and invalidate after claim_ready(), only for this
    # stale candidate to overwrite and delete that newer reference before the
    # post-commit revision check noticed.
    with reference_mutation(dataset_id):
        claim = reference_edit_jobs.claim_ready(
            dataset_id, engine, batch_id=batch_id)
        if not claim:
            return None
        dsdir = _dataset_dir(dataset_id)
        try:
            with open(os.path.join(dsdir, claim['candidate_filename']), 'rb') as fh:
                data = fh.read()
        except OSError:
            reference_edit_jobs.clear_claimed(
                dataset_id, claim['batch_token'], claim['claim_token'], dsdir)
            return None
        try:
            new_ref = commit_edited_reference(user_id, dataset_id, data)
        except Exception:
            # A failed write leaves both the old master and every candidate intact,
            # so the user can retry Keep after fixing the storage problem.
            reference_edit_jobs.release_claim(
                dataset_id, claim['batch_token'], claim['claim_token'])
            raise
        cleared = reference_edit_jobs.clear_claimed(
            dataset_id, claim['batch_token'], claim['claim_token'], dsdir,
            reference_mutated=True)
        if cleared is None:
            # Defensive for TTL/process-lifecycle anomalies. Real reference
            # mutations cannot interleave here because they use this same lock.
            reference_edit_jobs.invalidate(dataset_id, dsdir)
        return new_ref


def _clear_reference_edit(dataset_id):
    """Drop the pending edit, delete its candidate, and — for a LOCAL edit still
    rendering — cancel the ComfyUI job and close its activity. Without the cancel,
    abandoning a local edit left the GPU busy on a result nobody would ever see
    and the ✦ activity badge lit until the TTL."""
    reference_edit_jobs.clear(dataset_id, _dataset_dir(dataset_id))


def discard_reference_edit(dataset_id):
    """Drop a pending edit (running=abandon OR ready) and delete its candidate
    file. An API call already sent is still billed — honesty preserved, no
    'refund' implied; a local render is cancelled, because it can be."""
    _guard_not_bank_export(dataset_id)
    _clear_reference_edit(dataset_id)


def reference_mutation(dataset_id):
    """Context manager shared by every primary-reference mutation path."""
    _guard_not_bank_export(dataset_id)
    return reference_edit_jobs.reference_mutation(dataset_id)


def invalidate_reference_edit(dataset_id):
    """Drop any pending edit candidate when the reference itself changes
    (crop/recrop/change/keep): a Before/After computed from the OLD reference would
    be a visual lie. Idempotent — a no-op when nothing is pending."""
    with reference_mutation(dataset_id):
        reference_edit_jobs.invalidate(dataset_id, _dataset_dir(dataset_id))


def commit_edited_reference(user_id, dataset_id, image_bytes):
    """Serialize and promote edited bytes as the dataset's reference."""
    with reference_mutation(dataset_id):
        return _commit_edited_reference_locked(user_id, dataset_id, image_bytes)


def _commit_edited_reference_locked(user_id, dataset_id, image_bytes):
    """Promote an edited candidate (bytes) to BE the dataset reference. The edited
    image is the new source of truth, so it becomes BOTH ref_filename (working
    crop) and ref_original_filename (the full frame ✂ Crop re-reads) — a later
    crop widens back out INSIDE the edited frame; re-cropping the pre-edit
    original would drop the edit (e.g. the glasses just added).

    ATOMIC, fail-safe order: write the two NEW files and confirm they are on disk
    BEFORE unlinking the old ones, and only repoint the DB after. A failed write
    (unusable candidate bytes, full disk) leaves the dataset on its PREVIOUS
    reference — a Keep must never strand it with no reference. Deleting the old
    files is safe because every in-flight batch snapshotted the reference at
    launch (API reads the bytes before the thread starts; Klein copies the file
    into ComfyUI's input at enqueue), so nothing running depends on them.

    Returns the new ref_filename. Raises ValueError if the dataset/reference is
    gone; propagates the write error (old reference intact) on failure."""
    ds = get_dataset(user_id, dataset_id)
    if not ds or not ds.ref_filename:
        raise ValueError('reference image required')
    dsdir = _dataset_dir(dataset_id)
    old_ref, old_orig = ds.ref_filename, ds.ref_original_filename
    new_ref = f"{user_id}_datasetref_{uuid.uuid4().hex[:8]}.webp"
    new_orig = f"{user_id}_datasetreforig_{uuid.uuid4().hex[:8]}.webp"
    ref_path = os.path.join(dsdir, new_ref)
    orig_path = os.path.join(dsdir, new_orig)
    # 1) WRITE the new files (working ref ≤1024, full-frame original ≤2048).
    #    normalize_to_webp raises on unusable bytes BEFORE any file is created, so
    #    a corrupt candidate never touches the existing reference.
    try:
        webp = normalize_to_webp(image_bytes, size=1024)
        orig_webp = normalize_to_webp(image_bytes, size=2048)
        with open(ref_path, 'wb') as fh:
            fh.write(webp)
        with open(orig_path, 'wb') as fh:
            fh.write(orig_webp)
    except Exception:
        # Roll back any partial write; the old reference is untouched.
        for p in (ref_path, orig_path):
            try:
                os.remove(p)
            except OSError:
                pass
        raise
    # 2) VERIFY both landed before touching anything the dataset still points at.
    if not (os.path.exists(ref_path) and os.path.exists(orig_path)):
        for p in (ref_path, orig_path):
            try:
                os.remove(p)
            except OSError:
                pass
        raise RuntimeError('failed to write edited reference')
    # 3) REPOINT the dataset, then commit.
    ds.ref_filename = new_ref
    ds.ref_original_filename = new_orig
    db.session.commit()
    # 4) Only now delete the superseded files (nothing in flight depends on them).
    for fn in (old_ref, old_orig):
        if fn and fn not in (new_ref, new_orig):
            try:
                os.remove(os.path.join(dsdir, fn))
            except OSError:
                pass
    return new_ref

# --- Borrow: face_dataset_service.py primitives -----------------------------
# MUST stay at the bottom of this file, same reason as in the sibling split
# modules: this module and face_dataset_service.py import names from each
# other, and whichever side loads first must find the other fully defined by
# the time the reach-back import resolves. A name owned by ANOTHER split module
# is imported from that module directly, never through the parent's re-export.
from .face_dataset_service import (
    _guard_not_bank_export,
    get_dataset, dataset_klein_model, normalize_to_webp, write_image_atomic,
    link_completed_dataset_image, _ref_path, _crop_resize_file,
    _comfy_output_dir, _generation_steps, _generation_base_lora_strength,
    _dataset_dir,
    REF_CROP_PAD, MAX_EDIT_REFERENCE_UPLOADS, logger,
)
# Owned by sibling split modules -- imported from the owner, never through
# face_dataset_service's re-export, so no block here depends on the order in
# which the parent emits them.
from .dataset_generation_service import (
    API_ENGINES, LOCAL_ENGINES, KREA_ENGINE, _EMPTY_MSG, _api_generate_fn,
    engine_labels, editable_engines, edit_engine_choice_message,
)
from .dataset_import_service import face_crop_to_square_webp
from .reference_photos_service import _all_ref_bytes, sanitize_external_reference
