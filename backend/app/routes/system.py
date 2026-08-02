"""🖥️ System pickers — server-side folder selection for the Browse… field.

The folders the app works on live on the machine RUNNING the server (not the
browser's), so these two endpoints do the selecting there:

  POST /api/system/pick-folder   pops the OS-native dialog on the server desktop.
  GET  /api/system/list-folders  read-only in-app browser (drives + subfolders).

It also hosts the app-wide ComfyUI recovery surface (GET/POST
/api/system/comfyui-recovery…), which belongs here for the same reason: it is
about the machine running the server, not about any one dataset.

pick-folder never 500s on the expected "no desktop here" case — it answers 200
with {available:false} so the UI silently falls back to the in-app browser
(LAN/tablet/Linux) instead of flashing a scary error toast.
"""
import json
import logging

from flask import Blueprint, jsonify, request

from ..config import LOCAL_USER
from ..services import folder_picker
from ._common import _map_error, _require_comfyui

logger = logging.getLogger(__name__)

bp = Blueprint('system', __name__, url_prefix='/api/system')


@bp.post('/pick-folder')
def pick_folder():
    data = request.get_json(silent=True) or {}
    initial = (data.get('initial') or '').strip() or None
    try:
        path = folder_picker.open_native_folder_dialog(initial)
    except folder_picker.NativePickerUnavailable as e:
        # Expected on a headless / Linux / service-session server: 200 so the
        # front falls back to the in-app browser without an error toast.
        return jsonify({'available': False, 'reason': str(e)})
    if path is None:
        return jsonify({'available': True, 'cancelled': True})
    return jsonify({'available': True, 'path': path})


@bp.get('/list-folders')
def list_folders():
    path = request.args.get('path') or None
    try:
        return jsonify(folder_picker.list_subfolders(path))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except PermissionError:
        return jsonify({'error': 'Permission denied for this folder.'}), 403


# --- ComfyUI recovery, app-wide ---------------------------------------------
# The durable recovery barrier is global: one stalled prompt blocks EVERY local
# generation in the app. Its resolution used to be reachable from exactly one
# place — the Stop button of the dataset that happened to own the job — so a
# user working anywhere else met a refusal with no visible way out. These two
# endpoints are the app-wide surface: what is stuck, and one confirmed action
# to clear it, wherever the user happens to be standing.


def _stalled_job_facts(job_id):
    """(stalled_since, metadata) for the barrier's queue row, best effort."""
    from ..models import ImageGenerationQueue
    row = (ImageGenerationQueue.query
           .filter_by(job_id=str(job_id))
           .with_entities(ImageGenerationQueue.last_heartbeat,
                          ImageGenerationQueue.started_at,
                          ImageGenerationQueue.created_at,
                          ImageGenerationQueue.job_metadata).first())
    if row is None:
        return None, {}
    when = row.last_heartbeat or row.started_at or row.created_at
    try:
        metadata = json.loads(row.job_metadata or '{}')
    except (TypeError, ValueError):
        metadata = {}
    return (when.isoformat() + 'Z' if when else None,
            metadata if isinstance(metadata, dict) else {})


def _dataset_name(dataset_id):
    if dataset_id is None:
        return None
    from ..services import face_dataset_service as fds
    try:
        ds = fds.get_dataset(LOCAL_USER, int(dataset_id))
    except (TypeError, ValueError):
        return None
    return getattr(ds, 'name', None) if ds else None


def _recovery_snapshot():
    """What is blocking generation right now, or None when nothing is."""
    from ..job_queue import (COMFYUI_RECOVERY_REQUIRED_MESSAGE, queue_manager)
    owner = queue_manager.get_comfyui_stalled_barrier()
    if owner is None:
        if not queue_manager.has_comfyui_stalled_barrier():
            return None
        # Present but unreadable. It still blocks every generation, and no code
        # may guess its way out of it — but hiding it is how a user ends up
        # thinking the app is simply broken.
        return {'kind': 'unreadable', 'job_id': None, 'can_confirm_restart': False,
                'message': COMFYUI_RECOVERY_REQUIRED_MESSAGE,
                'detail': ('LDS found an invalid ComfyUI recovery record. Restart LDS '
                           'and check the server log before starting new generations.')}
    stalled_since, metadata = _stalled_job_facts(owner.get('job_id'))
    dataset_id = owner.get('dataset_id') or metadata.get('dataset_id')
    try:
        dataset_id = int(dataset_id) if dataset_id is not None else None
    except (TypeError, ValueError):
        dataset_id = None
    return {
        'kind': owner.get('kind'),
        'job_id': owner.get('job_id'),
        'dataset_id': dataset_id,
        'dataset_name': _dataset_name(dataset_id),
        'variation_label': metadata.get('variation_label'),
        'run_id': owner.get('run_id'),
        'cell_id': owner.get('cell_id'),
        'stalled_since': stalled_since,
        'message': COMFYUI_RECOVERY_REQUIRED_MESSAGE,
        'detail': owner.get('reason'),
        # Every readable barrier can be resolved from here once the user
        # confirms the restart; the backend still refuses anything it cannot
        # prove or identify, with its own message.
        'can_confirm_restart': True,
    }


@bp.get('/comfyui-recovery')
def comfyui_recovery_state():
    """Poll target for the app-wide banner.

    Reading this also attempts the provable automatic clear, so an install left
    blocked overnight heals as soon as any page is open and ComfyUI is back —
    without the user having to click the thing that was refused.
    """
    from ..job_queue import auto_resolve_comfyui_barrier, peek_auto_recovery_notice
    if auto_resolve_comfyui_barrier() is not None:
        logger.info('system: ComfyUI recovery barrier cleared automatically on poll')
    return jsonify({'ok': True,
                    'recovery': _recovery_snapshot(),
                    'auto_cleared': peek_auto_recovery_notice()})


@bp.post('/comfyui-recovery/resolve')
def comfyui_recovery_resolve():
    """"I restarted ComfyUI — clear it", from anywhere in the app."""
    data = request.get_json(silent=True) or {}
    if data.get('confirmed_comfyui_restart') is not True:
        return jsonify({'error': 'Confirm that you restarted ComfyUI before '
                                 'clearing this paused job.'}), 400
    state = _recovery_snapshot()
    if state is None:
        return jsonify({'ok': True, 'cleared': 0, 'already_clear': True})
    # A confirmation is only meaningful when the replacement process answers
    # NOW; a cached green probe is not a restart gate.
    gate = _require_comfyui(force=True)
    if gate:
        return gate
    if state['kind'] == 'unreadable':
        return jsonify({'ok': False, 'error': state['detail']}), 409

    from ..job_queue import auto_resolve_comfyui_barrier
    if state['kind'] == 'prompt':
        # A known prompt id is checkable, so the user's word is not the
        # authority here — ComfyUI's answer is. Still queued/running means the
        # job is alive and must not be cancelled behind the user's back.
        if auto_resolve_comfyui_barrier() is not None:
            return jsonify({'ok': True, 'cleared': 1})
        return jsonify({'ok': False, 'error': (
            'ComfyUI still reports this generation, or it did not answer. If you '
            'just restarted it, wait a few seconds and try again; if the job is '
            'still running there, let it finish.')}), 409

    try:
        if state.get('run_id'):
            from ..services import lora_test_studio as lts
            cleared = lts.confirm_unknown_comfyui_restart(
                LOCAL_USER, run_id=state['run_id'], restart_confirmed=True)
        elif state.get('cell_id') and state.get('dataset_id') is not None:
            from ..services import lora_test_studio as lts
            cleared = lts.confirm_unknown_comfyui_restart(
                LOCAL_USER, dataset_id=state['dataset_id'], restart_confirmed=True)
        elif state.get('dataset_id') is not None:
            from ..services import face_dataset_service as fds
            cleared = fds.confirm_unknown_generation_restart(
                LOCAL_USER, state['dataset_id'], restart_confirmed=True)
        else:
            return jsonify({'ok': False, 'error': (
                'LDS cannot tell which dataset or run this paused job belongs to. '
                'Restart LDS and check the server log.')}), 409
    except Exception as e:
        return _map_error(e)
    if not cleared:
        return jsonify({'ok': False, 'error': (
            'The paused job could not be cleared. Refresh the page and try again; '
            'if it persists, check the server log.')}), 409
    return jsonify({'ok': True, 'cleared': cleared})
