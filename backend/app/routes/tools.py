"""Model-file tools that are pure file-in / file-out work on this machine.

No ai-toolkit gate, no ComfyUI gate, no cloud: merging a LoRA into a base reads
two files and writes a third. It needs neither a GPU nor a training environment,
so gating it on either would refuse the one user who most needs it — someone
who trained on a rented machine and only has the weights.

It lives in its own blueprint rather than in ``training.py`` because it is not
training, and because ``training.py`` is already the largest route module in the
app. The fp8 tools are its neighbours in spirit; they stayed where they were
written, and moving them would break URLs the frontend already ships.
"""
from flask import Blueprint, current_app, jsonify, request

from ._common import _map_error

bp = Blueprint('tools', __name__, url_prefix='/api')


# --- Merge a LoRA into a base checkpoint -------------------------------------------
# The step between "I trained a LoRA" and "I have a model to publish". See
# `lora_merge` for what the arithmetic is and why the output says, in its own
# header, that it came from a merge and not from training.

@bp.post('/tools/lora-merge/plan')
def tools_lora_merge_plan():
    """What merging these LoRAs into this base would produce, or WHY it is refused.

    Always 200: the panel disables its button and shows the reason, instead of
    letting the user commit to a 26 GB write and meeting the refusal afterwards.
    Every condition the run needs is decided here — see ``lora_merge_job.plan``.
    """
    from ..services import lora_merge_job
    d = request.get_json(silent=True) or {}
    return jsonify(lora_merge_job.describe(
        d.get('base'), d.get('loras'),
        destination=d.get('destination'),
        destination_dir=d.get('destination_dir'),
        overwrite=bool(d.get('overwrite'))))


@bp.post('/tools/lora-merge')
def tools_lora_merge_start():
    from ..services import lora_merge_job
    d = request.get_json(silent=True) or {}
    try:
        info = lora_merge_job.start_async(
            current_app._get_current_object(), d.get('base'), d.get('loras'),
            destination=d.get('destination'),
            destination_dir=d.get('destination_dir'),
            overwrite=bool(d.get('overwrite')))
    except Exception as e:                     # noqa: BLE001 — mapped, not swallowed
        return _map_error(e)
    return jsonify({'ok': True, **info, 'status': lora_merge_job.status()})


@bp.get('/tools/lora-merge/status')
def tools_lora_merge_status():
    """Where the merge is. Also the place a state stranded by a restart is
    cleared: this is the first thing the panel asks after a reload, so a "merge"
    left running by a process that no longer exists is corrected before it can
    show a progress bar that never moves or refuse the next merge for six hours.
    """
    from ..services import lora_merge_job
    reconciled = lora_merge_job.reconcile()
    return jsonify({'ok': True, 'reconciled': reconciled,
                    **(lora_merge_job.status() or {})})


@bp.post('/tools/lora-merge/cancel')
def tools_lora_merge_cancel():
    from ..services import lora_merge_job
    return jsonify({'ok': True, **lora_merge_job.cancel()})
