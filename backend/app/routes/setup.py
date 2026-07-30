"""Setup API: auto-detect installed tools + run the whitelisted one-click installs."""
from flask import Blueprint, jsonify, request

from .. import capabilities
from .. import setup_installer
from ..services import comfyui_control

bp = Blueprint('setup', __name__, url_prefix='/api/setup')

_COMFYUI_START_LOOPBACKS = {'127.0.0.1', '::1', '::ffff:127.0.0.1'}


@bp.get('/autodetect')
def setup_autodetect():
    """Discover already-installed tools (Ollama/ComfyUI/ai-toolkit) so the wizard
    can fill config itself. Reachable-port hits are safe to apply; disk paths are
    suggestions the UI confirms."""
    return jsonify(capabilities.autodetect())


@bp.get('/comfyui-dir')
def setup_validate_comfyui_dir():
    """Classify a candidate ComfyUI folder WITHOUT saving it, so the wizard can give
    immediate, actionable feedback as the field is edited — a wrong path, an empty
    folder, or the launcher/parent folder (with the child to adopt) instead of a
    blanket "invalid" that only shows up after a save. Read-only, cheap (a couple of
    stat calls), never raises. `?path=` is the raw folder string the user typed."""
    return jsonify(capabilities.classify_comfyui_dir(request.args.get('path', '')))


@bp.post('/comfyui/start')
def start_comfyui():
    """Explicitly start LDS's narrowly validated portable ComfyUI instance.

    This endpoint intentionally accepts no path, flags, query values, or JSON. The
    service reads only the already-persisted local configuration and owns no process
    started by the user's normal ComfyUI launcher.
    """
    # The application-wide LAN guard can allow authenticated remote clients. Starting
    # a local executable is stricter: it is available only to the exact local peers.
    if request.remote_addr not in _COMFYUI_START_LOOPBACKS:
        return jsonify({'error': 'This action is only available from this computer.'}), 403
    # No client input may influence the executable, cwd, or arguments. Do not parse a
    # body at all; reject it instead. The CSRF header remains the only required POST
    # metadata and is enforced by Flask-WTF before this handler in normal operation.
    if request.query_string or request.content_length not in (None, 0) or request.content_type:
        return jsonify({'error': 'This action does not accept options.'}), 400
    return jsonify(comfyui_control.start_comfyui()), 200


@bp.get('/comfyui-folders')
def setup_resolve_comfyui_folders():
    """Resolve the four ComfyUI working folders (output/input/models/loras) for the
    values currently TYPED in Settings, without saving anything, so each override
    field can show the effective path it falls back to and flag one that is not on
    disk. `?base_dir=` plus optional `?output_dir=&input_dir=&models_dir=&loras_dir=`.

    `?detect=1` additionally asks the running ComfyUI which custom folders it was
    launched with (its own argv, via /system_stats) so the UI can offer them in one
    click. That is one short network call, hence opt-in; `detected` is {} whenever
    ComfyUI is unreachable, too old to report its argv, or was started with no
    custom folder flags — never a guessed path."""
    base_dir = request.args.get('base_dir', '')
    overrides = {k: request.args.get(k, '') for k in
                 ('output_dir', 'input_dir', 'models_dir', 'loras_dir')}
    payload = {'folders': capabilities.classify_comfyui_folders(base_dir, overrides),
               'detected': {}}
    if request.args.get('detect'):
        payload['detected'] = capabilities.detect_comfyui_folders()
    return jsonify(payload)


@bp.post('/install/<action>')
def start_install(action):
    if action not in setup_installer.INSTALL_ACTIONS:
        return jsonify({'error': f'unknown action: {action}'}), 404
    try:
        state = setup_installer.start(action)
    except setup_installer.AlreadyRunning:
        return jsonify({'error': 'install already running'}), 409
    except setup_installer.Precondition as e:
        return jsonify({'error': str(e)}), 400
    return jsonify(state)


@bp.get('/install/<action>/status')
def install_status(action):
    if action not in setup_installer.INSTALL_ACTIONS:
        return jsonify({'error': f'unknown action: {action}'}), 404
    return jsonify(setup_installer.status(action))


@bp.get('/install-all/plan')
def install_all_plan():
    """What 'Install everything' WOULD queue for the current machine — the missing
    components the app can install itself (ML extras, the vision model when Ollama is
    up, the Klein weights when a valid ComfyUI is set). Read-only, so the button can
    show the plan (and an accurate 'X items') before the user commits."""
    caps = capabilities.probe()
    return jsonify({'plan': setup_installer.install_all_plan(caps)})


@bp.post('/install-all')
def start_install_all():
    """One click that queues every install in the plan above. Reuses the per-action
    serialization (pip FIFO) and preconditions, so it's a safe fan-out — nothing new to
    race. Returns the plan + each action's status for the global progress bar."""
    caps = capabilities.probe()
    return jsonify(setup_installer.start_all(caps))


@bp.get('/install-group/<group>/plan')
def install_group_plan(group):
    """What the one-click install for a NAMED group (today: the Krea 2 Edit
    engine — node pack + four weights) would queue right now. Read-only, so the
    button can show its own count and stay honest about what is already there."""
    if group not in setup_installer._INSTALL_GROUPS:
        return jsonify({'error': f'unknown group: {group}'}), 404
    # force=True: this plan is read right after the ComfyUI folder is saved, and a
    # 30 s-stale probe would answer "nothing to install" for a machine that has
    # nothing installed — the exact opposite of the truth.
    return jsonify({'plan': setup_installer.install_group_plan(
        group, capabilities.probe(force=True))})


@bp.post('/install-group/<group>')
def start_install_group(group):
    """Install a whole engine in one click without dragging it into the
    unattended 'Install everything' plan — a second local engine is ~20 GB, so it
    downloads when it is ASKED for, not by default."""
    if group not in setup_installer._INSTALL_GROUPS:
        return jsonify({'error': f'unknown group: {group}'}), 404
    return jsonify(setup_installer.start_group(group, capabilities.probe(force=True)))


@bp.get('/install-all/status')
def install_all_status():
    """Batched status for the actions the caller is tracking (?actions=a,b,c) — one poll
    for the whole 'Install everything' run instead of one request per action."""
    actions = [a for a in (request.args.get('actions', '') or '').split(',') if a]
    return jsonify({'statuses': setup_installer.status_many(actions)})
