"""Let the POD talk to Hugging Face for us — push a master out, pull one back.

WHY THE POD AND NOT THIS MACHINE
--------------------------------
A dense Krea 2 checkpoint is ~26 GB. Both directions exist for one reason each,
and in both cases the pod is the only sane endpoint:

* **push** — the Hub copy of a finished run is what makes it RESUMABLE later.
  Uploading it from home would take hours of the user's uplink; the pod already
  has the file next to a datacenter link, and it is about to be destroyed
  anyway. It is a best-effort step: the local copy has already been harvested
  and verified by the time this runs, so a refused push (a full private quota,
  most often) costs resumability, never the run.
* **pull** — resuming a dense run means the checkpoint has to be sitting in the
  new pod's ``save_root`` before ai-toolkit starts, so its auto-resume finds it.
  A pod fetching it from the Hub is the FAST way to put it there: a datacenter
  link makes ~26 GB a matter of minutes, against hours of a home uplink billed
  at the pod's hourly rate the whole time.

  This paragraph used to say something else — that sending the local file was
  impossible, because the upload route built its whole request in memory "with
  a 300 s timeout". Both halves were true of the code as it stood and neither
  is true now (``pod_checkpoint_push``: the body is streamed, the file goes in
  resumable slices, and the timeout was never a duration cap — requests' is an
  inactivity timeout). It is corrected rather than deleted because this file's
  own reasoning was built on it: the Hub road is still the one to prefer, and
  it is now preferred for what it costs, not for what the other one could not
  do.

HOW THE CODE GETS THERE
-----------------------
The same two seams the fp8 export already uses, and nothing more:

* **write** — ``RemoteAiToolkit.seed_checkpoint`` places the credential file in
  a dedicated pod directory (the ai-toolkit UI has no shell, but its dataset
  upload route joins a relative path onto DATASETS_FOLDER).
* **execute** — ``vast_client.execute_command``, whose combined output appears
  at a URL when the command ends.

The programs below are CONSTANTS. Every user-influenced value (repository id,
filename, destination path) is passed as a shell-quoted ``argv`` entry and read
with ``sys.argv`` — nothing is ever interpolated into the program text, so a
repository name carrying quotes cannot become code.

WHAT IS PROVEN AND WHAT IS NOT
------------------------------
No pod is rented by the test suite, so what the tests assert is the exact
command, the exact payload and the exact handling of every answer — including
the ones that must NOT be read as success. The transfers themselves are proven
only in production.
"""
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

# Where the credential lands on the pod, relative to its DATASETS_FOLDER.
POD_DIR_NAME = '_lds_hub'
POD_TOKEN_NAME = 'hf_token.txt'

RESULT_PREFIX = 'LDS_HUB_RESULT'
_RESULT_RE = re.compile(re.escape(RESULT_PREFIX) + r'\s+(\{.*\})')

# Generous, and bounded on purpose: a 26 GB transfer on a datacenter link is
# minutes, but a degraded host must not hold a paid pod for ever.
DEFAULT_PUSH_BUDGET_SECONDS = 3600
DEFAULT_FETCH_BUDGET_SECONDS = 3600
_POLL_SECONDS = 15
_POLL_SLACK_SECONDS = 120

# The two programs. Kept free of single quotes: the whole text is passed to the
# shell inside single quotes, and escaping is not a thing to be clever about in
# a command that runs on someone else's machine.
PUSH_PROGRAM = (
    'import json,os,sys\n'
    'from huggingface_hub import HfApi\n'
    'src,repo,name,tokfile=sys.argv[1:5]\n'
    'out={"ok":False,"error":None,"uploaded":False,"bytes":0}\n'
    'try:\n'
    '    tok=None\n'
    '    if tokfile and os.path.isfile(tokfile):\n'
    '        tok=open(tokfile).read().strip() or None\n'
    '    out["bytes"]=os.path.getsize(src)\n'
    '    HfApi(token=tok).upload_file(path_or_fileobj=src,path_in_repo=name,'
    'repo_id=repo,repo_type="model")\n'
    '    out["ok"]=True\n'
    '    out["uploaded"]=True\n'
    'except Exception as e:\n'
    '    out["error"]=str(e)[:400]\n'
    'print("' + RESULT_PREFIX + ' "+json.dumps(out))\n'
)

FETCH_PROGRAM = (
    'import json,os,sys\n'
    'from huggingface_hub import hf_hub_download\n'
    'repo,name,dest,tokfile=sys.argv[1:5]\n'
    'out={"ok":False,"error":None,"bytes":0}\n'
    'try:\n'
    '    tok=None\n'
    '    if tokfile and os.path.isfile(tokfile):\n'
    '        tok=open(tokfile).read().strip() or None\n'
    '    d=os.path.dirname(dest)\n'
    '    os.makedirs(d,exist_ok=True)\n'
    # local_dir materialises a REAL file instead of a symlink into the Hub
    # cache: ai-toolkit globs the save_root and must find a file, and a moved
    # symlink would point at a blob path nothing else keeps alive.
    '    p=hf_hub_download(repo_id=repo,filename=name,repo_type="model",'
    'token=tok,local_dir=d)\n'
    '    if os.path.abspath(p)!=os.path.abspath(dest):\n'
    '        os.replace(p,dest)\n'
    '    out["bytes"]=os.path.getsize(dest)\n'
    '    out["ok"]=True\n'
    'except Exception as e:\n'
    '    out["error"]=str(e)[:400]\n'
    'print("' + RESULT_PREFIX + ' "+json.dumps(out))\n'
)


# vast's asynchronous command endpoint carries 16384 characters, and a lane that
# inlines its program into that ask has already been refused twice for going over
# (see cloud_quantize / dense_fp8_delivery.MAX_COMMAND_CHARS). The two programs
# here are inlined ON PURPOSE — they are ~600 characters of constant text, and
# shipping a file for them would need a second seam — so this ceiling is what
# keeps that decision honest: anything approaching it means something started
# being interpolated into the program instead of passed in argv.
MAX_COMMAND_CHARS = 4096


class PodHubError(RuntimeError):
    """The pod could not complete the transfer. The caller decides what it costs."""


def quote(value) -> str:
    """POSIX single-quoting for one argument. Never optional here: a repository
    id carries a dataset name, and a quoting mistake in a remote command is not
    a bug anyone finds twice."""
    return "'" + str(value).replace("'", "'\\''") + "'"


def pod_paths(datasets_folder: str) -> dict:
    root = str(datasets_folder or '').rstrip('/') or '/workspace/datasets'
    directory = f'{root}/{POD_DIR_NAME}'
    return {'root': root, 'dir': directory,
            'token': f'{directory}/{POD_TOKEN_NAME}'}


def _sized(command: str) -> str:
    if len(command) > MAX_COMMAND_CHARS:
        raise PodHubError(
            f'the pod command grew to {len(command)} characters (ceiling '
            f'{MAX_COMMAND_CHARS}) — refusing to send it')
    return command


def build_push_command(src_path, repo_id, path_in_repo, token_file='') -> str:
    return _sized(' '.join(
        ['python', '-c', quote(PUSH_PROGRAM), quote(src_path),
         quote(repo_id), quote(path_in_repo), quote(token_file or '')]))


def build_fetch_command(repo_id, filename, dest_path, token_file='') -> str:
    return _sized(' '.join(
        ['python', '-c', quote(FETCH_PROGRAM), quote(repo_id),
         quote(filename), quote(dest_path), quote(token_file or '')]))


def parse_result(output) -> dict | None:
    """The program's JSON verdict, out of a combined output that also carries
    whatever the pod's shell printed around it."""
    match = _RESULT_RE.search(str(output or ''))
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def upload_token(remote, paths, token, tmp_dir) -> bool:
    """Place the Hugging Face token in the pod's ``_lds_hub`` directory.

    In a FILE, never on the command line: a command line is visible in ``ps`` to
    every process on the pod and is stored verbatim by the provider next to the
    command's output.
    """
    if not token:
        return False
    os.makedirs(tmp_dir, exist_ok=True)
    local = os.path.join(tmp_dir, POD_TOKEN_NAME)
    with open(local, 'w', encoding='utf-8', newline='\n') as fh:
        fh.write(str(token))
    try:
        remote.seed_checkpoint(paths['root'], paths['dir'], POD_TOKEN_NAME, local)
    finally:
        try:
            os.remove(local)
        except OSError:
            pass
    return True


def _await_result(vast, result_url, budget_seconds, *, _sleep, _now,
                  should_cancel=None):
    deadline = _now() + budget_seconds + _POLL_SLACK_SECONDS
    while _now() < deadline:
        _sleep(_POLL_SECONDS)
        if should_cancel is not None and should_cancel():
            raise PodHubError('cancelled while the pod was transferring')
        try:
            output = vast.fetch_command_result(result_url)
        except Exception:
            output = None            # transient: keep waiting, never fail on it
        if output is not None:
            return output
    raise PodHubError('the pod did not report a transfer result in time')


def _run(remote, *, instance_id, token, command, budget_seconds, tmp_dir,
         vast=None, on_state=None, _sleep=time.sleep, _now=time.monotonic,
         should_cancel=None, need_token=True) -> dict:
    """Ship the credential, execute one command, wait for its verdict."""
    from . import vast_client
    vast = vast or vast_client
    if not instance_id:
        raise PodHubError('no pod instance to run the transfer on')
    settings = remote.get_settings() or {}
    paths = pod_paths(settings.get('DATASETS_FOLDER') or '')
    if need_token and not token:
        raise PodHubError('no Hugging Face token for the pod-side transfer')
    upload_token(remote, paths, token, tmp_dir)
    full = command(paths['token'] if token else '')
    if on_state:
        try:
            on_state('Asking the pod to talk to Hugging Face…')
        except Exception:
            pass
    output = _await_result(vast, vast.execute_command(instance_id, full),
                           budget_seconds, _sleep=_sleep, _now=_now,
                           should_cancel=should_cancel)
    parsed = parse_result(output)
    if parsed is None:
        raise PodHubError('the pod produced no result line')
    if not parsed.get('ok'):
        raise PodHubError(str(parsed.get('error') or 'the pod transfer failed'))
    return parsed


# The pod-side executor is not Hugging Face-specific: ship what the program
# needs, run ONE command, read ONE result line. `pod_checkpoint_push` runs its
# probe and assembly programs through this exact function rather than becoming a
# second `execute_command` caller — two implementations of "what does a failure
# on a rented pod look like" is the kind of divergence that only ever shows up
# in production, on the machine nobody can attach a debugger to.
run_program = _run


def push_master(remote, *, instance_id, src_path, repo_id, path_in_repo,
                hf_token, tmp_dir, budget_seconds=DEFAULT_PUSH_BUDGET_SECONDS,
                vast=None, on_state=None, _sleep=time.sleep,
                _now=time.monotonic) -> dict:
    """Upload ONE file from the pod to the private delivery repository.

    Returns ``{'state','detail','result'}`` and NEVER raises: this runs after
    the local copy is harvested and verified, so its only possible cost is
    resumability. A full private quota — the exact wall run #146 hit, and the
    reason this is no longer done during training — must read as a note, not as
    a failed run.
    """
    out = {'state': 'failed', 'detail': '', 'result': None}

    def note(detail):
        out['detail'] = detail
        if on_state:
            try:
                on_state(detail)
            except Exception:
                pass

    if not repo_id:
        out['state'] = 'skipped'
        note('no Hugging Face delivery repository for this run')
        return out
    try:
        note('Uploading the full model from the pod to Hugging Face…')
        parsed = _run(
            remote, instance_id=instance_id, token=hf_token,
            command=lambda token_file: build_push_command(
                src_path, repo_id, path_in_repo, token_file),
            budget_seconds=budget_seconds, tmp_dir=tmp_dir, vast=vast,
            on_state=None, _sleep=_sleep, _now=_now)
        out['result'] = parsed
        out['state'] = 'done'
        note(f'The Hugging Face backup copy of {path_in_repo} was uploaded '
             'from the pod.')
    except Exception as e:
        logger.warning('run: the pod could not upload the dense master (%s)', e)
        out['state'] = 'failed'
        note(f'No Hugging Face copy was made ({e}). The model is on this '
             'computer; without a Hub copy it cannot be resumed later.'[:400])
    return out


def fetch_checkpoint(remote, *, instance_id, repo_id, filename, dest_path,
                     hf_token, tmp_dir, budget_seconds=DEFAULT_FETCH_BUDGET_SECONDS,
                     vast=None, on_state=None, _sleep=time.sleep,
                     _now=time.monotonic, should_cancel=None) -> dict:
    """Have the pod download ONE checkpoint from the Hub into ``dest_path``.

    RAISES on failure, deliberately: this is the seeding step of a resume, and a
    resume that cannot place its checkpoint must fail loudly instead of silently
    training a brand-new model from step 0 on the user's money.
    """
    if not repo_id or not filename:
        raise PodHubError('no Hugging Face checkpoint to resume from')
    parsed = _run(
        remote, instance_id=instance_id, token=hf_token,
        command=lambda token_file: build_fetch_command(
            repo_id, filename, dest_path, token_file),
        budget_seconds=budget_seconds, tmp_dir=tmp_dir, vast=vast,
        on_state=on_state, _sleep=_sleep, _now=_now, should_cancel=should_cancel)
    if not parsed.get('bytes'):
        raise PodHubError('the pod reported an empty checkpoint download')
    return parsed
