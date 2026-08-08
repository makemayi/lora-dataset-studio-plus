"""One click, from a repository to a file ComfyUI loads — and the refusals first.

Nothing here touches Hugging Face and nothing downloads 26 GB: the Hub listing
is a fake, the transfer is a fake that writes a real (tiny) safetensors file, and
the conversion that runs on it is the production one. What is asserted is the
chain, the destination, and the four refusals that must happen BEFORE a byte
moves — a full disk, an output that already exists, an already-quantized source,
and an adapter.

The other half of this file is the one-file-per-repository contract. A dense
repository holds the final save AND every step snapshot, all ~26 GB with nearly
the same name, and the app used to designate two different ones on the same
screen: the card named `…_000002750.safetensors` (the delivery verifier sorted
and took the last, and `.` sorts before `_`) while the operation offered right
underneath took `….safetensors`. That is now one rule, and these tests are what
keeps it one.
"""
import os

import pytest

from app.services import dense_weights as dw
from app.services import fp8_export, fp8_local_delivery as fld

torch = pytest.importorskip('torch', reason='fp8 quantization needs torch')
safetensors_torch = pytest.importorskip('safetensors.torch')

REPO = 'me/krea-run-146'
JOB = 'Krea_lds146_subject_Krea-2-Raw'
FINAL = f'{JOB}.safetensors'
STEP = f'{JOB}_000002750.safetensors'
EARLIER = f'{JOB}_000002500.safetensors'
BF16 = 25_600_000_000
FAKE_TOKEN = 'hf_zzUNIQUEsecret999'


@pytest.fixture(autouse=True)
def _app_context(app):
    """Job state is stored in system_state, i.e. in the database."""
    with app.app_context():
        yield app


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setattr(fld.cfg, 'secret',
                        lambda name, *a, **k: FAKE_TOKEN
                        if name in ('HF_CLOUD_TOKEN', 'HF_TOKEN') else None)


def _files(*names):
    """A Hub listing: every dense checkpoint is the same size, as in reality."""
    return [(n, BF16) for n in names] + [('README.md', 900), ('LICENSE.md', 4_000)]


def _model(path):
    """A real (tiny) full model: two matrices above the 1 Mi quantization floor."""
    torch.manual_seed(3)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    safetensors_torch.save_file({
        'blocks.0.attn.wq.weight': torch.randn(1024, 1024).bfloat16(),
        'blocks.0.mlp.up.weight': torch.randn(1024, 2048).bfloat16(),
        'blocks.0.prenorm.scale': torch.ones(1024),
    }, str(path))
    return str(path)


@pytest.fixture()
def comfy(tmp_path, monkeypatch):
    """A configured ComfyUI, resolved through the real write_root resolver."""
    root = tmp_path / 'ComfyUI'
    (root / 'models' / 'diffusion_models').mkdir(parents=True)
    from app import config as cfg
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    monkeypatch.setattr(cfg, 'get', lambda key, *a, **k: (
        str(root) if key == 'comfyui.base_dir' else ''))
    yield root / 'models' / 'diffusion_models'
    comfy_model_paths.clear_cache()


# --- one file per repository, everywhere -------------------------------------------

def test_the_final_save_wins_over_every_step_snapshot():
    assert dw.pick_master(_files(FINAL, STEP, EARLIER)) == FINAL
    # ...and the order the Hub happens to list them in changes nothing. This is
    # the exact tie the cloud lane lost: same size, first one seen won.
    assert dw.pick_master(_files(STEP, FINAL)) == FINAL
    assert dw.pick_master(_files(EARLIER, STEP, FINAL)) == FINAL


def test_without_a_final_save_the_highest_step_is_the_model():
    assert dw.pick_master(_files(EARLIER, STEP)) == STEP
    assert dw.describe_choice(_files(EARLIER, STEP))['step'] == 2750


def test_an_fp8_export_is_never_a_master():
    fp8 = fp8_export.fp8_name_for(FINAL)
    assert dw.pick_master(_files(fp8)) is None
    assert dw.pick_master(_files(fp8, STEP)) == STEP
    assert dw.is_fp8_export(fp8) and not dw.is_fp8_export(FINAL)


def test_a_repository_with_no_weights_designates_nothing():
    assert dw.pick_master([('README.md', 10)]) is None
    assert dw.describe_choice([])['name'] is None


def test_the_delivery_verifier_stamps_the_same_file_the_quantizer_would_take(
        monkeypatch):
    """The bug, pinned: the card and the button must name the SAME checkpoint.

    Before the shared rule, this verifier sorted the matching names and took the
    last one — and `Krea….safetensors` sorts BEFORE `Krea…_000002750` because
    `.` is 0x2E and `_` is 0x5F. So the card advertised the step snapshot while
    the operation offered under it took the final save.
    """
    from app.services import cloud_training as ct

    class _Sibling:
        def __init__(self, name):
            self.rfilename = name
            self.size = BF16
            self.lfs = {'size': BF16, 'sha256': 'a' * 64}
            self.blob_id = 'b' * 40

    class _Api:
        def repo_info(self, **_kw):
            return type('I', (), {'siblings': [_Sibling(n) for n in
                                               (FINAL, STEP, EARLIER)]})()

    class _Run:
        id = 1
        job_name = JOB
        status = 'done'

    stamped = {}
    monkeypatch.setattr(ct, '_persist_artifact_state',
                        lambda run, state, **kw: stamped.update(state=state, **kw))
    monkeypatch.setattr(ct, '_apply_full_transformer_compliance',
                        lambda *a, **k: None)
    monkeypatch.setattr(ct, '_is_full_transformer_run', lambda _run: True)
    monkeypatch.setattr(ct, '_run_param',
                        lambda run, key: REPO if key == 'hf_repo_id' else None)
    assert ct._verify_full_transformer_artifact(_Run(), _api=_Api()) == 'available'
    assert stamped['hf_weight_filename'] == FINAL == dw.pick_master(
        _files(FINAL, STEP, EARLIER))
    assert stamped['hf_artifact_proof']['size_bytes'] == BF16


def test_the_cloud_lane_takes_the_same_file_when_it_is_told_which_one():
    """The dormant cloud lane has its own "largest sibling" fallback. Every caller
    passes the designated name, and with it the two lanes agree — which is the
    property that has to hold for as long as that fallback exists."""
    from app.services import cloud_quantize as cq

    class _Sibling:
        def __init__(self, name):
            self.rfilename, self.size, self.lfs = name, BF16, None

    info = type('I', (), {'siblings': [_Sibling(STEP), _Sibling(FINAL)]})()
    designated = dw.pick_master(_files(FINAL, STEP))
    assert cq._weight_sibling(info, filename=designated)[0] == designated


# --- the plan tells the truth before anything starts ---------------------------------

def test_the_plan_names_the_file_the_folder_and_what_it_costs(comfy):
    plan = fld.plan(repo_id=REPO, family='krea', _files=_files(FINAL, STEP))
    assert plan['weight_basename'] == FINAL
    assert plan['choice']['total'] == 2 and plan['choice']['is_final'] is True
    assert plan['destination_dir'] == os.path.normpath(str(comfy))
    assert plan['destination_dir_kind'] == 'comfyui'
    assert plan['destination_name'] == fp8_export.fp8_name_for(FINAL)
    assert plan['download_bytes'] == BF16
    # The forecast rounds UP (master + fp8 ceiling + headroom), never down.
    assert plan['required_bytes'] > BF16 + fp8_export.typical_fp8_bytes(BF16)


def test_the_run_s_own_recorded_file_is_honoured_and_reported_as_pinned(comfy):
    plan = fld.plan(repo_id=REPO, filename=STEP, family='krea',
                    _files=_files(FINAL, STEP))
    assert plan['weight_basename'] == STEP
    assert plan['choice']['pinned'] is True and plan['choice']['step'] == 2750


def test_an_sdxl_full_checkpoint_goes_to_the_checkpoints_folder_not_diffusion_models(
        tmp_path, monkeypatch):
    root = tmp_path / 'ComfyUI'
    (root / 'models' / 'checkpoints').mkdir(parents=True)
    from app import config as cfg
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    monkeypatch.setattr(cfg, 'get', lambda key, *a, **k: (
        str(root) if key == 'comfyui.base_dir' else ''))
    try:
        assert fld.folder_type_for('sdxl') == 'checkpoints'
        assert fld.folder_type_for('krea') == 'diffusion_models'
        plan = fld.plan(repo_id=REPO, family='sdxl', _files=_files(FINAL))
        assert plan['destination_dir'].endswith(os.path.join('models', 'checkpoints'))
    finally:
        comfy_model_paths.clear_cache()


def test_the_folder_that_already_exists_wins_over_comfyui_s_legacy_alias(
        tmp_path, monkeypatch):
    """ComfyUI reads models/unet AND models/diffusion_models and lists unet
    first. Writing there blindly would create a models/unet/ nobody has on a
    modern install — the file would load, from a directory the user has never
    opened."""
    root = tmp_path / 'ComfyUI'
    (root / 'models' / 'diffusion_models').mkdir(parents=True)
    from app import config as cfg
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    monkeypatch.setattr(cfg, 'get', lambda key, *a, **k: (
        str(root) if key == 'comfyui.base_dir' else ''))
    try:
        assert fld._comfy_write_dir('diffusion_models') == os.path.normpath(
            str(root / 'models' / 'diffusion_models'))
        # With the legacy folder present instead, ComfyUI's own priority wins.
        (root / 'models' / 'unet').mkdir()
        comfy_model_paths.clear_cache()
        assert fld._comfy_write_dir('diffusion_models') == os.path.normpath(
            str(root / 'models' / 'unet'))
    finally:
        comfy_model_paths.clear_cache()


def test_an_unconfigured_comfyui_falls_back_and_says_so_instead_of_guessing(monkeypatch):
    from app import config as cfg
    from app.services import comfy_model_paths
    comfy_model_paths.clear_cache()
    monkeypatch.setattr(cfg, 'get', lambda *_a, **_k: '')
    try:
        plan = fld.plan(repo_id=REPO, family='krea', _files=_files(FINAL))
        assert plan['destination_dir_kind'] == 'fallback'
        assert 'ComfyUI is not configured' in plan['destination_dir_note']
    finally:
        comfy_model_paths.clear_cache()


def test_a_full_disk_is_a_refusal_whose_arithmetic_is_written_out(comfy, monkeypatch):
    monkeypatch.setattr(fld, '_free_bytes', lambda _p: 5 * 1000 ** 3)
    plan = fld.plan(repo_id=REPO, family='krea', _files=_files(FINAL))
    assert plan['enough_space'] is False
    # Every term named: what is left to download, the fp8 ceiling, the headroom.
    # "~30 GB needed" next to a 12.8 GB output was a number nobody could act on.
    assert '5.0 GB free' in plan['space_error']
    assert 'still to download' in plan['space_error']
    assert 'for the fp8 file' in plan['space_error']
    assert '2 GB of working headroom' in plan['space_error']
    assert 'another folder' in plan['space_error']
    described = fld.describe(repo_id=REPO, family='krea', _files=_files(FINAL))
    assert described['ok'] is True and described['enough_space'] is False


def test_the_budget_is_derived_not_a_flat_threshold(comfy, monkeypatch):
    """A 12.8 GB output with 17.6 GB free must be allowed — it was refused."""
    monkeypatch.setattr(fld, '_free_bytes', lambda _p: 17_600_000_000)
    local_master = _model(comfy.parent / 'masters' / 'Krea_real.safetensors')
    monkeypatch.setattr(fld.fp8_export, 'plan_quantization', lambda header: {
        'quantize': ['w'], 'keep': [], 'bytes_before': 25_600_000_000,
        'bytes_after': 12_822_354_094})
    assert fld.plan(path=local_master, family='krea')['enough_space'] is True


def test_a_start_never_refuses_what_the_plan_accepted(app, comfy, tmp_path, monkeypatch):
    """The panel disables its button with the very verdict it shows.

    Real sequence this pins, from a 25.6 GB master: plan answered ``ok: true``
    with a 12.8 GB estimate and 17.6 GB free, and the start call refused with
    "~30 GB needed" — a threshold plan had never applied. So the button stayed
    enabled and the refusal landed after the user had committed.
    """
    source = _model(tmp_path / 'hub' / FINAL)
    files = [(FINAL, os.path.getsize(source))]
    # The thread is not the subject here: only whether start ACCEPTS.
    monkeypatch.setattr(fld.threading, 'Thread',
                        lambda **kw: type('T', (), {'start': lambda _s: None})())
    from app.job_queue import queue_manager
    for free in (10 ** 8, 3 * 1000 ** 3, 10 ** 13):
        queue_manager._set_system_state('fp8_local_delivery', None)
        monkeypatch.setattr(fld, '_free_bytes', lambda _p, f=free: f)
        described = fld.describe(repo_id=REPO, family='krea', _files=files)
        if described['ok'] and described['enough_space']:
            fld.start(app, repo_id=REPO, family='krea', _files=files)
        else:
            with pytest.raises(fld.DeliveryError):
                fld.start(app, repo_id=REPO, family='krea', _files=files)


def test_the_destination_can_be_overridden_when_the_resolved_one_is_full(
        comfy, tmp_path):
    other = tmp_path / 'bigger-drive'
    other.mkdir()
    plan = fld.plan(repo_id=REPO, family='krea', _files=_files(FINAL),
                    destination_dir=str(other))
    assert plan['destination_dir'] == os.path.normpath(str(other))
    assert plan['destination_dir_kind'] == 'chosen'
    with pytest.raises(fld.DeliveryError, match='full path'):
        fld.plan(repo_id=REPO, family='krea', _files=_files(FINAL),
                 destination_dir='relative/folder')


def test_free_space_is_measured_on_the_volume_that_really_holds_the_folder(
        tmp_path, monkeypatch):
    """ComfyUI model folders are very often junctions onto another drive."""
    seen = {}
    monkeypatch.setattr('shutil.disk_usage',
                        lambda p: seen.setdefault('path', p) or type('U', (), {'free': 1})())
    fld._free_bytes(str(tmp_path))
    assert seen['path'] == os.path.realpath(str(tmp_path))


def test_an_existing_output_is_never_overwritten(comfy):
    (comfy / fp8_export.fp8_name_for(FINAL)).write_bytes(b'x')
    with pytest.raises(fld.DeliveryError, match='already in'):
        fld.plan(repo_id=REPO, family='krea', _files=_files(FINAL))


def test_a_repository_holding_only_an_fp8_export_has_nothing_left_to_quantize(comfy):
    with pytest.raises(fld.DeliveryError, match='no full-precision checkpoint'):
        fld.plan(repo_id=REPO, family='krea',
                 _files=_files(fp8_export.fp8_name_for(FINAL)))


def test_a_machine_that_cannot_convert_is_refused_BEFORE_the_26_gb_download(
        comfy, monkeypatch):
    """The worst version of the plan/run gap: 26 GB pulled, then
    "No module named 'safetensors'". The interpreter is probed in the plan, for
    the Hugging Face source kind too — where the cost of finding out late is a
    whole download."""
    from app.services import fp8_quantize as fq
    fq.clear_probe_cache()
    monkeypatch.setattr(fq, 'candidates', lambda: ['/nowhere/python'])
    monkeypatch.setattr(fq, '_probe', lambda _p: {'torch': False, 'safetensors': False})
    described = fld.describe(repo_id=REPO, family='krea', _files=_files(FINAL))
    assert described['ok'] is False
    assert 'torch' in described['error'] and 'pip install' in described['error']
    with pytest.raises(fld.DeliveryError, match='missing'):
        fld.plan(repo_id=REPO, family='krea', _files=_files(FINAL))


def test_the_plan_names_the_interpreter_that_will_do_the_work(comfy):
    from app.services import fp8_quantize as fq
    plan = fld.plan(repo_id=REPO, family='krea', _files=_files(FINAL))
    assert plan['python'] == fq.interpreter()['python']


def test_no_token_is_a_sentence_that_names_the_setting(comfy, monkeypatch):
    monkeypatch.setattr(fld.cfg, 'secret', lambda *_a, **_k: None)
    described = fld.describe(repo_id=REPO, family='krea', _files=_files(FINAL))
    assert described['ok'] is False and 'HF_CLOUD_TOKEN' in described['error']


def test_a_partly_downloaded_master_is_counted_as_progress_not_restarted(comfy):
    part = comfy / f'{FINAL}.part'
    part.write_bytes(b'0' * 4096)
    plan = fld.plan(repo_id=REPO, family='krea', _files=_files(FINAL))
    assert plan['download_bytes'] == BF16 - 4096


# --- a model already on this machine --------------------------------------------------

def test_a_local_model_needs_no_download_and_still_lands_in_comfyui(comfy, tmp_path):
    src = _model(tmp_path / 'elsewhere' / 'MyBigModel.safetensors')
    plan = fld.plan(path=src, family='krea')
    assert plan['source_kind'] == 'local' and plan['download_bytes'] == 0
    assert plan['master_path'] == src
    assert plan['destination'] == os.path.join(os.path.normpath(str(comfy)),
                                               'MyBigModel_fp8.safetensors')
    # A file the user already had is never deleted, whatever the caller asked.
    assert fld.plan(path=src, family='krea', keep_master=False)['keep_master'] is True


def test_the_local_door_reuses_the_manual_tool_s_refusals(comfy, tmp_path):
    lora = tmp_path / 'my_lora.safetensors'
    safetensors_torch.save_file(
        {'lora_unet_blocks_0.lora_down.weight': torch.randn(32, 1024).bfloat16()},
        str(lora))
    with pytest.raises(fld.DeliveryError, match='not a full model'):
        fld.plan(path=str(lora), family='krea')
    missing = tmp_path / 'gone.safetensors'
    with pytest.raises(fld.DeliveryError, match='no longer where'):
        fld.plan(path=str(missing), family='krea')


# --- the chain, end to end (a real conversion, on a small file) ------------------------

def _fake_download(source):
    """Stand-in for the Hub transfer: it reports bytes and produces a real file."""
    def download(info, on_progress):
        import shutil
        os.makedirs(os.path.dirname(info['master_path']), exist_ok=True)
        shutil.copyfile(source, info['master_path'])
        total = os.path.getsize(info['master_path'])
        on_progress(total // 2, total)
        on_progress(total, total)
        return info['master_path']
    return download


def _run_chain(comfy, tmp_path, *, keep_master=True):
    """The job body, run inline.

    ``start`` only validates and hands this to a daemon thread; running the body
    here keeps the assertion on the CHAIN rather than on thread scheduling (and
    the in-memory test database belongs to this thread). The refusals ``start``
    itself makes are covered separately, below.
    """
    source = _model(tmp_path / 'hub' / FINAL)
    size = os.path.getsize(source)
    info = fld.plan(repo_id=REPO, family='krea', keep_master=keep_master,
                    _files=[(FINAL, size)])
    fld._execute(info, download=_fake_download(source))
    return fld.status()


def test_one_click_downloads_converts_verifies_and_leaves_the_file_in_comfyui(
        comfy, tmp_path):
    state = _run_chain(comfy, tmp_path)
    assert state['status'] == 'done', state.get('error')
    out = comfy / fp8_export.fp8_name_for(FINAL)
    assert out.is_file()
    assert state['destination'] == str(out)
    assert state['result']['verified'] is True
    assert state['result']['scaled_tensors'] == 2
    # Kept by default: it is the only file that can be trained again.
    assert (comfy / FINAL).is_file() and state['result']['master_kept'] is True
    loaded = safetensors_torch.load_file(str(out))
    assert loaded['scaled_fp8'].dtype is torch.float8_e4m3fn


def test_asking_to_drop_the_master_removes_it_only_after_the_output_is_proven(
        comfy, tmp_path):
    state = _run_chain(comfy, tmp_path, keep_master=False)
    assert state['status'] == 'done', state.get('error')
    assert state['result']['master_removed'] is True
    assert not (comfy / FINAL).exists()
    assert (comfy / fp8_export.fp8_name_for(FINAL)).is_file()


def test_stopping_mid_download_keeps_the_bytes_so_the_next_start_resumes(
        comfy, tmp_path):
    source = _model(tmp_path / 'hub' / FINAL)
    info = fld.plan(repo_id=REPO, family='krea',
                    _files=[(FINAL, os.path.getsize(source))])

    def _stopped(job, on_progress):
        with open(f'{job["master_path"]}.part', 'wb') as fh:
            fh.write(b'0' * 2048)
        fld._cancel.set()
        on_progress(2048, job['source_bytes'])

    fld._execute(info, download=_stopped)
    state = fld.status()
    assert state['status'] == 'cancelled'
    assert 'resumes' in state['error']
    assert (comfy / f'{FINAL}.part').is_file(), 'the downloaded bytes were thrown away'
    assert not (comfy / fp8_export.fp8_name_for(FINAL)).exists()
    # ...and the next plan asks only for what is left.
    resumed = fld.plan(repo_id=REPO, family='krea',
                       _files=[(FINAL, os.path.getsize(source))])
    assert resumed['download_bytes'] == os.path.getsize(source) - 2048


def test_stopping_during_the_conversion_reads_as_a_stop_not_a_failure(
        comfy, tmp_path, monkeypatch):
    """The worker refuses with "the conversion was stopped"; that must not reach
    the user as a red failure — it is the button they just pressed."""
    source = _model(tmp_path / 'hub' / FINAL)
    info = fld.plan(repo_id=REPO, family='krea',
                    _files=[(FINAL, os.path.getsize(source))])

    def _stop_in_the_middle(src, **kw):
        fld._cancel.set()
        raise fld.fp8_quantize.QuantizeError('the conversion was stopped')

    monkeypatch.setattr(fld.fp8_quantize, 'quantize', _stop_in_the_middle)
    fld._execute(info, download=_fake_download(source))
    state = fld.status()
    assert state['status'] == 'cancelled'
    assert 'resumes' in state['error']


def test_a_second_job_is_refused_while_one_is_running(app, comfy, tmp_path, monkeypatch):
    # `start()` now asks "is something already running?" BEFORE it plans and
    # before it looks at the disk, so this pin is no longer what makes the test
    # pass. IT STAYS ON PURPOSE. It was added because the space refusal used to
    # fire first and this test silently measured the wrong branch — red on a
    # release runner, green everywhere else. Kept, the pin means the test can
    # never re-couple itself to the host's free space if the order is ever
    # reorganised back; removed, that regression would come back invisible. The
    # sibling test below pins it low to cover the other side, and the one after
    # that pins it low WITH a job running, which is the ordering itself.
    monkeypatch.setattr(fld, '_free_bytes', lambda _p: 10 ** 15)
    from app.job_queue import queue_manager
    queue_manager._set_system_state('fp8_local_delivery', {'status': 'downloading'},
                                    ttl_seconds=60)
    with pytest.raises(fld.DeliveryError, match='already being prepared'):
        fld.start(app, repo_id=REPO, family='krea', _files=_files(FINAL))


def test_starting_with_too_little_disk_refuses_instead_of_launching_a_thread(
        app, comfy, monkeypatch):
    monkeypatch.setattr(fld, '_free_bytes', lambda _p: 1000 ** 3)
    with pytest.raises(fld.DeliveryError, match='not enough disk space'):
        fld.start(app, repo_id=REPO, family='krea', _files=_files(FINAL))
    assert fld.status().get('status') != 'downloading'


def test_a_running_job_is_named_even_when_the_disk_is_also_full(
        app, comfy, monkeypatch):
    """Both refusals are true at once — the ACTIONABLE one has to be the one said.

    A job in flight is a fact about right now that one gesture settles (wait, or
    stop it). Free space is a forecast about work that could not have started
    anyway, and hearing it first sends someone off to delete gigabytes for a run
    the disk was never blocking. So with a job running the space message must not
    appear at all, whatever the disk says."""
    monkeypatch.setattr(fld, '_free_bytes', lambda _p: 1000 ** 3)   # 1 GB: far too little
    from app.job_queue import queue_manager
    queue_manager._set_system_state('fp8_local_delivery', {'status': 'downloading'},
                                    ttl_seconds=60)
    with pytest.raises(fld.DeliveryError) as caught:
        fld.start(app, repo_id=REPO, family='krea', _files=_files(FINAL))
    assert 'already being prepared' in str(caught.value)
    assert 'disk space' not in str(caught.value)


def test_a_busy_machine_is_refused_without_planning_anything(app, comfy, monkeypatch):
    """...and the refusal costs nothing. `plan` lists a Hugging Face repository
    and probes interpreters for torch; none of that has to run to answer "one is
    already going". This asserts the call is not made, not merely that it was
    fast — a timing assertion would pass on a warm cache."""
    def _never(**_kw):
        raise AssertionError('plan() was called to refuse a job that is already running')
    monkeypatch.setattr(fld, 'plan', _never)
    from app.job_queue import queue_manager
    queue_manager._set_system_state('fp8_local_delivery', {'status': 'quantizing'},
                                    ttl_seconds=60)
    with pytest.raises(fld.DeliveryError, match='already being prepared'):
        fld.start(app, repo_id=REPO, family='krea', _files=_files(FINAL))


def test_cancel_reports_nothing_to_stop_when_idle(app):
    assert fld.cancel() is False
