"""A model file the app offers but the user's ComfyUI does not accept.

Reported by naniii2352 (Discord, displayed name Dexter): Krea 2 generation died on
a raw ComfyUI validation error he had to decode himself —

    WORKFLOW_INVALIDE (validation ComfyUI 400): "value_not_in_list",
    "details": "unet_name: 'krea2_turbo-Q4_K_M.gguf' not in ['a.safetensors', ...]"

He spent over an hour copying the file between model folders and suspecting
ComfyUI Desktop. Two INDEPENDENT causes produce that same message, and the app
knew about neither:

(a) EXTENSION — core ComfyUI's `folder_paths.supported_pt_extensions` is
    {.ckpt,.pt,.pt2,.bin,.pth,.safetensors,.pkl,.sft}. `.gguf` is NOT in it, so
    core never scans the file whatever folder it sits in — which is exactly why
    copying it into all three model roots changed nothing. A GGUF diffusion model
    needs the third-party ComfyUI-GGUF pack and its own `UnetLoaderGGUF` node;
    the `UNETLoader` our graphs use can never open one. The app nevertheless
    LISTS `.gguf` in its own pickers (_MODEL_SUFFIXES across capabilities.py,
    krea_edit_helper.py, klein_edit_helper.py, comfy_model_paths.py), so the app
    hands the user a choice it cannot honour.

(b) WRONG INSTALL — his LDS install directory pointed at one ComfyUI tree while
    his models-folder override pointed at a different one, and his ComfyUI
    Desktop declares three model roots. The app lists models with os.listdir on
    the disk; ComfyUI serves them from wherever the process that answers :8188
    was configured. Those two lists disagree by construction as soon as there is
    more than one install — a file can be in an LDS dropdown and absent from
    ComfyUI's list. Nothing here is GGUF-specific: it hits everyone running more
    than one ComfyUI.

The fix treats the API as the source of truth: ComfyUI publishes the filenames it
actually accepts in /object_info, so the gap is computable BEFORE queueing, the
same way `unsupported_enum_values` (shipped the same day for scheduler values)
computes it for capability values. Detection is deliberately conservative — a
name is only reported when no case/separator-normalised variant of it is in
ComfyUI's list, i.e. only when ComfyUI is certain to answer 400 anyway.

No network, no generation: /object_info is a fixture.
"""
import pytest


def _object_info(unets=('krea\\krea2_turbo_fp8.safetensors',), loras=('krea/id.safetensors',)):
    """A minimal /object_info in ComfyUI's real shape. The file inputs are combos
    exactly like the capability ones — the list ComfyUI quotes in its
    "Value not in list" error."""
    return {
        'KSampler': {'input': {'required': {
            'seed': ['INT', {'default': 0}],
            'sampler_name': [['euler'], {}],
            'scheduler': [['simple', 'beta'], {}],
            'model': ['MODEL'],
        }}},
        'UNETLoader': {'input': {'required': {
            'unet_name': [list(unets), {}],
            'weight_dtype': [['default', 'fp8_e4m3fn'], {}],
        }}},
        'LoraLoaderModelOnly': {'input': {'required': {
            'lora_name': [list(loras), {}],
            'strength_model': ['FLOAT', {'default': 1.0}],
            'model': ['MODEL'],
        }}},
    }


def _krea_like_workflow(unet='krea\\krea2_turbo_fp8.safetensors',
                        lora='krea/id.safetensors'):
    """The shape of the graph that actually fails: every NODE class exists (so the
    missing-node preflight sees nothing wrong) and every capability value is
    accepted — only the file name is not on the list."""
    return {
        '1': {'class_type': 'UNETLoader',
              'inputs': {'unet_name': unet, 'weight_dtype': 'fp8_e4m3fn'}},
        '2': {'class_type': 'LoraLoaderModelOnly',
              'inputs': {'lora_name': lora, 'strength_model': 1.0, 'model': ['1', 0]}},
        '3': {'class_type': 'KSampler',
              'inputs': {'seed': 1, 'sampler_name': 'euler', 'scheduler': 'simple',
                         'model': ['2', 0]}},
    }


@pytest.fixture
def probe(app, monkeypatch):
    """Serve a canned /object_info and count requests, so the file check can be
    proven to ride the EXISTING cache instead of adding a probe per generation."""
    from app.utils import comfyui

    class _Resp:
        def __init__(self, payload):
            self._p = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._p

    state = {'payload': _object_info(), 'calls': 0, 'boom': False}

    def fake_get(url, timeout=None):
        state['calls'] += 1
        if state['boom']:
            raise OSError('ComfyUI is not running')
        return _Resp(state['payload'])

    from app.services import klein_edit_helper as keh

    def _reset():
        comfyui.clear_model_caches()
        keh._enums_ok_until = 0.0
        keh._nodes_ok_until = 0.0

    with app.app_context():
        _reset()
        monkeypatch.setattr(comfyui.requests, 'get', fake_get)
        yield state
        _reset()


# --- (a) the GGUF case ----------------------------------------------------

def test_gguf_model_is_named_as_unloadable_with_the_pack_that_would_load_it(app, probe):
    """RED before the fix: the name went out as-is and ComfyUI answered a 400 whose
    only explanation was a truncated list of OTHER files."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unavailable_model_files(
            _krea_like_workflow(unet='krea2_turbo-Q4_K_M.gguf'))

    assert [(i['input'], i['value'], i['node_id']) for i in found] == [
        ('unet_name', 'krea2_turbo-Q4_K_M.gguf', '1')]
    assert found[0]['class_type'] == 'UNETLoader'
    # The reason must be the EXTENSION, not "put it in another folder" — that is the
    # distinction he lost an hour to.
    assert found[0]['reason'] == 'gguf'

    message = comfyui.format_unavailable_models_message(found)
    assert 'krea2_turbo-Q4_K_M.gguf' in message
    # Says WHY no folder works...
    assert '.gguf' in message and 'ComfyUI-GGUF' in message
    assert 'https://github.com/city96/ComfyUI-GGUF' in message
    # ...and offers the option that needs no pack at all.
    assert '.safetensors' in message
    # Must NOT send him copying files around again.
    assert 'copy' not in message.lower()


def test_gguf_is_flagged_even_when_the_gguf_pack_is_installed(app, probe):
    """Having ComfyUI-GGUF does not make `UNETLoader` open a .gguf — the pack adds a
    SEPARATE node (`UnetLoaderGGUF`) with its own folder key. Our graphs use
    `UNETLoader`, so the answer is the same and must not become "you have the pack,
    should be fine"."""
    from app.utils import comfyui
    probe['payload']['UnetLoaderGGUF'] = {'input': {'required': {
        'unet_name': [['krea2_turbo-Q4_K_M.gguf'], {}]}}}
    with app.app_context():
        comfyui.clear_model_caches()
        found = comfyui.unavailable_model_files(
            _krea_like_workflow(unet='krea2_turbo-Q4_K_M.gguf'))
    assert [i['value'] for i in found] == ['krea2_turbo-Q4_K_M.gguf']


# --- (b) the two-installs case --------------------------------------------

def test_model_absent_from_this_comfyui_says_it_may_live_in_another_install(app, probe):
    """His LDS install dir pointed at one ComfyUI tree, his models override at a
    different one, and Desktop declares three roots. A file read off the disk by the
    app is simply not in the list of the ComfyUI answering on the port."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unavailable_model_files(
            _krea_like_workflow(unet='krea\\some_other_base.safetensors'))

    assert [(i['input'], i['value']) for i in found] == [
        ('unet_name', 'krea\\some_other_base.safetensors')]
    assert found[0]['reason'] == 'not_listed'

    message = comfyui.format_unavailable_models_message(found)
    assert 'some_other_base.safetensors' in message
    # The actionable half: it is a DIFFERENT ComfyUI, not a missing download.
    assert 'another ComfyUI' in message or 'different ComfyUI' in message
    assert 'install' in message.lower()


def test_a_lora_name_is_checked_too_not_just_the_base_model(app, probe):
    """The identity LoRA lives in a models root exactly like the base model, so the
    same install mismatch hides it the same way."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unavailable_model_files(
            _krea_like_workflow(lora='krea/absent_lora.safetensors'))
    assert [(i['input'], i['class_type']) for i in found] == [
        ('lora_name', 'LoraLoaderModelOnly')]


# --- no false positives ---------------------------------------------------

def test_a_normal_safetensors_that_comfyui_lists_is_never_touched(app, probe):
    """Non-regression: the everyday case must stay completely unaffected."""
    from app.utils import comfyui
    with app.app_context():
        assert comfyui.unavailable_model_files(_krea_like_workflow()) == []


def test_separator_and_case_differences_are_not_a_gap(app, probe):
    """ComfyUI joins subfolders with the host os.sep and the app backslash-joins its
    own names; Windows paths are case-insensitive. Reporting those as missing would
    break working installs, so matching normalises both before deciding."""
    from app.utils import comfyui
    with app.app_context():
        # forward slash vs the backslash ComfyUI published, and a different case
        assert comfyui.unavailable_model_files(
            _krea_like_workflow(unet='Krea/Krea2_Turbo_FP8.safetensors')) == []
        # backslash vs the forward slash ComfyUI published (a Linux/container host)
        assert comfyui.unavailable_model_files(
            _krea_like_workflow(lora='krea\\id.safetensors')) == []


def test_unreachable_comfyui_fails_open(app, probe):
    """A probe that cannot answer must never block a working install."""
    from app.utils import comfyui
    probe['boom'] = True
    with app.app_context():
        comfyui.clear_model_caches()
        assert comfyui.fetch_object_info_model_files() is None
        assert comfyui.unavailable_model_files(
            _krea_like_workflow(unet='krea2_turbo-Q4_K_M.gguf')) == []


def test_a_class_this_comfyui_does_not_expose_is_left_to_the_node_preflight(app, probe):
    """An absent class_type is a MISSING NODE, already reported elsewhere with the
    right fix — claiming its file is missing would be a second, wrong diagnosis."""
    from app.utils import comfyui
    wf = {'9': {'class_type': 'SomePackLoader',
                'inputs': {'unet_name': 'whatever.gguf'}}}
    with app.app_context():
        assert comfyui.unavailable_model_files(wf) == []


def test_links_and_non_strings_are_never_treated_as_file_names(app, probe):
    from app.utils import comfyui
    wf = {'1': {'class_type': 'UNETLoader',
                'inputs': {'unet_name': ['2', 0], 'weight_dtype': 'default'}}}
    with app.app_context():
        assert comfyui.unavailable_model_files(wf) == []


# --- cost -----------------------------------------------------------------

def test_file_view_rides_the_existing_object_info_cache(app, probe):
    """/object_info is the heaviest probe in the app. The file check must add ZERO
    requests: same fetch, same TTL cache as the class and enum views."""
    from app.utils import comfyui
    with app.app_context():
        assert comfyui.fetch_object_info_classes() is not None
        assert comfyui.fetch_object_info_enums() is not None
        assert comfyui.fetch_object_info_model_files() is not None
        comfyui.unavailable_model_files(_krea_like_workflow())
        assert probe['calls'] == 1

        comfyui.clear_model_caches()
        assert comfyui.fetch_object_info_model_files() is not None
        assert probe['calls'] == 2


def test_only_the_loader_classes_we_ship_are_distilled(app, probe):
    """Keeping the file arrays of EVERY node out of the cache is what makes caching
    them affordable — a node-rich install repeats those arrays across hundreds of
    classes (the enum view drops them wholesale for that reason). We keep only the
    loader classes our own graphs use."""
    from app.utils import comfyui
    probe['payload']['SomeOtherPackLoader'] = {'input': {'required': {
        'unet_name': [['x.safetensors'], {}]}}}
    with app.app_context():
        comfyui.clear_model_caches()
        files = comfyui.fetch_object_info_model_files()
    assert set(files) == {'UNETLoader', 'LoraLoaderModelOnly'}
    assert 'SomeOtherPackLoader' not in files
    # and the capability view is unchanged by all this
    with app.app_context():
        assert set(comfyui.fetch_object_info_enums()['UNETLoader']) == {'weight_dtype'}


# --- the gap stops the job instead of becoming a raw 400 ------------------

def test_queue_refuses_before_sending_and_explains(app, probe, monkeypatch):
    """End to end: the message replaces the raw ComfyUI 400, and nothing is queued."""
    from app.utils import comfyui
    sent = {}

    class _Ok:
        def raise_for_status(self):
            pass

        def json(self):
            return {'prompt_id': 'p1'}

    monkeypatch.setattr(comfyui.requests, 'post',
                        lambda url, json=None, headers=None, timeout=None:
                        (sent.update(json or {}), _Ok())[1])
    monkeypatch.setattr(comfyui, '_ensure_comfyui_before_generation', lambda: (True, 'up'))

    with app.app_context():
        result, error = comfyui.queue_prompt_to_comfyui(
            _krea_like_workflow(unet='krea2_turbo-Q4_K_M.gguf'), 'client')

    assert result is None
    assert 'WORKFLOW_INVALIDE' in error          # deterministic: not an outage
    assert 'krea2_turbo-Q4_K_M.gguf' in error and 'ComfyUI-GGUF' in error
    assert sent == {}                            # nothing reached ComfyUI


def test_a_valid_graph_still_goes_out(app, probe, monkeypatch):
    from app.utils import comfyui
    sent = {}

    class _Ok:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {'prompt_id': 'p1'}

    def fake_post(url, json=None, headers=None, timeout=None, **kwargs):
        assert kwargs.get('allow_redirects') is False
        sent.update(json or {})
        return _Ok()

    monkeypatch.setattr(comfyui.requests, 'post', fake_post)
    monkeypatch.setattr(comfyui, '_ensure_comfyui_before_generation', lambda: (True, 'up'))

    with app.app_context():
        result, error = comfyui.queue_prompt_to_comfyui(_krea_like_workflow(), 'client')

    assert error is None and result == {'prompt_id': 'p1'}
    assert sent['prompt']['1']['inputs']['unet_name'] == 'krea\\krea2_turbo_fp8.safetensors'


# --- paste-safety ---------------------------------------------------------

def test_message_carries_no_filesystem_path(app, probe):
    """The text is meant to be pasted verbatim into a Discord thread."""
    from app.utils import comfyui
    with app.app_context():
        message = comfyui.format_unavailable_models_message(
            comfyui.unavailable_model_files(
                _krea_like_workflow(unet='krea\\absent.safetensors')))
    assert ':\\' not in message and ':/' not in message


def test_repeated_gap_is_reported_once(app, probe):
    """The same file pinned on three nodes is ONE thing to fix."""
    from app.utils import comfyui
    wf = _krea_like_workflow(unet='krea2_turbo-Q4_K_M.gguf')
    wf['4'] = {'class_type': 'UNETLoader',
               'inputs': {'unet_name': 'krea2_turbo-Q4_K_M.gguf',
                          'weight_dtype': 'default'}}
    with app.app_context():
        message = comfyui.format_unavailable_models_message(
            comfyui.unavailable_model_files(wf))
    assert message.count('krea2_turbo-Q4_K_M.gguf') == 1
