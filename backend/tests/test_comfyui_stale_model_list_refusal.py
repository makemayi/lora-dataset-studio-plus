"""A model deployed a moment ago must not be killed by a 60-second-old list.

WHY THIS FILE EXISTS (#help, 2026-08-08)
----------------------------------------
Reported after deploying a checkpoint from Canvas: "it feels like there is a delay
before LDS can use it — even though I can clearly select/use the deployed LoRA
directly inside Comfy". Both halves of that sentence were true, and the second one
is what makes the report decisive: ComfyUI had the file, LDS did not think so.

    Workflow refused before queuing — Your ComfyUI does not offer a model file this
    workflow requires: lora_name = "krea\\lora_....safetensors"
    (on LoraLoaderModelOnly). […] it is most likely a different ComfyUI install.
    job … was deterministically refused: WORKFLOW_INVALIDE (ComfyUI capability)

`_fetch_object_info` serves one snapshot for `_OBJECT_INFO_TTL` (60 s) and no
deploy path invalidates it, so the graph was judged against the model list as it
stood up to a minute BEFORE the model existed. The refusal it produced is
deterministic — `job_queue` never retries `WORKFLOW_INVALIDE` — so the job died for
good, and the message then sent its reader to go and check an API address that was
correct all along.

Two rules come out of that, and this file holds both:
  * a deterministic kill may not rest on a stale observation — re-ask once first;
  * a message may not assert a cause it has not checked.

Everything here is a canned /object_info dict. No network, no ComfyUI, no queue.
"""
import pytest


LORA = 'krea\\lora_bank_000003250_rc3_v3.safetensors'


def _object_info(loras=(), unets=('krea\\base_fp8.safetensors',)):
    return {
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


def _workflow(lora=LORA, unet='krea\\base_fp8.safetensors'):
    return {
        '1': {'class_type': 'UNETLoader',
              'inputs': {'unet_name': unet, 'weight_dtype': 'fp8_e4m3fn'}},
        '2': {'class_type': 'LoraLoaderModelOnly',
              'inputs': {'lora_name': lora, 'strength_model': 1.0,
                         'model': ['1', 0]}},
    }


@pytest.fixture
def probe(app, monkeypatch):
    """A ComfyUI whose model list can change between calls, and that counts how
    often it is asked — the cost of the re-check is part of the contract."""
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

    def deploy(name=LORA):
        """What the user does in Canvas: the file appears in ComfyUI's list."""
        state['payload'] = _object_info(loras=(name,))

    state['deploy'] = deploy
    with app.app_context():
        comfyui.clear_model_caches()
        monkeypatch.setattr(comfyui.requests, 'get', fake_get)
        yield state
        comfyui.clear_model_caches()


def _age_the_snapshot(seconds=30):
    """Make the cached snapshot `seconds` old without sleeping. Below the TTL, so
    the cache still SERVES it — which is precisely the reported situation."""
    from app.utils import comfyui
    comfyui._object_info_cache['timestamp'] -= seconds


# --- the reported symptom -------------------------------------------------

def test_a_lora_deployed_after_the_snapshot_is_not_refused(app, probe):
    """THE regression test. Something warms the cache while the LoRA does not exist
    (a capability poll, a Studio preflight, an earlier generation); the user then
    deploys and hits Generate. Before the fix the graph was refused off that
    snapshot for the rest of the minute."""
    from app.utils import comfyui
    with app.app_context():
        assert comfyui.unavailable_model_files(_workflow()), 'not deployed yet'
        probe['deploy']()
        _age_the_snapshot()

        items, wf, fresh = comfyui.confirm_unavailable_model_files(
            _workflow(), comfyui.unavailable_model_files(_workflow()))
        assert items == [], 'the deployed LoRA must not be refused'
        assert fresh is True
        assert wf['2']['inputs']['lora_name'] == LORA


def test_a_confirmed_model_is_respelled_against_the_FRESH_list(app, probe):
    """ComfyUI validates a model widget by exact string match. The spelling the
    graph carries was resolved against the STALE list — which did not contain this
    name at all, so it fell back to a guessed separator. Confirming the model
    exists and then POSTing a spelling this ComfyUI does not publish would trade
    the old bug for a raw 400."""
    from app.utils import comfyui
    published = 'krea/Lora_Bank_000003250_RC3_v3.safetensors'   # other sep, other case
    with app.app_context():
        found = comfyui.unavailable_model_files(_workflow())
        probe['deploy'](published)
        _age_the_snapshot()
        items, wf, _ = comfyui.confirm_unavailable_model_files(_workflow(), found)
        assert items == []
        assert wf['2']['inputs']['lora_name'] == published


def test_a_genuinely_absent_model_is_still_refused_after_the_recheck(app, probe):
    """The re-check may not turn the guard off: a model no ComfyUI has must still
    die before queuing, because ComfyUI would answer an undecodable 400."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unavailable_model_files(_workflow())
        _age_the_snapshot()
        items, _, fresh = comfyui.confirm_unavailable_model_files(_workflow(), found)
        assert [i['value'] for i in items] == [LORA]
        assert items[0]['reason'] == 'not_listed'
        assert fresh is True


# --- the cost of the re-check ---------------------------------------------

def test_the_recheck_costs_at_most_one_extra_probe(app, probe):
    """/object_info is the heaviest probe in the app. Confirming a refusal may cost
    ONE more, never one per node and never one per item."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unavailable_model_files(
            {'1': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'lora_name': LORA}},
             '2': {'class_type': 'LoraLoaderModelOnly',
                   'inputs': {'lora_name': 'krea\\other.safetensors'}}})
        assert len(found) == 2
        before = probe['calls']
        _age_the_snapshot()
        comfyui.confirm_unavailable_model_files(_workflow(), found)
        assert probe['calls'] - before == 1


def test_a_batch_cannot_storm_comfyui_with_probes(app, probe):
    """A grid of N tiles against a genuinely missing model must not fire N probes.
    The forced probe refreshes the shared snapshot, and a snapshot younger than
    _OBJECT_INFO_RECHECK_MAX_AGE is already young enough to be trusted — so the
    tiles behind the first one re-ask nothing."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unavailable_model_files(_workflow())
        _age_the_snapshot()
        before = probe['calls']
        for _ in range(25):
            items, _, fresh = comfyui.confirm_unavailable_model_files(
                _workflow(), found)
            assert items and fresh is True
        assert probe['calls'] - before == 1, 'one probe for the whole batch'


def test_a_gguf_is_never_re_probed(app, probe):
    """.gguf is an EXTENSION verdict: core ComfyUI never scans the file, so no list
    from anyone can change the answer and asking again is pure cost."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unavailable_model_files(_workflow(lora='x-Q4_K_M.gguf'))
        assert found and found[0]['reason'] == 'gguf'
        _age_the_snapshot()
        before = probe['calls']
        items, _, fresh = comfyui.confirm_unavailable_model_files(
            _workflow(lora='x-Q4_K_M.gguf'), found)
        assert probe['calls'] == before, 'no probe for an extension verdict'
        assert [i['reason'] for i in items] == ['gguf']
        assert fresh is True


def test_a_failed_recheck_fails_open_but_keeps_the_gguf_verdict(app, probe):
    """With no current evidence, handing the graph to ComfyUI is the honest move:
    its own 400 is ground truth and is already handled. A gguf item survives, since
    no probe was ever its basis."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unavailable_model_files(
            _workflow(lora='x-Q4_K_M.gguf', unet='krea\\absent.safetensors'))
        assert {i['reason'] for i in found} == {'gguf', 'not_listed'}
        _age_the_snapshot()
        probe['boom'] = True
        items, _, fresh = comfyui.confirm_unavailable_model_files(
            _workflow(lora='x-Q4_K_M.gguf', unet='krea\\absent.safetensors'), found)
        assert [i['reason'] for i in items] == ['gguf']
        assert fresh is True, 'the gguf verdict needs no probe to be firm'


# --- the message may not assert what it has not checked -------------------

def test_the_unchecked_message_names_the_stale_list_first(app, probe):
    """Without a fresh re-check, "it is most likely a different ComfyUI install" is
    a guess — and the wrong one for the person who just deployed. The stale list has
    to be offered as the thing to rule out first."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unavailable_model_files(_workflow())
    msg = comfyui.format_unavailable_models_message(found)
    assert 'most likely a different ComfyUI install' not in msg
    assert 'deployed this model' in msg and 'try once more' in msg
    assert msg.index('deployed this model') < msg.index('different ComfyUI installs')


def test_the_rechecked_message_earns_the_second_install_and_says_so(app, probe):
    """After a fresh re-check the install hypothesis is defensible, and the reader
    is told in so many words that a fresh deploy is NOT what they are seeing."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unavailable_model_files(_workflow())
    msg = comfyui.format_unavailable_models_message(found, rechecked=True)
    assert 're-read seconds ago' in msg
    assert 'fresh deploy is NOT what you are looking at' in msg
    assert 'second ComfyUI' in msg


@pytest.mark.parametrize('rechecked', [False, True])
def test_the_message_stays_paste_safe_and_never_says_copy_it(app, probe, rechecked):
    """It is written to be pasted into a Discord thread verbatim, and "copy it into
    the models folder" is the advice a previous reporter followed for an hour."""
    from app.utils import comfyui
    with app.app_context():
        found = comfyui.unavailable_model_files(_workflow())
    msg = comfyui.format_unavailable_models_message(found, rechecked=rechecked)
    assert 'copy' not in msg.lower()
    assert ':\\' not in msg and '/home/' not in msg and '@' not in msg
    assert LORA in msg


# --- the queueing path actually uses it -----------------------------------

def test_queueing_rechecks_before_it_kills_the_job(app, probe, monkeypatch):
    """The whole point: `queue_prompt_to_comfyui` is where the deterministic refusal is issued,
    so that is where the re-check has to sit. Deploy, then queue: the job must reach
    ComfyUI instead of dying."""
    from app.utils import comfyui
    posted = {}

    class _Posted:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {'prompt_id': 'ok'}

    def fake_post(url, json=None, headers=None, timeout=None, allow_redirects=None):
        posted['workflow'] = (json or {}).get('prompt')
        return _Posted()

    with app.app_context():
        assert comfyui.unavailable_model_files(_workflow()), 'not deployed yet'
        probe['deploy']()
        _age_the_snapshot()
        monkeypatch.setattr(comfyui, '_ensure_comfyui_before_generation',
                            lambda: (True, 'up'))
        monkeypatch.setattr(comfyui.requests, 'post', fake_post)
        result, err = comfyui.queue_prompt_to_comfyui(_workflow(), 't')
    assert err is None, f'the deployed LoRA was still refused: {err}'
    assert result == {'prompt_id': 'ok'}
    assert posted['workflow']['2']['inputs']['lora_name'] == LORA


def test_queueing_still_refuses_what_is_really_missing(app, probe, monkeypatch):
    """Non-regression on the guard this fix must not weaken — and the refusal it
    emits now carries the rechecked wording, because it was rechecked."""
    from app.utils import comfyui

    def no_post(*a, **k):
        raise AssertionError('a refused workflow must never be POSTed')

    with app.app_context():
        monkeypatch.setattr(comfyui, '_ensure_comfyui_before_generation',
                            lambda: (True, 'up'))
        monkeypatch.setattr(comfyui.requests, 'post', no_post)
        _age_the_snapshot()
        result, err = comfyui.queue_prompt_to_comfyui(_workflow(), 't')
    assert result is None
    assert err.startswith('WORKFLOW_INVALIDE (ComfyUI capability)')
    assert 're-read seconds ago' in err
