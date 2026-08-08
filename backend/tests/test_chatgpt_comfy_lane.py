"""The THIRD ChatGPT lane: the same shot through ComfyUI's OpenAI API node.

NOTHING here talks to OpenAI, to comfy.org, or to a running ComfyUI. The submit
call and the history poll are patched in every test. What is being locked is the
part that is invisible from the outside and expensive to get wrong:

  1. the credential travels in the prompt's `extra_data`, because ComfyUI reads
     API-node credentials from there and NOWHERE else — a headless caller
     inherits no browser session, so a lane that forgets this fails on the
     provider's side with an opaque message;
  2. the lane is chosen ONCE per batch and pinned, so a settings change or an
     expiring token mid-batch cannot move later rows onto a different bill;
  3. it is still an API engine: `chatgpt` must not drift into LOCAL_ENGINES
     because the request happens to leave through 127.0.0.1.
"""
import pytest

from app.services import chatgpt_comfy as lane

#: Captured before the autouse fixture below replaces it — the two tests that
#: exercise the wait itself need the real one.
REAL_WAIT = lane.wait_until_comfyui_answers


@pytest.fixture(autouse=True)
def _no_real_comfyui_wait(monkeypatch):
    """The readiness wait is exercised by its own two tests below. Everywhere
    else it would poll a ComfyUI that the test config does not point at, and
    spend the whole timeout doing it."""
    monkeypatch.setattr(lane, 'wait_until_comfyui_answers', lambda: None)


def _no_key(monkeypatch):
    monkeypatch.delenv(lane.API_KEY_ENV, raising=False)


def test_the_key_rides_in_extra_data_and_nowhere_else(app, monkeypatch, tmp_path):
    seen = {}

    def fake_submit(workflow, client_id, worker_url=None, *, extra_data=None):
        seen['workflow'] = workflow
        seen['extra_data'] = extra_data
        return {'prompt_id': 'p1'}, None

    monkeypatch.setenv(lane.API_KEY_ENV, 'comfy-key-123')
    monkeypatch.setattr(lane, 'queue_prompt_to_comfyui', fake_submit)
    monkeypatch.setattr(lane, 'resolve_node', lambda: lane.NODE_CLASS_V2)
    monkeypatch.setattr(lane.comfy_fs, 'ensure_input_usable', lambda d: str(tmp_path))
    monkeypatch.setattr(lane, 'get_comfyui_history', lambda pid: {
        'p1': {'outputs': {'101': {'images': [{'filename': 'out.png', 'subfolder': ''}]}}}})
    monkeypatch.setattr(lane, 'fetch_output_image_bytes', lambda fn, sub: b'PNGBYTES')

    with app.app_context():
        out = lane.generate_variation([_png()], 'a portrait', aspect_ratio='3:4')

    assert out == b'PNGBYTES'
    assert seen['extra_data'] == {'api_key_comfy_org': 'comfy-key-123'}
    # ...and never inside the graph, which ComfyUI stores in its history.
    assert 'comfy-key-123' not in str(seen['workflow'])


def test_the_graph_is_the_node_the_install_actually_exposes(app, monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setenv(lane.API_KEY_ENV, 'k')
    monkeypatch.setattr(lane, 'resolve_node', lambda: lane.NODE_CLASS_V2)
    monkeypatch.setattr(lane.comfy_fs, 'ensure_input_usable', lambda d: str(tmp_path))
    monkeypatch.setattr(lane, 'queue_prompt_to_comfyui',
                        lambda wf, cid, worker_url=None, *, extra_data=None:
                        (seen.update(workflow=wf), ({'prompt_id': 'p'}, None))[1])
    monkeypatch.setattr(lane, 'get_comfyui_history', lambda pid: {
        'p': {'outputs': {'101': {'images': [{'filename': 'o.png'}]}}}})
    monkeypatch.setattr(lane, 'fetch_output_image_bytes', lambda fn, sub: b'X')

    with app.app_context():
        lane.generate_variation([_png(), _png()], 'two references', aspect_ratio='16:9')

    classes = [n['class_type'] for n in seen['workflow'].values()]
    assert lane.NODE_CLASS_V2 in classes and 'SaveImage' in classes
    assert classes.count('LoadImage') == 2
    node = next(n for n in seen['workflow'].values()
                if n['class_type'] == lane.NODE_CLASS_V2)
    assert node['inputs']['model.size'] == '1536x1024'   # from the card's ratio
    assert node['inputs']['n'] == 1


def test_v2_takes_the_references_as_separate_slots():
    """The dynamic combo is filled with a dotted prefix, and `images` is an
    autogrow list of 16 NAMED slots — the shape gpt-image's edit endpoint has.
    Reading that combo as unfillable is what sent the first version of this lane
    to the older node."""
    graph = lane.build_workflow(['a.png', 'b.png', 'c.png'], 'p',
                                size='1024x1536', quality='low',
                                node_class=lane.NODE_CLASS_V2)
    node = next(n for n in graph.values() if n['class_type'] == lane.NODE_CLASS_V2)
    assert node['inputs']['model.quality'] == 'low'
    assert node['inputs']['model.size'] == '1024x1536'
    assert node['inputs']['model.background'] == 'auto'
    assert [k for k in node['inputs'] if k.startswith('model.images.')] == [
        'model.images.image_1', 'model.images.image_2', 'model.images.image_3']
    assert lane.BATCH_NODE_CLASS not in [n['class_type'] for n in graph.values()]


def test_v2_stops_at_the_sixteen_slots_it_declares():
    graph = lane.build_workflow([f'{i}.png' for i in range(20)], 'p',
                                node_class=lane.NODE_CLASS_V2)
    node = next(n for n in graph.values() if n['class_type'] == lane.NODE_CLASS_V2)
    slots = [k for k in node['inputs'] if k.startswith('model.images.')]
    assert len(slots) == lane.MAX_REFS_V2 == 16


def test_the_older_node_still_gets_its_batched_single_input():
    """The fallback is not decoration: an install without V2 must still run, and
    V1's single IMAGE input needs the references chained into one picture."""
    graph = lane.build_workflow(['a.png', 'b.png'], 'p', size='1024x1024',
                                quality='high', node_class=lane.NODE_CLASS)
    classes = [n['class_type'] for n in graph.values()]
    assert lane.BATCH_NODE_CLASS in classes
    node = next(n for n in graph.values() if n['class_type'] == lane.NODE_CLASS)
    assert node['inputs']['size'] == '1024x1024'         # flat, not dotted
    assert node['inputs']['quality'] == 'high'
    assert 'model.images.image_1' not in node['inputs']


def test_the_node_choice_prefers_v2_and_falls_back(monkeypatch):
    monkeypatch.setattr(lane, 'fetch_object_info_classes',
                        lambda: {lane.NODE_CLASS_V2, lane.NODE_CLASS})
    assert lane.resolve_node() == lane.NODE_CLASS_V2
    monkeypatch.setattr(lane, 'fetch_object_info_classes', lambda: {lane.NODE_CLASS})
    assert lane.resolve_node() == lane.NODE_CLASS
    monkeypatch.setattr(lane, 'fetch_object_info_classes', lambda: {'KSampler'})
    assert lane.resolve_node() is None
    # A probe that cannot answer must not report the engine as unavailable.
    monkeypatch.setattr(lane, 'fetch_object_info_classes',
                        lambda: (_ for _ in ()).throw(OSError('unreachable')))
    assert lane.resolve_node() == lane.NODE_CLASS_V2


def test_an_unusable_quality_costs_a_cheaper_picture_not_the_shot():
    assert lane.quality_for('low') == 'low'
    assert lane.quality_for('HIGH') == 'high'
    assert lane.quality_for('ultra') == lane.DEFAULT_QUALITY
    assert lane.quality_for(None) == lane.DEFAULT_QUALITY


def test_an_unknown_model_degrades_instead_of_failing_the_shot():
    """`engines.chatgpt_image_model` is free text on purpose (OpenAI renames
    models between our releases). The node's combo is not."""
    assert lane.model_for('gpt-image-1.5') == 'gpt-image-1.5'
    assert lane.model_for('gpt-image-9-turbo') == lane.DEFAULT_MODEL
    assert lane.model_for('') == lane.DEFAULT_MODEL
    assert lane.model_for(None) == lane.DEFAULT_MODEL


def test_an_unmapped_ratio_asks_the_node_for_auto():
    assert lane.size_for_aspect('1:1') == '1024x1024'
    assert lane.size_for_aspect('3:4') == '1024x1536'
    assert lane.size_for_aspect('7:5') == 'auto'
    assert lane.size_for_aspect(None) == 'auto'


def test_no_key_refuses_before_anything_is_staged_or_queued(app, monkeypatch, tmp_path):
    _no_key(monkeypatch)
    monkeypatch.setattr(lane, 'queue_prompt_to_comfyui',
                        lambda *a, **k: pytest.fail('must not queue without a key'))
    with app.app_context():
        with pytest.raises(lane.ComfyGptUnavailable, match='comfy.org API key'):
            lane.generate_variation([_png()], 'x')
    assert not list(tmp_path.glob('gptref*')), 'nothing may be staged before the refusal'


def test_a_comfyui_without_the_node_says_so_by_name(app, monkeypatch):
    monkeypatch.setenv(lane.API_KEY_ENV, 'k')
    monkeypatch.setattr(lane, 'resolve_node', lambda: None)
    with app.app_context():
        with pytest.raises(lane.ComfyGptUnavailable, match=lane.NODE_CLASS_V2):
            lane.generate_variation([_png()], 'x')


def test_a_failed_prompt_quotes_comfyuis_own_words(app, monkeypatch, tmp_path):
    """A refused key, an empty credit balance and a provider refusal all arrive
    as an execution error — inventing 'generation failed' would throw away the
    only sentence that says which one it was."""
    monkeypatch.setenv(lane.API_KEY_ENV, 'k')
    monkeypatch.setattr(lane, 'resolve_node', lambda: lane.NODE_CLASS_V2)
    monkeypatch.setattr(lane.comfy_fs, 'ensure_input_usable', lambda d: str(tmp_path))
    monkeypatch.setattr(lane, 'queue_prompt_to_comfyui',
                        lambda *a, **k: ({'prompt_id': 'p'}, None))
    monkeypatch.setattr(lane, 'get_comfyui_history', lambda pid: {
        'p': {'status': {'status_str': 'error', 'messages': [
            ['execution_error', {'exception_message': 'Insufficient credits'}]]}}})

    with app.app_context():
        with pytest.raises(lane.ComfyGptUnavailable, match='Insufficient credits'):
            lane.generate_variation([_png()], 'x')


def test_the_staged_reference_is_dropped_whatever_happened(app, monkeypatch, tmp_path):
    monkeypatch.setenv(lane.API_KEY_ENV, 'k')
    monkeypatch.setattr(lane, 'resolve_node', lambda: lane.NODE_CLASS_V2)
    monkeypatch.setattr(lane.comfy_fs, 'ensure_input_usable', lambda d: str(tmp_path))
    monkeypatch.setattr(lane, 'queue_prompt_to_comfyui',
                        lambda *a, **k: (None, 'ComfyUI is not running'))
    with app.app_context():
        with pytest.raises(lane.ComfyGptUnavailable):
            lane.generate_variation([_png()], 'x')
    assert not list(tmp_path.glob('gptref*')), 'a staged reference outlived its shot'


def test_the_staged_name_matches_the_sweep_the_app_relies_on():
    """comfy_fs's 48 h safety net only recognises `<lane>_<8 hex>_<name>`. A
    name outside that shape is a full-resolution copy of a user photo that
    nothing will ever collect."""
    from app.utils import comfy_fs
    assert comfy_fs.is_staged_input_name('gptref0_0123abcd_reference.png')
    assert comfy_fs.is_staged_input_name('h3_source_0123abcd_shot.png')


def test_the_lane_is_pinned_once_and_survives_a_settings_change(app):
    """A batch pins the lane so a save mid-run cannot move later rows onto a
    different bill — the same guarantee the subscription lane already had."""
    from app import config as cfg
    from app.services import chatgpt_image
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_auth': 'comfyui'}})
        assert chatgpt_image.resolve_lane() == 'comfyui'
        assert chatgpt_image.resolve_lane('api') == 'api'
        cfg.save_config({'engines': {'chatgpt_auth': 'api'}})
        # The pinned value still decides, whatever the config now says.
        assert chatgpt_image.resolve_lane('comfyui') == 'comfyui'
        assert chatgpt_image.resolve_lane() == 'api'


def test_the_comfyui_lane_never_turns_chatgpt_into_a_local_engine(app):
    """The picture is still made on a third-party server. If `chatgpt` ever
    joined LOCAL_ENGINES, every NSFW fail-closed rule in the app would start
    routing adult shots to OpenAI."""
    from app import config as cfg
    from app.services import dataset_generation_service as gen
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_auth': 'comfyui'}})
        assert 'chatgpt' in gen.API_ENGINES
        assert 'chatgpt' not in gen.LOCAL_ENGINES


def test_the_comfyui_lane_does_not_use_the_subscription(app):
    from app import config as cfg
    from app.services import chatgpt_image
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_auth': 'comfyui'}})
        assert chatgpt_image._use_subscription() is False


def _png(size=(64, 64)):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', size, (10, 20, 30)).save(buf, 'PNG')
    return buf.getvalue()


def test_readiness_answers_from_the_lane_that_will_run(app, monkeypatch):
    """The engine card was greyed out with "add an API key" while the ComfyUI
    lane was selected and its key was set: readiness still asked OpenAI's
    question. The lane is pinned, so the probe must ask the lane's own."""
    from app import capabilities
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_auth': 'comfyui'}})
        monkeypatch.setattr(lane, 'resolve_node', lambda: lane.NODE_CLASS_V2)

        monkeypatch.setenv(lane.API_KEY_ENV, 'comfy-key')
        ready = capabilities.probe_openai()
        assert ready['ok'] is True and 'ComfyUI' in ready['detail']

        # ...and an OpenAI key does NOT light it up on this lane, because
        # nothing here would ever read that key.
        monkeypatch.delenv(lane.API_KEY_ENV, raising=False)
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-whatever')
        blocked = capabilities.probe_openai()
        assert blocked['ok'] is False
        assert 'comfy.org' in blocked['detail']


def test_a_comfyui_missing_the_node_is_named_in_the_readiness_detail(app, monkeypatch):
    from app import capabilities
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_auth': 'comfyui'}})
        monkeypatch.setenv(lane.API_KEY_ENV, 'comfy-key')
        monkeypatch.setattr(lane, 'resolve_node', lambda: None)
        state = capabilities.probe_openai()
        assert state['ok'] is False and lane.NODE_CLASS_V2 in state['detail']


def test_the_other_lanes_are_unchanged_by_the_new_branch(app, monkeypatch):
    from app import capabilities
    from app import config as cfg
    with app.app_context():
        cfg.save_config({'engines': {'chatgpt_auth': 'api'}})
        monkeypatch.setenv('OPENAI_API_KEY', 'sk-whatever')
        assert capabilities.probe_openai()['ok'] is True
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        monkeypatch.setattr('app.services.chatgpt_oauth.status',
                            lambda: {'connected': False})
        assert capabilities.probe_openai()['ok'] is False


def test_a_rejected_key_says_what_to_do_in_this_app(app, monkeypatch, tmp_path):
    """ComfyUI's own words for a 401 are "Please login first to use this node" —
    written for someone sitting in its web UI. From here that is advice the user
    cannot follow: they DID set a key, and no login would be read anyway."""
    monkeypatch.setenv(lane.API_KEY_ENV, 'k')
    monkeypatch.setattr(lane, 'resolve_node', lambda: lane.NODE_CLASS_V2)
    monkeypatch.setattr(lane, 'wait_until_comfyui_answers', lambda: None)
    monkeypatch.setattr(lane.comfy_fs, 'ensure_input_usable', lambda d: str(tmp_path))
    monkeypatch.setattr(lane, 'queue_prompt_to_comfyui',
                        lambda *a, **k: ({'prompt_id': 'p'}, None))
    monkeypatch.setattr(lane, 'get_comfyui_history', lambda pid: {
        'p': {'status': {'status_str': 'error', 'messages': [
            ['execution_error',
             {'exception_message': 'Unauthorized: Please login first to use this node.'}]]}}})

    with app.app_context():
        with pytest.raises(lane.ComfyGptUnavailable, match='401'):
            lane.generate_variation([_png()], 'x')


def test_an_empty_balance_is_not_reported_as_a_bad_key(app, monkeypatch, tmp_path):
    monkeypatch.setenv(lane.API_KEY_ENV, 'k')
    monkeypatch.setattr(lane, 'resolve_node', lambda: lane.NODE_CLASS_V2)
    monkeypatch.setattr(lane, 'wait_until_comfyui_answers', lambda: None)
    monkeypatch.setattr(lane.comfy_fs, 'ensure_input_usable', lambda d: str(tmp_path))
    monkeypatch.setattr(lane, 'queue_prompt_to_comfyui',
                        lambda *a, **k: ({'prompt_id': 'p'}, None))
    monkeypatch.setattr(lane, 'get_comfyui_history', lambda pid: {
        'p': {'status': {'status_str': 'error', 'messages': [
            ['execution_error',
             {'exception_message': 'Payment Required: Please add credits to your account.'}]]}}})

    with app.app_context():
        with pytest.raises(lane.ComfyGptUnavailable, match='out of credits'):
            lane.generate_variation([_png()], 'x')


def test_a_busy_comfyui_is_waited_for_instead_of_failing_the_row(monkeypatch):
    """Three of four rows failed with "ComfyUI not running" against an instance
    that was demonstrably up: the submit path's readiness probe is a 3 s GET and
    the fan-out ran four of them at once. Waiting is free — the row would be
    waiting anyway."""
    answers = iter([(False, 'ComfyUI not running (Please start external supervisor)'),
                    (False, 'ComfyUI not running (Please start external supervisor)'),
                    (True, 'Running')])
    monkeypatch.setattr('app.services.comfyui_service.ensure_comfyui_before_generation',
                        lambda: next(answers))
    monkeypatch.setattr(lane, '_READY_POLL_SECONDS', 0)
    REAL_WAIT()                                # returns rather than raising


def test_a_comfyui_that_never_answers_still_fails_the_row(monkeypatch):
    monkeypatch.setattr('app.services.comfyui_service.ensure_comfyui_before_generation',
                        lambda: (False, 'ComfyUI not running (Please start external supervisor)'))
    monkeypatch.setattr(lane, '_READY_POLL_SECONDS', 0)
    monkeypatch.setattr(lane, '_READY_TIMEOUT_SECONDS', 0.01)
    with pytest.raises(lane.ComfyGptUnavailable, match='not running'):
        REAL_WAIT()
