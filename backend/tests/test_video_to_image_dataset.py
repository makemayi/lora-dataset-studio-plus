"""The promotion-to-images service: what it refuses, and when it refuses it.

Every refusal here has to happen BEFORE a dataset row or a folder exists — a
background job that dies on its first item leaves the user something to clean
up. So these tests mostly assert that a bad request raises and that nothing was
created, not that a good one produces frames (that path is covered by
`test_video_frame_extract.py`, without a video file).
"""

import pytest

from app.services import video_to_image_dataset as vtid


class _Bank:
    id = 1
    source_path = '/tmp/bank'


@pytest.fixture(autouse=True)
def _no_bank_lookup(monkeypatch):
    """`_require_free_bank` hits the database; the refusals under test come
    after it and do not care which bank it is."""
    monkeypatch.setattr(vtid.vbs, '_require_free_bank',
                        lambda user_id, bank_id: _Bank())


def call(**over):
    kw = dict(app=None, user_id='u', bank_id=1, name='frames',
              frames_per_clip=3, require_face=False)
    kw.update(over)
    return vtid.start_promote_to_images(**kw)


# ── refusals, all synchronous ────────────────────────────────────────────────

@pytest.mark.parametrize('name', ['', '   ', None])
def test_a_nameless_dataset_is_refused(name):
    with pytest.raises(ValueError, match='name'):
        call(name=name)


@pytest.mark.parametrize('n', [0, -1, vtid.FRAMES_PER_CLIP_MAX + 1])
def test_an_impossible_frame_budget_is_refused(n):
    with pytest.raises(ValueError, match='frames per clip'):
        call(frames_per_clip=n)


@pytest.mark.parametrize('n', ['many', None, 2.5j])
def test_a_non_numeric_frame_budget_is_refused(n):
    with pytest.raises(ValueError, match='whole number'):
        call(frames_per_clip=n)


def test_the_face_gate_without_an_interpreter_is_refused_not_degraded(monkeypatch):
    """Silently producing an unfiltered dataset from a request that asked for
    faces is the kind of mismatch nobody attributes to the right setting."""
    monkeypatch.setattr(vtid.cfg, 'get', lambda key, *a: '')
    monkeypatch.setattr(vtid, 'resolve_refs', lambda *a: ['/tmp/ref.png'])
    with pytest.raises(ValueError, match='face_scoring.python'):
        call(require_face=True, ref_dataset_id=5)


def test_asking_for_the_face_filter_with_nothing_to_compare_extracts_anyway():
    """There is no third answer. Refusing would block a perfectly reasonable
    run — a location set, or a fresh dataset whose reference comes later — and
    claiming it filtered would be a lie about the pictures. So it runs, OFF, and
    the caller is told (composition.face_filter_skipped)."""
    assert vtid.face_filter_decision(True, []) is False
    assert vtid.face_filter_decision(True, ['/a.png']) is True
    assert vtid.face_filter_decision(False, ['/a.png']) is False


def test_no_face_gate_needs_neither_interpreter_nor_reference(monkeypatch):
    monkeypatch.setattr(vtid.cfg, 'get', lambda key, *a: '')
    assert vtid._face_pass_or_raise(None, False) is None


def test_the_face_config_is_returned_whole_when_it_is_usable(monkeypatch):
    values = {'face_scoring.python': 'py.exe',
              'face_scoring.models_root': '/models'}
    monkeypatch.setattr(vtid.cfg, 'get', lambda key, *a: values.get(key, ''))
    python, script, root = vtid._face_pass_or_raise(['/ref.png'], True)
    assert python == 'py.exe'
    assert script.endswith('face_score_infer.py')
    assert root == '/models'


def test_a_blank_models_root_becomes_none_not_an_empty_path(monkeypatch):
    """insightface treats '' as a path; None means "use your default"."""
    monkeypatch.setattr(vtid.cfg, 'get',
                        lambda key, *a: 'py.exe' if 'python' in key else '   ')
    assert vtid._face_pass_or_raise(['/ref.png'], True)[2] is None


# ── the job key ──────────────────────────────────────────────────────────────

def test_it_shares_the_video_promotion_job_key():
    """One bank runs one job: a frame extraction and a clip encode would fight
    over the same decoder and the same disk, and the bank UI has one slot."""
    assert vtid.job_key(7) == vtid.vbs.job_key(7)


# ── references come from a dataset id, never from a path in the body ─────────

def test_the_client_cannot_name_file_paths_at_all():
    """A body that could hand arbitrary absolute paths to the face subprocess is
    a file-read primitive, and "it is only the face scorer" is not a boundary
    anything enforces. The API takes a dataset id; paths are built server-side."""
    import inspect
    params = inspect.signature(vtid.start_promote_to_images).parameters
    assert 'ref_dataset_id' in params
    assert 'refs' not in params


def test_an_unknown_reference_dataset_is_refused(monkeypatch):
    monkeypatch.setattr('app.services.face_dataset_service.get_dataset',
                        lambda *a: None)
    with pytest.raises(ValueError, match='does not exist'):
        vtid.resolve_refs('u', 999)


def test_no_reference_dataset_resolves_to_no_paths():
    assert vtid.resolve_refs('u', None) == []


def test_a_reference_dataset_with_no_photo_says_which_one(monkeypatch):
    class _DS:
        id, name, ref_filename = 4, 'ada', None
    monkeypatch.setattr('app.services.face_dataset_service.get_dataset',
                        lambda *a: _DS())
    monkeypatch.setattr('app.services.reference_photos_service._extra_ref_paths',
                        lambda ds: [])
    with pytest.raises(ValueError, match='ada'):
        vtid.resolve_refs('u', 4)


# ── which gates a CHARACTER dataset turns on ─────────────────────────────────

def test_a_character_set_gets_the_pixel_gate_and_a_style_set_does_not():
    """A style or location set wants the variety a passer-by's face brings."""
    px, _ = vtid.character_gates('character', False)
    assert px == vtid.video_frame_select.MIN_FACE_PX
    assert vtid.character_gates('style', True) == (None, None)
    assert vtid.character_gates('concept', True) == (None, None)


def test_an_unset_kind_is_treated_as_character():
    """`create_dataset` defaults to character, and the stricter reading is the
    safe one when the answer is unknown."""
    assert vtid.character_gates(None, False)[0] is not None
    assert vtid.character_gates('', False)[0] is not None


def test_the_identity_gate_needs_something_to_be_similar_to():
    """Without the face filter there is no `sim` at all, and a gate on it would
    reject every frame for a reason the user never chose."""
    assert vtid.character_gates('character', False)[1] is None
    assert vtid.character_gates('character', True)[1] == vtid.video_frame_select.MIN_SIM
