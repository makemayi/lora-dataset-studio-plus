"""Structural invariants of QUICK_GEN_COMPONENTS: every id unique within its
axis, every compatible_angles entry names a real angle id in that framing's
own angle pool, every phrase non-empty, and the expression pool never
contains a strong/face-distorting entry (identity comes first — see the
design spec's 'facial features are priority one' constraint)."""
from app.services import face_variations as fv

_STRONG_EXPRESSION_WORDS = ('laugh', 'laughing', 'surprised', 'shock',
                            'shocked', 'crying', 'angry', 'scream')


def test_every_axis_has_unique_ids():
    for framing, axes in fv.QUICK_GEN_COMPONENTS['human'].items():
        for axis, entries in axes.items():
            ids = [e['id'] for e in entries]
            assert len(ids) == len(set(ids)), f'{framing}.{axis} has duplicate ids: {ids}'


def test_every_phrase_is_nonempty_text():
    for framing, axes in fv.QUICK_GEN_COMPONENTS['human'].items():
        for axis, entries in axes.items():
            for e in entries:
                assert isinstance(e['phrase'], str) and e['phrase'].strip(), \
                    f"{framing}.{axis}.{e['id']} has an empty phrase"


def test_compatible_angles_only_name_real_angle_ids():
    for framing, axes in fv.QUICK_GEN_COMPONENTS['human'].items():
        angle_ids = {e['id'] for e in axes.get('angle', [])}
        for axis, entries in axes.items():
            if axis == 'angle':
                continue
            for e in entries:
                ca = e.get('compatible_angles')
                if ca is None:
                    continue
                unknown = set(ca) - angle_ids
                assert not unknown, (
                    f"{framing}.{axis}.{e['id']} names unknown angle(s) {unknown} "
                    f"— must be a subset of {sorted(angle_ids)}")


def test_face_has_no_pose_axis():
    """A close-up face shot has no body pose to vary — only angle/expression."""
    assert set(fv.QUICK_GEN_COMPONENTS['human']['face']) == {'angle', 'expression'}


def test_bust_and_body_have_no_expression_axis():
    """Expression is a face-only axis — bust/body never show enough of the
    face for it to matter, and it would just be dead weight in the pool."""
    for framing in ('bust', 'body'):
        assert 'expression' not in fv.QUICK_GEN_COMPONENTS['human'][framing]


def test_expression_pool_excludes_strong_face_distorting_entries():
    for e in fv.QUICK_GEN_COMPONENTS['human']['face']['expression']:
        low = e['phrase'].lower()
        for bad in _STRONG_EXPRESSION_WORDS:
            assert bad not in low, f"expression '{e['id']}' contains strong word '{bad}': {e['phrase']}"


def test_every_framing_angle_pool_is_nonempty():
    for framing, axes in fv.QUICK_GEN_COMPONENTS['human'].items():
        assert axes['angle'], f'{framing} has an empty angle pool'
