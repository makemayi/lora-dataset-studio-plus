"""The tile already says which way its head is turned — read it back out.

The H3 swap's instruction can only tell the model to "match the shoulders", so
orientation and expression are inferred from a cropped photo. They do not have
to be: the row carries the catalog prompt that GENERATED the tile, and that
prompt names both in the catalog's own words (counted over a real database: 490
rows name an expression, 112 say front view, 65 three-quarter, 56 profile).

These pin the two halves that matter — that a real catalog phrase is carried,
and that a row saying nothing usable produces NOTHING rather than a guess.
"""
from app.services.face_swap_pose import pose_hint


def test_a_real_catalog_prompt_yields_orientation_and_expression():
    hint = pose_hint(
        'upper body portrait, three-quarter view, wearing a soft open cardigan '
        'over a top, home library background, a calm neutral facial expression, '
        'not copying the expression from the reference', 'bust', 'Bust, cardigan')
    assert 'three-quarters' in hint
    assert 'neutral expression' in hint
    # The reference is ONE photo with ONE expression; importing it is the exact
    # failure the catalog prompts themselves spend a clause forbidding.
    assert '<Picture 1>' in hint and 'do not carry over' in hint


def test_each_kind_of_view_is_named_in_words_a_model_can_act_on():
    assert 'facing the camera straight on' in pose_hint('close-up portrait, front view')
    assert 'full profile' in pose_hint('bust shot, profile view')
    assert 'seen from behind' in pose_hint('full body, back view')
    assert 'looking upwards' in pose_hint('portrait, looking up, soft light')


def test_the_longest_phrase_wins():
    """"three-quarter view" must not be read as the "view" of something else,
    and "slight smile" must not degrade to a bare "smile"."""
    assert 'three-quarters' in pose_hint('three-quarter view, front lighting')
    assert 'closed-mouth smile' in pose_hint('portrait with a slight smile')


def test_tilt_rides_along_with_the_view():
    hint = pose_hint('bust shot, front view, head tilted slightly upward, smiling')
    assert 'facing the camera straight on' in hint
    assert 'tilted slightly up' in hint
    assert 'a smile' in hint


def test_a_back_shot_is_known_from_its_framing_alone():
    """A row whose prompt forgot to say it is still not facing the camera."""
    assert 'seen from behind' in pose_hint('full body shot, evening street', 'back')


def test_a_row_that_says_nothing_usable_gets_NO_hint(): 
    """An imported photo has no catalog prompt. A wrong orientation stated
    confidently is worse than the generic "match the shoulders" clause."""
    assert pose_hint(None, None, None) is None
    assert pose_hint('', '', '') is None
    assert pose_hint('a photograph of a person outdoors', 'bust') is None


def test_the_label_alone_is_enough_when_the_prompt_is_gone():
    assert 'full profile' in pose_hint(None, 'face', 'Face, profile view')


def test_a_tilt_without_a_view_still_reads_as_english():
    """A naive join produced "the head is the chin raised" on real rows — a tilt
    is not something the head IS."""
    hint = pose_hint('bust shot, one hand at the collarbone, chin up, '
                     'a calm neutral facial expression')
    assert 'the head is held with the chin raised' in hint
    assert 'the head is the chin' not in hint


def test_a_view_and_a_tilt_read_as_one_clause():
    hint = pose_hint('portrait, three-quarter view, head tilted slightly upward')
    assert 'the head is turned three-quarters towards the camera, with the head tilted slightly up' in hint
