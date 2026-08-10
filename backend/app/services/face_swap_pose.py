"""What the tile's own prompt already knows about how the head is turned, and
what its face is doing — turned back into one sentence for the swap.

WHY THIS IS WORTH DOING AT ALL
------------------------------
The H3 swap graph is prompt-driven, and its instruction can only say "match the
shoulders": the model has to INFER, from a cropped photo, which way the head was
turned and what the expression was. It often gets the first one roughly right
and the second one wrong — a laughing body with a calm face reads as a paste-up
just as loudly as a bad seam does.

But the app is not guessing. The row that made this tile carries the catalog
prompt it was generated from, and that prompt SAYS both things in words:

    "upper body portrait, three-quarter view, wearing a soft open cardigan …,
     a calm neutral facial expression, not copying the expression from the
     reference"

Counted over this database: 490 rows name an expression, 112 say front view, 65
three-quarter, 56 profile. So the orientation and the expression are already
written down for most tiles, in the vocabulary the catalog itself uses — this
module only reads them back out.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
* It does not INVENT a pose. A tile with no usable phrase returns None and the
  prompt keeps its generic "match the shoulders" clause, which is the honest
  answer: a wrong orientation stated confidently is worse than none.
* It does not read the picture. Estimating yaw from the pixels is possible here
  (InsightFace already computes it — see infer/face_score_infer.py, which reads
  `face.pose[1]`), but that costs a detection pass per swap and is only needed
  for IMPORTED photos, which carry no catalog prompt. That is the second source
  to add, not the first.
* It never takes the expression from the reference photo. The reference is ONE
  photo with ONE expression, and copying it is exactly what the catalog prompts
  spend a clause forbidding.

The vocabulary below is READ FROM the shipped catalog, not invented: every
phrase here is one the generation prompts actually use. A term that stops being
used costs nothing; a term that is added there and not here simply stops being
carried, which degrades to the generic clause rather than to a wrong one.
"""
from __future__ import annotations
import re

# Orientation, longest first: "three-quarter view from behind" must not match as
# "from behind". Each maps to how the SWAP prompt should say it — the model is
# being told where the face points, not being handed catalog jargon.
_VIEWS = (
    ('three-quarter view', 'turned three-quarters towards the camera'),
    ('three quarter view', 'turned three-quarters towards the camera'),
    ('profile view', 'in full profile, seen from the side'),
    ('side view', 'in full profile, seen from the side'),
    ('back view', 'seen from behind'),
    ('from behind', 'seen from behind'),
    ('front view', 'facing the camera straight on'),
    ('looking over the shoulder', 'looking back over the shoulder'),
    ('looking at the camera', 'looking straight at the camera'),
    ('looking away', 'looking away from the camera'),
    ('looking to the left', 'looking to the left'),
    ('looking to the right', 'looking to the right'),
    ('looking up', 'looking upwards'),
    ('looking down', 'looking downwards'),
)

# Tilt is independent of view — a three-quarter head can also be chin-up.
_TILTS = (
    ('head tilted slightly upward', 'the head tilted slightly up'),
    ('head tilted slightly downward', 'the head tilted slightly down'),
    ('head tilted slightly', 'the head tilted slightly'),
    ('head tilted', 'the head tilted'),
    ('chin up', 'the chin raised'),
    ('chin down', 'the chin lowered'),
    ('chin slightly up', 'the chin raised slightly'),
)

# Expression. 'calm neutral facial expression' is by far the most common (490
# rows), and it is the one the model most often gets wrong by inventing a smile.
_EXPRESSIONS = (
    ('calm neutral facial expression', 'a calm, neutral expression, mouth closed'),
    ('neutral facial expression', 'a neutral expression, mouth closed'),
    ('neutral expression', 'a neutral expression, mouth closed'),
    ('laughing', 'laughing, mouth open'),
    ('broad smile', 'a broad smile'),
    ('slight smile', 'a slight, closed-mouth smile'),
    ('soft smile', 'a soft, closed-mouth smile'),
    ('smiling', 'a smile'),
    ('smile', 'a smile'),
    ('serious', 'a serious expression'),
    ('surprised', 'a surprised expression'),
)

# Framing is the fallback for the ONE orientation a catalog prompt may leave
# implicit: a back shot's head is not facing the camera whatever else it says.
_FRAMING_VIEWS = {'back': 'seen from behind'}


def _first_match(haystack, table):
    for needle, phrasing in table:
        if needle in haystack:
            return phrasing
    return None


def pose_hint(variation_prompt=None, framing=None, variation_label=None):
    """One sentence describing how the head sits and what the face is doing, or
    None when the row says nothing usable.

    PURE — text in, text out. Reads the tile's own catalog prompt first, its
    label second (a label like "Bust, three-quarter" carries the same words),
    and the stored framing last."""
    text = ' '.join(str(part or '') for part in (variation_prompt, variation_label))
    text = re.sub(r'\s+', ' ', text).strip().lower()
    view = _first_match(text, _VIEWS)
    tilt = _first_match(text, _TILTS)
    expression = _first_match(text, _EXPRESSIONS)
    if view is None:
        view = _FRAMING_VIEWS.get(str(framing or '').strip().lower())
    if not (view or tilt or expression):
        return None

    # Built rather than joined: a tilt is not a thing the head "is", so
    # "the head is the chin raised" is what a naive join produces — and it did,
    # on 'bust shot, chin up' rows.
    if view and tilt:
        pose = f'the head is {view}, with {tilt}'
    elif view:
        pose = f'the head is {view}'
    elif tilt:
        pose = f'the head is held with {tilt}'
    else:
        pose = ''
    parts = []
    if pose:
        parts.append('In this shot ' + pose)
    if expression:
        parts.append(('and the face wears ' if pose else 'In this shot the face wears ')
                     + expression)
    sentence = ' '.join(parts).strip()
    # The whole point of stating it: the model must reproduce THIS, and must not
    # import the reference's own expression, which is a single frozen photo.
    return (sentence + ' — reproduce that exactly, and do not carry over the '
                       'expression or the head direction of <Picture 1>.')
