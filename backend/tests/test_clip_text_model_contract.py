"""The image half and the text half of CLIP must load the SAME model.

🔤 Text search ranks a query vector (from ``clip_text_infer.py``) against the
image vectors ``bank_score_infer.py`` cached. Cosine similarity between vectors
produced by two DIFFERENT CLIP configurations is meaningless — and, crucially,
it does not look meaningless: the dot product returns numbers in the usual range
and a ranking that is plausible at a glance but is noise. No exception, no
crash, nothing in a log. It is the worst failure mode this feature has, so it
gets a test of its own rather than a comment.

The pair is asserted textually, on the source, because the only environment that
can actually import open_clip is the ML interpreter — not the Flask venv the
suite runs in. A textual contract still catches the realistic regression: someone
"modernises" one script's model id (say to the `-quickgelu` variant that matches
how the openai weights were really trained) and silently invalidates every
comparison against every already-scored bank.

⚠️ Known, deliberate divergence from upstream: the `openai` ViT-L-14 weights were
trained WITH QuickGELU, and `'ViT-L-14'` builds a plain-GELU model — open_clip
warns about it on every load. Every embedding cached on every install out there
was produced that way. Switching to `'ViT-L-14-quickgelu'` would be more correct
in the abstract and would invalidate every scored bank in practice, so it is a
migration to schedule, NOT a one-line fix. Both halves stay wrong together, on
purpose, until that migration happens.
"""
import re
from pathlib import Path

INFER = Path(__file__).resolve().parents[1] / 'infer'
SCORE = INFER / 'bank_score_infer.py'
TEXT = INFER / 'clip_text_infer.py'


def _score_spec():
    """(model_name, pretrained) as bank_score_infer.py passes them."""
    src = SCORE.read_text(encoding='utf-8')
    m = re.search(r"create_model_and_transforms\(\s*'([^']+)'\s*,\s*"
                  r"pretrained='([^']+)'", src)
    assert m, 'could not find the CLIP model spec in bank_score_infer.py'
    return m.group(1), m.group(2)


def _text_spec():
    """(MODEL_NAME, PRETRAINED) as clip_text_infer.py declares them."""
    src = TEXT.read_text(encoding='utf-8')
    name = re.search(r"^MODEL_NAME\s*=\s*'([^']+)'", src, re.M)
    pre = re.search(r"^PRETRAINED\s*=\s*'([^']+)'", src, re.M)
    assert name and pre, 'could not find the CLIP model spec in clip_text_infer.py'
    return name.group(1), pre.group(1)


def test_text_and_image_towers_load_the_same_checkpoint():
    """The load-bearing assertion of the whole feature."""
    assert _text_spec() == _score_spec(), (
        'clip_text_infer.py and bank_score_infer.py must load the SAME CLIP '
        'model/pretrained pair — otherwise text search ranks by a cosine '
        'between incomparable vectors and silently returns nonsense.')


def test_the_text_side_uses_the_declared_constants():
    """The constants are not decoration: the actual call has to use them, or the
    contract above would pass while the code loaded something else."""
    src = TEXT.read_text(encoding='utf-8')
    assert re.search(r'create_model_and_transforms\(\s*\n?\s*MODEL_NAME\s*,\s*'
                     r'pretrained=PRETRAINED', src), \
        'clip_text_infer.py must build its model from MODEL_NAME/PRETRAINED'
    # The tokenizer must come from the same model id — a mismatched tokenizer is
    # the same class of silent bug.
    assert 'get_tokenizer(MODEL_NAME)' in src


def test_both_sides_l2_normalise():
    """A dot product only IS a cosine when both sides are unit vectors. The image
    side normalises; the text side must too, or every score is off by a factor
    nobody can see."""
    for path in (SCORE, TEXT):
        src = path.read_text(encoding='utf-8')
        assert re.search(r'emb\s*/\s*emb\.norm\(dim=-1,\s*keepdim=True\)', src), \
            f'{path.name} must L2-normalise its embeddings'


def test_the_text_child_never_takes_the_gpu():
    """Text encoding is ~20 ms of compute behind an ~8 s load: the GPU buys
    nothing and would make a search collide with a training run. The child must
    hide CUDA from itself BEFORE torch is imported."""
    src = TEXT.read_text(encoding='utf-8')
    hide = src.index("os.environ['CUDA_VISIBLE_DEVICES'] = ''")
    imp = src.index('import torch', hide - 4000 if hide > 4000 else 0)
    assert hide < src.index('import torch', imp), \
        'CUDA must be hidden before torch is imported'
    assert "model.to('cpu')" in src
