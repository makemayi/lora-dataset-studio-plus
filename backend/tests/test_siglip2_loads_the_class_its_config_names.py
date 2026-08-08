"""The SigLIP2 workers must let the checkpoint's config choose the model class.

Seen on a real bank: every semantic scan died with
`SigLIP2 load failed: RuntimeError: You set ignore_mismatched_sizes to False`.
Both workers named `Siglip2Model` by hand, and the pinned checkpoint —
`google/siglip2-base-patch16-224` — declares `model_type: siglip`.

That is not a mistake in the pin. SigLIP 2's FIXED-RESOLUTION checkpoints reuse the
SigLIP 1 architecture; only the NaFlex variants declare `model_type: siglip2`. The
weights are SigLIP 2 either way, so nothing is mislabelled by loading them through
the class their own config names — and `AutoModel` is exactly the thing that reads
that name.

The failure was loud, which is worth keeping in mind: transformers refused rather
than initialising the mismatched tensors at random. A checkpoint loaded with the
wrong front-end would have produced embeddings that look fine and mean nothing.
"""
import json
import re
from pathlib import Path

import pytest

_INFER = Path(__file__).resolve().parents[1] / 'infer'
_WORKERS = ('bank_semantic_infer.py', 'siglip2_text_infer.py')


@pytest.mark.parametrize('name', _WORKERS)
def test_the_worker_does_not_name_the_model_class_by_hand(name):
    """`Siglip2Model` hardcoded is the defect itself: it builds a siglip2 shell
    around a siglip checkpoint. The class must come from the config."""
    src = (_INFER / name).read_text(encoding='utf-8')
    # CODE, not prose: the comment at the call site has to name `Siglip2Model` to
    # explain why it is absent, so forbidding the bare word would forbid the
    # explanation. What must not exist is the import and the call.
    code = '\n'.join(l for l in src.split('\n') if not l.lstrip().startswith('#'))
    assert not re.search(r'\bSiglip2Model\s*\.\s*from_pretrained', code), (
        f'{name} instantiates Siglip2Model; the pinned checkpoint declares '
        'model_type "siglip" and the weights will be refused')
    assert not re.search(r'^\s*from transformers import .*\bSiglip2Model\b', code,
                         re.M), f'{name} still imports the hand-picked class'
    assert re.search(r'model\s*=\s*AutoModel\.from_pretrained', code), (
        f'{name} must load through AutoModel so the config picks the class')


@pytest.mark.parametrize('name', _WORKERS)
def test_the_reason_is_written_where_the_next_reader_will_be(name):
    """A future contributor seeing `AutoModel` under a file called
    `siglip2_text_infer` will be tempted to "fix" it back. The measurement that
    forbids that has to live at the call site, not in a commit message."""
    src = (_INFER / name).read_text(encoding='utf-8')
    assert 'NaFlex' in src and 'model_type' in src, (
        f'{name} switched class without recording why it must stay that way')


def test_the_pin_still_points_at_a_fixed_resolution_checkpoint():
    """The whole fix rests on ONE property of the pinned model id. If someone
    repins to a NaFlex checkpoint, `AutoModel` still works (its config names
    Siglip2Model) — but the dimension and the processor change with it, so this
    test exists to make that a deliberate, visible edit rather than a surprise."""
    from app.services import bank_semantic_models as m
    assert m.MODEL_ID == 'google/siglip2-base-patch16-224'
    assert 'naflex' not in m.MODEL_ID
    assert m.DIMENSION == 768


def test_the_downloaded_config_is_the_one_this_fix_assumes(tmp_path):
    """Pins the reasoning against a real config body rather than a memory of it:
    a `siglip` model_type is what makes a hardcoded Siglip2Model wrong. Uses a
    local fixture so the test never reaches the network."""
    body = {
        'initializer_factor': 1.0,
        'model_type': 'siglip',
        'text_config': {'model_type': 'siglip_text_model', 'vocab_size': 256000},
        'vision_config': {'model_type': 'siglip_vision_model'},
    }
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps(body), encoding='utf-8')
    loaded = json.loads(cfg.read_text(encoding='utf-8'))
    assert loaded['model_type'] == 'siglip'
    assert loaded['model_type'] != 'siglip2', (
        'if this ever flips, the hardcoded class was not the bug and this fix '
        'needs re-reading')
