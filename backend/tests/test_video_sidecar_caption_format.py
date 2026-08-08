"""📝 The exported caption is the caption — for EVERY target, MiniMax H3 included.

WHY THIS FILE EXISTS. The obvious next thought about H3 is that its sidecars need
a special shape: the architecture is described everywhere as taking
``"<Picture i>: "``-labelled prompts, and it trains audio jointly, so a caption
that never mentions the soundtrack looks half-written. Both readings are wrong,
and acting on either would corrupt every H3 dataset the app promotes. Read in the
ai-toolkit INSTALLED on this machine, not in a model card:

  * THE LABEL IS SYNTHESISED BY THE ENCODER, NEVER READ FROM THE CAPTION.
    ``C:\\ai-toolkit\\extensions_built_in\\diffusion_models\\minimax_h3\\src\\
    text_encoder.py:65-72`` builds one ``f"<Picture {i + 1}>: "`` per KEYFRAME
    IMAGE and follows it with a vision block; the caption is appended after all
    of them, tokenized on its own at line 74 with ``add_special_tokens=False``.
    Its own module docstring (lines 8-15) states the presentation is raw tokens,
    "no chat template, no special tokens". A sidecar that carried a literal
    "<Picture 1>:" would be tokenized as ORDINARY TEXT and would announce an
    image that is not there.

  * THE CAPTION REACHES THE MODEL VERBATIM. ai-toolkit reads the homonym .txt
    (``toolkit/dataloader_mixins.py:141-155``), passes it through a
    ``clean_caption`` whose every normalising line is commented out (lines
    98-109), and hands it to the arch's ``get_prompt_embeds``, which encodes
    ``p.strip()`` and nothing else (``minimax_h3.py:552-561``). LTX-2 does the
    same (``ltx2/ltx2.py:1068``). There is no per-architecture caption formatting
    anywhere on that path.

  * THE SOUNDTRACK HAS ITS OWN CHANNEL AND IT IS NOT TEXT. The audio is decoded
    from the video file's own track (``dataloader_mixins.py:718-761``) and
    encoded by the audio VAE into latent rows (``minimax_h3.py:733-760``). No
    audio tag, no audio field, is ever parsed out of a caption. (Describing the
    soundtrack in the prose is not forbidden — the packed sequence is one joint
    attention over [text | keyframes | audio | video], so text rows do condition
    the audio rows — but that is a captioning-style question with no measurement
    behind it, not a format the trainer demands.)

So the contract is: ONE canonical prose caption, stored once, exported
unchanged, whatever the target. That is what these tests pin — because the
failure mode of the opposite belief is silent, and lands on the datasets rather
than on a stack trace.
"""
from pathlib import Path

import pytest

from app.services import video_clip_export as ex
from app.services import video_targets


def _promote(app, tmp_path, monkeypatch, *, caption, target_profile, frames):
    """Run a one-clip promotion onto `target_profile` with ffmpeg faked, and
    return the sidecar the exporter wrote.

    monkeypatch ONLY — a bare module assignment leaks the fake ffmpeg into every
    later test in the same process."""
    from app.extensions import db
    from app.models import VideoClip, VideoSource
    from app.services import video_bank_service as svc

    folder = tmp_path / target_profile
    folder.mkdir(exist_ok=True)
    (folder / 's0.mp4').write_bytes(b'\x00')
    calls = []

    def _run(args):
        calls.append(list(args))
        with open(args[-1], 'wb') as fh:
            fh.write(b'\x00')
        return 0, ''

    monkeypatch.setattr(svc, '_run_ffmpeg', _run)
    monkeypatch.setattr(svc, '_ffmpeg_or_raise', lambda: '/usr/bin/ffmpeg')
    with app.app_context():
        bank, _ = svc.create_bank('local', target_profile, str(folder))
        bank_id = bank.id
        src = VideoSource.query.filter_by(bank_id=bank_id).first()
        src.probe_state = 'ok'
        src.duration_s = 600.0
        src.fps_native = 30.0
        db.session.add(VideoClip(bank_id=bank_id, source_id=src.id, status='keep',
                                 start_s=0.0, end_s=20.0, caption=caption,
                                 caption_state='ok' if caption else None))
        db.session.commit()
        svc.start_promote(app, 'local', bank_id, name='Set',
                          target_profile=target_profile, frames=frames)

    clip = Path(calls[0][-1])
    return clip.parent / (clip.stem + '.txt')


CAPTION = ('A woman turns from the window and walks out of frame as the camera '
           'pushes in.')


def test_a_promoted_h3_clip_gets_its_caption_and_nothing_else(app, tmp_path,
                                                              monkeypatch):
    """The whole point of the file. text_encoder.py:67 makes the "<Picture i>: "
    label out of the KEYFRAMES it was handed; one written into the caption is
    plain text that claims an image nobody passed."""
    sidecar = _promote(app, tmp_path, monkeypatch, caption=CAPTION,
                       target_profile='minimax_h3', frames=107)

    assert sidecar.read_text(encoding='utf-8') == CAPTION


def test_the_target_profile_never_reshapes_the_prompt(app, tmp_path, monkeypatch):
    """Two targets, one caption, identical bytes. ai-toolkit's caption path is
    architecture-blind (dataloader_mixins.py:141-155 -> clean_caption, a no-op ->
    the arch's get_prompt_embeds), so a per-target sidecar dialect would be an
    invention of ours with nothing on the other end reading it."""
    h3 = _promote(app, tmp_path, monkeypatch, caption=CAPTION,
                  target_profile='minimax_h3', frames=107)
    wan = _promote(app, tmp_path, monkeypatch, caption=CAPTION,
                   target_profile='wan22_14b', frames=81)

    assert h3.read_bytes() == wan.read_bytes()


def test_no_target_declares_a_caption_style_the_exporter_would_have_to_apply():
    """`caption_style` is advice to the CAPTIONER, not a transformation the
    exporter owes the trainer — nothing downstream parses it. Pinned as a closed
    set so adding a value cannot quietly imply a rewrite step that does not
    exist: a new style must arrive with the code that honours it."""
    styles = {video_targets.get(k)['caption_style']
              for k in video_targets.PROFILE_KEYS}

    assert styles == {'freeform', 'paragraph_with_audio'}


@pytest.mark.parametrize('caption', [
    'A <Picture 1>: sign hangs over the door.',       # the label as real content
    'She says "meet me at 8" over a rising synth pad.',
])
def test_a_caption_that_looks_like_markup_still_rides_through_untouched(tmp_path,
                                                                       caption):
    """The exporter has no opinion about the caption's contents. A sanitiser here
    would silently edit the one thing the user wrote by hand — and the encoder
    tokenizes the caption with add_special_tokens=False, so there is no token
    boundary to protect (text_encoder.py:74)."""
    clip = tmp_path / 'clip_0001.mp4'
    ex.write_sidecar(str(clip), caption)

    assert (tmp_path / 'clip_0001.txt').read_text(encoding='utf-8') == caption
