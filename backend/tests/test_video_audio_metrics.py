"""🔊 Measuring the audio nobody was measuring.

For the joint audio-video targets (LTX, MiniMax H3) the promotion MUXES the
source's track into every clip — `video_clip_export.clip_command` lines 141-156,
and the per-target policy in `video_targets.py` (H3: 32 kHz stereo). Nothing
looked at that track before this. A clip whose audio is a silent stretch, a
dropout or a muted camera passed exactly like a clip with sound, and a dataset
of silent clips teaches the model to be silent — with nothing anywhere to say so,
because the file is present, the right length and the right sample rate.

THE THREE STATES THAT MUST NOT COLLAPSE, which is what most of this file is
about. "This clip has no audio track", "this clip has a track and it is silent"
and "nobody has measured this clip's audio yet" are three different facts with
three different remedies (nothing to do / re-cut or drop / re-run the pass). Any
two of them collapsed into one produce a bank that lies: silence read as
'no track' hides a real defect, and 'not measured' read as silence flags half a
bank that is perfectly fine. The same rule the rest of this lane already follows
for unmeasured video — `video_metrics.verdicts` never flags a missing score.

THE RESUME CONTRACT IS THE SUBTLE ONE. `run_metrics` skips clips that already
have a `metrics_json` (video_metrics_scan.py:149), so a bank measured before
this feature existed keeps its summaries and they carry NO audio keys at all.
That absence has to mean "an earlier pass measured this, without audio" — not
"silent" — and the only way to get the numbers is an explicit re-measure.

The audio decode sits behind `_read_clip_audio`; these tests monkeypatch it and
stay green with no PyAV. The one integration test skips itself, loudly, when the
extra is absent.
"""
import json
import math

import pytest

from app.services import video_metrics as vm
from app.services import video_metrics_scan as scan


# --- the summary ---------------------------------------------------------------

def test_a_clip_with_sound_reports_a_silent_share_and_a_level():
    out = vm.summarise_audio([0.2, 0.25, 0.18, 0.22])

    assert out['audio_state'] == 'ok'
    assert out['silence_ratio'] == 0.0
    # dBFS: negative, and a fifth of full scale is around -14 dB.
    assert -20 < out['rms_dbfs'] < -10


def test_a_track_that_is_silent_throughout_is_measured_as_silent_not_as_absent():
    """The defect this whole feature exists to catch. A muted camera produces a
    perfectly valid AAC track of digital silence, and every check that looks at
    the FILE says the clip is fine."""
    out = vm.summarise_audio([0.0, 0.0, 0.0, 0.0])

    assert out['audio_state'] == 'ok'
    assert out['silence_ratio'] == 1.0
    assert out['rms_dbfs'] is not None and out['rms_dbfs'] < -100


def test_a_half_silent_clip_reports_the_share_and_not_only_the_average():
    """A dropout in the middle leaves the average level perfectly healthy — the
    same reason `freeze_ratio` exists next to `motion_mean`."""
    out = vm.summarise_audio([0.3, 0.3, 0.0, 0.0])

    assert out['silence_ratio'] == 0.5
    assert out['rms_dbfs'] > -30       # the average alone would say "fine"


def test_a_file_with_no_audio_track_says_so_and_invents_no_zero():
    """'No track' is not 'silent'. A Wan dataset has no audio by design (video
    _targets.py forces `-an`), and flagging every one of its clips as silent
    would be an app-wide false alarm."""
    out = vm.summarise_audio(None)

    assert out['audio_state'] == 'none'
    assert out['silence_ratio'] is None
    assert out['rms_dbfs'] is None


def test_a_track_that_could_not_be_decoded_is_unreadable_not_silent():
    """Same distinction the video side already makes between an unreadable clip
    and a black one — collapsing them filters a file for the wrong reason."""
    out = vm.summarise_audio([])

    assert out['audio_state'] == 'unreadable'
    assert out['silence_ratio'] is None


# --- what an UNMEASURED clip looks like -----------------------------------------

def test_a_summary_made_without_audio_carries_no_audio_keys_at_all():
    """Exactly the shape of a metrics_json written by an earlier pass. It has to
    stay distinguishable from 'measured, no track' — otherwise re-measuring a
    bank looks unnecessary and nobody ever gets the numbers."""
    out = vm.summarise(_frames(), fps=25)

    assert 'audio_state' not in out
    assert 'silence_ratio' not in out


def test_a_summary_made_with_audio_carries_the_keys_alongside_the_video_ones():
    out = vm.summarise(_frames(), fps=25, audio=[0.2, 0.2])

    assert out['metrics_state'] == 'ok'
    assert out['audio_state'] == 'ok'
    assert out['motion_mean'] is not None      # the video half is untouched


def test_an_unreadable_clip_makes_no_audio_claim_either():
    """If the segment could not be decoded, its audio was not measured — saying
    'no track' about a file we could not open would be a claim we cannot back."""
    out = vm.summarise([], fps=25, audio=None)

    assert out['metrics_state'] == 'unreadable'
    assert 'audio_state' not in out


# --- the flags ------------------------------------------------------------------

def test_a_mostly_silent_clip_is_flagged_only_when_a_cut_is_set():
    """No default threshold, like every other cut in this lane: the published
    ones do not transfer between corpora, and a cut belongs to the bank being
    worked on."""
    scores = {'audio_state': 'ok', 'silence_ratio': 0.8, 'rms_dbfs': -55.0}

    assert vm.verdicts(scores, {}) == set()
    assert 'silent' in vm.verdicts(scores, {'silence_max': 0.5})
    assert 'silent' not in vm.verdicts(scores, {'silence_max': 0.9})


def test_a_quiet_clip_is_its_own_flag_and_not_a_silent_one():
    """Different defects, different remedies — a quiet clip can be normalised, a
    silent one cannot be rescued at all. Same reasoning as freeze vs still."""
    scores = {'audio_state': 'ok', 'silence_ratio': 0.0, 'rms_dbfs': -48.0}

    flags = vm.verdicts(scores, {'audio_floor': -40.0, 'silence_max': 0.5})
    assert 'quiet' in flags and 'silent' not in flags


def test_a_clip_with_no_track_is_never_flagged_by_an_audio_cut():
    scores = {'audio_state': 'none', 'silence_ratio': None, 'rms_dbfs': None}

    assert vm.verdicts(scores, {'silence_max': 0.1, 'audio_floor': -20.0}) == set()


def test_a_clip_measured_before_audio_existed_is_never_flagged_by_an_audio_cut():
    """The load-bearing one for the resume contract. A bank measured last week
    has no audio keys; a cut set today must not retro-flag all of it."""
    scores = vm.summarise(_frames(), fps=25)          # no audio argument

    assert vm.verdicts(scores, {'silence_max': 0.0, 'audio_floor': 0.0}) == set()


# --- the scan -------------------------------------------------------------------

def test_the_scan_stores_the_audio_numbers_next_to_the_video_ones(app, monkeypatch):
    bank_id = _bank_with_clips(app, 2)
    monkeypatch.setattr(scan, '_read_clip_frames',
                        lambda path, start, end, fps: _frames())
    monkeypatch.setattr(scan, '_read_clip_audio',
                        lambda path, start, end: [0.2, 0.0])

    with app.app_context():
        scan.run_metrics(bank_id)

    stored = _summaries(app, bank_id)
    assert all(s['audio_state'] == 'ok' for s in stored)
    assert all(s['silence_ratio'] == 0.5 for s in stored)


def test_an_audio_decode_that_explodes_costs_the_audio_not_the_clip(app, monkeypatch):
    """The video measurements are the expensive ones and the ones the lane was
    built for. A codec that defeats the audio decoder must not throw them away."""
    bank_id = _bank_with_clips(app, 1)
    monkeypatch.setattr(scan, '_read_clip_frames',
                        lambda path, start, end, fps: _frames())

    def boom(path, start, end):
        raise OSError('no decoder for this audio codec')
    monkeypatch.setattr(scan, '_read_clip_audio', boom)

    with app.app_context():
        scan.run_metrics(bank_id)

    stored = _summaries(app, bank_id)[0]
    assert stored['metrics_state'] == 'ok'
    assert stored['motion_mean'] is not None
    assert stored['audio_state'] == 'unreadable'


def test_a_clip_already_measured_is_not_redecoded_for_the_new_metric(app, monkeypatch):
    """The resume contract, and the reason the absence of audio keys means 'an
    earlier pass'. Re-decoding a whole bank because a new metric shipped would
    turn every update into hours of work nobody asked for."""
    bank_id = _bank_with_clips(app, 2)
    monkeypatch.setattr(scan, '_read_clip_frames',
                        lambda path, start, end, fps: _frames())
    monkeypatch.setattr(scan, '_read_clip_audio', lambda path, start, end: None)

    with app.app_context():
        scan.run_metrics(bank_id)
        seen = []
        monkeypatch.setattr(scan, '_read_clip_audio',
                            lambda path, start, end: (seen.append(path), [0.2])[1])
        scan.run_metrics(bank_id)

    assert seen == []
    # …and an explicit re-measure is what gets the numbers.
    with app.app_context():
        scan.run_metrics(bank_id, remeasure=True)
    assert len(seen) == 2


# --- the cuts have to be reachable ------------------------------------------------

def test_the_audio_cuts_are_readable_as_thresholds(app):
    """A cut `verdicts()` honours but `metric_thresholds()` never reads is a
    feature that exists only in the source. This lane already carries one — see
    `first_frame_floor`, supported since wave 2 and absent from the config
    reader, so `soft_start` can never fire — and it must not gain a second."""
    from app.services import video_bank_service as svc
    with app.app_context():
        keys = set(svc.metric_thresholds())

    assert {'silence_max', 'audio_floor'} <= keys


def test_the_dry_run_counts_the_audio_cuts_instead_of_dropping_them(app, client):
    """The preview is what keeps a mis-set cut from gutting a bank, so a cut the
    route silently drops is worse than no preview at all: it reports zero and
    reassures. Asserted through the COUNT rather than through an echo of the
    request — what matters is that the cut reached the counting."""
    from app.extensions import db
    from app.models import VideoClip
    bank_id = _bank_with_clips(app, 1)
    with app.app_context():
        clip = VideoClip.query.filter_by(bank_id=bank_id).first()
        clip.metrics_json = json.dumps(vm.summarise(_frames(), fps=25,
                                                    audio=[0.0, 0.0]))
        db.session.commit()

    r = client.post(f'/api/video-bank/{bank_id}/metrics-dry-run',
                    json={'silence_max': 0.5, 'audio_floor': -40.0})

    body = r.get_json()
    assert r.status_code == 200
    assert body['silent'] == 1 and body['quiet'] == 1
    assert body['total_flagged'] == 1        # one CLIP, two rules


# --- the real audio seam, when the extra is present ------------------------------

def test_real_decode_reads_a_track_and_reports_its_absence(tmp_path):
    """Integration: the true PyAV path on two synthesised files — one with an
    audio track, one without. Skips, loudly, when the video extra is absent."""
    av = pytest.importorskip('av', reason='video extra not installed')
    import numpy as np

    with_audio = str(tmp_path / 'sound.mp4')
    _write_clip(av, np, with_audio, with_audio_track=True)
    silent_file = str(tmp_path / 'mute.mp4')
    _write_clip(av, np, silent_file, with_audio_track=False)

    windows = scan._read_clip_audio(with_audio, 0.0, 1.5)
    assert windows is not None and len(windows) >= 1
    assert all(0.0 <= w <= 1.0 for w in windows)
    # A real tone is nowhere near digital silence.
    assert vm.summarise_audio(windows)['silence_ratio'] < 1.0

    # No track at all — the state, not a zero.
    assert scan._read_clip_audio(silent_file, 0.0, 1.5) is None


# --- helpers ---------------------------------------------------------------------

def _write_clip(av, np, path, *, with_audio_track):
    with av.open(path, 'w') as container:
        vs = container.add_stream('h264', rate=10)
        vs.width, vs.height, vs.pix_fmt = 64, 64, 'yuv420p'
        audio_stream = None
        if with_audio_track:
            audio_stream = container.add_stream('aac', rate=44100)
        for i in range(20):
            img = np.full((64, 64, 3), (i * 12) % 255, dtype=np.uint8)
            for packet in vs.encode(av.VideoFrame.from_ndarray(img, format='rgb24')):
                container.mux(packet)
        if audio_stream is not None:
            # A 440 Hz tone, stereo, in 1024-sample chunks.
            t0 = 0
            for _ in range(80):
                t = (np.arange(1024) + t0) / 44100.0
                tone = (0.3 * np.sin(2 * math.pi * 440 * t)).astype('float32')
                frame = av.AudioFrame.from_ndarray(
                    np.vstack([tone, tone]), format='fltp', layout='stereo')
                frame.sample_rate = 44100
                frame.pts = t0
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
                t0 += 1024
            for packet in audio_stream.encode():
                container.mux(packet)
        for packet in vs.encode():
            container.mux(packet)


def _frames(n=24, luma=0.5, sharp=100.0, motion=0.003):
    return [{'luma': luma, 'sharp': sharp, 'motion': motion} for _ in range(n)]


def _bank_with_clips(app, n):
    from app.extensions import db
    from app.models import VideoBank, VideoClip, VideoSource
    with app.app_context():
        bank = VideoBank(name='b', source_path='/srv/rushes')
        db.session.add(bank)
        db.session.flush()
        src = VideoSource(bank_id=bank.id, relpath='a.mp4', duration_s=60.0,
                          fps_native=25.0, probe_state='ok')
        db.session.add(src)
        db.session.flush()
        for i in range(n):
            db.session.add(VideoClip(bank_id=bank.id, source_id=src.id,
                                     start_s=float(i * 10), end_s=float(i * 10 + 5)))
        db.session.commit()
        return bank.id


def _summaries(app, bank_id):
    from app.models import VideoClip
    with app.app_context():
        rows = VideoClip.query.filter_by(bank_id=bank_id).order_by(VideoClip.id).all()
        return [json.loads(r.metrics_json) if r.metrics_json else {} for r in rows]
