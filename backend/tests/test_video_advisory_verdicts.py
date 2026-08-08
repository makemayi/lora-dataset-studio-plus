"""The two verdicts that come from OTHER passes — near-duplicate and watermark.

Every other flag in this lane is derived from numbers one decode produced. These
two are not: they are produced by passes with their own cost (a dedup over the
search vectors, a classifier over one frame per shot), they land in the SAME
metrics_json blob, and that shared blob is where they can quietly be lost.

So what is pinned here is the plumbing, not the arithmetic:

  * the flags fire, and they fire on the right member of a group;
  * they fire on a clip the metrics pass never measured — the dedup pass only
    needs vectors, and a bank that was embedded but not measured is a real state;
  * re-measuring a bank does not erase them;
  * an un-scanned clip is never flagged, so "not evaluated" cannot read as
    "clean".
"""
import json

from app.services import video_metrics


def test_a_non_representative_member_of_a_group_is_flagged():
    flags = video_metrics.verdicts({'duplicate_group': 1, 'duplicate_of': 12}, {})
    assert 'duplicate' in flags


def test_the_representative_of_a_group_is_not_flagged():
    """It is IN the group — the grid says "kept, 1 of 3" — and it is the shot the
    user is being told to keep. Flagging it would say drop all of them."""
    flags = video_metrics.verdicts({'duplicate_group': 1, 'duplicate_of': None}, {})
    assert 'duplicate' not in flags


def test_a_clip_the_dedup_pass_never_saw_is_not_flagged():
    assert 'duplicate' not in video_metrics.verdicts({'metrics_state': 'ok'}, {})


def test_a_watermark_above_the_cut_is_flagged():
    flags = video_metrics.verdicts({'watermark_state': 'ok', 'watermark_score': 0.91},
                                   {'watermark_max': 0.6})
    assert 'watermark' in flags


def test_a_watermark_below_the_cut_is_not():
    flags = video_metrics.verdicts({'watermark_state': 'ok', 'watermark_score': 0.04},
                                   {'watermark_max': 0.6})
    assert 'watermark' not in flags


def test_an_unscanned_clip_is_never_flagged_whatever_the_cut():
    """No score is not a low score. A bank half-scanned must not report its
    unscanned half as clean."""
    assert 'watermark' not in video_metrics.verdicts({'metrics_state': 'ok'},
                                                     {'watermark_max': 0.0})


def test_a_frame_the_detector_could_not_judge_is_never_flagged():
    flags = video_metrics.verdicts(
        {'watermark_state': 'unreadable', 'watermark_score': None},
        {'watermark_max': 0.0})
    assert 'watermark' not in flags


def test_the_watermark_cut_is_one_the_panel_and_the_route_both_know():
    """The canonical list, or the cut is settable only by hand-editing
    config.json — the exact failure `first_frame_floor` shipped with."""
    assert 'watermark_max' in video_metrics.THRESHOLD_KEYS


# --- the shared blob ------------------------------------------------------------

def test_re_measuring_a_bank_keeps_the_verdicts_the_other_passes_produced():
    """The metrics scan rewrites metrics_json wholesale. Without a merge, every
    'measure again' would silently throw away a dedup and a watermark pass —
    invisibly, since the flags simply stop appearing."""
    previous = {'metrics_state': 'ok', 'sharpness_p90': 3.0,
                'duplicate_group': 2, 'duplicate_of': 9,
                'watermark_score': 0.8, 'watermark_state': 'ok'}
    fresh = {'metrics_state': 'ok', 'sharpness_p90': 99.0}
    merged = video_metrics.merge_advisory(previous, fresh)
    assert merged['sharpness_p90'] == 99.0        # the re-measure wins its own keys
    assert merged['duplicate_of'] == 9
    assert merged['watermark_score'] == 0.8


def test_the_scan_merges_rather_than_replaces(app, monkeypatch):
    from app.services import video_metrics_scan as scan
    bank_id, clip_id = _bank_with_one_clip(app)
    _stored(app, clip_id, {'metrics_state': 'ok', 'sharpness_p90': 1.0,
                           'watermark_score': 0.8, 'watermark_state': 'ok'})
    monkeypatch.setattr(scan, '_read_clip_frames',
                        lambda path, start, end, fps:
                        [{'luma': 0.5, 'sharp': 100.0, 'motion': 0.01}] * 10)
    monkeypatch.setattr(scan, '_audio_of', lambda path, clip: None)

    with app.app_context():
        scan.run_metrics(bank_id, remeasure=True)

    assert _summary(app, clip_id)['watermark_score'] == 0.8


# --- the grid's view ------------------------------------------------------------

def test_a_clip_nobody_measured_still_carries_its_duplicate_flag(app):
    """The dedup pass reads VECTORS, not measurements, so an embedded-but-never-
    measured bank can legitimately have duplicate groups. Reading the flags off
    the 'ok'-only summary would drop every one of them, with no error to see."""
    from app.services import video_bank_service as svc
    bank_id, clip_id = _bank_with_one_clip(app)
    _stored(app, clip_id, {'duplicate_group': 1, 'duplicate_of': 4})

    with app.app_context():
        from app.models import VideoClip
        clip = VideoClip.query.filter_by(bank_id=bank_id).one()
        row = svc._clip_row(clip, {}, svc.metric_thresholds())

    assert 'duplicate' in row['flags']
    # …and the metrics payload stays None, because nothing MEASURED this clip.
    assert row['metrics'] is None


def test_the_preview_counts_the_watermark_cut_on_an_unmeasured_clip(app):
    """The dry run and the grid must agree about the same bank. The preview used
    to drop the blob of any clip whose metrics_state was not 'ok' — right while
    every cut came from the metrics decode, and wrong for a score the 🔖 pass
    wrote on a clip nobody measured. A preview that says "0 would be flagged"
    over a bank the grid then flags is worse than no preview."""
    from app.config import LOCAL_USER
    from app.services import video_bank_service as svc
    bank_id, clip_id = _bank_with_one_clip(app)
    _stored(app, clip_id, {'watermark_state': 'ok', 'watermark_score': 0.99})

    with app.app_context():
        out = svc.metrics_dry_run(LOCAL_USER, bank_id, {'watermark_max': 0.94})

    assert out['watermark'] == 1
    assert out['total_flagged'] == 1


# --- helpers ---------------------------------------------------------------------

def _bank_with_one_clip(app):
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
        clip = VideoClip(bank_id=bank.id, source_id=src.id, start_s=0.0, end_s=5.0)
        db.session.add(clip)
        db.session.commit()
        return bank.id, clip.id


def _stored(app, clip_id, summary):
    from app.extensions import db
    from app.models import VideoClip
    with app.app_context():
        db.session.get(VideoClip, clip_id).metrics_json = json.dumps(summary)
        db.session.commit()


def _summary(app, clip_id):
    from app.extensions import db
    from app.models import VideoClip
    with app.app_context():
        return json.loads(db.session.get(VideoClip, clip_id).metrics_json)
