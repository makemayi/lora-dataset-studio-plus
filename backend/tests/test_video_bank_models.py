"""🎬 Video bank schema — the tables that let one source file become many clips.

The image lane rests on "one row = one file". The video lane cannot: a two-hour
rush is one file and four hundred training clips. These tests pin the consequences
of that — the two-level ownership, what a delete takes with it, and the fact that
clip bounds are sub-second quantities, not integers.

Pure schema behaviour. Cutting, detecting and encoding live elsewhere.
"""
from app.extensions import db
from app.models import (VideoBank, VideoClip, VideoDataset, VideoDatasetClip,
                        VideoSource)


def _bank_with_one_source(name='rushes', relpath='a.mp4'):
    bank = VideoBank(name=name, source_path='/srv/rushes')
    db.session.add(bank)
    db.session.flush()
    source = VideoSource(bank_id=bank.id, relpath=relpath,
                         duration_s=142.5, fps_native=29.97, width=1920, height=1080)
    db.session.add(source)
    db.session.flush()
    return bank, source


def test_a_bank_holds_sources_which_hold_clips(app):
    """The two-level shape, round-tripped: many clips point at one source, many
    sources point at one bank."""
    with app.app_context():
        bank, source = _bank_with_one_source()
        db.session.add_all([
            VideoClip(bank_id=bank.id, source_id=source.id, start_s=0.0, end_s=5.0),
            VideoClip(bank_id=bank.id, source_id=source.id, start_s=41.2, end_s=46.3),
        ])
        db.session.commit()

        assert VideoSource.query.filter_by(bank_id=bank.id).count() == 1
        assert VideoClip.query.filter_by(source_id=source.id).count() == 2


def test_clip_bounds_keep_subsecond_precision(app):
    """Bounds are PTS seconds, not frame indices and not whole seconds. Storing
    them as integers would silently round every cut to the nearest second — which
    on a 2-second clip is a quarter of the sample."""
    with app.app_context():
        bank, source = _bank_with_one_source()
        db.session.add(VideoClip(bank_id=bank.id, source_id=source.id,
                                 start_s=41.24, end_s=46.31))
        db.session.commit()

        clip = VideoClip.query.one()
        assert clip.start_s == 41.24
        assert clip.end_s == 46.31


def test_a_new_clip_starts_pending(app):
    """Same triage vocabulary as the image lane: nothing is kept or rejected until
    somebody says so."""
    with app.app_context():
        bank, source = _bank_with_one_source()
        db.session.add(VideoClip(bank_id=bank.id, source_id=source.id,
                                 start_s=0.0, end_s=5.0))
        db.session.commit()

        assert VideoClip.query.one().status == 'pending'


def test_deleting_a_bank_takes_its_sources_and_clips_with_it(app):
    """A bank is a scratch container over a read-only folder. Dropping it must not
    leave orphan rows pointing at a bank id that will be reused."""
    with app.app_context():
        bank, source = _bank_with_one_source()
        db.session.add(VideoClip(bank_id=bank.id, source_id=source.id,
                                 start_s=0.0, end_s=5.0))
        db.session.commit()

        db.session.delete(bank)
        db.session.commit()

        assert VideoSource.query.count() == 0
        assert VideoClip.query.count() == 0


def test_deleting_one_source_leaves_the_other_sources_clips_alone(app):
    """Re-scanning a bank after a file is removed deletes that source. It must take
    exactly its own clips."""
    with app.app_context():
        bank, source_a = _bank_with_one_source()
        source_b = VideoSource(bank_id=bank.id, relpath='b.mp4', duration_s=10.0)
        db.session.add(source_b)
        db.session.flush()
        db.session.add_all([
            VideoClip(bank_id=bank.id, source_id=source_a.id, start_s=0.0, end_s=5.0),
            VideoClip(bank_id=bank.id, source_id=source_b.id, start_s=0.0, end_s=5.0),
        ])
        db.session.commit()

        db.session.delete(source_a)
        db.session.commit()

        remaining = VideoClip.query.all()
        assert [c.source_id for c in remaining] == [source_b.id]


def test_a_video_dataset_records_the_target_profile_it_was_built_for(app):
    """The profile is not decoration: it is what says at which fps and at which
    length the clips on disk were encoded. Without it a dataset cannot be
    re-exported, and cannot be checked against the trainer that will read it."""
    with app.app_context():
        db.session.add(VideoDataset(name='Rushes', target_profile='wan22_14b',
                                    fps=16, frames=81, output_dir='/srv/ds/rushes'))
        db.session.commit()

        assert VideoDataset.query.one().target_profile == 'wan22_14b'


def test_a_dataset_clip_remembers_where_it_was_cut_from(app):
    """Provenance survives the encode. The dataset holds .mp4 files, but each one
    still knows its source file and its bounds — which is what makes a later
    re-export to another target possible instead of a re-scan from scratch."""
    with app.app_context():
        bank, source = _bank_with_one_source(relpath='sub/holiday.mp4')
        clip = VideoClip(bank_id=bank.id, source_id=source.id,
                         start_s=41.24, end_s=46.31)
        db.session.add(clip)
        dataset = VideoDataset(name='Rushes', target_profile='wan22_14b',
                               fps=16, frames=81, output_dir='/srv/ds/rushes')
        db.session.add(dataset)
        db.session.flush()
        db.session.add(VideoDatasetClip(
            dataset_id=dataset.id, filename='clip_0001.mp4',
            source_bank_id=bank.id, source_clip_id=clip.id,
            src_relpath='sub/holiday.mp4', start_s=41.24, end_s=46.31))
        db.session.commit()

        stored = VideoDatasetClip.query.one()
        assert stored.src_relpath == 'sub/holiday.mp4'
        assert stored.start_s == 41.24


def test_deleting_a_dataset_does_not_delete_the_bank_clips_it_came_from(app):
    """The bank outlives the dataset. Throwing away a badly-cut dataset must cost
    the triage work, not redo it: the clips stay, they just stop claiming to have
    been promoted."""
    with app.app_context():
        bank, source = _bank_with_one_source()
        dataset = VideoDataset(name='Rushes', target_profile='wan22_14b',
                               fps=16, frames=81, output_dir='/srv/ds/rushes')
        db.session.add(dataset)
        db.session.flush()
        clip = VideoClip(bank_id=bank.id, source_id=source.id, start_s=0.0,
                         end_s=5.0, status='keep', promoted_dataset_id=dataset.id)
        db.session.add(clip)
        db.session.commit()

        db.session.delete(dataset)
        db.session.commit()

        survivor = VideoClip.query.one()
        assert survivor.status == 'keep'
        assert survivor.promoted_dataset_id is None
