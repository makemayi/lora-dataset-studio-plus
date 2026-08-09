"""Bringing a finished render out of ComfyUI's output folder.

THE BUG THIS FILE EXISTS FOR
----------------------------
Four call sites used `shutil.move`. On Windows that has two jaws, and only the
second one bites:

  1. ComfyUI's output and the app's data folder are routinely on DIFFERENT
     DRIVES (portable ComfyUI on one, datasets on another). `os.rename` fails
     with WinError 17 and `shutil.move` degrades to copy + unlink. Correct.
  2. The unlink then fails with WinError 32 — ComfyUI has just written the file
     and Windows still holds the handle. `shutil.move` raises AFTER the copy
     succeeded.

So a render that ARRIVED INTACT raised anyway, the completion callback died, and
the batch's progress indicator stalled forever: the app reporting "generating"
with nothing running and the image already on disk.

Measured 2026-08-09 on dataset 11, `local_FaceSwap_82fdc980_00001_.png` —
identical 3,169,043 bytes at both paths, and a 1-of-2 counter that never moved.

The rule these tests encode: **the content arriving is the success condition.**
A leftover file costs disk; a raised exception costs the run.
"""
import os

import pytest

from app.utils import comfy_fs

PIXELS = b'\x89PNG\r\n\x1a\n' + b'render-bytes' * 64


@pytest.fixture
def paths(tmp_path):
    out = tmp_path / 'comfy' / 'output'
    out.mkdir(parents=True)
    src = out / 'local_Klein_abc123_00001_.png'
    src.write_bytes(PIXELS)
    dst = tmp_path / 'data' / 'datasets' / '11' / 'local_Klein_abc123_00001_.png'
    return str(src), str(dst)


def test_the_ordinary_case_moves_and_leaves_nothing_behind(paths):
    src, dst = paths
    assert comfy_fs.collect_output(src, dst) is True
    assert open(dst, 'rb').read() == PIXELS
    assert not os.path.exists(src)


def test_it_creates_the_destination_folder(tmp_path):
    src = tmp_path / 'in.png'
    src.write_bytes(PIXELS)
    dst = tmp_path / 'never' / 'existed' / 'out.png'
    assert comfy_fs.collect_output(str(src), str(dst)) is True
    assert dst.read_bytes() == PIXELS


def test_a_cross_drive_source_still_arrives(paths, monkeypatch):
    """WinError 17. `os.replace` cannot cross volumes, so the copy path runs."""
    src, dst = paths
    monkeypatch.setattr(comfy_fs.os, 'replace',
                        lambda *a: (_ for _ in ()).throw(OSError(17, 'cross-device')))
    assert comfy_fs.collect_output(src, dst) is True
    assert open(dst, 'rb').read() == PIXELS
    assert not os.path.exists(src)


# --- THE regression ----------------------------------------------------------

def test_a_held_open_source_does_not_lose_the_render(paths, monkeypatch, caplog):
    """WinError 32 on every retry. The bytes are at `dst`, so this MUST NOT
    raise — raising is what killed the completion callback and stalled the
    batch. It reports False and says so in the log instead."""
    src, dst = paths
    monkeypatch.setattr(comfy_fs.os, 'replace',
                        lambda *a: (_ for _ in ()).throw(OSError(17, 'cross-device')))
    monkeypatch.setattr(comfy_fs.os, 'unlink',
                        lambda *a: (_ for _ in ()).throw(PermissionError(32, 'in use')))
    monkeypatch.setattr(comfy_fs.time, 'sleep', lambda _s: None)

    assert comfy_fs.collect_output(src, dst) is False    # no exception
    assert open(dst, 'rb').read() == PIXELS              # the render arrived
    assert os.path.exists(src)                           # original left behind
    assert 'could not remove the original' in caplog.text


def test_the_unlink_is_retried_before_giving_up(paths, monkeypatch):
    """The handle is normally released within a second or two, so a brief retry
    turns the common case back into a clean move."""
    src, dst = paths
    monkeypatch.setattr(comfy_fs.os, 'replace',
                        lambda *a: (_ for _ in ()).throw(OSError(17, 'cross-device')))
    monkeypatch.setattr(comfy_fs.time, 'sleep', lambda _s: None)
    real_unlink = os.unlink
    calls = {'n': 0}

    def flaky(path):
        calls['n'] += 1
        if calls['n'] < 3:
            raise PermissionError(32, 'in use')
        real_unlink(path)

    monkeypatch.setattr(comfy_fs.os, 'unlink', flaky)
    assert comfy_fs.collect_output(src, dst) is True
    assert calls['n'] == 3
    assert not os.path.exists(src)


def test_a_source_that_vanished_mid_flight_is_not_a_failure(paths, monkeypatch):
    """Something else cleaned it up between copy and unlink. The render is ours
    either way."""
    src, dst = paths
    monkeypatch.setattr(comfy_fs.os, 'replace',
                        lambda *a: (_ for _ in ()).throw(OSError(17, 'cross-device')))
    monkeypatch.setattr(comfy_fs.os, 'unlink',
                        lambda *a: (_ for _ in ()).throw(FileNotFoundError()))
    assert comfy_fs.collect_output(src, dst) is True


# --- the other direction: a real loss must still be loud ---------------------

def test_a_copy_that_fails_still_raises(paths, monkeypatch):
    """Tolerating the unlink must not tolerate losing the image. If the CONTENT
    did not arrive, the caller has to hear about it."""
    src, dst = paths
    monkeypatch.setattr(comfy_fs.os, 'replace',
                        lambda *a: (_ for _ in ()).throw(OSError(17, 'cross-device')))
    monkeypatch.setattr(comfy_fs.shutil, 'copy2',
                        lambda *a, **k: (_ for _ in ()).throw(OSError(28, 'no space')))
    with pytest.raises(OSError):
        comfy_fs.collect_output(src, dst)


def test_a_missing_source_raises(tmp_path):
    dst = tmp_path / 'out.png'
    with pytest.raises(OSError):
        comfy_fs.collect_output(str(tmp_path / 'nope.png'), str(dst))


def test_the_warning_is_paste_safe(paths, monkeypatch, caplog):
    """Every message in this module is written to be dropped into a public help
    thread — the module docstring says so, and this one names a path."""
    src, dst = paths
    monkeypatch.setattr(comfy_fs.os, 'replace',
                        lambda *a: (_ for _ in ()).throw(OSError(17, 'x')))
    monkeypatch.setattr(comfy_fs.os, 'unlink',
                        lambda *a: (_ for _ in ()).throw(PermissionError(32, 'in use')))
    monkeypatch.setattr(comfy_fs.time, 'sleep', lambda _s: None)
    comfy_fs.collect_output(src, dst)
    assert 'C:\\Users\\' not in caplog.text
    assert '/home/' not in caplog.text
