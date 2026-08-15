"""The H3 single-frame patch has to be safe to run twice and honest when the
file it targets has moved on.

These tests never touch a real ComfyUI: they build a miniature of the two dozen
lines the patch anchors on, which is enough because every edit is anchored on
code text, not on line numbers.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import patch_comfyui_h3_single_frame as patcher  # noqa: E402


STOCK = '''\
FPS = 24


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * 8)


class A:
    inputs = [
        io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="a"),
    ]


class B:
    inputs = [
        io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="b"),
    ]


class C:
    inputs = [
        io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="c"),
    ]
'''


def test_stock_is_reported_as_unpatched():
    applied, missing = patcher.patch_state(STOCK)
    assert applied == []
    assert len(missing) == 4


def test_every_edit_lands_and_the_result_is_patched():
    patched, problems = patcher.apply_edits(STOCK)
    assert problems == []
    assert patcher.patch_state(patched)[1] == []
    # All THREE length inputs, not just the first one: the swap graph and the
    # generation graph use different nodes, and a partial patch would fail one
    # lane at queue time while the other worked.
    assert patched.count('min=1, max=3600') == 3


def test_the_patched_file_still_compiles():
    patched, _ = patcher.apply_edits(STOCK)
    ok, why = patcher._compiles(patched, 'fixture')
    assert ok, why


def test_running_twice_does_not_insert_twice():
    once, _ = patcher.apply_edits(STOCK)
    twice, problems = patcher.apply_edits(once)
    # The second pass finds nothing to do, which the script reads as "already
    # patched" BEFORE it ever calls apply_edits — the point here is that the
    # text is not corrupted even if it did.
    assert problems, 'a second pass must report zero matches, not silently edit'
    assert twice == once


def test_a_file_it_does_not_recognise_is_refused_rather_than_mangled():
    moved_on = STOCK.replace('while n % 17 != 5:', 'while n % 19 != 5:')
    _, problems = patcher.apply_edits(moved_on)
    assert any('align_frame_count' in p for p in problems)


def test_check_mode_reports_two_on_an_unpatched_tree(tmp_path, capsys):
    root = tmp_path / 'ComfyUI'
    (root / 'comfy_extras').mkdir(parents=True)
    target = root / 'comfy_extras' / 'nodes_minimax_h3.py'
    target.write_text(STOCK, encoding='utf-8')
    assert patcher.main(['--comfy-dir', str(root), '--check']) == 2
    assert 'NOT patched' in capsys.readouterr().out
    assert target.read_text(encoding='utf-8') == STOCK, '--check must not write'


def test_apply_then_second_run_is_a_no_op(tmp_path, capsys):
    root = tmp_path / 'ComfyUI'
    (root / 'comfy_extras').mkdir(parents=True)
    target = root / 'comfy_extras' / 'nodes_minimax_h3.py'
    target.write_text(STOCK, encoding='utf-8')

    assert patcher.main(['--comfy-dir', str(root)]) == 0
    first = target.read_text(encoding='utf-8')
    assert patcher.main(['--comfy-dir', str(root)]) == 0
    assert 'already patched' in capsys.readouterr().out
    assert target.read_text(encoding='utf-8') == first


def test_revert_restores_the_original_bytes(tmp_path):
    root = tmp_path / 'ComfyUI'
    (root / 'comfy_extras').mkdir(parents=True)
    target = root / 'comfy_extras' / 'nodes_minimax_h3.py'
    target.write_text(STOCK, encoding='utf-8')

    assert patcher.main(['--comfy-dir', str(root)]) == 0
    assert target.read_text(encoding='utf-8') != STOCK
    assert patcher.main(['--comfy-dir', str(root), '--revert']) == 0
    assert target.read_text(encoding='utf-8') == STOCK


def test_a_portable_root_one_level_up_is_found(tmp_path):
    portable = tmp_path / 'ComfyUI_portable'
    inner = portable / 'ComfyUI' / 'comfy_extras'
    inner.mkdir(parents=True)
    (inner / 'nodes_minimax_h3.py').write_text(STOCK, encoding='utf-8')
    assert patcher.find_comfyui_dir(str(portable)) == str(portable / 'ComfyUI')


def test_a_directory_without_the_node_file_is_not_comfyui(tmp_path):
    assert patcher.find_comfyui_dir(str(tmp_path)) is None


@pytest.mark.parametrize('missing_dir', ['', '   '])
def test_a_blank_hint_is_ignored_rather_than_resolved_to_cwd(missing_dir):
    assert patcher.find_comfyui_dir(missing_dir) in (None, patcher.find_comfyui_dir(None))
