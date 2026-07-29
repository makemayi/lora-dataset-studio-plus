"""The path comparison behind "a dataset and a bank never share files".

Pure path logic, no app: one canonical spelling per folder, and a containment
test that only ever fires on a real separator boundary. Its JS twin lives in
`frontend/src/utils/pathRelation.js` and is held to the same table.
"""
import os

import pytest

from app.services import path_guard

CASE_INSENSITIVE = os.path.normcase('A') == 'a'


def test_the_same_folder_spelled_differently_is_the_same_folder(tmp_path):
    d = tmp_path / 'pics'
    d.mkdir()
    assert path_guard.relation(str(d), str(d) + os.sep) == 'same'
    assert path_guard.relation(str(d), f'"{d}"') == 'same'          # Copy-as-path
    assert path_guard.relation(str(d), str(d / 'sub' / '..')) == 'same'


def test_containment_only_on_a_separator_boundary(tmp_path):
    """`.../data2` must never read as being inside `.../data` — a bare
    startswith would say it does, and would refuse innocent folders forever."""
    a = tmp_path / 'data'
    b = tmp_path / 'data2'
    a.mkdir(); b.mkdir()
    assert path_guard.relation(str(b), str(a)) is None
    assert path_guard.relation(str(a / 'x'), str(a)) == 'inside'
    assert path_guard.relation(str(a), str(a / 'x')) == 'contains'


@pytest.mark.skipif(not CASE_INSENSITIVE, reason='POSIX paths are case-sensitive')
def test_case_folds_where_the_filesystem_folds_it(tmp_path):
    d = tmp_path / 'Pics'
    d.mkdir()
    assert path_guard.relation(str(d).upper(), str(d)) == 'same'


@pytest.mark.skipif(CASE_INSENSITIVE, reason='Windows folds case')
def test_case_is_kept_where_the_filesystem_keeps_it(tmp_path):
    """On POSIX two spellings that differ in case really are two folders, and
    saying otherwise would refuse a legitimate one."""
    d = tmp_path / 'Pics'
    d.mkdir()
    (tmp_path / 'PICS').mkdir()
    assert path_guard.relation(str(tmp_path / 'PICS'), str(d)) is None


def test_empty_and_junk_never_claim_a_relation():
    assert path_guard.relation('', '/anything') is None
    assert path_guard.relation(None, None) is None
    assert path_guard.norm('   ') is None


def test_conflict_names_the_dataset_and_the_way_out(tmp_path):
    root = tmp_path / 'datasets'
    (root / '7').mkdir(parents=True)
    c = path_guard.dataset_folder_conflict(str(root / '7'), datasets_root=str(root))
    assert c['scope'] == 'dataset' and c['dataset_id'] == 7
    assert 'Import to bank' in c['message']
    # the root itself, and anything above it, are refused as a whole
    assert path_guard.dataset_folder_conflict(str(root),
                                              datasets_root=str(root))['scope'] == 'root'
    assert path_guard.dataset_folder_conflict(str(tmp_path),
                                              datasets_root=str(root))['scope'] == 'root'
    # and a folder next door is not
    (tmp_path / 'elsewhere').mkdir()
    assert path_guard.dataset_folder_conflict(str(tmp_path / 'elsewhere'),
                                              datasets_root=str(root)) is None
