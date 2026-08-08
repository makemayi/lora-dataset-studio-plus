import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _bank(app, folder, *, user_id='local'):
    from app.extensions import db
    from app.models import ImageBank

    with app.app_context():
        bank = ImageBank(user_id=user_id, name='Source folder', source_path=str(folder))
        db.session.add(bank)
        db.session.commit()
        return bank.id


def _guard_open_source_dependencies(monkeypatch, banks):
    stored_path = r'C:\forbidden-bank-source'
    lookup = Mock(return_value=SimpleNamespace(source_path=stored_path))
    filesystem = SimpleNamespace(
        isabs=Mock(return_value=True),
        realpath=Mock(return_value=stored_path),
        isdir=Mock(return_value=True),
    )
    opener = Mock()
    monkeypatch.setattr(banks, 'get_bank', lookup)
    monkeypatch.setattr(banks, 'os', SimpleNamespace(path=filesystem))
    monkeypatch.setattr(banks, '_open_host_folder', opener)
    return lookup, filesystem, opener


def _assert_open_source_dependencies_unused(lookup, filesystem, opener):
    lookup.assert_not_called()
    filesystem.isabs.assert_not_called()
    filesystem.realpath.assert_not_called()
    filesystem.isdir.assert_not_called()
    opener.assert_not_called()


def test_open_source_route_uses_only_the_owned_bank_path(
        app, client, tmp_path, monkeypatch):
    from app.services import bank_jobs, image_bank_service as banks

    source = tmp_path / 'owned-source'
    source.mkdir()
    attacker_path = tmp_path / 'client-controlled'
    attacker_path.mkdir()
    bank_id = _bank(app, source)
    opened = []
    monkeypatch.setattr(banks, '_open_host_folder', opened.append)
    # Opening the folder is read-only and remains available during a Bank pass.
    monkeypatch.setattr(bank_jobs, 'running', lambda *_args: True)

    response = client.post(
        f'/api/bank/{bank_id}/open-source-folder',
        json={'path': str(attacker_path), 'source_path': str(attacker_path)},
    )

    assert response.status_code == 200
    assert response.get_json() == {'ok': True}
    assert opened == [str(source)]


@pytest.mark.parametrize('bank_id_factory', ('missing', 'other-user'))
def test_open_source_route_does_not_reveal_unowned_banks(
        app, client, tmp_path, monkeypatch, bank_id_factory):
    from app.services import image_bank_service as banks

    source = tmp_path / 'private-source'
    source.mkdir()
    bank_id = (987654 if bank_id_factory == 'missing'
               else _bank(app, source, user_id='somebody-else'))
    monkeypatch.setattr(
        banks, '_open_host_folder',
        lambda _path: pytest.fail('an unowned Bank folder must not be opened'))

    response = client.post(f'/api/bank/{bank_id}/open-source-folder', json={})

    assert response.status_code == 404
    assert response.get_json()['error'] == 'not found'


@pytest.mark.parametrize('bank_id', (0, 2**63))
def test_open_source_route_rejects_out_of_sqlite_range_without_side_effects(
        client, monkeypatch, bank_id):
    from app.services import image_bank_service as banks

    guarded = _guard_open_source_dependencies(monkeypatch, banks)

    response = client.post(f'/api/bank/{bank_id}/open-source-folder')

    assert response.status_code == 404
    assert response.get_json() == {'error': 'not found'}
    _assert_open_source_dependencies_unused(*guarded)


@pytest.mark.parametrize('source_kind', ('missing', 'file'))
def test_open_source_route_refuses_unavailable_sources_without_creating_them(
        app, client, tmp_path, monkeypatch, source_kind):
    from app.services import image_bank_service as banks

    source = tmp_path / source_kind
    if source_kind == 'file':
        source.write_text('not a directory', encoding='utf-8')
    bank_id = _bank(app, source)
    monkeypatch.setattr(
        banks, '_open_host_folder',
        lambda _path: pytest.fail('an unavailable source must not reach the opener'))

    response = client.post(f'/api/bank/{bank_id}/open-source-folder', json={})

    assert response.status_code == 409
    assert 'unavailable' in response.get_json()['error']
    if source_kind == 'missing':
        assert not source.exists()
    else:
        assert source.is_file()


def test_open_source_route_maps_launcher_failure_to_json_500(
        app, client, tmp_path, monkeypatch):
    from app.services import image_bank_service as banks

    source = tmp_path / 'source'
    source.mkdir()
    bank_id = _bank(app, source)

    def fail_cleanly(_path):
        raise OSError('no desktop session')

    monkeypatch.setattr(banks, '_open_host_folder', fail_cleanly)
    response = client.post(f'/api/bank/{bank_id}/open-source-folder', json={})

    assert response.status_code == 500
    assert response.is_json
    assert response.get_json()['error'] == 'could not open the bank source folder'


@pytest.mark.parametrize(
    ('os_name', 'platform', 'expected'),
    [
        ('nt', 'win32', 'startfile'),
        ('posix', 'darwin', '/usr/bin/open'),
        ('posix', 'linux', 'xdg-open'),
    ],
)
def test_host_folder_launcher_never_uses_a_shell(
        tmp_path, monkeypatch, os_name, platform, expected):
    from app.services import image_bank_service as banks

    path = str(tmp_path)
    calls = []
    monkeypatch.setattr(banks.os, 'name', os_name)
    monkeypatch.setattr(banks.sys, 'platform', platform)
    monkeypatch.setattr(
        banks.os, 'startfile',
        lambda value, verb: calls.append(('startfile', value, verb)),
        raising=False)
    monkeypatch.setattr(
        banks.subprocess, 'Popen',
        lambda argv, **kwargs: calls.append(('popen', argv, kwargs)))

    banks._open_host_folder(path)

    if expected == 'startfile':
        assert calls == [('startfile', path, 'explore')]
    elif expected == '/usr/bin/open':
        assert calls == [
            ('popen', ['/usr/bin/open', '-R', path], {'shell': False})]
    else:
        assert calls == [('popen', [expected, path], {'shell': False})]


def test_macos_app_bundle_is_revealed_in_finder_not_launched(monkeypatch):
    from app.services import image_bank_service as banks

    app_bundle = '/Applications/Example.app'
    calls = []
    monkeypatch.setattr(banks.os, 'name', 'posix')
    monkeypatch.setattr(banks.sys, 'platform', 'darwin')
    monkeypatch.setattr(
        banks.subprocess, 'Popen',
        lambda argv, **kwargs: calls.append((argv, kwargs)))

    banks._open_host_folder(app_bundle)

    assert calls == [
        (['/usr/bin/open', '-R', app_bundle], {'shell': False})]


def test_service_returns_none_before_touching_an_unowned_path(monkeypatch):
    from app.services import image_bank_service as banks

    monkeypatch.setattr(banks, 'get_bank', lambda *_args: None)
    monkeypatch.setattr(
        banks.os.path, 'isdir',
        lambda _path: pytest.fail('ownership must be checked before the filesystem'))

    assert banks.open_bank_source_folder('local', 123) is None


@pytest.mark.parametrize('bank_id', (0, 2**63))
def test_service_rejects_out_of_sqlite_range_without_side_effects(
        monkeypatch, bank_id):
    from app.services import image_bank_service as banks

    guarded = _guard_open_source_dependencies(monkeypatch, banks)

    assert banks.open_bank_source_folder('local', bank_id) is None
    _assert_open_source_dependencies_unused(*guarded)


def test_service_validates_the_stored_path_not_a_caller_value(tmp_path, monkeypatch):
    from app.services import image_bank_service as banks

    source = tmp_path / 'stored'
    source.mkdir()
    stored_path = os.path.join(str(source), '..', source.name)
    canonical_path = os.path.realpath(stored_path, strict=True)
    opened = []
    monkeypatch.setattr(
        banks, 'get_bank',
        lambda user_id, bank_id: SimpleNamespace(
            id=bank_id, user_id=user_id, source_path=stored_path))
    monkeypatch.setattr(banks, '_open_host_folder', opened.append)

    assert banks.open_bank_source_folder('local', 7) == canonical_path
    assert opened == [canonical_path]


def test_service_rejects_a_relative_stored_path_without_resolving_it(monkeypatch):
    from app.services import image_bank_service as banks

    monkeypatch.setattr(
        banks, 'get_bank',
        lambda *_args: SimpleNamespace(id=7, source_path='relative/bank'))
    monkeypatch.setattr(
        banks.os.path, 'realpath',
        lambda *_args, **_kwargs: pytest.fail('relative paths must not be resolved'))

    with pytest.raises(banks.BankSourceFolderUnavailable):
        banks.open_bank_source_folder('local', 7)


def test_service_allows_a_reachable_unc_bank_source(monkeypatch):
    from app.services import image_bank_service as banks

    unc = r'\\photoserver\library\bank'
    opened = []
    monkeypatch.setattr(
        banks, 'get_bank', lambda *_args: SimpleNamespace(id=7, source_path=unc))
    monkeypatch.setattr(banks.os.path, 'isabs', lambda value: value == unc)
    monkeypatch.setattr(
        banks.os.path, 'realpath',
        lambda value, *, strict: value if strict else pytest.fail('strict required'))
    monkeypatch.setattr(banks.os.path, 'isdir', lambda value: value == unc)
    monkeypatch.setattr(banks, '_open_host_folder', opened.append)

    assert banks.open_bank_source_folder('local', 7) == unc
    assert opened == [unc]
