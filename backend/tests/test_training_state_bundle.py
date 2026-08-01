"""Unit contract for portable, exact training-state bundles.

The service treats torch state as opaque bytes.  These tests therefore need no
ML dependency and exercise the filesystem contract directly.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import pytest

from app.services import training_state_bundle as state


def _metadata(step=10, **changes):
    values = {
        'completed_step': step,
        'next_step': step + 1,
        'optimizer_updates_completed': step,
        'toolkit_revision': 'aitoolkit@abc123',
        'toolkit_runtime': {
            'python': '3.12.4',
            'torch': '2.7.1+cu128',
            'cuda': '12.8',
        },
        'config_hash': state.sha256_json({'steps': 1000, 'lr': 0.0001}),
        'dataset_hash': state.sha256_json({'images': ['a.webp', 'b.webp']}),
        'base_model_hash': state.sha256_json({'model': 'flux-dev'}),
        'network_hash': state.sha256_json({'type': 'lora', 'rank': 16}),
        'capabilities': (
            'weights', 'optimizer', 'scheduler', 'scaler', 'ema', 'rng',
            'dataloader', 'trainer',
        ),
        'state_level': 'exact',
    }
    values.update(changes)
    return state.BundleMetadata(**values)


def _sources(tmp_path):
    source_root = tmp_path / 'sources'
    source_root.mkdir()
    weights = source_root / 'weights.bin'
    optimizer = source_root / 'optimizer.pt'
    weights.write_bytes(b'weights\x00\x01' * 101)
    optimizer.write_bytes(b'opaque optimizer state' * 37)
    return {
        'raw_weights/adapter.safetensors': weights,
        'optimizer/optimizer.pt': optimizer,
    }


def _create(tmp_path, *, step=10, retention=2, metadata=None):
    save_root = tmp_path / 'save'
    inspection = state.create_bundle(
        save_root,
        _sources(tmp_path),
        metadata or _metadata(step),
        retention=retention,
    )
    return save_root, inspection


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob('*')
        if path.is_file()
    }


def _make_directory_link(link: Path, target: Path) -> None:
    if os.name == 'nt':
        completed = subprocess.run(
            ['cmd.exe', '/d', '/c', 'mklink', '/J', str(link), str(target)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            pytest.skip(f'junction creation unavailable: {completed.stderr}')
    else:
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f'directory symlink creation unavailable: {exc}')


def _remove_directory_link(link: Path) -> None:
    if not os.path.lexists(link):
        return
    if os.name == 'nt':
        os.rmdir(link)
    else:
        link.unlink()


def test_numbered_checkpoint_accepts_zero_based_optimizer_update_count(tmp_path):
    metadata = _metadata(step=10, optimizer_updates_completed=11)
    save_root = tmp_path / 'save'
    sources = _sources(tmp_path)

    created = state.create_bundle(save_root, sources, metadata)

    assert created.optimizer_updates_completed == 11
    with pytest.raises(ValueError, match=r'completed_step \+ 1'):
        state.create_bundle(
            save_root,
            sources,
            replace(metadata, optimizer_updates_completed=12),
        )


def _rewrite_manifest(bundle, mutate):
    manifest_path = bundle / state.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    mutate(manifest)
    raw = (
        json.dumps(
            manifest, ensure_ascii=False, allow_nan=False, sort_keys=True,
            separators=(',', ':'))
        + '\n'
    ).encode('utf-8')
    manifest_path.write_bytes(raw)
    marker = (
        f'{state.SCHEMA} sha256={hashlib.sha256(raw).hexdigest()}\n'
    ).encode('ascii')
    (bundle / state.COMPLETE_FILENAME).write_bytes(marker)


def test_create_writes_complete_canonical_manifest_and_hashes(tmp_path):
    save_root, inspection = _create(tmp_path)
    bundle = save_root / state.STATE_DIRECTORY / inspection.bundle_id

    assert inspection.restorable
    assert bundle.parent.name == '.lds-state'
    assert len(inspection.bundle_id) == 32
    assert (bundle / state.COMPLETE_FILENAME).is_file()
    manifest_bytes = (bundle / state.MANIFEST_FILENAME).read_bytes()
    manifest = json.loads(manifest_bytes)
    marker = (bundle / state.COMPLETE_FILENAME).read_text(encoding='ascii')

    assert manifest_bytes.endswith(b'\n')
    assert marker == (
        f'{state.SCHEMA} sha256='
        f'{hashlib.sha256(manifest_bytes).hexdigest()}\n'
    )
    assert set(manifest) == {
        'schema', 'bundle_id', 'created_at', 'created_at_ns',
        'completed_step', 'next_step', 'optimizer_updates_completed',
        'toolkit_revision', 'toolkit_runtime',
        'config_hash', 'dataset_hash', 'base_model_hash', 'network_hash',
        'capabilities', 'state_level', 'artifacts',
    }
    assert manifest['schema'] == state.SCHEMA
    assert manifest['completed_step'] == 10
    assert manifest['next_step'] == 11
    assert manifest['optimizer_updates_completed'] == 10
    assert manifest['state_level'] == 'exact'
    assert manifest['capabilities'] == sorted(manifest['capabilities'])

    for artifact in manifest['artifacts']:
        path = bundle.joinpath(*Path(artifact['path']).parts)
        assert path.stat().st_size == artifact['size_bytes']
        assert state.sha256_file(path) == artifact['sha256']

    ui = inspection.to_ui_dict()
    assert set(ui) == {
        'bundle_id', 'status', 'integrity', 'state_level', 'reason',
        'size_bytes', 'capabilities',
    }
    assert ui['status'] == 'complete'
    assert ui['integrity'] == 'verified'
    assert str(save_root) not in repr(ui)


def test_publish_is_one_directory_rename_after_complete_and_manifest(
        tmp_path, monkeypatch):
    save_root = tmp_path / 'save'
    writes = []
    observed = {}
    original_write = state._write_bytes_fsynced
    original_replace = state.os.replace

    def recording_write(path, data):
        writes.append(path.name)
        return original_write(path, data)

    def recording_replace(source, destination):
        source = Path(source)
        destination = Path(destination)
        observed['source_name'] = source.name
        observed['destination_existed'] = destination.exists()
        observed['files'] = {
            path.relative_to(source).as_posix()
            for path in source.rglob('*') if path.is_file()
        }
        return original_replace(source, destination)

    monkeypatch.setattr(state, '_write_bytes_fsynced', recording_write)
    monkeypatch.setattr(state.os, 'replace', recording_replace)

    result = state.create_bundle(save_root, _sources(tmp_path), _metadata())

    assert writes[-2:] == [state.COMPLETE_FILENAME, state.MANIFEST_FILENAME]
    assert observed['source_name'].startswith('.partial-')
    assert observed['destination_existed'] is False
    assert {state.COMPLETE_FILENAME, state.MANIFEST_FILENAME}.issubset(
        observed['files'])
    assert all(
        path.name == result.bundle_id
        for path in (save_root / state.STATE_DIRECTORY).iterdir())


def test_failed_publish_leaves_no_bundle_or_partial(tmp_path, monkeypatch):
    save_root = tmp_path / 'save'

    def fail_publish(_source, _destination):
        raise OSError('injected rename failure')

    monkeypatch.setattr(state.os, 'replace', fail_publish)
    with pytest.raises(OSError, match='injected'):
        state.create_bundle(save_root, _sources(tmp_path), _metadata())

    assert list((save_root / state.STATE_DIRECTORY).iterdir()) == []


def test_low_disk_rejects_before_partial_and_preserves_existing_bundle(
        tmp_path, monkeypatch):
    save_root, existing = _create(tmp_path)
    bundle = state.resolve_bundle_path(save_root, existing.bundle_id)
    before = _tree_bytes(bundle)
    source = tmp_path / 'next-state.bin'
    source.write_bytes(b'next state' * 100)
    monkeypatch.setattr(
        state.shutil,
        'disk_usage',
        lambda _path: SimpleNamespace(total=100, used=99, free=1),
    )

    with pytest.raises(
            state.InsufficientSpaceError, match='insufficient_free_space'):
        state.create_bundle(
            save_root,
            {'trainer/next-state.bin': source},
            _metadata(step=20),
            reserve_bytes=1,
        )

    assert not list((save_root / state.STATE_DIRECTORY).glob('.partial-*'))
    assert _tree_bytes(bundle) == before
    assert state.verify_bundle(save_root, existing.bundle_id).restorable


def test_restore_low_disk_rejects_before_copy_or_partial(tmp_path, monkeypatch):
    save_root, existing = _create(tmp_path)
    reserve = 4096
    free = existing.size_bytes + reserve - 1
    destination_volume = tmp_path / 'destination-volume'
    destination_volume.mkdir()
    restore_root = destination_volume / 'restore-low-disk'
    copy_calls = 0
    disk_usage_paths = []
    original_copy = state._copy_regular_file

    def recording_copy(source, destination):
        nonlocal copy_calls
        copy_calls += 1
        return original_copy(source, destination)

    monkeypatch.setattr(state, '_copy_regular_file', recording_copy)
    def low_disk(path):
        disk_usage_paths.append(Path(path))
        return SimpleNamespace(total=free, used=0, free=free)

    monkeypatch.setattr(state.shutil, 'disk_usage', low_disk)

    with pytest.raises(
            state.InsufficientSpaceError, match='insufficient_restore_space'):
        state.stage_restore(
            save_root,
            existing.bundle_id,
            restore_root,
            reserve_bytes=reserve,
        )

    assert copy_calls == 0
    assert disk_usage_paths == [destination_volume]
    assert not restore_root.exists()
    assert not list(destination_volume.glob('.restore-low-disk.partial-*'))


def test_oversize_rejects_before_partial_and_preserves_existing_bundle(tmp_path):
    save_root, existing = _create(tmp_path)
    bundle = state.resolve_bundle_path(save_root, existing.bundle_id)
    before = _tree_bytes(bundle)
    source = tmp_path / 'oversize.bin'
    source.write_bytes(b'X' * ((1 << 20) + 1))

    with pytest.raises(
            state.InsufficientSpaceError, match='bundle_size_limit_exceeded'):
        state.create_bundle(
            save_root,
            {'trainer/oversize.bin': source},
            _metadata(step=20),
            reserve_bytes=1,
            max_bundle_bytes=1 << 20,
        )

    assert not list((save_root / state.STATE_DIRECTORY).glob('.partial-*'))
    assert _tree_bytes(bundle) == before
    assert state.verify_bundle(save_root, existing.bundle_id).restorable


def test_store_quota_accounts_for_bundle_overhead_after_retention(tmp_path):
    save_root, existing = _create(tmp_path)
    projected = (
        existing.size_bytes
        + state._BUNDLE_OVERHEAD_BYTES
        + 1
        + state._BUNDLE_OVERHEAD_BYTES
    )

    with pytest.raises(
            state.InsufficientSpaceError, match='state_store_quota_exceeded'):
        state.preflight_bundle_space(
            save_root,
            1,
            retention=2,
            reserve_bytes=1,
            max_store_bytes=projected - 1,
        )


def test_stale_partial_scavenger_uses_exact_allowlist_and_commit_evidence(
        tmp_path):
    save_root = tmp_path / 'save'
    state_root = save_root / state.STATE_DIRECTORY
    state_root.mkdir(parents=True)
    now = time.time()

    def directory(name, *, old=True, evidence=None):
        path = state_root / name
        path.mkdir()
        (path / 'payload.bin').write_bytes(b'incomplete')
        if evidence is not None:
            (path / evidence).write_bytes(b'evidence')
        stamp = now - 100_000 if old else now
        os.utime(path, (stamp, stamp))
        return path

    removable = directory('.partial-' + 'a' * 32)
    recent = directory('.partial-' + 'b' * 32, old=False)
    has_complete = directory(
        '.partial-' + 'c' * 32, evidence=state.COMPLETE_FILENAME)
    has_manifest = directory(
        '.partial-' + 'd' * 32, evidence=state.MANIFEST_FILENAME)
    malformed = directory('.partial-not-a-uuid')
    published = directory('e' * 32)

    removed = state.scavenge_stale_partials(
        save_root,
        older_than_seconds=24 * 60 * 60,
        now=now,
    )

    assert removed == (removable.name,)
    assert not removable.exists()
    assert recent.exists()
    assert has_complete.exists()
    assert has_manifest.exists()
    assert malformed.exists()
    assert published.exists()


def test_state_root_link_is_rejected_before_external_store_iteration_or_cleanup(
        tmp_path):
    save_root = tmp_path / 'save'
    save_root.mkdir()
    outside = tmp_path / 'outside-state'
    outside.mkdir()
    partial = outside / ('.partial-' + 'f' * 32)
    partial.mkdir()
    sentinel = partial / 'external.bin'
    sentinel.write_bytes(b'preserve')
    state_link = save_root / state.STATE_DIRECTORY
    _make_directory_link(state_link, outside)
    source = tmp_path / 'source.bin'
    source.write_bytes(b'state')
    try:
        with pytest.raises(
                state.InvalidBundleError, match='unsafe_state_root'):
            state.list_bundles(save_root)
        with pytest.raises(
                state.InvalidBundleError, match='unsafe_state_root'):
            state.scavenge_stale_partials(
                save_root, older_than_seconds=0, now=time.time() + 1)
        with pytest.raises(
                state.InvalidBundleError, match='unsafe_state_root'):
            state.create_bundle(
                save_root,
                {'trainer/state.bin': source},
                _metadata(),
                reserve_bytes=1,
            )
        assert sentinel.read_bytes() == b'preserve'
    finally:
        _remove_directory_link(state_link)


def test_state_root_rejects_linked_save_root_ancestor(tmp_path):
    outside_save = tmp_path / 'outside-save'
    outside_state = outside_save / state.STATE_DIRECTORY
    outside_state.mkdir(parents=True)
    sentinel = outside_state / 'external.bin'
    sentinel.write_bytes(b'preserve')
    linked_save = tmp_path / 'linked-save'
    _make_directory_link(linked_save, outside_save)
    try:
        with pytest.raises(
                state.InvalidBundleError, match='unsafe_state_root'):
            state.list_bundles(linked_save)
        assert sentinel.read_bytes() == b'preserve'
    finally:
        _remove_directory_link(linked_save)


def test_default_retention_keeps_the_two_newest_complete_bundles(tmp_path):
    save_root = tmp_path / 'save'
    created = []
    for step in (10, 20, 30):
        source_root = tmp_path / f'sources-{step}'
        source_root.mkdir()
        source = source_root / 'state.bin'
        source.write_bytes(str(step).encode() * 50)
        created.append(state.create_bundle(
            save_root, {'trainer/state.bin': source}, _metadata(step)))

    present = {item.bundle_id for item in state.list_bundles(
        save_root, verify=True)}
    assert present == {created[1].bundle_id, created[2].bundle_id}
    assert not (
        save_root / state.STATE_DIRECTORY / created[0].bundle_id).exists()


def test_content_corruption_is_detected_and_restore_refuses_it(tmp_path):
    save_root, created = _create(tmp_path)
    bundle = state.resolve_bundle_path(
        save_root, created.bundle_id, require_exists=True)
    artifact = bundle / created.artifacts[0].path
    original = artifact.read_bytes()
    artifact.write_bytes(b'X' * len(original))

    inspected = state.inspect_bundle(save_root, created.bundle_id)
    assert inspected.status == 'invalid'
    assert inspected.integrity == 'failed'
    assert inspected.reason == 'artifact_sha256_mismatch'
    # The corrupt state is still attributed to its checkpoint, so the UI can
    # explain the fallback instead of calling a modern save "legacy".
    assert state.checkpoint_resume_state(
        save_root, 10).bundle_id == created.bundle_id
    with pytest.raises(state.InvalidBundleError,
                       match='artifact_sha256_mismatch'):
        state.verify_bundle(save_root, created.bundle_id)
    restore_root = tmp_path / 'restore'
    with pytest.raises(state.InvalidBundleError):
        state.stage_restore(save_root, created.bundle_id, restore_root)
    assert not restore_root.exists()


@pytest.mark.parametrize(
    ('damage', 'reason'),
    [
        ('marker_missing', 'complete_marker_missing'),
        ('manifest_changed', 'manifest_sha256_mismatch'),
        ('extra_artifact', 'artifact_set_mismatch'),
    ],
)
def test_incomplete_or_untracked_bundle_is_rejected(tmp_path, damage, reason):
    save_root, created = _create(tmp_path)
    bundle = state.resolve_bundle_path(save_root, created.bundle_id)
    if damage == 'marker_missing':
        (bundle / state.COMPLETE_FILENAME).unlink()
    elif damage == 'manifest_changed':
        with open(bundle / state.MANIFEST_FILENAME, 'ab') as handle:
            handle.write(b' ')
    else:
        (bundle / 'artifacts' / 'surprise.bin').write_bytes(b'untracked')

    inspected = state.inspect_bundle(save_root, created.bundle_id)
    assert inspected.status == 'invalid'
    assert inspected.reason == reason


def test_matching_compatibility_passes_and_each_identity_mismatch_refuses(
        tmp_path):
    metadata = _metadata()
    save_root, created = _create(tmp_path, metadata=metadata)
    expected = state.CompatibilitySpec.from_metadata(
        metadata,
        required_capabilities=('weights', 'rng'),
        minimum_state_level='exact',
    )

    assert state.verify_bundle(
        save_root, created.bundle_id, expected=expected).restorable
    for field_name in (
            'config_hash', 'dataset_hash', 'base_model_hash', 'network_hash',
            'toolkit_revision'):
        incompatible = replace(expected, **{field_name: 'different'})
        inspected = state.inspect_bundle(
            save_root, created.bundle_id, expected=incompatible)
        assert inspected.status == 'incompatible'
        assert inspected.integrity == 'verified'
        assert inspected.reason == f'{field_name}_mismatch'
        with pytest.raises(state.IncompatibleBundleError):
            state.verify_bundle(
                save_root, created.bundle_id, expected=incompatible)

    assert state.inspect_bundle(
        save_root,
        created.bundle_id,
        expected=replace(
            expected, toolkit_runtime={'torch': '0.0'}),
    ).reason == 'toolkit_runtime_mismatch'
    assert state.inspect_bundle(
        save_root,
        created.bundle_id,
        expected=replace(
            expected, required_capabilities=frozenset({'missing'})),
    ).reason == 'capability_missing'


def test_state_level_compatibility_is_ordered(tmp_path):
    metadata = _metadata(
        capabilities=('weights',),
        state_level='weights',
    )
    save_root, created = _create(tmp_path, metadata=metadata)

    inspected = state.inspect_bundle(
        save_root,
        created.bundle_id,
        expected=state.CompatibilitySpec(minimum_state_level='exact'),
    )
    assert inspected.status == 'incompatible'
    assert inspected.reason == 'state_level_insufficient'


def test_final_save_can_resume_at_its_target_step(tmp_path):
    metadata = _metadata(step=100)
    metadata = replace(metadata, next_step=metadata.completed_step)

    save_root, created = _create(tmp_path, metadata=metadata)

    assert created.completed_step == 100
    assert created.next_step == 100
    assert state.verify_bundle(save_root, created.bundle_id).restorable


def test_capabilities_must_be_a_sequence_of_names(tmp_path):
    source = tmp_path / 'state.bin'
    source.write_bytes(b'state')
    with pytest.raises(ValueError, match='invalid capabilities'):
        state.create_bundle(
            tmp_path / 'save',
            {'state.bin': source},
            _metadata(capabilities='weights'),
        )


@pytest.mark.parametrize(
    'bundle_id',
    [
        '../outside',
        '..\\outside',
        'step-10',
        '/absolute',
        'a' * 31,
        'A' * 32,
    ],
)
def test_opaque_bundle_id_rejects_traversal_and_semantic_names(
        tmp_path, bundle_id):
    with pytest.raises(state.InvalidBundleError, match='invalid_bundle_id'):
        state.inspect_bundle(tmp_path, bundle_id)


@pytest.mark.parametrize(
    'artifact_name',
    [
        '../outside.bin',
        'dir/../../outside.bin',
        r'dir\outside.bin',
        '/absolute.bin',
        'C:/drive.bin',
        './state.bin',
    ],
)
def test_create_rejects_unsafe_artifact_names(tmp_path, artifact_name):
    source = tmp_path / 'source.bin'
    source.write_bytes(b'state')
    save_root = tmp_path / 'save'

    with pytest.raises(ValueError, match='unsafe artifact name'):
        state.create_bundle(
            save_root, {artifact_name: source}, _metadata())
    assert not (tmp_path / 'outside.bin').exists()
    assert not (save_root / state.STATE_DIRECTORY).exists()


def test_committed_manifest_cannot_redirect_an_artifact_outside_bundle(tmp_path):
    save_root, created = _create(tmp_path)
    bundle = state.resolve_bundle_path(save_root, created.bundle_id)
    outside = save_root / 'outside.bin'
    outside.write_bytes(b'do not read me')

    def redirect(manifest):
        manifest['artifacts'][0]['path'] = '../outside.bin'

    _rewrite_manifest(bundle, redirect)
    inspected = state.inspect_bundle(save_root, created.bundle_id)

    assert inspected.status == 'invalid'
    assert inspected.reason == 'unsafe_artifact_path'
    assert outside.read_bytes() == b'do not read me'


def test_restore_materializes_only_artifacts_into_a_fresh_root(tmp_path):
    metadata = _metadata()
    save_root, created = _create(tmp_path, metadata=metadata)
    restore_root = tmp_path / 'fresh-save-root'
    expected = state.CompatibilitySpec.from_metadata(metadata)

    restored = state.stage_restore(
        save_root, created.bundle_id, restore_root, expected=expected)

    assert restored.restore_root == restore_root
    assert set(restored.artifacts) == {
        'raw_weights/adapter.safetensors',
        'optimizer/optimizer.pt',
    }
    assert (restore_root / 'raw_weights' / 'adapter.safetensors').read_bytes()
    assert (restore_root / 'optimizer' / 'optimizer.pt').read_bytes()
    assert not (restore_root / state.MANIFEST_FILENAME).exists()
    assert not list(tmp_path.glob('.fresh-save-root.partial-*'))


def test_restore_rehash_rejects_source_mutation_after_initial_verification(
        tmp_path, monkeypatch):
    save_root, created = _create(tmp_path)
    bundle = state.resolve_bundle_path(save_root, created.bundle_id)
    restore_root = tmp_path / 'restore-race'
    original_copy = state._copy_regular_file
    changed = False

    def mutate_then_copy(source, destination):
        nonlocal changed
        source = Path(source)
        if not changed and bundle in source.parents:
            changed = True
            source.write_bytes(b'X' * source.stat().st_size)
        return original_copy(source, destination)

    monkeypatch.setattr(state, '_copy_regular_file', mutate_then_copy)

    with pytest.raises(
            state.InvalidBundleError, match='artifact_sha256_mismatch'):
        state.stage_restore(save_root, created.bundle_id, restore_root)

    assert changed is True
    assert not restore_root.exists()
    assert not list(tmp_path.glob('.restore-race.partial-*'))


@pytest.mark.parametrize('nonempty', [False, True])
def test_restore_refuses_any_existing_root(tmp_path, nonempty):
    save_root, created = _create(tmp_path)
    restore_root = tmp_path / 'existing'
    restore_root.mkdir()
    if nonempty:
        (restore_root / 'owned.txt').write_text('keep', encoding='utf-8')

    with pytest.raises(state.RestoreTargetError,
                       match='restore_root_exists'):
        state.stage_restore(save_root, created.bundle_id, restore_root)
    assert not nonempty or (
        restore_root / 'owned.txt').read_text(encoding='utf-8') == 'keep'


def test_restore_refuses_a_target_inside_the_bundle_store(tmp_path):
    save_root, created = _create(tmp_path)
    target = save_root / state.STATE_DIRECTORY / 'restore-target'

    with pytest.raises(state.RestoreTargetError,
                       match='restore_root_unsafe'):
        state.stage_restore(save_root, created.bundle_id, target)
    assert not target.exists()


def test_checkpoint_lookup_maps_completed_step_to_newest_bundle(tmp_path):
    save_root = tmp_path / 'save'
    items = []
    for index, step in enumerate((10, 20, 20)):
        source = tmp_path / f'state-{index}.bin'
        source.write_bytes(bytes([index]) * 20)
        items.append(state.create_bundle(
            save_root,
            {'trainer/state.bin': source},
            _metadata(step),
            retention=3,
        ))

    assert state.checkpoint_resume_state(
        save_root, 20).bundle_id == items[-1].bundle_id
    assert state.checkpoint_resume_state(save_root, 999) is None
    with pytest.raises(ValueError):
        state.checkpoint_resume_state(save_root, -1)
