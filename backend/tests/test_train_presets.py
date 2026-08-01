"""Training presets: named snapshots of the advanced settings, import/export
friendly. The core promise is SCHEMA TOLERANCE — a preset from another app
version applies with unknown keys ignored and invalid values reported, never
a hard failure.
"""


def _create_ds(client, name='Preset', trigger='pres', train_type='krea',
               kind=None):
    payload = {'name': name, 'trigger_word': trigger,
               'train_type': train_type, 'kind': kind}
    if kind == 'concept':
        # A concept dataset requires the description the captioner must omit.
        payload['concept_desc'] = 'a red vintage telephone'
    return client.post('/api/dataset/create', json=payload).get_json()['id']


def test_save_from_dataset_snapshot_and_list(client, app):
    ds_id = _create_ds(client)
    # give the dataset one explicit setting to snapshot
    with app.app_context():
        from app.services import lora_training as lt
        lt.update_train_settings('local', ds_id, {'rank': 32})
    r = client.post('/api/train/presets', json={'name': 'My Krea', 'dataset_id': ds_id})
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True and body['created'] is True
    assert body['train_type'] == 'krea'
    assert body['dataset_kind'] == 'character'
    assert body['variants'] == ['base']
    assert body['settings'] == {'rank': 32}
    listed = client.get('/api/train/presets').get_json()['presets']
    assert any(p['name'] == 'My Krea' and p['settings'] == {'rank': 32} for p in listed)


def test_save_overwrites_by_name(client):
    r1 = client.post('/api/train/presets',
                     json={'name': 'Dup', 'train_type': 'zimage', 'settings': {'rank': 16}})
    r2 = client.post('/api/train/presets',
                     json={'name': 'Dup', 'train_type': 'zimage', 'settings': {'rank': 64}})
    assert r1.get_json()['created'] is True
    assert r2.get_json()['created'] is False
    listed = client.get('/api/train/presets').get_json()['presets']
    assert [p['settings'] for p in listed if p['name'] == 'Dup'] == [{'rank': 64}]


def test_apply_is_schema_tolerant(client, app):
    """Unknown keys (future app versions) are ignored, invalid values rejected,
    valid keys land — all in one call, never fatal."""
    ds_id = _create_ds(client)
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={'settings': {
        'rank': 32,                       # valid → applied
        'dropout': 0.1,                   # valid → applied
        'rank_v2_search_space': [1, 2],   # unknown → ignored
        'save_every': 123,                # invalid value → rejected
    }})
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is True
    assert body['ignored'] == ['rank_v2_search_space']
    assert [x['key'] for x in body['rejected']] == ['save_every']
    with app.app_context():
        from app.services import lora_training as lt
        stored = lt.snapshot_train_settings('local', ds_id)
        assert stored == {'rank': 32, 'dropout': 0.1}


def test_apply_validates_then_commits_the_complete_replacement_once(app, monkeypatch):
    """A concurrent Train may see the old preset or the final one, never a prefix.

    The count is the regression guard: the old implementation committed once to
    clear and once per accepted key, exposing partial combinations between them.
    """
    from app.services import lora_training as lt
    from app.services import face_dataset_service as svc
    from app.config import LOCAL_USER
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Atomic', 'atomic', train_type='krea')
        lt.update_train_settings(LOCAL_USER, ds.id, {'rank': 64, 'dropout': 0.3})
        commits = []
        real_commit = svc.db.session.commit

        def commit_once():
            commits.append(True)
            return real_commit()

        monkeypatch.setattr(svc.db.session, 'commit', commit_once)
        _, ignored, rejected = lt.apply_train_settings_dict(LOCAL_USER, ds.id, {
            'rank': 32, 'dropout': 0.1, 'save_every': 123,
        })
        assert commits == [True]
        assert ignored == []
        assert [r['key'] for r in rejected] == ['save_every']
        assert lt.snapshot_train_settings(LOCAL_USER, ds.id) == {
            'rank': 32, 'dropout': 0.1,
        }


def test_apply_replaces_previous_settings(client, app):
    """A preset REPLACES the explicit settings — keys absent from the preset
    fall back to defaults instead of surviving from before."""
    ds_id = _create_ds(client)
    with app.app_context():
        from app.services import lora_training as lt
        lt.update_train_settings('local', ds_id, {'rank': 64, 'dropout': 0.3})
    client.post(f'/api/dataset/{ds_id}/train/presets/apply',
                json={'settings': {'rank': 16}})
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {'rank': 16}


def test_builtin_presets_listed_first_and_undeletable(client):
    listed = client.get('/api/train/presets').get_json()['presets']
    assert listed and listed[0]['id'] == 'builtin-krea-character'
    assert listed[0]['builtin'] is True
    assert listed[0]['dataset_kind'] == 'character'
    ids = {p['id'] for p in listed}
    assert 'builtin-style' not in ids
    assert {
        'builtin-style-krea-raw',
        'builtin-style-klein-base',
        'builtin-style-zimage-base',
        'builtin-style-flux1',
        'builtin-style-sdxl',
    } <= ids
    # the delete route only matches integer ids — built-ins are unreachable
    assert client.delete('/api/train/presets/builtin-krea-character').status_code == 404


def test_style_builtin_catalogue_has_researched_family_settings(client):
    listed = client.get('/api/train/presets').get_json()['presets']
    styles = {p['id']: p for p in listed if p.get('dataset_kind') == 'style'}
    assert styles['builtin-style-krea-raw']['variants'] == ['base', 'raw']
    assert styles['builtin-style-klein-base']['variants'] == ['4b', '9b']
    assert styles['builtin-style-zimage-base']['variants'] == []
    expected = {
        'builtin-style-krea-raw': (32, 32, '768,1024', 'linear'),
        'builtin-style-klein-base': (32, 32, '768,1024', 'weighted'),
        'builtin-style-zimage-base': (32, 32, '768,1024', 'weighted'),
        # Style wants full capacity/strength: FLUX corrected 16/16 → 32/32
        # (research groups FLUX with the 32/32 prose family for style) and
        # SDXL alpha corrected 16 → 32 (alpha = rank recommended for style).
        'builtin-style-flux1': (32, 32, '768,1024', 'weighted'),
        'builtin-style-sdxl': (32, 32, '1024', None),
    }
    for preset_id, (rank, alpha, resolution, timestep) in expected.items():
        settings = styles[preset_id]['settings']
        assert (settings['rank'], settings['alpha'], settings['resolution']) == (
            rank, alpha, resolution)
        assert settings.get('timestep_type') == timestep
        assert settings['save_every'] == settings['sample_every'] == 250
        assert settings['max_step_saves'] == 10
        assert len(settings['sample_prompts']) == 8
        assert all('{trigger}' not in prompt for prompt in settings['sample_prompts'])
        # ``dropout`` is network dropout, not caption dropout. Style caption
        # dropout is applied by the service's family policy.
        assert 'dropout' not in settings
        assert 'ema' not in settings


def test_every_builtin_applies_cleanly(client, app):
    """The shipped presets must ALWAYS apply with zero ignored keys and zero
    rejected values — this is the guard that catches a choice-list drifting
    away from what the built-ins promise."""
    from app.services.lora_training import BUILTIN_TRAIN_PRESETS
    ds_id = _create_ds(client)
    for preset in BUILTIN_TRAIN_PRESETS:
        r = client.post(f'/api/dataset/{ds_id}/train/presets/apply',
                        json={'settings': preset['settings']})
        body = r.get_json()
        assert body['ok'] is True, preset['id']
        assert body['ignored'] == [], preset['id']
        assert body['rejected'] == [], preset['id']


# --- Built-in quick presets: core coverage + source-labelled specialties -------
# The shipped catalogue promise: every supported model family exposes one
# general-purpose Character, Style and Concept quick preset where that kind is
# supported. A source-labelled specialty may share a scope without replacing its
# general-purpose recipe (Krea Raw LoKr likeness is the first such preset).

_QUICK_PRESET_MATRIX = {
    ('zimage', 'character'): 'builtin-character-zimage',
    ('sdxl', 'character'): 'builtin-character-sdxl',
    ('krea', 'character'): 'builtin-krea-character',
    ('flux', 'character'): 'builtin-character-flux1',
    ('flux2klein', 'character'): 'builtin-character-klein',
    ('zimage', 'style'): 'builtin-style-zimage-base',
    ('sdxl', 'style'): 'builtin-style-sdxl',
    ('krea', 'style'): 'builtin-style-krea-raw',
    ('flux', 'style'): 'builtin-style-flux1',
    ('flux2klein', 'style'): 'builtin-style-klein-base',
    ('zimage', 'concept'): 'builtin-concept',
    ('sdxl', 'concept'): 'builtin-concept-sdxl',
    ('krea', 'concept'): 'builtin-concept-krea',
    ('flux', 'concept'): 'builtin-concept-flux1',
    ('flux2klein', 'concept'): 'builtin-concept-klein',
    # Anima ships Character + Concept only (Style is out of scope for this family;
    # it trains and deploys, no per-family style recipe yet).
    ('anima', 'character'): 'builtin-character-anima',
    ('anima', 'concept'): 'builtin-concept-anima',
}

_ORIGINAL_BUILTIN_IDS = set(_QUICK_PRESET_MATRIX.values()) | {
    'builtin-krea-raw-lokr-likeness',
}

_APPROVED_BUILTIN_IDS = {
    'builtin-krea-raw-character-balanced',
    'builtin-krea-raw-character-lokr-fast',
    'builtin-krea-raw-style-compact',
    'builtin-krea-raw-concept-16gb',
    'builtin-zimage-turbo-character-balanced',
}


def test_quick_preset_catalogue_covers_every_family_and_kind(client):
    listed = client.get('/api/train/presets').get_json()['presets']
    builtins = [p for p in listed if p.get('builtin')]
    assert len(builtins) == 23
    assert {p['id'] for p in builtins} == (
        _ORIGINAL_BUILTIN_IDS | _APPROVED_BUILTIN_IDS)
    coverage = {}
    for preset in builtins:
        coverage.setdefault((preset['train_type'], preset['dataset_kind']), set()).add(preset['id'])
    assert set(coverage) == set(_QUICK_PRESET_MATRIX)
    for scope, preset_id in _QUICK_PRESET_MATRIX.items():
        assert preset_id in coverage[scope]
    assert coverage[('krea', 'character')] == {
        'builtin-krea-character',
        'builtin-krea-raw-lokr-likeness',
        'builtin-krea-raw-character-balanced',
        'builtin-krea-raw-character-lokr-fast',
    }
    assert coverage[('krea', 'style')] == {
        'builtin-style-krea-raw', 'builtin-krea-raw-style-compact'}
    assert coverage[('krea', 'concept')] == {
        'builtin-concept-krea', 'builtin-krea-raw-concept-16gb'}
    assert coverage[('zimage', 'character')] == {
        'builtin-character-zimage', 'builtin-zimage-turbo-character-balanced'}
    for p in builtins:
        # Why culture: every quick preset explains itself in one line.
        assert p.get('description'), p['id']
        assert len(p['settings']['sample_prompts']) == 8, p['id']
        if p['id'] in _ORIGINAL_BUILTIN_IDS:
            # The original 18 entries remain byte-for-byte explicit about their
            # researched capacity. Two new community recipes intentionally omit
            # unpublished/irrelevant rank and alpha rather than inventing them.
            assert p['settings'].get('rank'), p['id']
            assert p['settings'].get('alpha'), p['id']


def test_every_quick_preset_applies_by_id_with_announced_values(client, app):
    """Apply each general-purpose quick preset by id on its family and kind:
    the scope check passes, nothing is ignored/rejected, and the STORED raw
    settings reproduce the announced settings dict exactly."""
    listed = client.get('/api/train/presets').get_json()['presets']
    by_id = {p['id']: p for p in listed if p.get('builtin')}
    # Families whose built-ins restrict variants: request an allowed one
    # (zimage style is Base-only; Klein has only its two sizes).
    request_variant = {'zimage': 'base', 'krea': 'base', 'flux2klein': '4b'}
    for i, ((family, kind), preset_id) in enumerate(
            sorted(_QUICK_PRESET_MATRIX.items())):
        preset = by_id[preset_id]
        ds_id = _create_ds(client, name=f'Quick {i}', trigger=f'quick{i}',
                           train_type=family, kind=kind)
        payload = {'preset_id': preset_id, 'train_type': family}
        if family in request_variant:
            payload['variant'] = request_variant[family]
        r = client.post(f'/api/dataset/{ds_id}/train/presets/apply',
                        json=payload)
        assert r.status_code == 200, (preset_id, r.get_json())
        body = r.get_json()
        assert body['ok'] is True and body['preset_id'] == preset_id
        assert body['ignored'] == [] and body['rejected'] == [], preset_id
        with app.app_context():
            from app.services import lora_training as lt
            stored = lt.snapshot_train_settings('local', ds_id)
            assert stored == preset['settings'], preset_id


def test_krea_raw_lokr_likeness_builtin_is_scoped_and_self_describing(client, app):
    """The community Krea recipe is an extra Character option, never a replacement
    for the default Krea preset; all of its announced controls survive apply."""
    listed = client.get('/api/train/presets').get_json()['presets']
    preset = next(p for p in listed if p['id'] == 'builtin-krea-raw-lokr-likeness')
    assert preset['dataset_kind'] == 'character'
    assert preset['variants'] == ['base', 'raw']
    assert preset['community'] is True
    assert {k: preset['settings'][k] for k in (
        'network_type', 'lokr_factor', 'rank', 'alpha', 'resolution',
        'timestep_type', 'optimizer', 'learning_rate', 'content_or_style',
        'do_differential_guidance', 'differential_guidance_scale',
    )} == {
        'network_type': 'lokr', 'lokr_factor': 16, 'rank': 32, 'alpha': 32,
        'resolution': '768', 'timestep_type': 'sigmoid', 'optimizer': 'automagic2',
        'learning_rate': 1e-4, 'content_or_style': 'balanced',
        'do_differential_guidance': True, 'differential_guidance_scale': 3.0,
    }
    ds_id = _create_ds(client, train_type='krea', kind='character')
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
        'preset_id': preset['id'], 'train_type': 'krea', 'variant': 'base',
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body['ignored'] == [] and body['rejected'] == []
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == preset['settings']


def test_approved_builtin_metadata_scopes_and_clean_apply(client, app):
    listed = client.get('/api/train/presets').get_json()['presets']
    approved = {p['id']: p for p in listed if p['id'] in _APPROVED_BUILTIN_IDS}
    assert set(approved) == _APPROVED_BUILTIN_IDS
    expected_scope = {
        'builtin-krea-raw-character-balanced': ('krea', 'character', ['base', 'raw']),
        'builtin-krea-raw-character-lokr-fast': ('krea', 'character', ['base', 'raw']),
        'builtin-krea-raw-style-compact': ('krea', 'style', ['base', 'raw']),
        'builtin-krea-raw-concept-16gb': ('krea', 'concept', ['base', 'raw']),
        'builtin-zimage-turbo-character-balanced': ('zimage', 'character', ['turbo']),
    }
    expected_sources = {
        'builtin-krea-raw-character-balanced':
            'https://www.reddit.com/r/StableDiffusion/comments/1upiocf/'
            'character_loras_with_krea2_again/',
        'builtin-krea-raw-character-lokr-fast':
            'https://www.reddit.com/r/StableDiffusion/comments/1uyk9fz/'
            'struggling_with_krea2_lora_training_looking_for/',
        'builtin-krea-raw-style-compact':
            'https://www.reddit.com/r/StableDiffusion/comments/1uzuypa/'
            'made_another_style_lora_on_krea2/',
        'builtin-krea-raw-concept-16gb':
            'https://www.reddit.com/r/StableDiffusion/comments/1v9yl1u/'
            'krea_2_lora_training_the_very_easy_guide_for_16gb/',
        'builtin-zimage-turbo-character-balanced':
            'https://www.reddit.com/r/StableDiffusion/comments/1q1ahx9/'
            'some_zimageturbo_training_presets_for_12gb_vram/',
    }
    for idx, (preset_id, (family, kind, variants)) in enumerate(expected_scope.items()):
        preset = approved[preset_id]
        assert (preset['train_type'], preset['dataset_kind'], preset['variants']) == (
            family, kind, variants)
        assert preset['approved'] is preset['community'] is True
        assert preset['confidence'] == 'medium'
        assert preset['evidence_label'] == 'community-tested'
        assert preset['source_url'] == expected_sources[preset_id]
        assert {'min', 'max'} <= set(preset['recommended_images'])
        assert preset['recommended_images']['min'] <= preset['recommended_images']['max']
        steps = preset['recommended_steps']
        assert ('fixed' in steps) ^ ({'per_image', 'min', 'max'} <= set(steps))
        assert preset['checkpoint_targets']
        assert all(type(step) is int and step > 0
                   for step in preset['checkpoint_targets'])
        assert preset['caption_guidance']
        assert preset['limitations'] and all(
            isinstance(item, str) and item for item in preset['limitations'])

        ds_id = _create_ds(
            client, name=f'Approved {idx}', trigger=f'approved{idx}',
            train_type=family, kind=kind)
        r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
            'preset_id': preset_id,
            'train_type': family,
            'variant': variants[0],
        })
        assert r.status_code == 200, (preset_id, r.get_json())
        body = r.get_json()
        assert body['ignored'] == [] and body['rejected'] == []
        with app.app_context():
            from app.services import lora_training as lt
            assert lt.snapshot_train_settings('local', ds_id) == preset['settings']


def test_approved_presets_emit_exact_krea_and_zimage_primitives(client, app, tmp_path):
    cases = [
        ('builtin-krea-raw-character-balanced', 'krea', 'character', 'base'),
        ('builtin-krea-raw-character-lokr-fast', 'krea', 'character', 'base'),
        ('builtin-krea-raw-style-compact', 'krea', 'style', 'base'),
        ('builtin-krea-raw-concept-16gb', 'krea', 'concept', 'base'),
        ('builtin-zimage-turbo-character-balanced', 'zimage', 'character', 'turbo'),
    ]
    processes = {}
    for idx, (preset_id, family, kind, variant) in enumerate(cases):
        ds_id = _create_ds(
            client, name=f'Config {idx}', trigger=f'cfg{idx}',
            train_type=family, kind=kind)
        r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
            'preset_id': preset_id, 'train_type': family, 'variant': variant,
        })
        assert r.status_code == 200
        with app.app_context():
            from app.services import face_dataset_service as svc
            from app.services import lora_training as lt
            ds = svc.get_dataset('local', ds_id)
            ds.train_variant = variant
            processes[preset_id] = lt.build_job_config(
                ds, str(tmp_path / preset_id), steps=2222,
                training_folder=str(tmp_path / 'runs'))['config']['process'][0]

    balanced = processes['builtin-krea-raw-character-balanced']
    assert balanced['network'] == {
        'type': 'lora', 'linear': 32, 'linear_alpha': 32}
    assert balanced['train']['optimizer'] == 'automagic3'
    assert balanced['train']['optimizer_params'] == {'weight_decay': 1e-4}
    assert balanced['train']['timestep_type'] == 'sigmoid'
    assert balanced['train']['content_or_style'] == 'balanced'
    assert balanced['datasets'][0]['resolution'] == [1024]
    assert balanced['save']['save_every'] == 500

    fast = processes['builtin-krea-raw-character-lokr-fast']
    assert {k: fast['network'][k] for k in (
        'type', 'lokr_full_rank', 'lokr_factor')} == {
            'type': 'lokr', 'lokr_full_rank': True, 'lokr_factor': 4}
    assert fast['train']['optimizer'] == 'automagic2'
    assert fast['train']['optimizer_params'] == {'weight_decay': 1e-4}
    assert fast['train']['loss_type'] == 'mse'
    assert fast['train']['ema_config'] == {'use_ema': True, 'ema_decay': 0.99}
    assert fast['train']['do_differential_guidance'] is True
    assert fast['train']['differential_guidance_scale'] == 3.0
    assert fast['datasets'][0]['cache_text_embeddings'] is True

    compact = processes['builtin-krea-raw-style-compact']
    assert compact['network']['type'] == 'lora'
    assert compact['network']['linear'] == compact['network']['linear_alpha'] == 16
    assert compact['datasets'][0]['resolution'] == [512, 768]
    assert compact['save']['save_every'] == compact['sample']['sample_every'] == 250

    low_vram = processes['builtin-krea-raw-concept-16gb']
    assert {k: low_vram['model'][k] for k in (
        'low_vram', 'layer_offloading',
        'layer_offloading_transformer_percent',
        'layer_offloading_text_encoder_percent', 'qtype', 'qtype_te')} == {
            'low_vram': True,
            'layer_offloading': True,
            'layer_offloading_transformer_percent': 0.5,
            'layer_offloading_text_encoder_percent': 0.5,
            'qtype': 'int8',
            'qtype_te': 'int8',
        }
    assert low_vram['datasets'][0]['cache_text_embeddings'] is True
    assert low_vram['train']['optimizer_params'] == {'weight_decay': 1e-4}

    zimage = processes['builtin-zimage-turbo-character-balanced']
    assert {k: zimage['network'][k] for k in (
        'type', 'linear', 'linear_alpha', 'conv', 'conv_alpha')} == {
            'type': 'lora', 'linear': 32, 'linear_alpha': 32,
            'conv': 16, 'conv_alpha': 16,
        }
    assert zimage['train']['optimizer'] == 'adamw8bit'
    assert zimage['train']['lr'] == 1e-4
    assert zimage['train']['content_or_style'] == 'balanced'
    assert zimage['model']['qtype'] == zimage['model']['qtype_te'] == 'float8'
    assert zimage['save']['dtype'] == 'bf16'
    assert zimage['datasets'][0]['cache_text_embeddings'] is True
    assert zimage['datasets'][0]['resolution'] == [512, 768]


def test_zimage_cached_preset_reports_dual_captions_as_unsupported(
        client, app, tmp_path):
    """The preset's cache override must not silently promise two captions."""
    ds_id = _create_ds(
        client, name='Z cache', trigger='zcache',
        train_type='zimage', kind='character')
    applied = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
        'preset_id': 'builtin-zimage-turbo-character-balanced',
        'train_type': 'zimage',
        'variant': 'turbo',
    })
    assert applied.status_code == 200

    with app.app_context():
        from app.services import face_dataset_service as svc
        from app.services import lora_training as lt

        lt.update_train_settings('local', ds_id, {'dual_captions': True})
        ds = svc.get_dataset('local', ds_id)
        ds.train_variant = 'turbo'
        process = lt.build_job_config(
            ds, str(tmp_path / 'dataset'), steps=3000,
            training_folder=str(tmp_path / 'runs'))['config']['process'][0]

        assert process['datasets'][0]['cache_text_embeddings'] is True
        assert 'short_and_long_captions' not in process['train']
        assert lt.launch_settings_snapshot(ds, family='zimage')['dual_captions'] is False
        preflight = lt.training_preflight(
            'local', ds_id, train_type='zimage', variant='turbo')
        dual = next(check for check in preflight['checks']
                    if check['id'] == 'dual_captions')
        assert dual['status'] == 'warn'
        assert 'short caption is ignored' in dual['detail']


def test_approved_preset_step_policies_drive_recommendation_and_info(client, app):
    cases = [
        ('builtin-krea-raw-character-balanced', 'krea', 'character', 'base', 10, 2000),
        ('builtin-krea-raw-character-lokr-fast', 'krea', 'character', 'base', 20, 2000),
        ('builtin-krea-raw-style-compact', 'krea', 'style', 'base', 70, 2250),
        ('builtin-krea-raw-concept-16gb', 'krea', 'concept', 'base', 60, 3250),
        ('builtin-zimage-turbo-character-balanced',
         'zimage', 'character', 'turbo', 35, 3500),
    ]
    for idx, (preset_id, family, kind, variant, count, expected) in enumerate(cases):
        ds_id = _create_ds(
            client, name=f'Steps {idx}', trigger=f'steps{idx}',
            train_type=family, kind=kind)
        r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
            'preset_id': preset_id, 'train_type': family, 'variant': variant,
        })
        assert r.status_code == 200
        with app.app_context():
            from app.extensions import db
            from app.models import FaceDatasetImage
            from app.services import lora_training as lt
            for image_idx in range(count):
                db.session.add(FaceDatasetImage(
                    dataset_id=ds_id, status='keep',
                    filename=f'{idx}-{image_idx}.webp'))
            db.session.commit()
            assert lt.recommended_steps(ds_id, family, variant) == expected
            info = lt.recommended_steps_info(ds_id, family, variant)
            assert info['steps'] == expected
            assert info['recipe'].startswith('preset_')
            if preset_id == 'builtin-krea-raw-style-compact':
                assert info['preset_steps_fixed'] == info['fixed_steps'] == 2250
            else:
                assert info['preset_steps_per_image'] > 0
                assert info['preset_steps_min'] <= expected <= info['preset_steps_max']


def test_approved_builtin_scope_mismatches_are_atomic(client, app):
    cases = [
        ('builtin-krea-raw-character-balanced', 'krea', 'character', 'turbo'),
        ('builtin-krea-raw-character-lokr-fast', 'krea', 'character', 'turbo'),
        ('builtin-krea-raw-style-compact', 'krea', 'style', 'turbo'),
        ('builtin-krea-raw-concept-16gb', 'krea', 'concept', 'turbo'),
        ('builtin-zimage-turbo-character-balanced', 'zimage', 'character', 'base'),
    ]
    for idx, (preset_id, family, kind, variant) in enumerate(cases):
        ds_id = _create_ds(
            client, name=f'Bad scope {idx}', trigger=f'badscope{idx}',
            train_type=family, kind=kind)
        with app.app_context():
            from app.services import lora_training as lt
            lt.update_train_settings('local', ds_id, {'rank': 64})
        r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
            'preset_id': preset_id, 'train_type': family, 'variant': variant,
        })
        assert r.status_code == 409
        assert r.get_json()['error_code'] == 'PRESET_SCOPE'
        with app.app_context():
            from app.services import lora_training as lt
            assert lt.snapshot_train_settings('local', ds_id) == {'rank': 64}


def test_applied_approved_preset_cannot_leak_into_another_family(client, app):
    """Hidden recipe fields are invalidated when the model family changes."""
    import json
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt

    ds_id = _create_ds(
        client, name='Scoped approved recipe', trigger='scoped-approved',
        kind='character', train_type='krea')
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, ds_id)
        ds.train_variant = 'raw'
        svc.db.session.commit()

    applied = client.post(
        f'/api/dataset/{ds_id}/train/presets/apply',
        json={'preset_id': 'builtin-krea-raw-character-balanced',
              'train_type': 'krea', 'variant': 'raw'})
    assert applied.status_code == 200

    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, ds_id)
        raw = json.loads(ds.train_settings)
        assert raw['_active_preset_scope']['train_type'] == 'krea'
        assert raw['optimizer'] == 'automagic3'
        assert svc.set_train_type(LOCAL_USER, ds_id, 'zimage') is True
        ds = svc.get_dataset(LOCAL_USER, ds_id)
        active = lt._train_settings(ds)
        assert 'optimizer' not in active
        assert 'preset_steps_per_image' not in active
        persisted = json.loads(ds.train_settings or '{}')
        assert '_active_preset_scope' not in persisted


def test_variant_override_preflight_and_steps_ignore_out_of_scope_preset(
        client, app, monkeypatch):
    """The UI-selected variant is authoritative before launch persists it."""
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt

    ds_id = _create_ds(
        client, name='Variant context', trigger='variant-context',
        kind='character', train_type='krea')
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, ds_id)
        ds.train_variant = 'base'
        svc.db.session.commit()
    applied = client.post(
        f'/api/dataset/{ds_id}/train/presets/apply',
        json={'preset_id': 'builtin-krea-raw-character-balanced',
              'train_type': 'krea', 'variant': 'base'})
    assert applied.status_code == 200

    with app.app_context():
        monkeypatch.setattr(lt, '_aitoolkit_supports_automagic3', lambda: False)
        matching = lt.training_preflight(
            LOCAL_USER, ds_id, train_type='krea', variant='base')
        assert any(c['id'] == 'automagic3' for c in matching['checks'])
        switched = lt.training_preflight(
            LOCAL_USER, ds_id, train_type='krea', variant='turbo')
        assert all(c['id'] != 'automagic3' for c in switched['checks'])
        assert lt.recommended_steps(ds_id, 'krea', 'base') == 2000
        assert lt.recommended_steps(ds_id, 'krea', 'turbo') == 1500


def test_automagic3_grad_accum_rejection_never_leaves_invalid_candidate(client, app):
    ds_id = _create_ds(client)
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
        'settings': {'optimizer': 'automagic3', 'grad_accum': 2},
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body['ignored'] == []
    assert [item['key'] for item in body['rejected']] == ['grad_accum']
    assert 'Automagic3' in body['rejected'][0]['reason']
    assert 'above 1' in body['rejected'][0]['reason']
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {
            'optimizer': 'automagic3'}


def test_automagic3_capability_preflight_and_16gb_vram_warning_survive(
        client, app, monkeypatch):
    ds_id = _create_ds(
        client, name='16 GB', trigger='sixteen', train_type='krea', kind='concept')
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
        'preset_id': 'builtin-krea-raw-concept-16gb',
        'train_type': 'krea', 'variant': 'base',
    })
    assert r.status_code == 200
    with app.app_context():
        from app import capabilities
        from app.services import lora_training as lt
        monkeypatch.setattr(lt, '_aitoolkit_supports_automagic3', lambda: False)
        monkeypatch.setattr(capabilities, 'gpu_vram_gb', lambda: 16)
        local = lt.training_preflight('local', ds_id, train_type='krea')
        automagic = next(c for c in local['checks'] if c['id'] == 'automagic3')
        assert automagic['status'] == 'fail'
        assert automagic['bypassable'] is False
        vram = next(c for c in local['checks'] if c['id'] == 'vram')
        assert vram['status'] == 'warn'
        assert '24 GB' in vram['detail']
        cloud = lt.training_preflight(
            'local', ds_id, train_type='krea', lane='cloud')
        assert all(c['id'] != 'automagic3' for c in cloud['checks'])


def test_apply_by_preset_id_and_delete(client):
    ds_id = _create_ds(client)
    pid = client.post('/api/train/presets',
                      json={'name': 'ById', 'train_type': 'krea',
                            'settings': {'rank': 32}}).get_json()['id']
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={'preset_id': pid})
    assert r.get_json()['ok'] is True
    assert client.delete(f'/api/train/presets/{pid}').get_json()['ok'] is True
    assert client.delete(f'/api/train/presets/{pid}').status_code == 404


def test_apply_style_builtin_by_string_id_and_legacy_alias(client, app):
    ds_id = _create_ds(client, train_type='krea', kind='style')
    url = f'/api/dataset/{ds_id}/train/presets/apply'
    r = client.post(url, json={
        'preset_id': 'builtin-style-krea-raw',
        'train_type': 'krea',
        'variant': 'base',
    })
    assert r.status_code == 200
    body = r.get_json()
    assert body['preset_id'] == 'builtin-style-krea-raw'
    assert body['ignored'] == [] and body['rejected'] == []
    assert body['train_settings']['effective_rank'] == 32
    with app.app_context():
        from app.services import lora_training as lt
        stored = lt.snapshot_train_settings('local', ds_id)
        assert stored['rank'] == stored['alpha'] == 32
        assert 'dropout' not in stored and 'ema' not in stored

    alias = client.post(url, json={
        'preset_id': 'builtin-style',
        'train_type': 'krea',
        'variant': 'raw',
    })
    assert alias.status_code == 200
    assert alias.get_json()['preset_id'] == 'builtin-style-krea-raw'


def test_builtin_scope_mismatches_never_mutate_dataset(client, app):
    cases = [
        # kind mismatch
        ('krea', None, 'base', 'builtin-style-krea-raw'),
        # family mismatch
        ('zimage', 'style', 'base', 'builtin-style-krea-raw'),
        # variant mismatch (Krea style is scoped to base/raw, not turbo)
        ('krea', 'style', 'turbo', 'builtin-style-krea-raw'),
        # Krea Raw LoKr likeness is likewise never offered to the distilled Turbo base.
        ('krea', 'character', 'turbo', 'builtin-krea-raw-lokr-likeness'),
    ]
    for idx, (family, kind, variant, preset_id) in enumerate(cases):
        ds_id = _create_ds(client, name=f'Scope {idx}', trigger=f'scope{idx}',
                           train_type=family, kind=kind)
        with app.app_context():
            from app.services import lora_training as lt
            lt.update_train_settings('local', ds_id, {'rank': 64})
        r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
            'preset_id': preset_id,
            'train_type': family,
            'variant': variant,
        })
        assert r.status_code == 409
        assert r.get_json()['error_code'] == 'PRESET_SCOPE'
        with app.app_context():
            from app.services import lora_training as lt
            assert lt.snapshot_train_settings('local', ds_id) == {'rank': 64}

    # The Z-Image style preset is now variant-agnostic (weighted timesteps are the
    # Z-Image arch default, not a Base-only choice), so applying it with no
    # requested variant SUCCEEDS on a Turbo-default dataset — a Turbo Z-Image
    # style dataset is no longer left with no built-in style preset.
    ds_id = _create_ds(client, name='Scope absent', trigger='scope_absent',
                       train_type='zimage', kind='style')
    with app.app_context():
        from app.services import lora_training as lt
        lt.update_train_settings('local', ds_id, {'rank': 64})
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
        'preset_id': 'builtin-style-zimage-base',
        'train_type': 'zimage',
    })
    assert r.status_code == 200
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id)['rank'] == 32


def test_numeric_preset_family_mismatch_is_409_without_mutation(client, app):
    ds_id = _create_ds(client, train_type='zimage', kind='style')
    pid = client.post('/api/train/presets', json={
        'name': 'Krea only', 'train_type': 'krea', 'settings': {'rank': 32},
    }).get_json()['id']
    with app.app_context():
        from app.services import lora_training as lt
        lt.update_train_settings('local', ds_id, {'rank': 64})
    r = client.post(f'/api/dataset/{ds_id}/train/presets/apply', json={
        'preset_id': pid, 'train_type': 'zimage', 'variant': 'base',
    })
    assert r.status_code == 409
    assert r.get_json()['error_code'] == 'PRESET_SCOPE'
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {'rank': 64}


def test_new_numeric_preset_kind_and_variant_scope_is_enforced(client, app):
    pid = client.post('/api/train/presets', json={
        'name': 'Scoped Style Base',
        'train_type': 'zimage',
        'dataset_kind': 'style',
        'variants': ['base'],
        'settings': {'rank': 32},
    }).get_json()['id']
    ds_id = _create_ds(client, train_type='zimage', kind='style')
    with app.app_context():
        from app.services import lora_training as lt
        lt.update_train_settings('local', ds_id, {'rank': 64})
    url = f'/api/dataset/{ds_id}/train/presets/apply'
    r = client.post(url, json={
        'preset_id': pid, 'train_type': 'zimage', 'variant': 'turbo',
    })
    assert r.status_code == 409
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {'rank': 64}

    r = client.post(url, json={
        'preset_id': pid, 'train_type': 'zimage', 'variant': 'base',
    })
    assert r.status_code == 200
    with app.app_context():
        from app.services import lora_training as lt
        assert lt.snapshot_train_settings('local', ds_id) == {'rank': 32}


def test_training_preset_scope_columns_exist(app):
    with app.app_context():
        from sqlalchemy import text
        from app.extensions import db
        cols = {row[1] for row in db.session.execute(
            text('PRAGMA table_info(training_preset)'))}
    assert {'dataset_kind', 'variants'} <= cols


def test_save_and_import_reject_invalid_family_without_zimage_fallback(client):
    ds_id = _create_ds(client, train_type='krea')
    snapshot = client.post('/api/train/presets', json={
        'name': 'Bad snapshot family', 'dataset_id': ds_id,
        'train_type': 'not-a-family', 'variant': 'base',
    })
    assert snapshot.status_code == 400
    assert snapshot.get_json()['error'] == 'invalid train_type'

    imported = client.post('/api/train/presets', json={
        'name': 'Bad import family', 'train_type': 'not-a-family',
        'settings': {'rank': 32},
    })
    assert imported.status_code == 400
    assert imported.get_json()['error'] == 'invalid train_type'
    names = {p['name'] for p in client.get('/api/train/presets').get_json()['presets']}
    assert 'Bad snapshot family' not in names
    assert 'Bad import family' not in names


def test_save_and_import_reject_invalid_variant_without_empty_scope(client):
    ds_id = _create_ds(client, train_type='krea')
    snapshot = client.post('/api/train/presets', json={
        'name': 'Bad snapshot variant', 'dataset_id': ds_id,
        'train_type': 'krea', 'variant': '9b',
    })
    assert snapshot.status_code == 400

    imported = client.post('/api/train/presets', json={
        'name': 'Bad import variant', 'train_type': 'zimage',
        'variants': ['9b'], 'settings': {'rank': 32},
    })
    assert imported.status_code == 400
    names = {p['name'] for p in client.get('/api/train/presets').get_json()['presets']}
    assert 'Bad snapshot variant' not in names
    assert 'Bad import variant' not in names
