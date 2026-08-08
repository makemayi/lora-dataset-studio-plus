"""Full models in the Checkpoints panel — the lane, and the line it must not cross.

The panel's other two lists deploy LoRA adapters into ComfyUI's ``loras/<family>``.
A dense run produces a ~26 GB transformer and its ~13 GB fp8 twin, and those are
not adapters: the guard that keeps them out of those lists stays, and this second
lane carries the verbs a full model actually has.

The line: THE MASTER IS NEVER SENT TO COMFYUI. Not by a button, not by an
endpoint, not by a filename that happens to sort first. Several tests here exist
only to hold that.
"""
import json
import os

import pytest


def _dense_run(dataset_id, tmp_path, *, files=(), status='done',
               hf_repo='acme/dense-1', delivery='both', **params):
    from app.extensions import db
    from app.models import CloudTrainingRun
    staging = tmp_path / f'store{dataset_id}'
    staging.mkdir(parents=True, exist_ok=True)
    for name, size in files:
        (staging / name).write_bytes(b'W' * size)
    p = {'training_mode': 'full_transformer', 'train_type': 'krea',
         'variant': 'Raw', 'steps': 3000, 'dense_delivery': delivery,
         'version': 2, 'pod_image': 'ai-toolkit:cu124', **params}
    if hf_repo:
        p.update({'hf_repo_id': hf_repo, 'hf_url': f'https://huggingface.co/{hf_repo}',
                  'hf_weight_filename': 'Krea_full_x.safetensors',
                  'artifact_status': 'available'})
    run = CloudTrainingRun(dataset_id=dataset_id, status=status, job_name='j',
                           staging_dir=str(staging), train_params=json.dumps(p))
    db.session.add(run)
    db.session.commit()
    return run


def _lora_run(dataset_id, tmp_path):
    from app.extensions import db
    from app.models import CloudTrainingRun
    d = tmp_path / f'lora{dataset_id}'
    d.mkdir(parents=True, exist_ok=True)
    (d / 'lora_x_000001000.safetensors').write_bytes(b'L')
    run = CloudTrainingRun(dataset_id=dataset_id, status='done', job_name='j',
                           staging_dir=str(d),
                           train_params=json.dumps({'train_type': 'krea',
                                                    'steps': 1000}))
    db.session.add(run)
    db.session.commit()
    return run


# --- what the lane lists ---------------------------------------------------------

def test_a_lora_only_dataset_has_no_full_models(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        _lora_run(1, tmp_path)
        assert da.list_dense_models(1) == []


def test_a_hub_only_run_is_listed_with_no_local_file(app, tmp_path):
    """Every dense run trained before the local delivery is in this state. Hiding
    those is exactly what made the dense lane invisible."""
    from app.services import dense_artifacts as da
    with app.app_context():
        _dense_run(2, tmp_path, delivery='hub')
        [entry] = da.list_dense_models(2)
        assert entry['master'] is None and entry['fp8'] is None
        assert entry['hub']['repo_id'] == 'acme/dense-1'
        assert entry['delivery'] == 'hub'
        # …and it can still be quantized: the job downloads the master first.
        assert entry['can_send_to_comfyui'] is False


def test_a_delivered_run_names_the_master_by_the_shared_rule(app, tmp_path):
    """`dense_weights.pick_master` decides, here as everywhere: the FINAL save
    beats every step snapshot, even though it sorts before them."""
    from app.services import dense_artifacts as da
    with app.app_context():
        _dense_run(3, tmp_path, files=(
            ('Krea_full_x.safetensors', 40),                 # the final
            ('Krea_full_x_000002750.safetensors', 40),       # a step snapshot
            ('Krea_full_x_fp8.safetensors', 20),             # the twin
        ))
        [entry] = da.list_dense_models(3)
        assert entry['master']['filename'] == 'Krea_full_x.safetensors'
        assert entry['master']['is_final'] is True
        # and it says what it was picked over, so the card cannot disagree with
        # the button underneath it
        assert entry['master']['total_candidates'] == 2
        assert entry['master']['others'] == ['Krea_full_x_000002750.safetensors']


def test_the_fp8_twin_is_never_mistaken_for_the_master(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        _dense_run(4, tmp_path, files=(('Krea_full_x_fp8.safetensors', 20),))
        [entry] = da.list_dense_models(4)
        assert entry['master'] is None
        assert entry['fp8']['filename'] == 'Krea_full_x_fp8.safetensors'


def test_the_entry_carries_the_raw_sampler_settings_and_the_trainer_stamp(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        _dense_run(5, tmp_path, files=(('Krea_full_x.safetensors', 40),))
        [entry] = da.list_dense_models(5)
        assert entry['inference_hint']['guidance_scale'] == 4
        assert entry['inference_hint']['steps'] == 25
        # pod_image was stamped since the host blacklist landed and shown nowhere
        assert entry['trainer'] == 'ai-toolkit:cu124'


def test_an_active_run_offers_nothing_yet(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        _dense_run(6, tmp_path, status='training',
                   files=(('Krea_full_x.safetensors', 40),))
        [entry] = da.list_dense_models(6)
        assert entry['active'] is True
        assert entry['can_quantize'] is False
        assert entry['can_send_to_comfyui'] is False
        assert entry['can_delete'] is False


def test_quantizing_is_offered_only_when_there_is_no_twin_yet(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        _dense_run(7, tmp_path, files=(('Krea_full_x.safetensors', 40),))
        assert da.list_dense_models(7)[0]['can_quantize'] is True
        _dense_run(8, tmp_path, files=(('Krea_full_x.safetensors', 40),
                                       ('Krea_full_x_fp8.safetensors', 20)))
        assert da.list_dense_models(8)[0]['can_quantize'] is False


def test_the_family_filter_does_not_hide_a_run_with_no_stamped_family(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        _dense_run(9, tmp_path, files=(('Krea_full_x.safetensors', 40),),
                   train_type=None)
        assert len(da.list_dense_models(9, 'krea')) == 1


def test_the_endpoint_carries_the_lane(app, client, tmp_path):
    ds = client.post('/api/dataset/create',
                     json={'name': 'Dense', 'trigger_word': 'person'}).get_json()['id']
    with app.app_context():
        _dense_run(ds, tmp_path, files=(('Krea_full_x.safetensors', 40),))
    body = client.get(f'/api/dataset/{ds}/train/checkpoints?train_type=krea').get_json()
    assert len(body['dense_models']) == 1
    # the LoRA lanes stay empty — the guard that keeps a full model out of them
    # is untouched, on purpose
    assert body['cloud_checkpoints'] == []
    assert body['cloud_checkpoint_groups'] == []


# --- the line: the master is never sent ------------------------------------------

def test_sending_refuses_a_run_that_has_only_a_master(app, tmp_path):
    """The master is 26 GB of the wrong format. There is no flag, no override and
    no filename that turns this refusal into a copy."""
    from app.services import dense_artifacts as da
    with app.app_context():
        run = _dense_run(10, tmp_path, files=(('Krea_full_x.safetensors', 40),))
        plan = da.send_plan(10, run.id)
        assert plan['ok'] is False
        assert 'Quantize the full model first' in plan['error']
        with pytest.raises(da.DenseArtifactError):
            da.send_to_comfyui(app, 10, run.id)


def test_the_send_plan_only_ever_names_the_twin(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        run = _dense_run(11, tmp_path, files=(('Krea_full_x.safetensors', 40),
                                              ('Krea_full_x_fp8.safetensors', 20)))
        plan = da.send_plan(11, run.id)
        assert plan['ok'] is True
        assert plan['filename'] == 'Krea_full_x_fp8.safetensors'
        assert plan['source'].endswith('Krea_full_x_fp8.safetensors')
        assert plan['destination'].endswith('Krea_full_x_fp8.safetensors')
        # a diffusion model goes to diffusion_models, never to loras/
        assert 'diffusion_models' in plan['destination_dir'].replace('\\', '/')
        assert 'loras' not in plan['destination_dir'].replace('\\', '/')


def test_sending_lands_the_twin_and_says_how(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        run = _dense_run(12, tmp_path, files=(('Krea_full_x_fp8.safetensors', 20),))
        out = da.send_to_comfyui(app, 12, run.id)
        assert out['status'] == 'done'          # same volume -> hard link, instant
        assert out['method'] == da.LINKED
        assert os.path.isfile(out['destination'])
        # …and the button is gone, because the file is where the app puts it.
        [entry] = da.list_dense_models(12)
        assert entry['fp8']['delivered'] is True
        assert entry['can_send_to_comfyui'] is False
        # No ComfyUI is configured in tests, so the destination is the app's OWN
        # models folder (fp8_local_delivery says so out loud) and no ComfyUI scan
        # will ever list it. `in_comfyui` must NOT claim otherwise: it is what
        # lights the "test this in the Studio" link, and a base ComfyUI cannot
        # resolve is a link to a screen where the model is absent.
        assert entry['fp8']['in_comfyui'] is False
        assert entry['fp8']['comfyui_name'] is None
        assert entry['comfyui']['kind'] == 'fallback'


def test_sending_never_overwrites_a_file_already_there(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        run = _dense_run(13, tmp_path, files=(('Krea_full_x_fp8.safetensors', 20),))
        da.send_to_comfyui(app, 13, run.id)
        plan = da.send_plan(13, run.id)
        assert plan['ok'] is False
        assert 'already' in plan['error']


def test_a_twin_ComfyUI_really_lists_is_testable_in_the_studio(app, tmp_path, monkeypatch):
    """The other half of the pair: when the destination IS a ComfyUI folder, the
    twin gets a loader-relative name and the Studio link may appear."""
    from app.services import comfy_model_paths, dense_artifacts as da
    comfy = tmp_path / 'ComfyUI' / 'models' / 'diffusion_models'
    comfy.mkdir(parents=True)
    monkeypatch.setattr(comfy_model_paths, 'search_roots',
                        lambda folder_type: [str(comfy)])
    with app.app_context():
        run = _dense_run(19, tmp_path, files=(('Krea_full_x_fp8.safetensors', 20),))
        da.send_to_comfyui(app, 19, run.id)
        [entry] = da.list_dense_models(19)
        assert entry['fp8']['in_comfyui'] is True
        assert entry['fp8']['comfyui_name'] == 'Krea_full_x_fp8.safetensors'
        assert entry['can_send_to_comfyui'] is False


def test_the_store_sitting_inside_a_model_root_is_not_a_deployment(app, tmp_path, monkeypatch):
    """A checkpoint store declared as a model root would make every twin look
    deployed — and hide the one action that matters."""
    from app import config as cfg
    from app.services import comfy_model_paths, dense_artifacts as da
    with app.app_context():
        store = str(cfg.checkpoints_root(create=True))
        monkeypatch.setattr(comfy_model_paths, 'search_roots',
                            lambda folder_type: [store])
        run = _dense_run(20, tmp_path, files=(('Krea_full_x_fp8.safetensors', 20),))
        # the twin lives in <store>/run_<id>, i.e. under that very root
        import os as _os
        _os.makedirs(_os.path.join(store, 'run_%d' % run.id), exist_ok=True)
        [entry] = da.list_dense_models(20)
        if entry['fp8'] and entry['fp8']['path'].startswith(store):
            assert entry['fp8']['in_comfyui'] is False


def test_a_run_of_another_dataset_is_refused(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        run = _dense_run(14, tmp_path, files=(('Krea_full_x_fp8.safetensors', 20),))
        assert da.send_plan(99, run.id)['ok'] is False
        with pytest.raises(da.DenseArtifactError):
            da.delete_artifact(99, run.id, 'Krea_full_x_fp8.safetensors')


def test_a_lora_run_cannot_be_driven_through_this_lane(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        run = _lora_run(15, tmp_path)
        with pytest.raises(da.DenseArtifactError):
            da.delete_artifact(15, run.id, 'lora_x_000001000.safetensors')


# --- the trash --------------------------------------------------------------------

def test_deleting_moves_the_file_to_the_trash_and_the_lane_follows(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        run = _dense_run(16, tmp_path, files=(('Krea_full_x.safetensors', 40),
                                              ('Krea_full_x_fp8.safetensors', 20)))
        out = da.delete_artifact(16, run.id, 'Krea_full_x.safetensors')
        assert os.path.isfile(out['trashed_to'])          # recoverable, not destroyed
        [entry] = da.list_dense_models(16)
        # the listing reads the DISK, never the stamped filename, so it follows
        assert entry['master'] is None
        assert entry['fp8'] is not None


def test_deleting_is_whitelisted_against_this_run_only(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        run = _dense_run(17, tmp_path, files=(('Krea_full_x.safetensors', 40),))
        for bad in ('nope.safetensors', '../../secret.safetensors',
                    'C:/windows/system32/x.safetensors', ''):
            with pytest.raises(da.DenseArtifactError):
                da.delete_artifact(17, run.id, bad)


def test_deleting_is_refused_while_the_run_is_working(app, tmp_path):
    from app.services import dense_artifacts as da
    with app.app_context():
        run = _dense_run(18, tmp_path, status='training',
                         files=(('Krea_full_x.safetensors', 40),))
        with pytest.raises(da.DenseArtifactError):
            da.delete_artifact(18, run.id, 'Krea_full_x.safetensors')


def test_the_routes_answer_the_same_refusals(app, client, tmp_path):
    ds = client.post('/api/dataset/create',
                     json={'name': 'Dense routes', 'trigger_word': 'p'}).get_json()['id']
    with app.app_context():
        run_id = _dense_run(ds, tmp_path,
                            files=(('Krea_full_x.safetensors', 40),)).id
    plan = client.post(f'/api/dataset/{ds}/train/dense/send-plan',
                       json={'run_id': run_id}).get_json()
    assert plan['ok'] is False                       # master only: nothing to send
    res = client.post(f'/api/dataset/{ds}/train/dense/delete',
                      json={'run_id': run_id, 'filename': 'nope.safetensors'})
    assert res.status_code == 400
    ok = client.post(f'/api/dataset/{ds}/train/dense/delete',
                     json={'run_id': run_id, 'filename': 'Krea_full_x.safetensors'})
    assert ok.status_code == 200 and ok.get_json()['ok'] is True
