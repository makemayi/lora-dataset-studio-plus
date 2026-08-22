"""The two-sided declaration, held to the two job builders.

A hand-written map of "which setting reaches which trainer" drifts the first
time a field moves, and a drifted map is the same bug as no map — the panel goes
on describing a run that is not happening. So nothing here trusts the
declaration: each lane's REAL job config is built with every applying setting
populated, and the claim is checked against what came out.

The check is by VALUE, not by field name. The two trainers spell the same idea
differently (min-SNR gamma is `min_snr_gamma` on one side and
loss_weight_fn + loss_weight_strength on the other), so asserting field names
would mean maintaining a second map to keep the first one honest.
"""
import json

import pytest

from app.services import training_settings_map as tsm


# One distinctive value per setting, chosen so it cannot be confused with a
# default or with another setting's value once the config is a JSON string.
DISTINCTIVE = {
    'learning_rate': 0.000123,
    'resolution': '1024',
    'lr_scheduler': 'cosine',
    'warmup': 500,
    'min_snr_gamma': 7,
    'epochs': 13,
    'batch_size': 5,
    'te1_lr': 0.0000111,
    'te2_lr': 0.0000222,
    'rank': 16,
    'alpha': 24,
    'dropout': 0.15,
    'grad_accum': 4,
    'ema': 0.99,
    'timestep_type': 'weighted',
    'optimizer': 'adafactor',
}


def _dataset(tmp_path, name, **settings):
    from app import config as cfg
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    cfg.save_config({'aitoolkit': {'dir': str(tmp_path / 'aitoolkit')}})
    ds = svc.create_dataset(LOCAL_USER, name, 'zmap', train_type='krea')
    if settings:
        lt.update_train_settings(LOCAL_USER, ds.id, settings)
    return ds


def test_every_lane_is_declared_for_every_setting(app):
    """A setting that names one lane and forgets the other is how "absent" and
    "nobody checked" become indistinguishable."""
    for key, entry in tsm.SETTINGS.items():
        assert set(entry['lanes']) == set(tsm.LANES), f'{key} does not name both lanes'
        assert entry['group'] in dict(tsm.GROUPS), f'{key} is in an unknown group'


def test_a_pinned_setting_always_says_why(app):
    """A greyed control with no reason teaches nothing — the user cannot tell a
    deliberate limitation from a bug."""
    for key, entry in tsm.SETTINGS.items():
        for lane, (state, why) in entry['lanes'].items():
            if state == tsm.PINNED:
                assert why.strip(), f'{key} is pinned on {lane} and does not say why'


def test_an_unknown_setting_or_lane_is_absent(app):
    """Failing safe: this module cannot promise a trainer reads something it has
    never heard of."""
    assert tsm.status('invented_later', tsm.LANE_ONETRAINER)[0] == tsm.ABSENT
    assert tsm.status('learning_rate', 'some_third_trainer')[0] == tsm.ABSENT


def test_ai_toolkit_only_is_not_the_complement_of_onetrainer(app):
    """The reason this map is two-sided at all.

    If "ai-toolkit only" could be taken as "everything OneTrainer skips", one
    declaration would have done. It cannot: `qtype`, `layer_offloading`,
    `cache_text_embeddings` and `save_dtype` are stored settings with no control
    on either lane, reachable only through a preset. Taking the complement would
    have invented a category and put them in a panel.
    """
    ot_absent = {k for k in tsm.SETTINGS
                 if tsm.status(k, tsm.LANE_ONETRAINER)[0] == tsm.ABSENT}
    ai_applies = {k for k in tsm.SETTINGS
                  if tsm.status(k, tsm.LANE_AITOOLKIT)[0] == tsm.APPLIES}
    # They overlap heavily, which is the point — but the map is what decides,
    # not a set operation.
    assert ot_absent & ai_applies, 'expected the ai-toolkit-only group to exist'
    for k in ('epochs', 'batch_size', 'te1_lr', 'te2_lr'):
        assert k in tsm.SETTINGS and k not in ai_applies, (
            f'{k} is OneTrainer vocabulary; the complement would have handed it '
            'to ai-toolkit')


def test_what_the_onetrainer_lane_claims_to_apply_really_reaches_its_config(app, tmp_path):
    """Built for real, then read back. A claim the builder does not honour fails
    HERE rather than in a panel nobody can check."""
    from app.services import onetrainer_service as ots
    with app.app_context():
        _dataset(tmp_path, 'map-ot')
        cfgd = ots.build_job_config(
            trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
            steps=1000, num_images=20, rank=DISTINCTIVE['rank'],
            learning_rate=DISTINCTIVE['learning_rate'],
            resolution=1024, epochs=DISTINCTIVE['epochs'],
            batch_size=DISTINCTIVE['batch_size'],
            te1_lr=DISTINCTIVE['te1_lr'], te2_lr=DISTINCTIVE['te2_lr'],
            lr_scheduler=DISTINCTIVE['lr_scheduler'],
            warmup_steps=DISTINCTIVE['warmup'],
            min_snr_gamma=DISTINCTIVE['min_snr_gamma'],
            grad_accum=DISTINCTIVE['grad_accum'],
            dropout=DISTINCTIVE['dropout'],
            ema=DISTINCTIVE['ema'])
    blob = json.dumps(cfgd)
    # Only the ones this builder is actually handed — `dual_captions` shapes the
    # exported dataset rather than the job config, and `resolution` is checked
    # by value below because it is stringified.
    for key in ('learning_rate', 'epochs', 'batch_size', 'te1_lr', 'te2_lr',
                'min_snr_gamma', 'grad_accum', 'dropout', 'ema', 'rank'):
        if tsm.status(key, tsm.LANE_ONETRAINER)[0] != tsm.APPLIES:
            continue
        assert str(DISTINCTIVE[key]) in blob, (
            f'{key} is declared as applying on OneTrainer and its value is not '
            'in the job config this lane writes')
    assert cfgd['resolution'] == '1024'
    assert cfgd['learning_rate_scheduler'] == 'COSINE'


def test_what_the_ai_toolkit_lane_claims_to_apply_really_reaches_its_config(app, tmp_path):
    from app.config import LOCAL_USER
    from app.services import lora_training as lt
    keys = ('learning_rate', 'rank', 'alpha', 'dropout', 'grad_accum', 'ema',
            'timestep_type', 'optimizer', 'min_snr_gamma', 'lr_scheduler')
    with app.app_context():
        ds = _dataset(tmp_path, 'map-ai',
                      **{k: DISTINCTIVE[k] for k in keys})
        folder = tmp_path / 'ds'
        folder.mkdir()
        job = lt.build_job_config(ds, str(folder), steps=1000)
    blob = json.dumps(job)
    for key in keys:
        if tsm.status(key, tsm.LANE_AITOOLKIT)[0] != tsm.APPLIES:
            continue
        assert str(DISTINCTIVE[key]) in blob, (
            f'{key} is declared as applying on ai-toolkit and its value is not '
            'in the job config that lane writes')


def test_for_lane_is_shaped_for_the_panel(app):
    view = tsm.for_lane(tsm.LANE_ONETRAINER)
    assert view['rank']['state'] == tsm.APPLIES
    assert view['alpha']['state'] == tsm.PINNED and view['alpha']['why']
    assert view['epochs']['state'] == tsm.APPLIES
    assert view['optimizer']['state'] == tsm.ABSENT
    assert view['epochs']['group'] == 'iteration'


def test_the_route_serves_both_lanes_and_the_group_order(client):
    """The panel needs both sides in one answer — it renders the shared block
    plus the lane's own, and it cannot do that from a one-sided list."""
    r = client.get('/api/train/settings-map')
    assert r.status_code == 200
    body = r.get_json()
    assert set(body['lanes']) == set(tsm.LANES)
    assert body['lanes']['onetrainer']['epochs']['state'] == 'applies'
    assert body['lanes']['ai_toolkit']['epochs']['state'] == 'absent'
    assert body['lanes']['onetrainer']['rank']['state'] == 'applies'
    assert body['lanes']['onetrainer']['alpha']['state'] == 'pinned'
    assert body['lanes']['onetrainer']['alpha']['why']
    # Ordered: the panel renders groups in this order and a stable order is the
    # difference between a layout and a shuffle on every reload.
    assert [g['key'] for g in body['groups']] == [k for k, _ in tsm.GROUPS]
