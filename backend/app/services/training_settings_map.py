"""Which training setting reaches which trainer, declared once.

WHY THIS EXISTS
---------------
The Advanced-options panel was built for the ai-toolkit lane and shows around
forty settings. The app then grew a second lane, OneTrainer, and for a long time
nothing said which of those forty it actually read — so a user could pick a
rank, an optimiser and a scheduler, watch a OneTrainer run start, and get a job
that used none of them. That is worse than not offering them, because the run
looks like it obeyed.

A one-sided declaration fixed half of it. It could not group the panel, because
"ai-toolkit only" is NOT the complement of "OneTrainer reads it": several stored
settings (`qtype`, `layer_offloading`, `cache_text_embeddings`, `save_dtype`)
have no control on either lane and are reachable only through a preset. Taking
the complement would have invented a category.

So each setting names the lanes that read it, per lane, and the panel groups
itself from this. The rule that makes it worth having is that it is HELD TO THE
CODE: `tests/test_training_settings_map.py` builds a real job config for each
lane with every setting populated and checks that everything claiming to apply
shows up in it. A claim here that the builder does not honour fails the suite
rather than quietly making the panel wrong again — the drift being the entire
bug this file exists to end.

THE THREE STATES, per lane
--------------------------
  'applies'  the panel's value is what runs on that lane.
  'pinned'   that lane forces a value of its own; the panel's is ignored.
             Carries a reason — a greyed control with no reason teaches
             nothing.
  'absent'   that lane has no such concept. Not an error and not a defect:
             epochs mean nothing to a trainer that counts steps.

A setting this module has never heard of is 'absent' everywhere, which fails
safe: a control it does not know about is certainly not one it can promise
reaches a trainer.
"""
from __future__ import annotations

LANE_AITOOLKIT = 'ai_toolkit'
LANE_ONETRAINER = 'onetrainer'
LANES = (LANE_AITOOLKIT, LANE_ONETRAINER)

APPLIES = 'applies'
PINNED = 'pinned'
ABSENT = 'absent'

# Panel groups. Ordered — the panel renders them in this order, and a group
# with nothing in it for the current lane is not rendered at all.
GROUPS = (
    ('core', 'Core'),
    ('iteration', 'Iteration'),
    ('network', 'Network'),
    ('optimisation', 'Optimisation'),
    ('text_encoders', 'Text encoders'),
    ('memory', 'Memory'),
    ('quality', 'Quality'),
)

_A = (APPLIES, '')


def _pinned(why):
    return (PINNED, why)


_ABSENT = (ABSENT, '')

# key -> {'group': str, 'lanes': {lane: (state, why)}}
#
# Every entry's OneTrainer side was verified against `onetrainer_service`'s own
# build_job_config; every ai-toolkit side against `lora_training`'s. Neither was
# taken from a field list written elsewhere — that is how a panel ends up
# offering `bucket_strategy` to a trainer that has never heard of it.
SETTINGS: dict[str, dict] = {
    # --- shared: both lanes read these ---
    'learning_rate': {'group': 'core',
                      'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},
    'resolution': {'group': 'core',
                   'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},
    'dual_captions': {'group': 'core',
                      'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},
    'lr_scheduler': {'group': 'optimisation',
                     'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},
    'warmup': {'group': 'optimisation',
               'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},
    # One idea, two shapes: ai-toolkit takes a bare `min_snr_gamma`, OneTrainer
    # takes loss_weight_fn + loss_weight_strength. Translated in each service.
    'min_snr_gamma': {'group': 'quality',
                      'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},

    # --- OneTrainer's own vocabulary ---
    'epochs': {'group': 'iteration',
               'lanes': {LANE_AITOOLKIT: _ABSENT, LANE_ONETRAINER: _A}},
    'batch_size': {'group': 'iteration',
                   'lanes': {LANE_AITOOLKIT: _ABSENT, LANE_ONETRAINER: _A}},
    'te1_lr': {'group': 'text_encoders',
               'lanes': {LANE_AITOOLKIT: _ABSENT, LANE_ONETRAINER: _A}},
    'te2_lr': {'group': 'text_encoders',
               'lanes': {LANE_AITOOLKIT: _ABSENT, LANE_ONETRAINER: _A}},

    # --- set on the panel, forced by the OneTrainer lane ---
    'rank': {'group': 'network', 'lanes': {
        LANE_AITOOLKIT: _A,
        LANE_ONETRAINER: _A}},
    'alpha': {'group': 'network', 'lanes': {
        LANE_AITOOLKIT: _A,
        LANE_ONETRAINER: _pinned(
            'Always equals the chosen rank (scale 1.0). A LoRA trained at any '
            'other alpha against this app’s rank came out ~1/32 of its intended '
            'strength.')}},
    'network_type': {'group': 'network', 'lanes': {
        LANE_AITOOLKIT: _A,
        LANE_ONETRAINER: _pinned(
            'This lane reads Settings ▸ OneTrainer ▸ PEFT type, not this '
            'control.')}},

    # --- ai-toolkit only. NOT "everything OneTrainer skips" — several stored
    #     settings below have no control anywhere and are preset-only, which is
    #     precisely why the complement could not be taken. ---
    'optimizer': {'group': 'optimisation',
                  'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'grad_accum': {'group': 'optimisation',
                   'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},
    'dropout': {'group': 'network',
                'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},
    'lokr_factor': {'group': 'network',
                    'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'lokr_full_rank': {'group': 'network',
                       'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'conv': {'group': 'network',
             'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'conv_alpha': {'group': 'network',
                   'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'timestep_type': {'group': 'quality',
                      'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'ema': {'group': 'quality',
            'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},
    'content_or_style': {'group': 'quality',
                         'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'do_differential_guidance': {'group': 'quality',
                                 'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'differential_guidance_scale': {'group': 'quality',
                                    'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'weight_decay': {'group': 'optimisation',
                     'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'loss_type': {'group': 'quality',
                  'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'quantize': {'group': 'memory',
                 'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'quantize_te': {'group': 'memory',
                    'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'low_vram': {'group': 'memory',
                 'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'qtype': {'group': 'memory',
              'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'qtype_te': {'group': 'memory',
                 'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'layer_offloading': {'group': 'memory',
                         'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'layer_offloading_transformer_percent': {'group': 'memory',
                                             'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'layer_offloading_text_encoder_percent': {'group': 'memory',
                                              'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'cache_text_embeddings': {'group': 'memory',
                              'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
    'save_dtype': {'group': 'memory',
                   'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _ABSENT}},
}


def status(setting_key, lane):
    """``(state, why)`` for one setting on one lane.

    Unknown key or unknown lane is ABSENT, which fails safe: a setting this
    module has never heard of is certainly not one it can promise reaches a
    trainer.
    """
    entry = SETTINGS.get(setting_key)
    if not entry:
        return _ABSENT
    return entry['lanes'].get(lane, _ABSENT)


def for_lane(lane):
    """``{key: {state, why, group}}`` for everything this lane has an opinion
    about — shaped for the UI, which groups itself from it."""
    out = {}
    for key, entry in SETTINGS.items():
        state, why = entry['lanes'].get(lane, _ABSENT)
        out[key] = {'state': state, 'why': why, 'group': entry['group']}
    return out


def keys_applying_on(lane):
    """Every setting that this lane's job builder is expected to honour. The
    contract test compares this against a real config rather than a list."""
    return sorted(k for k, e in SETTINGS.items()
                  if e['lanes'].get(lane, _ABSENT)[0] == APPLIES)
