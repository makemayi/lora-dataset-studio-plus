# OneTrainer honors three shared training settings (grad_accum, dropout, ema)

*2026-08-22 · design + implementation record (the slice has since been
implemented and shipped). This file is version-controlled under `docs/specs/`.
The base OneTrainer design spec it builds on
(`2026-07-30-onetrainer-backend-design.md`) lives in the gitignored
`docs/superpowers/` and is deliberately local-only.*

## Why

The two-sided settings declaration (`training_settings_map.py`) says the
OneTrainer lane reads only a handful of the shared Advanced-options settings.
`grad_accum`, `dropout` and `ema` are declared `absent` on OneTrainer, so on
that lane the three controls are greyed with the reason "does not read this
setting" — even though OneTrainer has an exact field for each of them. This
closes that gap for these three.

## Verified facts (against the real install at `E:/OneTrainer`)

Reread from `modules/util/config/TrainConfig.py` and
`modules/util/enum/EMAMode.py` on 2026-08-22, not guessed:

- `gradient_accumulation_steps: int` — top-level field (TrainConfig.py:425).
- `ema: EMAMode` — top-level enum field (TrainConfig.py:426). `EMAMode` is
  `OFF='OFF' | GPU='GPU' | CPU='CPU'` (`modules/util/enum/EMAMode.py`).
- `ema_decay: float` — top-level, paired with `ema` (TrainConfig.py:427).
- `dropout_probability: float` — appears TWICE:
  - TrainConfig.py:266 inside the text-encoder/model-part config, commented
    `#this is text encoder caption dropout!`
  - TrainConfig.py:447 as a TOP-LEVEL field, commented `#this is LoRA dropout!`
  This app's panel `dropout` control is NETWORK (adapter) dropout
  (`TrainingAdvancedGroups.jsx`: "this is *network* dropout: it randomly drops
  adapter updates"). It maps to the **top-level field at :447**, never to :266.
- `lora_rank` / `lora_alpha` are also top-level (:537/:538) — same class, so
  adding these three beside the existing `lora_rank`/`lora_alpha` top-level keys
  stays consistent.

The shipped Krea 2 preset file does NOT contain any of these three, so writing
them via `--config-path` supplies a value the preset did not decide — consistent
with the ownership rule at the top of `build_job_config`.

## Scope

**In this pass:**

- `grad_accum` → `gradient_accumulation_steps`
- `dropout` → top-level `dropout_probability` (LoRA/network)
- `ema` → `ema: EMAMode.CPU` + `ema_decay`

**Out of scope** (deliberate, noted so the panel-slicing is honest):

- `weight_decay` — its OneTrainer field lives inside `TrainOptimizerConfig`
  (:83), which this app does not own (the optimizer vocabulary differs; the
  app's own Krea preset uses `automagic3`, not an OneTrainer optimizer). Also
  the panel has no `weight_decay` control. **Skip** — not a clean shared land.
- `rank` / `alpha` — currently pinned (32 / =rank) by design. Letting the panel
  drive them would reverse a deliberate pin and needs a separate reappraisal of
  the preset's rank/alpha semantics. Separate pass.
- `save_every` / `sample_every` — not settings-map keys yet; their panel
  controls live outside `TrainingAdvancedGroups`. New keys + new merge, bigger.
- `masked` → `masked_training` — requires exporting masks (currently hardcoded
  `masked=False`). Separate pass.
- OneTrainer queueing, live progress %, non-Krea-2 families — already declared
  non-goals in the base design.

## Field mapping

| Panel setting (app key) | OneTrainer field | Written when | Value translation |
| --- | --- | --- | --- |
| `grad_accum` | `gradient_accumulation_steps` (int) | value `> 1` | pass int |
| `dropout` | `dropout_probability` (float) | value `> 0` | pass float |
| `ema` | `ema` (enum) + `ema_decay` (float) | value is `0.99` or `0.999` | `ema: 'CPU'`, `ema_decay: float(v)` |

"Written when" mirrors the ai-toolkit lane's own semantics (see
`lora_training._ema_eff` / `_ema_fields` / the `if d: net['dropout'] = d`
guard): only an explicit, non-default choice reaches the config — the default /
preset decides otherwise, and "off" (ema) or "0" (dropout) is simply not
written rather than being written as an explicit false.

`EMA mode`: **CPU**, chosen here. The EMA weights are tiny (a rank-32 LoRA
adapter), so CPU EMA is fast enough, and this app deliberately keeps VRAM
pressure off an already-tight 12B training run. GPU mode is the faster option
but costs training VRAM; we do not use it.

## Changes

### 1. `backend/app/services/onetrainer_service.py` — `build_job_config`

Add three optional keyword params (`grad_accum`, `dropout`, `ema`) and emit
the mapped top-level keys under the same "only when the user chose" conditional
shape already used for `lr_scheduler` / `warmup` / `min_snr_gamma`:

```python
**({'gradient_accumulation_steps': max(1, int(grad_accum))}
   if grad_accum and int(grad_accum) > 1 else {}),
**({'dropout_probability': float(dropout)} if dropout else {}),
**({'ema': 'CPU', 'ema_decay': float(ema)}
   if ema in (0.99, 0.999) else {}),
```

### 2. `backend/app/services/onetrainer_service.py` — `launch_training`

Read the three from the same `_s = _train_settings(ds)` already resolved, and
pass them through:

```python
grad_accum=_s.get('grad_accum'), dropout=_s.get('dropout'), ema=_s.get('ema')
```

### 3. `backend/app/services/training_settings_map.py`

Flip the OneTrainer side of these three from `_ABSENT` to `_A` (applies, no
reason — the meanings now genuinely agree across lanes):

- `grad_accum` (group `optimisation`)
- `dropout` (group `network`)
- `ema` (group `quality`)

### 4. UI impact (automatic, no JSX change)

`TrainingAdvancedGroups.jsx` classifies by the map. Once these three become
`applies` on OneTrainer, the three controls leave the "The other lane … does
not read these" collapsed block and render active under "OneTrainer settings" /
"Shared". No frontend change is required for that reclassification.

### 5. Contract test — `backend/tests/test_training_settings_map.py`

In `test_what_the_onetrainer_lane_claims_to_apply_really_reaches_its_config`,
build the config with the new params and assert the mapped values land in the
emitted config (value-by-value, not by field name):

- `grad_accum` = 4 → `"gradient_accumulation_steps": 4` present
- `dropout` = 0.15 → `"dropout_probability": 0.15` present
- `ema` = 0.99 → `"ema": "CPU"` and `"ema_decay": 0.99` present

Add `grad_accum`/`dropout`/`ema` to `DISTINCTIVE` and to the passed-in kwarg
list / assertion tuple.

### 6. Service tests — `backend/tests/test_onetrainer_service.py`

Add a focused test asserting the exact emitted keys for the three, including
the negative cases (grad_accum `1` is NOT written; dropout `0` is NOT written;
ema `'off'` is NOT written), so the "only when the user chose" contract is
pinned.

## Testing

- Backend: `python -m pytest tests/test_training_settings_map.py tests/test_onetrainer_service.py`
- Full affected suite: topaz + bank + the two training-map/service files.
- Frontend: `node --test` (no JSX change, but the settingDefaults / help-registry
  contract tests must stay green — the map change flows to the panel through
  `/api/train/settings-map`).

## Risks / guards

| Risk | Guard |
| --- | --- |
| `dropout` maps to the caption (text-encoder) field instead of the network one | write the TOP-LEVEL `dropout_probability` (:447); service test asserts the emitted flat key |
| `ema` written as a float instead of an enum | service test asserts `ema == 'CPU'` and `ema_decay` separately |
| "off"/default leaked into the config | "written when >1 / >0 / in (0.99,0.999)" gates; negative-case service test |
| A settings-map claim the builder does not honour | contract test builds the real config and checks by VALUE |
