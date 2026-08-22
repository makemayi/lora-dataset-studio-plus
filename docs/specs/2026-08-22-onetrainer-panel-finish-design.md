# OneTrainer panel — finish the remaining shared settings (save/sample, rank, masked)

*2026-08-22 · design + execution record (implemented in one wave). Version-controlled under
`docs/specs/`. Builds on `2026-08-22-onetrainer-shared-settings-design.md`.*

## Scope (three items, completed as one wave)

1. **`save_every` / `sample_every`** reach the OneTrainer config.
2. **`rank`** is un-pinned on OneTrainer (the panel drives it); **`alpha`** stays pinned to equal the chosen rank.
3. **`masked`** — person-masked datasets train with masks on the OneTrainer lane.

## Verified facts (real install `E:/OneTrainer`)

- `save_every: int` + `save_every_unit: TimeUnit` (TrainConfig.py:585-586);
  `sample_after: float` + `sample_after_unit: TimeUnit` (567-568). `TimeUnit.STEP='STEP'`.
  `save_every_unit` must be pinned to STEP or the number is read as epochs.
- `lora_rank: int` / `lora_alpha: float` top-level (537-538).
- Masked data: OneTrainer builds each sample's mask via
  `ModifyPath(in_name='image_path', out_name='mask_path', postfix='-masklabel', extension='.png')`
  (`modules/dataLoader/mixin/DataLoaderText2ImageMixin.py`), i.e. a `-masklabel.png` sidecar
  beside the image; `-masklabel` / `-condlabel` are excluded from image collection.
  `masked_training: bool` (TrainConfig.py:518) turns on the mask input.
- This app exports (masked) person masks to a `-masklabel`-compatible form: subject white,
  background weighted at `mask_min_value` (default 0.1). That matches OneTrainer's per-pixel
  loss-weight use (white=keep, dark=downweight). PERSON masking only; the face-mask polarity
  (face dark) is inverted and stays a separate `mask_faces` concern.

## Field mappings

| app setting | OneTrainer field | written when |
| --- | --- | --- |
| `save_every` | `save_every` (int) + `save_every_unit: 'STEP'` | value < a threshold? | value present |
| `sample_every` | `sample_after` (float) + `sample_after_unit: 'STEP'` | value present |
| `rank` | `lora_rank` (int), then `lora_alpha` = rank | rank resolved (default 32 when unset) |
| `masked` | `masked_training: True` + `-masklabel.png` sidecars exported | person masking enabled on the dataset |

## Changes

### 1. `onetrainer_service.build_job_config`
Add `save_every`, `sample_every`, `masked` params (rank already present). Emit:
- `save_every` → `{'save_every': int(v), 'save_every_unit': 'STEP'}`
- `sample_every` → `{'sample_after': float(v), 'sample_after_unit': 'STEP'}`
- `masked` → `{'masked_training': True}` when truthy.

### 2. `onetrainer_service.launch` — thread `save_every` / `sample_every` / `masked` through to `build_job_config`.

### 3. `onetrainer_service.launch_training`
- Read `save_every` / `sample_every` from `_s`; pass through.
- Resolve `masked` via `lora_training.resolve_masked(ds)` and pass it to `launch`. When masked, export with `masked=True` so the `-masklabel` sidecars are written, and set `masked_training`.
- `rank` comes from the dataset's stored setting when present, else the pinned default 32.

### 4. `training_settings_map`
- `rank`: OneTrainer side `PINNED` → `APPLIES`.
- `alpha`: keep `PINNED`, update the reason to "Always equals the chosen rank (scale 1.0)."
- `save_every` / `sample_every`: NOT added to the map (they are dense/separate settings outside the Expert-lane declaration; wiring makes them work rather than greying them).

### 5. `export_dataset_to_aitoolkit`
When `masked=True`, also write the `-masklabel.png` sidecar beside each exported image (the OneTrainer convention), so the same export serves both lanes.

## Tests

- Service: exact keys + unit pinning for save/sample; rank flows from settings; masked_training emitted when masked and sidecars exported; absent cases write nothing.
- Contract (`test_training_settings_map`): rank now applies (update the OneTrainer assertion); add save_every/sample_every not needed (not in map).
- Update `test_the_pinned_rank_is_the_rank_the_launch_actually_passes` → rank is no longer pinned; assert the panel/default value lands in the config.
- Frontend: what's-new entry; no JSX change required for rank/save/sample/masked reclassification beyond the map (rank becomes active; alpha greyed with updated reason).

## Non-goals
- Face-mask polarity (`mask_faces`) stays separate.
- `weight_decay` — field lives in `TrainOptimizerConfig` (nested) which this app does not own, and there is no panel control. Not a panel item.
- OneTrainer queueing / live % / non-Krea-2 — already declared non-goals.
