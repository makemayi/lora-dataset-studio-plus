# OneTrainer honors grad_accum / dropout / ema — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the OneTrainer lane honour three shared Advanced-options settings — `grad_accum`, `dropout`, `ema` — so their panel controls leave the greyed "the other lane" block and become live on OneTrainer.

**Architecture:** The three settings are translated inside `onetrainer_service.build_job_config` into OneTrainer's own TOP-LEVEL fields (`gradient_accumulation_steps`, `dropout_probability`, `ema`+`ema_decay`), read from the dataset's stored train-settings in `launch_training`, and flipped from `absent` to `applies` in `training_settings_map`. The front-end needs no JSX change: `TrainingAdvancedGroups` reclassifies these controls from the served map automatically.

**Tech Stack:** Python / Flask / SQLAlchemy (backend), pytest; `node --test` (frontend, contract tests only — no JSX change).

**Spec:** `docs/specs/2026-08-22-onetrainer-shared-settings-design.md`.

**Verified facts (read from the real install `E:/OneTrainer`):** `gradient_accumulation_steps: int` (TrainConfig.py:425), `ema: EMAMode` (:426) with `CPU` member (`modules/util/enum/EMAMode.py`), `ema_decay: float` (:427), and `dropout_probability: float` top-level at :447 commented `#this is LoRA dropout!` (NOT the :266 text-encoder caption dropout). The shipped Krea 2 preset sets none of these, so writing them via `--config-path` supplies values the preset did not decide.

**Panel value domains** (from `lora_training.py`): `_GRAD_ACCUM_CHOICES=(1,2,4)`, `_DROPOUT_CHOICES=(0.05,0.1,0.15,0.2,0.3)`, `_EMA_CHOICES=(0.99,0.999)`. A value at the default (grad_accum `1`, dropout `0`/off, ema `off`) is NOT written — the preset/default decides, per the ownership rule.

---

## Task 1: `build_job_config` translates the three settings

**Files:**
- Modify: `backend/app/services/onetrainer_service.py` (the `build_job_config` signature + return dict)
- Test: `backend/tests/test_onetrainer_service.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_onetrainer_service.py`:

```python
def test_build_job_config_translates_grad_accum_dropout_ema(onetrainer, tmp_path):
    """The three settings reach OneTrainer in ITS own top-level vocabulary.

    All three are TOP-LEVEL TrainConfig fields (gradient_accumulation_steps,
    dropout_probability = the LoRA/network one, ema+ema_decay). ema is chosen
    as CPU here — the EMA weights are a rank-32 LoRA, tiny, and this app keeps
    VRAM off an already-tight 12B run.
    """
    ots, _cfg = onetrainer
    c = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32,
        grad_accum=4, dropout=0.15, ema=0.99)
    assert c['gradient_accumulation_steps'] == 4
    assert c['dropout_probability'] == 0.15
    assert c['ema'] == 'CPU'
    assert c['ema_decay'] == 0.99
    # The LoRA/network dropout must land TOP-LEVEL, never as a text-encoder
    # caption-dropout field.
    assert 'caption_dropout' not in c


def test_build_job_config_does_not_write_the_defaults(onetrainer, tmp_path):
    """Ownership rule: a default choice (grad_accum 1, dropout off, ema off) is
    not written — the preset / OneTrainer default decides."""
    ots, _cfg = onetrainer
    c = ots.build_job_config(
        trigger='x', dataset_folder=str(tmp_path), training_folder=str(tmp_path),
        steps=100, num_images=10, rank=32, grad_accum=1, dropout=0, ema='off')
    assert 'gradient_accumulation_steps' not in c
    assert 'dropout_probability' not in c
    assert 'ema' not in c and 'ema_decay' not in c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_onetrainer_service.py -k "grad_accum_dropout_ema or does_not_write_the_defaults" -q`
Expected: `FAIL` — `TypeError: build_job_config() got an unexpected keyword argument 'grad_accum'`

- [ ] **Step 3: Implement**

In `backend/app/services/onetrainer_service.py`, add the three params to the `build_job_config` signature (after `min_snr_gamma`):

```python
def build_job_config(trigger: str, dataset_folder: str, training_folder: str,
                     steps: int, num_images: int, rank: int,
                     peft_type: str = PEFT_TYPE_LORA,
                     learning_rate: float | None = None,
                     resolution: int | None = None,
                     epochs: int | None = None,
                     batch_size: int | None = None,
                     te1_lr: float | None = None,
                     te2_lr: float | None = None,
                     lr_scheduler: str | None = None,
                     warmup_steps: int | None = None,
                     min_snr_gamma: float | None = None,
                     grad_accum: int | None = None,
                     dropout: float | None = None,
                     ema: float | None = None) -> dict:
```

And add these inside the returned dict, right after the `min_snr_gamma` block (the last `**{}` before the closing brace):

```python
        # grad_accum / dropout / ema, in OneTrainer's OWN top-level fields.
        # `dropout_probability` here is the TOP-LEVEL field (TrainConfig :447,
        # "this is LoRA dropout!") — never the text-encoder caption-dropout
        # field (:266), which the panel does not control. ema is CPU by design:
        # the EMA weights are a rank-32 LoRA, tiny, and this app keeps VRAM off
        # an already-tight 12B run. Each is written only when the user chose a
        # non-default value — 1 (grad accum) / 0 (dropout) / off (ema) means
        # the preset or OneTrainer's own default decides.
        **({'gradient_accumulation_steps': max(1, int(grad_accum))}
           if grad_accum and int(grad_accum) > 1 else {}),
        **({'dropout_probability': float(dropout)} if dropout else {}),
        **({'ema': 'CPU', 'ema_decay': float(ema)}
           if ema in (0.99, 0.999) else {}),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_onetrainer_service.py -k "grad_accum_dropout_ema or does_not_write_the_defaults" -q`
Expected: `PASS` (2 passed)

- [ ] **Step 5: Verify the existing suite is still green (no map change yet, so the lane-view / contract tests are untouched)**

Run: `cd backend && python -m pytest tests/test_onetrainer_service.py -q`
Expected: `PASS` — no existing assertion breaks, because `training_settings_map` still says these three are `absent` on OneTrainer (so `test_the_lane_view_names_exactly_what_this_lane_honours`'s `applying` list is unchanged).

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/onetrainer_service.py backend/tests/test_onetrainer_service.py
git commit -m "feat(onetrainer): translate grad_accum / dropout / ema into OneTrainer's own fields"
```

---

## Task 2: Flip the settings map, update the lane-view set and the contract test

**Files:**
- Modify: `backend/app/services/training_settings_map.py` (three entries)
- Modify: `backend/tests/test_onetrainer_service.py` (`test_the_lane_view_names_exactly_what_this_lane_honours`)
- Modify: `backend/tests/test_training_settings_map.py` (`DISTINCTIVE` + the OneTrainer contract assertion)

- [ ] **Step 1: Update the lane-view test FIRST so it documents the new set (it will fail after the map flips)**

In `backend/tests/test_onetrainer_service.py`, change the assertion in `test_the_lane_view_names_exactly_what_this_lane_honours`:

```python
    applying = sorted(k for k, v in view.items() if v['state'] == ots.SETTING_APPLIES)
    assert applying == ['batch_size', 'dropout', 'dual_captions', 'ema',
                        'epochs', 'grad_accum', 'learning_rate', 'lr_scheduler',
                        'min_snr_gamma', 'resolution', 'te1_lr', 'te2_lr',
                        'warmup']
```

(Note `grad_accum` / `dropout` / `ema` join the `applies` set; every other key's state across the two lanes is unchanged.)

- [ ] **Step 2: Flip the three map entries**

In `backend/app/services/training_settings_map.py`, in the `SETTINGS` dict, change these three entries so the OneTrainer side becomes `_A` (`(APPLIES, '')`):

```python
    'grad_accum': {'group': 'optimisation',
                   'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},
    'dropout': {'group': 'network',
                'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},
    'ema': {'group': 'quality',
            'lanes': {LANE_AITOOLKIT: _A, LANE_ONETRAINER: _A}},
```

(The ai-toolkit side already was `_A`; only the OneTrainer side changes from `_ABSENT` to `_A`. `dropout` maps to OneTrainer's LoRA/network `dropout_probability`, so "applies, no reason" is correct — it is not the caption dropout, which is a different field.)

- [ ] **Step 3: Extend the contract test**

In `backend/tests/test_training_settings_map.py`, in `test_what_the_onetrainer_lane_claims_to_apply_really_reaches_its_config`, add the three to the `build_job_config` call and to the assertion tuple:

```python
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
```

```python
    for key in ('learning_rate', 'epochs', 'batch_size', 'te1_lr', 'te2_lr',
                'min_snr_gamma', 'grad_accum', 'dropout', 'ema'):
```

(`DISTINCTIVE` already carries `grad_accum=4`, `dropout=0.15`, `ema=0.99`; the existing ai-toolkit branch already uses them, so no new DISTINCTIVE entries are needed. `ema` maps to `ema_decay: 0.99`, which appears in the blob as `'0.99'`.)

- [ ] **Step 4: Run the changed tests**

Run: `cd backend && python -m pytest tests/test_training_settings_map.py tests/test_onetrainer_service.py -q`
Expected: `PASS` — the contract test now proves the three `applies` claims reach a real config, and the lane-view set matches the map.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/training_settings_map.py backend/tests/test_training_settings_map.py backend/tests/test_onetrainer_service.py
git commit -m "feat(training): graduate grad_accum / dropout / ema to OneTrainer (declaration + contract)"
```

---

## Task 3: `launch_training` reads and passes the three

**Files:**
- Modify: `backend/app/services/onetrainer_service.py` (`launch_training`, the `launched = launch(...)` call)
- Test: `backend/tests/test_onetrainer_service.py`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_onetrainer_service.py`:

```python
def test_launch_training_forwards_grad_accum_dropout_ema(onetrainer, tmp_path, monkeypatch, app):
    import json as _j
    from app.config import LOCAL_USER
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    ots, cfg = onetrainer
    root = tmp_path / 'OneTrainer'
    (root / 'venv' / 'Scripts').mkdir(parents=True, exist_ok=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('')
    cfg.save_config({'onetrainer': {'dir': str(root)}})
    monkeypatch.setattr(lt, '_effective_resolution', lambda _ds: [1024])

    class FakeProc:
        pid = 890

        def poll(self):
            return None
    monkeypatch.setattr(ots.subprocess, 'Popen', lambda *a, **k: FakeProc())

    with app.app_context():
        ds = _trainable_krea_dataset(svc, LOCAL_USER, 'OT shared')
        lt.update_train_settings(LOCAL_USER, ds.id, {
            'grad_accum': 4, 'dropout': 0.15, 'ema': 0.999})
        r = ots.launch_training(LOCAL_USER, ds.id, steps=100, check_captions=False)
        written = _j.loads(open(r['config_path'], encoding='utf-8').read())
    assert written['gradient_accumulation_steps'] == 4
    assert written['dropout_probability'] == 0.15
    assert written['ema'] == 'CPU'
    assert written['ema_decay'] == 0.999
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest tests/test_onetrainer_service.py::test_launch_training_forwards_grad_accum_dropout_ema -q`
Expected: `FAIL` — `KeyError: 'gradient_accumulation_steps'` (launch_training does not yet read/pass these, so they are absent from the written config).

- [ ] **Step 3: Implement**

The new params must thread through the intermediate `launch()` too (it forwards to `build_job_config`), not just the `launch_training` caller — otherwise `launch_training` passes `grad_accum=` to a `launch()` that does not accept it. Do BOTH:

**(a) `launch()` signature** — add the three params after `min_snr_gamma`, and forward them in its `build_job_config(...)` call (which currently ends `min_snr_gamma=min_snr_gamma)`

```python
          grad_accum: int | None = None,
          dropout: float | None = None,
          ema: float | None = None) -> dict:
# ... and in build_job_config(...):
                              min_snr_gamma=min_snr_gamma,
                              grad_accum=grad_accum, dropout=dropout, ema=ema)
```

**(b) `launch_training`** — the `launched = launch(...)` call inside it. Add the three reads from `_s` and pass them through. Find the call that currently ends `min_snr_gamma=_s.get('min_snr_gamma'))` and change the `launch(...)` arguments section to:

```python
    launched = launch(trigger=trigger, dataset_folder=dataset_folder,
                      training_folder=str(training_folder), steps=steps,
                      num_images=max(1, num_images), rank=32, peft_type=peft_type,
                      learning_rate=lr, resolution=resolution,
                      epochs=ot_epochs, batch_size=ot_batch,
                      te1_lr=_s.get('te1_lr'), te2_lr=_s.get('te2_lr'),
                      lr_scheduler=_s.get('lr_scheduler'),
                      warmup_steps=_s.get('warmup'),
                      min_snr_gamma=_s.get('min_snr_gamma'),
                      grad_accum=_s.get('grad_accum'),
                      dropout=_s.get('dropout'),
                      ema=_s.get('ema'))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && python -m pytest tests/test_onetrainer_service.py::test_launch_training_forwards_grad_accum_dropout_ema -q`
Expected: `PASS`

- [ ] **Step 5: Run the full OneTrainer service suite**

Run: `cd backend && python -m pytest tests/test_onetrainer_service.py -q`
Expected: `PASS`

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/onetrainer_service.py backend/tests/test_onetrainer_service.py
git commit -m "feat(onetrainer): launch_training forwards grad_accum / dropout / ema"
```

---

## Task 4: Full regression + what's-new

**Files:**
- Modify: `frontend/src/whatsNew.js`

- [ ] **Step 1: Backend regression (the touched areas)**

Run: `cd backend && python -m pytest tests/test_training_settings_map.py tests/test_onetrainer_service.py tests/test_min_snr_gamma.py tests/test_train_presets.py tests/test_train_settings_family_scope.py tests/test_dataset_routes.py -q`
Expected: `PASS` (same count as before; the map flip only widens OneTrainer's `applies` set, no other lane changed).

- [ ] **Step 2: Frontend contract tests (no JSX change, but the served map flows to the panel)**

Run: `cd frontend && node --test`
Expected: `PASS` (3677+). The panel reclassifies the three controls from the served `/api/train/settings-map` — nothing in the JSX or its tests changes.

- [ ] **Step 3: What's-new entry**

Prepend to `frontend/src/whatsNew.js`:

```js
  {
    id: '2026-08-22-onetrainer-shared-settings',
    date: '2026-08-22',
    title: 'OneTrainer uses the same advanced settings as ai-toolkit',
    blurb: 'On the OneTrainer lane, Effective batch, Network dropout and EMA '
      + 'are no longer greyed out — they now reach the run exactly as set. '
      + 'The other advanced levers still show which lane reads them.',
  },
```

- [ ] **Step 4: Verify the what's-new contract**

Run: `cd frontend && node --test`
Expected: `PASS`.

- [ ] **Step 5: Rebuild dist (separate commit, per repo convention)**

Run: `cd frontend && npm run build`

- [ ] **Step 6: Commits (source, then dist)**

```bash
git add frontend/src/whatsNew.js
git commit -m "docs(whats-new): OneTrainer honours the shared advanced settings"
git add frontend/dist
git commit -m "build(frontend): rebuild dist for the OneTrainer shared-settings wave"
```

---

## Self-review notes

- **Spec coverage:** every spec section has a task — Task 1 builds the translation (mapping + "only when chosen" gates), Task 2 flips the declaration and holds it to the contract test, Task 3 wires the dataset values through the launch, Task 4 documents + verifies. The excluded items (`weight_decay`, `rank`/`alpha`, `save_every`/`sample_every`, `masked`) are explicitly non-goals and unchanged.
- **Type consistency:** all three map params are `int|None` / `float|None`; the emitted keys are `gradient_accumulation_steps` (int), `dropout_probability` (float), `ema` (`'CPU'`) + `ema_decay` (float). The service tests and the contract test use the same names.
- **The `ema in (0.99, 0.999)` gate** returns `False` for `'off'`, so `float(ema)` is only reached for a real decay value — no `ValueError` on the off/string path.
- **`test_the_lane_view_names_exactly_what_this_lane_honours`** must change in the SAME commit as the map flip (Task 2), or the suite turns red.
