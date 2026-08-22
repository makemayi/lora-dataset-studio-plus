# Topaz upscale — the worker thread never boots

*2026-08-19 · diagnosis + fix plan (no code changed yet)*

## Symptom

Since 2026-08-18 (the Topaz batch wave), 🔍 Upscale in place — single image
**and** bulk selection — appears to start and then does nothing:

- the bulk button posts, the app answers `{ok: true, queued: N}`, the
  "Topaz batch started" toast fires;
- every selected tile flips to *pending* and stays pending forever;
- the Task Center row appears and sits at `queued` with no progress;
- nothing is ever written to the dataset folder.

The SeedVR2 lane is unaffected (it rides the ComfyUI queue, which does boot).

## Root cause

`backend/app/__init__.py`, `_start_workers(app)`:

```python
def _start_workers(app):
    """Boot background machinery. Idempotent; nothing GPU-ish is required."""
    from .job_queue import queue_manager
    queue_manager.init_app(app)
    queue_manager.start()          # ComfyUI lane — started
    ...                            # training scheduler, cloud boot-recover, supervisor
```

The Topaz lane's worker is **never started**. `topaz_queue`
(`backend/app/services/topaz_job_queue.py:424`) is the single global
`TopazJobManager` instance; a repo-wide grep finds zero calls to
`topaz_queue.start()` or `topaz_queue.init_app()`. Enqueue/cancel/retry
touch the row in SQLite and wake an event no thread is listening to:

```
enqueue_batch -> TopazJob(status='queued') + tile pending   <- this works
                  _wake.set()                                <- nobody waits on it
_run_loop (thread)                                            <- never created
```

So every Topaz job — single or batch — parks in `queued` forever.

### Why the suite is green

All 28 Topaz tests drive `topaz_queue.process_one()` directly (the documented
synchronous seam) and none of them assert that `create_app()` actually starts
the worker. The boot wiring is therefore completely outside the test
perimeter — a seam-only suite cannot see a missing thread.

### Runtime evidence (repro, 2026-08-19)

Real `create_app()` (non-TESTING), then:

```
worker threads after create_app: ['job-queue-worker']      # no topaz-job-worker
enqueued job: topaz-408d...
job status after 4s real time: queued                       # worker polls every 2s
topaz_queue._thread: None | _running flag: False
```

The config confirms the batch path really does dispatch to Topaz:
`engines.upscale_engine = "topaz"`.

## Data state (no cleanup needed)

The stuck batch row is still in `topaz_job` as `queued`, the tiles are
`pending` with `swap_restore` snapshots, and the original files are still on
disk (swap-restore keeps the source until the result lands). After the fix,
the worker picks up the old `queued` row on the next poll — the tile set
completes without re-selecting. `done_images`/`retry` bookkeeping is
additive, nothing to migrate.

## Fix

`backend/app/__init__.py`, inside `_start_workers(app)`, alongside the
`queue_manager` block:

```python
    from .services.topaz_job_queue import topaz_queue
    topaz_queue.init_app(app)
    topaz_queue.start()
```

`start()` is already idempotent (refuses to double-spawn when the thread is
alive), so the `if not app.config.get('TESTING')` guard keeps the unit suite
thread-free exactly as it is today.

### Regression test (closes the seam-only blind spot)

New `backend/tests/test_topaz_boot.py` (or appended to
`test_topaz_job_queue.py`): after `create_app()` in a non-TESTING context,
assert `topaz_queue._running is True` **and** that a live thread named
`topaz-job-worker` exists, then `topaz_queue.stop()`. That test fails on the
current tree and passes after the two-line fix — it is the property the
feature actually depends on.

### What's-new entry (user-visible fix)

Prepend to `frontend/src/whatsNew.js`:

```js
  {
    id: '2026-08-19-topaz-worker-boot',
    date: '2026-08-19',
    title: 'Topaz upscales actually run now',
    blurb: 'Fixed an issue where Topaz upscaling (single or bulk) was queued '
      + 'but never picked up. Queued tasks are now taken up right away — '
      + 'anything still stuck pending from yesterday will finish after the '
      + 'restart.',
  },
```

(No `to` target — reliability fix with nothing to click; no new
setting/section, so no help-registry entry needed.)

## Verification

1. `python -m pytest` (backend) — green, including the new boot test
   (and red on the pre-fix tree).
2. `node --test` in `frontend/` — green (whatsNew list still valid; no
   `to` target means registry lookup stays satisfied).
3. Manual: boot the app, enqueue one topaz job via the bulk button with the
   engine set to Topaz; the tile leaves `pending` within seconds and the Task
   Center row advances k/N. A pre-existing `queued` row (today's stuck batch)
   also completes.

## Commit plan (source-only, then dist)

1. `fix(topaz): start the worker thread at boot — jobs no longer park in queued`
   — `backend/app/__init__.py`, `backend/tests/test_topaz_boot.py` (or
   appended test), `frontend/src/whatsNew.js`.
2. `build(frontend): rebuild dist for the Topaz boot wave` — `frontend/dist/**`.
