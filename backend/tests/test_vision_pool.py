"""Bounded concurrency for the vision passes (services/vision_pool.py).

The passes that call Ollama once per image over tens of thousands of files now
overlap those calls. Three properties make that safe, and each has a test here:
the pool never runs wider than configured, a cancel stops feeding it while
keeping every answer already paid for, and a single failing call is carried back
as a value instead of sinking the pass. No Ollama and no database involved —
this is the mechanism on its own.
"""
import threading
import time

import pytest

from app.services import vision_pool
from app.services.vision_pool import (DEFAULT_VISION_CONCURRENCY,
                                      MAX_VISION_CONCURRENCY, map_vision,
                                      vision_concurrency)


# --- the concurrency knob ----------------------------------------------------
def test_concurrency_defaults_when_unset(monkeypatch):
    monkeypatch.setattr(vision_pool.cfg, 'get', lambda *_a, **_k: None)
    assert vision_concurrency() == DEFAULT_VISION_CONCURRENCY


def test_concurrency_reads_config(monkeypatch):
    monkeypatch.setattr(vision_pool.cfg, 'get', lambda *_a, **_k: 6)
    assert vision_concurrency() == 6


def test_explicit_override_wins_over_config(monkeypatch):
    monkeypatch.setattr(vision_pool.cfg, 'get', lambda *_a, **_k: 6)
    assert vision_concurrency(2) == 2


@pytest.mark.parametrize('stored,expected', [
    ('', DEFAULT_VISION_CONCURRENCY),      # blank field in Settings
    ('   ', DEFAULT_VISION_CONCURRENCY),
    ('lots', DEFAULT_VISION_CONCURRENCY),  # free text
    (None, DEFAULT_VISION_CONCURRENCY),
    ('3', 3),                              # a form always sends strings
    (0, 1),                                # clamped, never a zero-wide pool
    (-4, 1),
    (999, MAX_VISION_CONCURRENCY),         # absurd values degrade, never raise
])
def test_unusable_settings_degrade_instead_of_failing(monkeypatch, stored, expected):
    """The Settings form stores whatever was typed. A bad value must cost the
    user the speed-up at worst — never the pass."""
    monkeypatch.setattr(vision_pool.cfg, 'get', lambda *_a, **_k: stored)
    assert vision_concurrency() == expected


# --- ordering and results ----------------------------------------------------
def test_yields_every_item_in_order():
    out = list(map_vision(range(20), lambda i: i * 2, workers=4))
    assert [item for item, _r, _e in out] == list(range(20))
    assert [res for _i, res, _e in out] == [i * 2 for i in range(20)]
    assert all(err is None for _i, _r, err in out)


def test_single_worker_runs_inline():
    """workers=1 is the escape hatch: same results, no pool at all."""
    threads = []
    list(map_vision(range(5), lambda i: threads.append(threading.current_thread()),
                    workers=1))
    assert threads == [threading.current_thread()] * 5


def test_empty_input_is_a_no_op():
    assert list(map_vision([], lambda i: i, workers=4)) == []


def test_parallel_worker_inherits_only_the_vision_token(app):
    """The heartbeat token crosses threads; Flask contexts never do."""
    from flask import has_app_context, has_request_context
    from app import gpu_window

    with app.app_context():
        token = gpu_window._vision_window_context.set(('vision-token', 60))
        try:
            out = list(map_vision(
                [0],
                lambda _item: (gpu_window.vision_window_is_owned(),
                               has_app_context(), has_request_context()),
                workers=2))
        finally:
            gpu_window._vision_window_context.reset(token)
    assert out == [(0, (True, False, False), None)]


# --- failure accounting ------------------------------------------------------
def test_one_failing_call_does_not_sink_the_pass():
    def work(i):
        if i == 3:
            raise RuntimeError('ollama said no')
        return i

    out = list(map_vision(range(8), work, workers=4))
    assert len(out) == 8
    failed = [(item, err) for item, _r, err in out if err is not None]
    assert len(failed) == 1
    assert failed[0][0] == 3
    assert isinstance(failed[0][1], RuntimeError)
    # Everything else still came back with its answer.
    assert [r for i, r, e in out if e is None] == [0, 1, 2, 4, 5, 6, 7]


# --- the bound ---------------------------------------------------------------
def test_never_runs_wider_than_configured():
    peak = 0
    live = 0
    lock = threading.Lock()

    def work(i):
        nonlocal peak, live
        with lock:
            live += 1
            peak = max(peak, live)
        time.sleep(0.02)
        with lock:
            live -= 1
        return i

    list(map_vision(range(40), work, workers=3))
    assert peak <= 3
    assert peak > 1, 'the pool must actually overlap calls, not just serialise them'


def test_calls_actually_overlap():
    """The whole point: waiting overlaps. Six 0.1 s calls at 3 wide must finish
    closer to two rounds than to six."""
    started = time.perf_counter()
    list(map_vision(range(6), lambda i: time.sleep(0.1), workers=3))
    elapsed = time.perf_counter() - started
    assert elapsed < 0.4, f'no overlap: {elapsed:.2f}s for what should take ~0.2s'


# --- cancellation ------------------------------------------------------------
def test_cancel_stops_feeding_and_keeps_what_was_paid_for():
    """The contract that makes Stop safe: once cancelled we submit nothing more,
    but every call already in flight is still handed back so the caller persists
    it. Nothing is answered and then thrown away."""
    answered = []
    stop = threading.Event()

    def work(i):
        answered.append(i)
        if len(answered) >= 5:
            stop.set()
        time.sleep(0.01)
        return i

    out = list(map_vision(range(200), work, workers=4,
                          should_cancel=stop.is_set))
    yielded = [item for item, _r, _e in out]
    assert stop.is_set()
    assert len(yielded) < 200, 'a cancel must actually cut the pass short'
    # Every answer we paid for reached the caller — no work silently discarded.
    assert sorted(yielded) == sorted(answered)


def test_cancel_never_pulls_an_item_it_will_not_process():
    """The caller's iterable is where per-image preparation happens (for the
    watermark pass it is destructive: it drops the stale cleaned blob). So an
    item pulled from it must always be processed — the pool may not read ahead
    and then drop what it read."""
    pulled = []
    stop = threading.Event()

    def source():
        for i in range(200):
            pulled.append(i)
            yield i

    def work(i):
        if i >= 6:
            stop.set()
        time.sleep(0.01)
        return i

    out = list(map_vision(source(), work, workers=4, should_cancel=stop.is_set))
    yielded = [item for item, _r, _e in out]
    assert yielded == pulled, 'items were prepared and then never processed'


@pytest.mark.parametrize('workers', [1, 4])
def test_cancel_never_pulls_an_item_it_will_not_process_at_any_width(workers):
    """Same contract at workers=1 — and it is the width where it mattered most.

    `for item in iterator:` pulls the item BEFORE the loop body can test the
    cancel flag, so the sequential path ran the caller's preparation and then
    threw the item away. For the watermark pass that preparation drops the
    cleaned file the user is looking at: a Stop destroyed a cleaned image and
    left nothing analysed in its place, so a resumed pass could not even redo
    it. The parallel path has always tested the flag before pulling.
    """
    pulled = []
    stop = threading.Event()

    def source():
        for i in range(60):
            pulled.append(i)
            yield i

    def work(i):
        if i >= 4:
            stop.set()
        return i

    out = list(map_vision(source(), work, workers=workers,
                          should_cancel=stop.is_set))
    yielded = [item for item, _r, _e in out]
    assert stop.is_set()
    assert len(yielded) < 60, 'a cancel must actually cut the pass short'
    assert yielded == pulled, 'items were prepared and then never processed'


def test_cancel_before_the_first_call_does_nothing_sequentially():
    """workers=1, cancelled from the start: not a single item may be pulled out
    of the caller's iterable — pulling one is already destructive."""
    pulled = []
    calls = []

    def source():
        for i in range(50):
            pulled.append(i)
            yield i

    out = list(map_vision(source(), calls.append, workers=1,
                          should_cancel=lambda: True))
    assert out == []
    assert calls == []
    assert pulled == []


def test_cancel_before_the_first_call_does_nothing():
    calls = []
    out = list(map_vision(range(50), calls.append, workers=4,
                          should_cancel=lambda: True))
    assert out == []
    assert calls == []


def test_abandoning_the_generator_leaves_no_worker_behind():
    """A consumer that breaks out (an unrelated crash in the job) must not leave
    threads still calling Ollama."""
    before = threading.active_count()
    for _item, _r, _e in map_vision(range(200), lambda i: time.sleep(0.01), workers=4):
        break
    deadline = time.time() + 5
    while threading.active_count() > before and time.time() < deadline:
        time.sleep(0.02)
    assert threading.active_count() <= before
