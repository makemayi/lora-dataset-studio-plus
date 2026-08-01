"""A refused launch must not pay for the freeze it will never use.

`prepare_launch` hashes every image in the dataset, probes nvidia-smi and reads
the ai-toolkit revision. It deliberately runs OUTSIDE the queue/GPU-arbiter lock
pair (doing it under the lock is what once lost cloud runs to `database is
locked`), which means the authoritative "a training is already in progress" test
— which lives inside that pair — necessarily comes after it.

The cheap flag test at the top of `launch_training` covers the common case, but a
real dataset export sits between the two, and that takes minutes: long enough for
another launch to win the process slot. So the flag is re-read immediately before
the freeze. Two in-memory reads; the lock keeps the authority.

Asserted on the SOURCE because `launch_training` cannot be driven here without a
GPU, an ai-toolkit install and a real dataset export — and the property being
protected is an ORDER of statements, which is exactly what a rewrite silently
loses.
"""
import pathlib
import re

SRC = (pathlib.Path(__file__).resolve().parents[1]
       / 'app' / 'services' / 'lora_training.py').read_text(encoding='utf-8')


def _launch_training_body() -> str:
    start = SRC.index('def launch_training(')
    end = SRC.index('\ndef ', start + 1)
    return SRC[start:end]


def test_the_busy_flag_is_re_read_before_the_expensive_freeze():
    body = _launch_training_body()
    prepare = body.index('checkpoint_registry.prepare_launch(')
    busy = [m.start() for m in re.finditer(
        r"queue_manager\._get_system_state\('training_in_progress'", body)]
    assert len(busy) >= 2, 'the busy flag must be read again after the export'
    assert any(b < prepare for b in busy), \
        'a launch that is going to be refused must not pay for the freeze'
    # …and the authoritative copy still lives under the queue/GPU lock pair,
    # after it. The order is part of the GPU admission contract.
    lock = body.index('with _queue_lock, GPU_ARBITER_LOCK:')
    assert any(b > lock for b in busy), \
        'the check inside the queue/GPU lock pair is the authority and must stay'
    assert prepare < lock, \
        'the freeze must stay OUTSIDE the lock — holding it there is what caused ' \
        '"database is locked" on the cloud launch path'
