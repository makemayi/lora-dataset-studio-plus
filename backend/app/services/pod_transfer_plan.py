"""What each way of getting a checkpoint onto a pod COSTS, before the click.

THE NUMBER NOBODY WAS SHOWN
---------------------------
A pod is rented by the hour and is billed from the moment it boots. Seeding its
checkpoint happens on a pod that is already running: the GPU is paid for, and
idle, for the entire duration of the transfer. Three hours of uplink at $1.40/h
is $4.20 of graphics card computing nothing, and until now that was invisible —
the choice of lane was made in the source code, so the price of the choice had
nowhere to appear.

So every estimate here ends in a currency amount, not a duration. A duration is
a fact about a network; a price is a fact the user can act on.

MEASURED, OR HONESTLY GUESSED — AND IT SAYS WHICH
-------------------------------------------------
Uplink speed is the one input that cannot be looked up: it belongs to the
user's line, not to the app or to the pod. So it is MEASURED. Every transfer
this app makes to a pod records how many bytes went out in how many seconds,
and the estimate is the median of the recent ones. The median rather than the
mean because one transfer throttled by a sick host would otherwise poison every
forecast after it.

TWO KINDS OF TRANSFER, AND THEY ARE NOT THE SAME MEASUREMENT
A dataset upload is thousands of small files, eight per POST: what it measures
is dominated by per-request latency. A checkpoint push is one continuous stream:
what it measures is raw throughput. Averaging them together produces a number
that describes NEITHER — and it would describe the slower one while being used
to forecast the faster. So each sample carries its ``kind`` and a forecast only
ever reads its own.

The honest cost of that separation, stated rather than hidden: ten dataset
uploads still leave a checkpoint-push forecast labelled "estimated", because
none of them measured the thing being forecast. That is the right side to be
wrong on — the alternative is a number that looks measured and is not.

With no history there is no pretending: the estimate says it is an assumption
and names the number it assumed. A forecast labelled "measured" that was in
fact a guess is worth less than no forecast, because it will be believed.

The samples live in SystemState. They are a few integers about the user's own
line, they never leave the machine, and they are worthless to anyone else.
"""
import json
import logging
import statistics
import time

from .. import config as cfg
from ..extensions import db
from ..models import SystemState

logger = logging.getLogger(__name__)

_SAMPLES_KEY = 'cloud_uplink_samples'
_MAX_SAMPLES = 12

# One continuous file (a checkpoint push) — raw throughput, and the only kind a
# checkpoint-push forecast may read.
KIND_STREAM = 'stream'
# Many small files, eight per request (a dataset upload) — dominated by
# per-request latency. Recorded because it is a real observation of this line
# and will serve another forecast; never mixed into this one.
KIND_BULK = 'bulk'
# A sample written before kinds existed. Its shape is unknown, so it is kept
# (deleting a user's history to fix our own oversight would be worse) and read
# by nothing: guessing which kind it was is exactly the mixing this avoids.
KIND_LEGACY = 'unknown'

# A sample has to be big enough and long enough to be about the LINE rather
# than about a round-trip. A 2 KB token upload measures latency, and letting it
# into the median would forecast a 26 GB transfer from the speed of a
# handshake.
MIN_SAMPLE_BYTES = 32 * 1024 * 1024
MIN_SAMPLE_SECONDS = 2.0

# The fallback when nothing has been measured and nothing configured. Chosen
# LOW on purpose: this number becomes a price, and the direction to be wrong in
# is the one where the transfer turns out cheaper than announced. It is also
# roughly the consumer upstream that still dominates outside fibre.
ASSUMED_UPLINK_MBPS = 50.0

# What the POD's link is worth, for the Hugging Face lane. Not a guess about a
# datacenter in the abstract: it is the floor this app already refuses to rent
# below (cloud.min_inet_down_mbps), so a pod that exists at all is at least
# this fast. Deliberately the floor and not a typical value — same direction of
# error as the uplink assumption.
DEFAULT_POD_DOWNLINK_MBPS = 200.0

# Neither lane is pure transfer. The Hub lane pays a fixed toll for the command
# round-trip and the Hub's own handshake; the direct lane pays for the pod-side
# assembly, which reads and rewrites the whole file on local disk.
_HUB_OVERHEAD_SECONDS = 90
_ASSEMBLE_BYTES_PER_SECOND = 700e6


def _load_samples(kind=None) -> list:
    """Recorded samples, optionally of ONE kind. A sample with no kind predates
    the distinction and is never returned for a specific one."""
    row = db.session.get(SystemState, _SAMPLES_KEY)
    if not row or not row.value:
        return []
    try:
        parsed = json.loads(row.value)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    out = [s for s in parsed
           if isinstance(s, dict) and float(s.get('bps') or 0) > 0]
    if kind is None:
        return out
    return [s for s in out if (s.get('kind') or KIND_LEGACY) == kind]


def record_uplink_sample(num_bytes, seconds, kind=KIND_STREAM) -> bool:
    """Remember one measured upload speed. Returns whether it was kept.

    ``kind`` says WHAT was measured (see the module docstring): a continuous
    file or a heap of small ones. It is not decoration — a forecast reads only
    its own kind, because the two numbers describe different bottlenecks.

    NEVER raises and never rolls anything back the caller cares about: this is
    bookkeeping about a transfer that already succeeded, and a failure to write
    it must not turn a landed 26 GB upload into an error.
    """
    try:
        size = int(num_bytes or 0)
        elapsed = float(seconds or 0)
        if size < MIN_SAMPLE_BYTES or elapsed < MIN_SAMPLE_SECONDS:
            return False
        samples = _load_samples()
        samples.append({'bytes': size, 'seconds': round(elapsed, 2),
                        'bps': size / elapsed, 'at': int(time.time()),
                        'kind': str(kind or KIND_STREAM)})
        samples = samples[-_MAX_SAMPLES:]
        row = db.session.get(SystemState, _SAMPLES_KEY)
        if row is None:
            row = SystemState(key=_SAMPLES_KEY)
            db.session.add(row)
        row.value = json.dumps(samples)
        db.session.commit()
        return True
    except Exception:
        try:
            db.session.rollback()
        except Exception:
            pass
        logger.debug('could not record an uplink sample', exc_info=True)
        return False


def uplink_bytes_per_second(kind=KIND_STREAM) -> dict:
    """``{'bps', 'source', 'samples', 'mbps'}`` for ONE kind of transfer.

    ``source`` is 'measured' (this machine's own recent transfers OF THAT KIND),
    'configured' (the user told us their uplink) or 'assumed'. The caller SHOWS
    this word — that is the whole point of returning it rather than a bare
    number.

    Dataset uploads do not raise the sample count here, by design: they measured
    a different bottleneck. See the module docstring.
    """
    configured = 0.0
    try:
        configured = float((cfg.get('cloud') or {}).get('uplink_mbps') or 0)
    except (TypeError, ValueError):
        configured = 0.0
    try:
        samples = _load_samples(kind)
    except Exception:
        logger.debug('uplink samples unreadable', exc_info=True)
        samples = []
    if samples:
        bps = statistics.median(float(s['bps']) for s in samples)
        return {'bps': bps, 'source': 'measured', 'samples': len(samples),
                'mbps': round(bps * 8 / 1e6, 1)}
    if configured > 0:
        bps = configured * 1e6 / 8
        return {'bps': bps, 'source': 'configured', 'samples': 0,
                'mbps': round(configured, 1)}
    bps = ASSUMED_UPLINK_MBPS * 1e6 / 8
    return {'bps': bps, 'source': 'assumed', 'samples': 0,
            'mbps': ASSUMED_UPLINK_MBPS}


def pod_downlink_bytes_per_second() -> float:
    try:
        mbps = float((cfg.get('cloud') or {}).get('min_inet_down_mbps') or 0)
    except (TypeError, ValueError):
        mbps = 0.0
    return (mbps or DEFAULT_POD_DOWNLINK_MBPS) * 1e6 / 8


def _cost(seconds, price_per_hour) -> float:
    return round(max(0.0, float(price_per_hour or 0)) * (seconds / 3600.0), 2)


def estimate_direct(size_bytes, price_per_hour=0.0) -> dict:
    """Sending the file from THIS computer, over the user's uplink."""
    size = max(0, int(size_bytes or 0))
    link = uplink_bytes_per_second()
    transfer = size / link['bps'] if link['bps'] else 0.0
    assemble = size / _ASSEMBLE_BYTES_PER_SECOND
    seconds = transfer + assemble
    return {
        'lane': 'direct',
        'bytes': size,
        'seconds': int(seconds),
        'transfer_seconds': int(transfer),
        'assemble_seconds': int(assemble),
        'rate_bytes_per_second': link['bps'],
        'rate_mbps': link['mbps'],
        'rate_source': link['source'],
        'rate_samples': link['samples'],
        'price_per_hour': round(float(price_per_hour or 0), 4),
        'gpu_cost': _cost(seconds, price_per_hour),
    }


def estimate_hub(size_bytes, price_per_hour=0.0) -> dict:
    """Having the POD pull the file from Hugging Face, over its own link."""
    size = max(0, int(size_bytes or 0))
    bps = pod_downlink_bytes_per_second()
    transfer = size / bps if bps else 0.0
    seconds = transfer + _HUB_OVERHEAD_SECONDS
    return {
        'lane': 'hub',
        'bytes': size,
        'seconds': int(seconds),
        'transfer_seconds': int(transfer),
        'rate_bytes_per_second': bps,
        'rate_mbps': round(bps * 8 / 1e6, 1),
        # Never 'measured': this is the floor the app refuses to rent below,
        # not something observed on this user's runs.
        'rate_source': 'floor',
        'rate_samples': 0,
        'price_per_hour': round(float(price_per_hour or 0), 4),
        'gpu_cost': _cost(seconds, price_per_hour),
    }


def duration_label(seconds) -> str:
    """A duration a human reads without converting. Rounded UP: a forecast that
    rounds 89 minutes down to "1 h" is the kind of small lie that makes the
    whole panel untrustworthy."""
    s = max(0, int(seconds or 0))
    if s < 90:
        return f'{max(1, s)} s'
    minutes = -(-s // 60)
    if minutes < 90:
        return f'{minutes} min'
    hours = s / 3600.0
    return f'{hours:.1f} h'


def size_label(num_bytes) -> str:
    v = float(num_bytes or 0)
    if v < 1e6:
        return f'{v / 1e3:.0f} kB'
    if v < 1e9:
        return f'{v / 1e6:.0f} MB'
    return f'{v / 1e9:.1f} GB'
