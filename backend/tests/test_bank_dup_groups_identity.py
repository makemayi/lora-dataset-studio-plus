"""🗃️ Image bank — the duplicate grouping still returns EXACTLY the old groups.

The pairwise comparison inside `rebuild_dup_groups` was rewritten from a Python
double loop (plus an unbounded `set` of every pair already looked at — 829 MB at
16 000 images, and the garbage collector sweeping it was ~79 % of the phase's
wall time) to the blocked numpy form `rebuild_semantic_dup_groups` already uses.

A rewrite of a grouping is not a refactor the tests can take on trust: `dup_group`
is what the user SEES in the duplicates panel, and a version that groups "the same
in spirit" would change their bank without anyone noticing. So the old algorithm
lives on below, verbatim, as an oracle, and these tests assert the two produce the
same groups — the same ids, on the same images — over random hashes, over hashes
built to be adversarial (exact ties, chains, one huge group), and at every
distance the setting allows.
"""
import hashlib
import random

import pytest
from PIL import Image


# --- the algorithm as it stood, character for character ----------------------
# Copied from image_bank_service.rebuild_dup_groups at 2912e86c/5011b2d1, with
# only the DB read and write stripped off. Do NOT "tidy" it: its value is being
# the code that produced the groups users already have.
def legacy_groups(hashes, d, hamming):
    n = len(hashes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    bands = max(1, min(16, d + 1))
    band_bits = 64 // bands
    buckets: dict = {}
    for i, h in enumerate(hashes):
        for b in range(bands):
            key = (b, (h >> (b * band_bits)) & ((1 << band_bits) - 1))
            buckets.setdefault(key, []).append(i)
    seen_pairs = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                if find(a) == find(b) or (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                if hamming(hashes[a], hashes[b]) <= d:
                    union(a, b)
    comps: dict = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return sorted((m for m in comps.values() if len(m) >= 2),
                  key=lambda m: (-len(m), m[0]))


MASK64 = (1 << 64) - 1


def _flip(h, bits, rnd):
    """`h` with `bits` random bit positions flipped — Hamming distance `bits`."""
    for pos in rnd.sample(range(64), bits):
        h ^= (1 << pos)
    return h


def _random_pool(seed, n, d):
    """A pool where duplicates are PLANTED, so groups actually exist: a third of
    the hashes are near-copies of an earlier one at distance 0..d+1 — straddling
    the threshold on purpose, so a version that rounds the comparison the other
    way is caught."""
    rnd = random.Random(seed)
    out = [rnd.getrandbits(64)]
    while len(out) < n:
        if rnd.random() < 0.35:
            out.append(_flip(rnd.choice(out), rnd.randint(0, d + 1), rnd))
        else:
            out.append(rnd.getrandbits(64))
    return out


def _adversarial_pools(d):
    """The shapes a random pool almost never produces."""
    base = 0x0123456789ABCDEF
    rnd = random.Random(99)
    return {
        'empty': [],
        'one': [base],
        'all identical': [base] * 40,
        # A chain: each link is within d of the previous one, the ends are not.
        # Union-find must still fuse the whole chain into ONE group.
        'chain': [(base ^ ((1 << (i * max(1, d))) - 1 if i else 0)) & MASK64
                  for i in range(12)],
        'ties at the threshold': [base] + [_flip(base, d, rnd) for _ in range(6)]
                                 + [_flip(base, d + 1, rnd) for _ in range(6)],
        # Bigger than one comparison block, so the blocked path really blocks.
        'one huge bucket': [base] * 900 + [_flip(base, 1, rnd) for _ in range(900)],
        'zero and all ones': [0, 0, MASK64, MASK64, 1, MASK64 - 1],
    }


@pytest.mark.parametrize('d', [0, 1, 3, 5, 8, 16, 24])
def test_the_rewritten_grouping_returns_the_legacy_groups_exactly(d):
    from app.services.image_bank_service import _dup_groups_from_hashes
    from app.services.face_dataset_service import _hamming

    for seed in range(6):
        pool = _random_pool(seed, 220, d)
        assert _dup_groups_from_hashes(pool, d) == legacy_groups(pool, d, _hamming), (
            f'the grouping changed at dup_distance={d} (seed {seed}) — the '
            'duplicates panel would show different groups than before')


@pytest.mark.parametrize('d', [0, 1, 5, 12])
def test_the_rewritten_grouping_survives_the_shapes_random_data_never_makes(d):
    from app.services.image_bank_service import _dup_groups_from_hashes
    from app.services.face_dataset_service import _hamming

    for name, pool in _adversarial_pools(d).items():
        assert _dup_groups_from_hashes(pool, d) == legacy_groups(pool, d, _hamming), (
            f'{name!r} grouped differently at dup_distance={d}')


def test_the_block_ceiling_is_really_crossed_by_the_big_pool(monkeypatch):
    """The blocked path is only tested if a pool is big enough to be cut in
    pieces. Force a tiny ceiling and re-run: with one row per block, the numpy
    branch takes a different code path at every iteration and must still agree."""
    from app.services import image_bank_service as banks
    from app.services.face_dataset_service import _hamming

    pool = _random_pool(7, 260, 5)
    monkeypatch.setattr(banks, '_DUP_BLOCK_CELLS', 260)   # → 1 row per block
    monkeypatch.setattr(banks, '_DUP_NUMPY_FROM', 2)      # → never the Python path
    assert banks._dup_groups_from_hashes(pool, 5) == legacy_groups(pool, 5, _hamming)


def test_the_oracle_is_not_trivially_equal_to_everything():
    """A guard on the two tests above: they would pass just as green if the
    fixtures produced no group at all, or if `legacy_groups` returned the same
    thing whatever it was given."""
    from app.services.face_dataset_service import _hamming

    pool = _random_pool(0, 220, 5)
    groups = legacy_groups(pool, 5, _hamming)
    assert len(groups) >= 5, 'the fixture must actually contain duplicate groups'
    assert sum(len(g) for g in groups) >= 20
    assert legacy_groups(pool, 0, _hamming) != groups, (
        'the oracle returns the same groups at distance 0 and 5 — it is not '
        'reading its input')


# --- and the same, end to end, through the database --------------------------
def _bank_of_hashes(banks, db, hashes, tmp_path):
    from app.models import BankImage
    src = tmp_path / 'hashes'
    src.mkdir(parents=True, exist_ok=True)
    bank, _added = banks.create_bank('local', 'Hashes', str(src))
    for i, h in enumerate(hashes):
        relpath = f'{i:05d}.jpg'
        path = src / relpath
        Image.new(
            'RGB', (8, 8), (i % 256, (i // 256) % 256, h & 0xff),
        ).save(str(path), 'JPEG')
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        db.session.add(BankImage(
            bank_id=bank.id, relpath=relpath, status='pending',
            dhash=f'{h:016x}', analysis_fingerprint=fingerprint))
    db.session.commit()
    return bank.id


def test_the_stored_dup_group_ids_are_the_legacy_ones(app, tmp_path):
    """The numbers actually written to the rows, not just the in-memory groups:
    the ids are 1-based, biggest group first, and they must land on the same
    images as before."""
    from app.extensions import db
    from app.models import BankImage
    from app.services import image_bank_service as banks
    from app.services.face_dataset_service import _hamming

    pool = _random_pool(3, 240, 5)
    with app.app_context():
        bank_id = _bank_of_hashes(banks, db, pool, tmp_path)
        n = banks.rebuild_dup_groups(bank_id, max_distance=5)
        rows = (BankImage.query.filter_by(bank_id=bank_id)
                .order_by(BankImage.id.asc()).all())
        stored = [r.dup_group for r in rows]

    expected = [None] * len(pool)
    for gid, members in enumerate(legacy_groups(pool, 5, _hamming), start=1):
        for i in members:
            expected[i] = gid
    assert stored == expected, 'the stored dup_group ids drifted from the old ones'
    assert n == max(g for g in expected if g), 'the returned count is not the group count'
