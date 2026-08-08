"""🗃️ Bank chips: the number ON a chip is the number the chip OPENS.

The defect, reported after a review session: "we always leave the total. We
never adapt the figures to the filters in use." Measured on a 50 397-image bank
with ✕ Rejected picked, the chips advertised "📺 Noisy 1 136" and "🔞 NSFW
20 540" over grids of 527 and 3 364 rows. The counts were never wrong ABOUT THE
BANK — they just stopped describing anything the user could see, which is the
same thing from where they sit.

So no test here asserts a facet count on its own. Each one asserts the count
AGAINST the page that clicking that chip really returns, with the same filters
in force — the two halves are never checked separately, because a gap between
them IS the bug.

The second half is the trap that a naive fix walks into: counting a facet WITH
its own value applied. Picking 🌫 Noise would then print 0 next to 🔍 Blurry,
🎞 Flat and every neighbour, and there would be no way back to them without
clearing the whole filter. That is a worse bug than the one being fixed, so it
gets its own test.
"""
from PIL import ImageFilter

from test_image_bank import _mkbank, checkerboard, flat, noisy


def _scan(client, bank_id):
    r = client.post(f'/api/bank/{bank_id}/scan', json={})
    assert r.status_code in (200, 202), r.get_json()


def _payload(client, bank_id):
    r = client.get(f'/api/bank/{bank_id}')
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def _facets(client, bank_id, **filters):
    r = client.get(f'/api/bank/{bank_id}/facets', query_string=filters)
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def _shown(client, bank_id, **filters):
    """How many images the grid REALLY returns for that filter — the number the
    chip is claiming to predict."""
    r = client.get(f'/api/bank/{bank_id}/images',
                   query_string={**filters, 'limit': '1'})
    assert r.status_code == 200, r.get_json()
    return r.get_json()['total']


def _ids(client, bank_id, **filters):
    r = client.get(f'/api/bank/{bank_id}/images', query_string=filters)
    assert r.status_code == 200, r.get_json()
    return [im['id'] for im in r.get_json()['images']]


def _set_status(client, bank_id, ids, status):
    r = client.post(f'/api/bank/{bank_id}/images/status',
                    json={'ids': ids, 'status': status})
    assert r.status_code == 200, r.get_json()


FLAGS = ('blur', 'noise', 'uniform', 'small', 'soft_detail', 'bars',
         'unreadable')


def _mixed_bank(client, tmp_path):
    """Blurry + flat + noisy + tiny + a big one: several flags and two
    resolution tiers hold something after one scan, whatever the shipped
    thresholds are."""
    return _mkbank(client, tmp_path, {
        'sharp.png': checkerboard(),
        'soft1.png': checkerboard().filter(ImageFilter.GaussianBlur(6)),
        'soft2.png': checkerboard().filter(ImageFilter.GaussianBlur(8)),
        'soft3.png': checkerboard().filter(ImageFilter.GaussianBlur(7)),
        'flat1.png': flat(),
        'flat2.png': flat(value=200),
        'noise.png': noisy(),
        'noise2.png': noisy(seed=11),
        'tiny.png': checkerboard(size=32, cell=2),
        'big.png': noisy(size=600, seed=3),
    })


def _reject_half(client, bank_id):
    """Decide on part of the bank, so status and the flags stop agreeing. Every
    other test in this file starts from a bank where a filter can actually
    CHANGE a number — a fresh bank hides this defect completely."""
    ids = _ids(client, bank_id)
    assert len(ids) >= 6
    _set_status(client, bank_id, ids[:len(ids) // 2], 'reject')
    _set_status(client, bank_id, ids[len(ids) // 2:len(ids) // 2 + 2], 'keep')


def test_with_no_filter_the_chips_still_describe_the_whole_bank(client, tmp_path):
    """The unfiltered answer must not move: 'All' selected IS the bank-wide
    question the payload has always answered, and the workspace keeps reading
    the payload there."""
    bank_id, _src = _mixed_bank(client, tmp_path)
    _scan(client, bank_id)
    payload = _payload(client, bank_id)
    facets = _facets(client, bank_id)

    assert facets['flags'] == payload['flags']
    assert facets['res_buckets'] == payload['res_buckets']
    assert facets['framing'] == payload['framing']
    assert facets['origins'] == payload['origins']
    assert facets['mediums'] == payload['mediums']
    assert facets['angles'] == payload['angles']
    for k in ('total', 'pending', 'keep', 'reject'):
        assert facets['counts'][k] == payload['counts'][k], k


def test_every_flag_chip_counts_what_it_would_open(client, tmp_path):
    """The one assertion that matters, made per flag under a real filter: the
    printed number against the page the click returns."""
    bank_id, _src = _mixed_bank(client, tmp_path)
    _scan(client, bank_id)
    _reject_half(client, bank_id)

    counted = _facets(client, bank_id, status='reject')['flags']
    for flag in FLAGS:
        assert counted[flag] == _shown(client, bank_id, status='reject',
                                       flag=flag), flag

    # ...and the probe discriminates: at least one of those numbers really is
    # smaller than the bank-wide one, otherwise this file proves nothing.
    wide = _payload(client, bank_id)['flags']
    assert any(counted[f] < wide[f] for f in FLAGS if wide[f]), (counted, wide)


def test_picking_one_flag_does_not_zero_its_neighbours(client, tmp_path):
    """The counting rule, pinned: a facet is measured with every OTHER filter
    applied and its OWN value lifted. Get this wrong and choosing 🌫 Noise
    prints 0 on every sibling chip, and the user can never change their mind
    without resetting the whole filter."""
    bank_id, _src = _mixed_bank(client, tmp_path)
    _scan(client, bank_id)
    _reject_half(client, bank_id)

    free = _facets(client, bank_id, status='reject')['flags']
    held = _facets(client, bank_id, status='reject', flag='blur')['flags']
    assert held == free, 'the flag row must not narrow itself'
    assert sum(free[f] for f in FLAGS) > 0, 'the fixture must flag something'
    # And each sibling still predicts its own page — the click swaps the flag,
    # it does not add to it.
    for flag in FLAGS:
        assert held[flag] == _shown(client, bank_id, status='reject',
                                    flag=flag), flag


def test_the_status_row_follows_the_flag_in_force(client, tmp_path):
    """Symmetry: status is a facet like any other, so ✓ Kept / ✕ Rejected /
    Undecided are counted inside the active flag, and 'total' is the size of the
    filtered pool — which is what the 'All' chip opens on."""
    bank_id, _src = _mixed_bank(client, tmp_path)
    _scan(client, bank_id)
    _reject_half(client, bank_id)

    counts = _facets(client, bank_id, flag='blur')['counts']
    assert counts['total'] == _shown(client, bank_id, flag='blur')
    for status in ('pending', 'keep', 'reject'):
        assert counts[status] == _shown(client, bank_id, flag='blur',
                                        status=status), status
    wide = _payload(client, bank_id)['counts']
    assert counts['total'] < wide['total'], 'the flag must really narrow it'


def test_resolution_tiers_follow_the_other_filters(client, tmp_path):
    """A second facet family, to prove the rule is the endpoint's and not one
    flag row's."""
    bank_id, _src = _mixed_bank(client, tmp_path)
    _scan(client, bank_id)
    _reject_half(client, bank_id)

    wide = _payload(client, bank_id)['res_buckets']
    tiers = _facets(client, bank_id, status='reject')['res_buckets']
    for tier, n in tiers.items():
        assert n == _shown(client, bank_id, status='reject',
                           res_bucket=tier), tier
    assert any(tiers[t] < wide[t] for t in wide if wide[t]), (tiers, wide)


def test_a_search_term_narrows_the_chips_too(client, tmp_path):
    """The text lane is a filter like the chips, and was the easiest one to
    forget: it lives in a box, not in the chip rows."""
    bank_id, _src = _mixed_bank(client, tmp_path)
    _scan(client, bank_id)

    found = _facets(client, bank_id, search='soft')
    assert found['counts']['total'] == _shown(client, bank_id, search='soft')
    assert found['counts']['total'] == 3, 'three files are named soft*'
    for flag in FLAGS:
        assert found['flags'][flag] == _shown(client, bank_id, search='soft',
                                              flag=flag), flag
    wide = _payload(client, bank_id)['flags']
    assert any(found['flags'][f] < wide[f] for f in FLAGS if wide[f])


def test_an_unknown_bank_is_a_404_not_an_empty_chip_row(client):
    r = client.get('/api/bank/999999/facets')
    assert r.status_code == 404
