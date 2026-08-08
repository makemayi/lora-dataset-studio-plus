"""🧹 Auto-reject: the number the popover SHOWS is the number the button MOVES.

The defect this file pins, reported as "auto-reject doesn't work" and measured
on a real 99 000-image bank: the popover printed its per-flag count from the
payload's ``flags`` map, which counts every image carrying the flag regardless
of status. ``apply_flags`` only ever touches ``status='pending'`` — that is its
contract, and it is the correct one (a decision, manual or from an earlier pass,
is never re-flipped). So the second time the user ran it the panel offered
"5 930 flagged", the click rejected 0, and the only reasonable conclusion was
that the feature was broken.

The trap is that the two numbers AGREE on a fresh bank, where everything is
pending. They diverge exactly once the tool has been used — i.e. once the user
has started trusting it.

So these tests never assert the displayed count and the rejected count
separately: they assert the displayed count AGAINST a real ``apply_flags``
played behind it. Asserting the two halves independently is how the gap would be
rebuilt.

They also pin the second, opposite reading of a zero: an image no scan ever
measured is invisible to every quality flag (``_flag_filter`` gates them all on
``quality_state == 'ok'``), so "0 blurry" on an unscanned bank is not good news.
The payload names that pile, and names what a 🔎 Scan would really reach.
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


def _apply(client, bank_id, flags):
    """Run the real pass and return how many images it actually rejected."""
    r = client.post(f'/api/bank/{bank_id}/apply-flags', json={'flags': flags})
    assert r.status_code == 200, r.get_json()
    return sum(r.get_json()['rejected'].values())


def _ids(client, bank_id, **facets):
    r = client.get(f'/api/bank/{bank_id}/images', query_string=facets)
    assert r.status_code == 200, r.get_json()
    return [im['id'] for im in r.get_json()['images']]


def _set_status(client, bank_id, ids, status):
    r = client.post(f'/api/bank/{bank_id}/images/status',
                    json={'ids': ids, 'status': status})
    assert r.status_code == 200, r.get_json()


def _mixed_bank(client, tmp_path):
    """Blurry + flat + noisy + one tiny shot: enough for several flags to hold
    something after one scan, whatever the shipped thresholds are."""
    return _mkbank(client, tmp_path, {
        'sharp.png': checkerboard(),
        'soft1.png': checkerboard().filter(ImageFilter.GaussianBlur(6)),
        'soft2.png': checkerboard().filter(ImageFilter.GaussianBlur(8)),
        'flat1.png': flat(),
        'flat2.png': flat(value=200),
        'noise.png': noisy(),
        'tiny.png': checkerboard(size=32, cell=2),
    })


def test_the_shown_count_is_what_the_click_rejects(client, tmp_path):
    """The one assertion that matters, made per flag: read the number the
    popover would print, run the pass, compare. Not the two separately."""
    bank_id, _src = _mixed_bank(client, tmp_path)
    _scan(client, bank_id)

    for flag in ('blur', 'uniform', 'small', 'noise'):
        shown = _payload(client, bank_id)['flags_actionable'][flag]
        assert _apply(client, bank_id, [flag]) == shown, flag


def test_a_second_pass_offers_zero_instead_of_the_old_number(client, tmp_path):
    """The reported bug, reproduced small. After a first auto-reject the images
    still CARRY the flag — so the facet count is unchanged and the chip still
    finds them — but there is nothing left for the button to do, and the popover
    now says 0 instead of advertising the facet's number."""
    bank_id, _src = _mixed_bank(client, tmp_path)
    _scan(client, bank_id)

    before = _payload(client, bank_id)
    flagged = before['flags']['blur']
    actionable = before['flags_actionable']['blur']
    assert flagged > 0, 'the fixture must actually produce blurry images'
    # Nothing decided yet: this is the case where the old counter looked right.
    assert actionable == flagged

    assert _apply(client, bank_id, ['blur']) == actionable

    after = _payload(client, bank_id)
    # The facet keeps its number on purpose — clicking 🌫 Blurry must still show
    # the images it rejected, which is why this could not be a one-line rename.
    assert after['flags']['blur'] == flagged
    # ...and the button's number collapses to the truth.
    assert after['flags_actionable']['blur'] == 0
    assert _apply(client, bank_id, ['blur']) == 0


def test_a_manual_keep_is_never_counted_as_a_candidate(client, tmp_path):
    """apply_flags leaves ✓ Kept alone; the count must leave it alone too, or
    the panel would promise to undo a decision the user made by hand."""
    bank_id, _src = _mixed_bank(client, tmp_path)
    _scan(client, bank_id)
    blurry = _ids(client, bank_id, flag='blur')
    assert len(blurry) >= 2, 'need a blurry image to keep and one to reject'
    _set_status(client, bank_id, blurry[:1], 'keep')

    shown = _payload(client, bank_id)['flags_actionable']['blur']
    assert _apply(client, bank_id, ['blur']) == shown
    # The kept one survived the pass, and was never advertised as a candidate.
    assert _payload(client, bank_id)['counts']['keep'] == 1


def test_never_scanned_images_are_counted_and_named(client, tmp_path):
    """A quality flag can only match a scanned image, so the pile no scan ever
    reached is a blind spot, not a clean bill of health. The payload says how
    big it is, and how much of it 🔎 Scan would actually pick up (rejected rows
    are out of the scan pool)."""
    bank_id, _src = _mixed_bank(client, tmp_path)

    fresh = _payload(client, bank_id)['counts']
    assert fresh['scanned'] == 0
    assert fresh['unscanned'] == fresh['total']
    assert fresh['unscanned_scannable'] == fresh['total']
    # ...and every quality flag reads 0 while knowing nothing about the images.
    assert _payload(client, bank_id)['flags_actionable']['blur'] == 0

    # Reject one by hand before it is ever measured: it stays unscanned, but a
    # 🔎 Scan will skip it, so the two numbers must part ways by exactly one.
    _set_status(client, bank_id, _ids(client, bank_id)[:1], 'reject')

    counts = _payload(client, bank_id)['counts']
    assert counts['unscanned'] == counts['total']
    assert counts['unscanned_scannable'] == counts['total'] - 1

    # The scan closes the blind spot for everything it is allowed to touch.
    _scan(client, bank_id)
    counts = _payload(client, bank_id)['counts']
    assert counts['unscanned'] == 1
    assert counts['unscanned_scannable'] == 0


def test_the_preview_endpoint_answers_both_questions_too(client, tmp_path):
    """The threshold panel reads the same producer, so its map has to carry the
    same keys — otherwise the two screens could drift apart flag by flag."""
    bank_id, _src = _mixed_bank(client, tmp_path)
    _scan(client, bank_id)
    r = client.post(f'/api/bank/{bank_id}/flag-preview', json={})
    assert r.status_code == 200, r.get_json()
    preview = r.get_json()
    payload = _payload(client, bank_id)
    assert set(preview['flags_actionable']) == set(payload['flags_actionable'])
    assert preview['flags_actionable'] == payload['flags_actionable']
