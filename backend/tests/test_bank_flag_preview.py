"""🎚 Bank threshold preview: how many images a CANDIDATE threshold WOULD flag.

This is what makes the Bank's threshold panel a decision instead of a guess —
the user sees "1 240 → 3 019 images flagged" before saving anything. It works
only because the quality scan persists RAW scores and every verdict is
recomputed at read time, so a different threshold is a different COUNT over the
same rows, not a rescan.

What must hold, and what these tests pin:
  * the preview MOVES with the candidate value (otherwise it is decoration);
  * it CHANGES NOTHING — the saved config and the live payload are identical
    afterwards, because the whole promise is "before you save";
  * junk in the body degrades to the saved thresholds instead of a 400, since it
    fires while the user is mid-keystroke;
  * a value that is not the app's is ignored, not merged into the config.
"""
from PIL import ImageFilter

from test_image_bank import _mkbank, checkerboard, flat


def _scan(client, bank_id):
    r = client.post(f'/api/bank/{bank_id}/scan', json={})
    assert r.status_code in (200, 202), r.get_json()


def _preview(client, bank_id, thresholds=None, body=None):
    payload = body if body is not None else (
        {} if thresholds is None else {'thresholds': thresholds})
    r = client.post(f'/api/bank/{bank_id}/flag-preview', json=payload)
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def test_preview_follows_the_candidate_threshold(client, tmp_path):
    """A sharpness floor above everything flags every scanned image; a floor of
    zero flags none. Same rows, same scan — only the number moved."""
    bank_id, _src = _mkbank(client, tmp_path, {
        'sharp.png': checkerboard(),
        'soft.png': checkerboard().filter(ImageFilter.GaussianBlur(6)),
    })
    _scan(client, bank_id)

    none_flagged = _preview(client, bank_id, {'sharpness_min': 0})
    assert none_flagged['flags']['blur'] == 0
    all_flagged = _preview(client, bank_id, {'sharpness_min': 10 ** 9})
    assert all_flagged['flags']['blur'] == 2
    # The echoed thresholds carry the candidate, so the client can tell WHICH
    # numbers produced the counts it is displaying.
    assert all_flagged['thresholds']['sharpness_min'] == 10 ** 9
    assert all_flagged['total'] == 2


def test_preview_saves_nothing(client, tmp_path):
    """The point of a preview is that it is not a save."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat()})
    _scan(client, bank_id)
    before_cfg = client.get('/api/settings').get_json()['config']['bank']
    before_flags = client.get(f'/api/bank/{bank_id}').get_json()['flags']

    _preview(client, bank_id, {'sharpness_min': 10 ** 9, 'min_side': 99999})

    after_cfg = client.get('/api/settings').get_json()['config']['bank']
    after_flags = client.get(f'/api/bank/{bank_id}').get_json()['flags']
    assert after_cfg == before_cfg
    assert after_flags == before_flags


def test_preview_degrades_instead_of_erroring(client, tmp_path):
    """It fires on every keystroke: a half-typed value must read as 'no change
    yet', never as an error the user has to dismiss."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat()})
    _scan(client, bank_id)
    saved = _preview(client, bank_id)

    # No body, junk body, junk value, unknown key — all fall back to the saved
    # thresholds rather than 400.
    assert _preview(client, bank_id, body={})['flags'] == saved['flags']
    assert _preview(client, bank_id, body={'thresholds': 'nope'})['flags'] == saved['flags']
    assert _preview(client, bank_id, {'sharpness_min': 'abc'})['flags'] == saved['flags']
    assert _preview(client, bank_id, {'sharpness_min': None})['flags'] == saved['flags']
    # ...but a number that merely LOOKS half-typed is a number: '0.' is 0.0, and
    # honouring it is what makes the readout track the input character by
    # character instead of freezing on the last "clean" value.
    assert _preview(client, bank_id, {'sharpness_min': '0.'})['thresholds']['sharpness_min'] == 0.0
    assert _preview(client, bank_id, {'not_a_threshold': 3})['flags'] == saved['flags']
    # An unknown key is dropped, not smuggled into the echoed thresholds.
    assert 'not_a_threshold' not in _preview(
        client, bank_id, {'not_a_threshold': 3})['thresholds']


def test_preview_covers_every_flag_the_payload_counts(client, tmp_path):
    """The panel reads its 'before' numbers from this endpoint and its 'after'
    from the same one, so the two maps must have identical keys — otherwise a
    flag would silently lose its effect line."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat()})
    _scan(client, bank_id)
    assert (set(_preview(client, bank_id)['flags'])
            == set(client.get(f'/api/bank/{bank_id}').get_json()['flags']))


def test_preview_404s_on_an_unknown_bank(client):
    r = client.post('/api/bank/999999/flag-preview', json={})
    assert r.status_code == 404
