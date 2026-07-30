"""🕸 Scrape → BANK: the scraper's second destination.

Until now the scraper had exactly ONE outlet, `scrape_import_urls` (straight into
a dataset), and that path applies training-grade filters BEFORE anything is
stored: side < 768 px and ratio > 3:1 are dropped, and a dHash pass removes
perceptual near-duplicates. That is the right call for a dataset — and the wrong
one for a bank, whose whole job is to be shown the raw pile and decide. These
tests pin the difference:

* the bank path stores what it downloaded (a small image is a bank decision, not
  a download-time rejection),
* it NEVER perceptually de-duplicates — the bank's own scan pass owns that, and
  running a second rule here would mean two different definitions of "duplicate"
  on the same images,
* a second scrape on the same bank APPENDS (resume) instead of replacing,
* byte-identical bytes re-downloaded land on the same file, so a resume is
  idempotent without that being a curation decision,
* rows are inventoried through the existing folder walk (refresh_bank), not a
  third hand-rolled insert path,
* and the historical dataset outlet still behaves exactly as before.
"""
import io
import os
import struct
import zlib
from unittest.mock import patch

import pytest
from PIL import Image

from app.config import LOCAL_USER
from app.services import face_dataset_service as fsvc
from app.services import image_bank_service as banks


def _img_bytes(w=1280, h=960, fmt='JPEG', grad=None, shade=(120, 40, 40)):
    if grad:
        ramp = list(range(0, 256, 32))
        if grad == 'rtl':
            ramp = ramp[::-1]
        small = Image.new('L', (8, 8))
        small.putdata([ramp[x] for _ in range(8) for x in range(8)])
        im = small.resize((w, h), Image.BILINEAR).convert('RGB')
    else:
        im = Image.new('RGB', (w, h), shade)
    b = io.BytesIO()
    im.save(b, fmt)
    return b.getvalue()


def _compact_png_header(width, height):
    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))
    return (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
            + chunk(b'IEND', b''))


def _item(url):
    return {'url': url, 'title': ''}


def _fake_downloader(by_url):
    def _dl(item):
        data = by_url.get(item['url'])
        return ('ok', data) if data is not None else ('errors', None)
    return _dl


def _files(bank):
    return sorted(os.listdir(bank.source_path))


# --- new bank ---------------------------------------------------------------
def test_scrape_creates_a_bank_and_inventories_it(app):
    with app.app_context():
        by_url = {'http://x/a.jpg': _img_bytes(grad='ltr'),
                  'http://x/b.jpg': _img_bytes(grad='rtl')}
        with patch.object(banks, '_download_scrape_item', _fake_downloader(by_url)):
            res = banks.scrape_import_to_bank(
                LOCAL_USER, [_item('http://x/a.jpg'), _item('http://x/b.jpg')],
                name='Scraped pile')
        assert res['created'] is True
        assert res['saved'] == 2 and res['added'] == 2
        bank = banks.get_bank(LOCAL_USER, res['bank_id'])
        assert bank is not None and bank.name == 'Scraped pile'
        assert len(_files(bank)) == 2
        # inventoried through the normal folder walk, like any other bank
        from app.models import BankImage
        assert BankImage.query.filter_by(bank_id=bank.id).count() == 2


def test_small_and_extreme_images_reach_the_bank_instead_of_being_dropped(app):
    """The dataset outlet drops these two before storing them. The bank must NOT:
    'too small' and 'panorama' are triage verdicts the user adjusts with the 🎚
    thresholds, and an image the bank never received cannot be reviewed."""
    with app.app_context():
        tiny = _img_bytes(320, 240, grad='ltr')
        wide = _img_bytes(2000, 200, grad='rtl')
        assert min(320, 240) < fsvc.SCRAPE_IMPORT_MIN_SIDE
        assert 2000 > fsvc.SCRAPE_IMPORT_MAX_RATIO * 200
        by_url = {'http://x/tiny.jpg': tiny, 'http://x/wide.jpg': wide}
        with patch.object(banks, '_download_scrape_item', _fake_downloader(by_url)):
            res = banks.scrape_import_to_bank(
                LOCAL_USER, [_item('http://x/tiny.jpg'), _item('http://x/wide.jpg')],
                name='Raw pile')
        assert res['saved'] == 2, res
        bank = banks.get_bank(LOCAL_USER, res['bank_id'])
        assert len(_files(bank)) == 2


def test_bank_path_never_runs_a_second_perceptual_dedup(app):
    """Two visually near-identical (but not byte-identical) shots: the DATASET
    outlet keeps one, the bank must keep BOTH and leave the verdict to its own
    duplicate-group pass. One pile, one definition of 'duplicate'."""
    with app.app_context():
        a = _img_bytes(grad='ltr', fmt='JPEG')
        b = _img_bytes(grad='ltr', fmt='PNG')       # same picture, different bytes
        assert a != b
        with Image.open(io.BytesIO(a)) as ia, Image.open(io.BytesIO(b)) as ib:
            assert fsvc._hamming(fsvc._dhash(ia), fsvc._dhash(ib)) \
                <= fsvc.SCRAPE_DHASH_MAX_DISTANCE   # the dataset rule WOULD drop one

        by_url = {'http://x/a.jpg': a, 'http://x/b.png': b}
        with patch.object(banks, '_download_scrape_item', _fake_downloader(by_url)):
            res = banks.scrape_import_to_bank(
                LOCAL_USER, [_item('http://x/a.jpg'), _item('http://x/b.png')],
                name='Near dupes')
        assert res['saved'] == 2
        assert 'duplicates' not in res['skipped']
        bank = banks.get_bank(LOCAL_USER, res['bank_id'])
        assert len(_files(bank)) == 2


# --- resume -----------------------------------------------------------------
def test_second_scrape_on_the_same_bank_appends(app):
    with app.app_context():
        first = {'http://x/a.jpg': _img_bytes(grad='ltr')}
        with patch.object(banks, '_download_scrape_item', _fake_downloader(first)):
            res1 = banks.scrape_import_to_bank(
                LOCAL_USER, [_item('http://x/a.jpg')], name='Growing')
        bank_id = res1['bank_id']
        second = {'http://x/b.jpg': _img_bytes(grad='rtl')}
        with patch.object(banks, '_download_scrape_item', _fake_downloader(second)):
            res2 = banks.scrape_import_to_bank(
                LOCAL_USER, [_item('http://x/b.jpg')], bank_id=bank_id)
        assert res2['created'] is False and res2['bank_id'] == bank_id
        assert res2['saved'] == 1 and res2['added'] == 1
        bank = banks.get_bank(LOCAL_USER, bank_id)
        assert len(_files(bank)) == 2
        from app.models import BankImage
        assert BankImage.query.filter_by(bank_id=bank_id).count() == 2


def test_re_downloading_the_same_bytes_is_idempotent_not_a_dedup_verdict(app):
    """Identical bytes = the same FILE, which is file identity, not curation: it
    lands on the same name and is reported as `already_there`, never as a
    'duplicate' skip (that word belongs to the bank's own pass)."""
    with app.app_context():
        blob = _img_bytes(grad='ltr')
        by_url = {'http://x/a.jpg': blob}
        with patch.object(banks, '_download_scrape_item', _fake_downloader(by_url)):
            res1 = banks.scrape_import_to_bank(
                LOCAL_USER, [_item('http://x/a.jpg')], name='Same twice')
            res2 = banks.scrape_import_to_bank(
                LOCAL_USER, [_item('http://x/a.jpg')], bank_id=res1['bank_id'])
        assert res2['saved'] == 0 and res2['already_there'] == 1
        assert 'duplicates' not in res2['skipped']
        bank = banks.get_bank(LOCAL_USER, res1['bank_id'])
        assert len(_files(bank)) == 1


def test_a_busy_bank_refuses_the_scrape(app):
    with app.app_context():
        with patch.object(banks, '_download_scrape_item', _fake_downloader({})):
            res = banks.scrape_import_to_bank(
                LOCAL_USER, [_item('http://x/a.jpg')], name='Busy')
        bank_id = res['bank_id']
        with patch.object(banks.bank_jobs, 'running', lambda bid: bid == bank_id), \
             pytest.raises(banks.bank_jobs.BankJobBusy):
            banks.scrape_import_to_bank(
                LOCAL_USER, [_item('http://x/b.jpg')], bank_id=bank_id)


# --- validation -------------------------------------------------------------
def test_scrape_blob_gate_rejects_compact_pixel_bombs_but_keeps_valid_bmp():
    """Bank scrape uses the same static/header budget as Dataset import."""
    assert banks._scrape_blob_name(_compact_png_header(9000, 9000)) is None
    name = banks._scrape_blob_name(_img_bytes(32, 20, fmt='BMP'))
    assert name is not None and name.endswith('.bmp')


def test_a_failed_download_is_counted_and_never_stored(app):
    with app.app_context():
        by_url = {'http://x/a.jpg': _img_bytes(grad='ltr'), 'http://x/dead.jpg': None}
        with patch.object(banks, '_download_scrape_item', _fake_downloader(by_url)):
            res = banks.scrape_import_to_bank(
                LOCAL_USER, [_item('http://x/a.jpg'), _item('http://x/dead.jpg')],
                name='Half dead')
        assert res['saved'] == 1 and res['skipped'].get('errors') == 1


def test_name_is_required_for_a_new_bank(app):
    with app.app_context():
        with pytest.raises(ValueError):
            banks.scrape_import_to_bank(LOCAL_USER, [_item('http://x/a.jpg')], name='  ')


def test_unknown_bank_id_is_rejected(app):
    with app.app_context():
        with pytest.raises(ValueError):
            banks.scrape_import_to_bank(LOCAL_USER, [_item('http://x/a.jpg')],
                                        bank_id=9999)


def test_batch_cap_is_the_same_number_as_the_dataset_outlet(app):
    with app.app_context():
        items = [_item(f'http://x/{i}.jpg') for i in range(fsvc.SCRAPE_IMPORT_MAX + 1)]
        with pytest.raises(ValueError):
            banks.scrape_import_to_bank(LOCAL_USER, items, name='Too many')


# --- route ------------------------------------------------------------------
def test_route_creates_then_resumes(app, client):
    by_url = {'http://x/a.jpg': _img_bytes(grad='ltr'),
              'http://x/b.jpg': _img_bytes(grad='rtl')}
    with patch.object(banks, '_download_scrape_item', _fake_downloader(by_url)):
        r = client.post('/api/bank/scrape-import',
                        json={'items': [_item('http://x/a.jpg')], 'name': 'Via HTTP'})
        assert r.status_code == 200, r.get_json()
        body = r.get_json()
        assert body['ok'] and body['created'] and body['saved'] == 1
        r2 = client.post('/api/bank/scrape-import',
                         json={'items': [_item('http://x/b.jpg')],
                               'bank_id': body['bank_id']})
        assert r2.status_code == 200
        assert r2.get_json()['created'] is False and r2.get_json()['saved'] == 1


def test_route_rejects_an_empty_selection(app, client):
    r = client.post('/api/bank/scrape-import', json={'items': [], 'name': 'x'})
    assert r.status_code == 400


def test_route_reports_a_busy_bank_as_409(app, client):
    with patch.object(banks, '_download_scrape_item', _fake_downloader({})):
        r = client.post('/api/bank/scrape-import',
                        json={'items': [_item('http://x/a.jpg')], 'name': 'Busy HTTP'})
    bank_id = r.get_json()['bank_id']
    with patch.object(banks.bank_jobs, 'running', lambda bid: bid == bank_id):
        r2 = client.post('/api/bank/scrape-import',
                         json={'items': [_item('http://x/b.jpg')], 'bank_id': bank_id})
    assert r2.status_code == 409 and r2.get_json().get('busy_kind')


# --- anti-regression: the dataset outlet is untouched ------------------------
def test_dataset_scrape_import_still_filters_and_dedupes(app):
    """The historical path keeps its training-grade gate: small dropped, near
    duplicate dropped. Adding a second destination must not soften the first."""
    with app.app_context():
        c = fsvc.create_dataset(LOCAL_USER, 'CIM', 'cim_act', kind='concept',
                                concept_desc='an ice cream cone')
        by_url = {'http://x/a.jpg': _img_bytes(grad='ltr', fmt='JPEG'),
                  'http://x/dupe.png': _img_bytes(grad='ltr', fmt='PNG'),
                  'http://x/tiny.jpg': _img_bytes(320, 240, grad='rtl')}
        with patch.object(fsvc, '_download_scrape_item', _fake_downloader(by_url)):
            res = fsvc.scrape_import_urls(
                LOCAL_USER, c.id, [_item('http://x/a.jpg'), _item('http://x/dupe.png'),
                                   _item('http://x/tiny.jpg')])
        assert res['imported'] == 1
        assert res['skipped']['duplicates'] == 1
        assert res['skipped']['low_res'] == 1
