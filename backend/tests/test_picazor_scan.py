"""Picazor scan() : distinguer un résultat vide LÉGITIME (listing/profil sans
média, page HTML chargée sans incident) d'un vrai échec de parsing (finding #3).

Tout est mocké (`_request_html`) : aucun appel réseau / curl_cffi."""
from types import SimpleNamespace

from app.scrape.sources import picazor


def test_an_empty_listing_is_a_result_not_an_error(monkeypatch):
    monkeypatch.setattr(picazor, '_request_html', lambda url: ('<html>no tiles here</html>', None))
    validation = SimpleNamespace(original_url='https://picazor.com/fr/videos/week',
                                 value='', url_type=None)

    items, err = picazor.scan(validation)

    assert items is None
    assert getattr(err, 'kind', None) == 'empty'


def test_an_empty_profile_is_a_result_not_an_error(monkeypatch):
    monkeypatch.setattr(picazor, '_request_html', lambda url: ('<html>no tiles here</html>', None))
    validation = SimpleNamespace(original_url='https://picazor.com/fr/nobody',
                                 value='nobody', url_type=None)

    items, err = picazor.scan(validation)

    assert items is None
    assert getattr(err, 'kind', None) == 'empty'


def test_a_detail_page_with_no_extractable_media_stays_a_real_failure(monkeypatch):
    """La page de détail décrit toujours un média précis — l'absence de match
    régex signale un layout changé (échec de parsing), pas une page vide.
    Volontairement PAS convertie en kind='empty' (cf. rapport)."""
    monkeypatch.setattr(picazor, '_request_html', lambda url: ('<html>layout changed</html>', None))
    validation = SimpleNamespace(original_url='https://picazor.com/fr/someone/42',
                                 value='someone', url_type=None)

    items, err = picazor.scan(validation)

    assert items is None
    assert getattr(err, 'kind', None) is None
    assert 'detail page' in err


def test_a_populated_listing_still_returns_its_items(monkeypatch):
    html = '"/uploads/a/b/300px_x.jpg"'
    monkeypatch.setattr(picazor, '_request_html', lambda url: (html, None))
    validation = SimpleNamespace(original_url='https://picazor.com/fr/models',
                                 value='', url_type=None)

    items, err = picazor.scan(validation)

    assert err is None
    assert len(items) == 1


def test_profile_scan_marks_a_mid_pagination_failure_as_partial(monkeypatch):
    """A profile has 3 pages worth of media (per the /fr/{creator}/{index} links
    on page 1). Page 1 loads fine and yields an item; page 2 gets blocked by
    Cloudflare mid-pagination. Before this fix, `scan()` degraded gracefully by
    just `break`ing out of the loop and returning `all_items[:MAX_ITEMS], None`
    — a plain list with no truncation signal, presenting a harvest cut short by
    a real HTTP failure as a complete profile. Patches at the HTTP-layer
    boundary (`_request_html`, same function the real pagination loop calls for
    every page) so the actual loop logic runs, not a stubbed-out scan()."""
    html_page1 = '"/uploads/a/b/300px_x.jpg"' + '<a href="/fr/someone/50">50</a>'

    def fake_request_html(url):
        if url.endswith('/page/2'):
            return None, "Picazor (Cloudflare) blocked access."
        return html_page1, None

    monkeypatch.setattr(picazor, '_request_html', fake_request_html)
    validation = SimpleNamespace(original_url='https://picazor.com/fr/someone',
                                 value='someone', url_type=None)

    items, err = picazor.scan(validation)

    assert err is None
    assert len(items) == 1
    assert getattr(items, 'partial', False) is True


def test_profile_scan_marks_hitting_max_items_as_partial(monkeypatch):
    """A profile that has more media than MAX_ITEMS (300) must not look complete
    just because every page loaded fine. Patches `_request_html` (the same
    function the real pagination loop calls) so the real loop really counts up
    to the cap, instead of stubbing scan() itself."""
    monkeypatch.setattr(picazor, 'MAX_ITEMS', 300)

    def fake_request_html(url):
        if '/page/' in url:
            page_num = int(url.rsplit('/page/', 1)[1])
        else:
            page_num = 1
        # 24 items/page, page-unique names so nothing dedupes across pages.
        tiles = ''.join(f'"/uploads/a/b/300px_p{page_num}_{i}.jpg"' for i in range(24))
        # A huge max_index => total_pages way beyond MAX_PAGES, so the cap that
        # fires here is MAX_ITEMS, not the MAX_PAGES exhaustion tested below.
        links = '<a href="/fr/someone/1000">1000</a>'
        return tiles + links, None

    monkeypatch.setattr(picazor, '_request_html', fake_request_html)
    validation = SimpleNamespace(original_url='https://picazor.com/fr/someone',
                                 value='someone', url_type=None)

    items, err = picazor.scan(validation)

    assert err is None
    assert len(items) == 300
    assert getattr(items, 'partial', False) is True


def test_profile_scan_marks_exhausting_max_pages_as_partial(monkeypatch):
    """A profile whose pages never fill up fast enough to hit MAX_ITEMS, but
    which has more real pages than the MAX_PAGES time guard allows to visit,
    must still be flagged partial — the MAX_PAGES cap is a truncation too, not
    just MAX_ITEMS. Before this fix, exhausting the `for _ in range(MAX_PAGES)`
    loop without an explicit break left `interrupted` False and the response
    looked complete."""
    monkeypatch.setattr(picazor, 'MAX_ITEMS', 300)

    def fake_request_html(url):
        if '/page/' in url:
            page_num = int(url.rsplit('/page/', 1)[1])
        else:
            page_num = 1
        # Only 10 unique items/page (14 pages * 10 = 140, well under MAX_ITEMS)
        # but a max_index implying far more real pages than MAX_PAGES (14).
        tiles = ''.join(f'"/uploads/a/b/300px_p{page_num}_{i}.jpg"' for i in range(10))
        links = '<a href="/fr/someone/2000">2000</a>'
        return tiles + links, None

    monkeypatch.setattr(picazor, '_request_html', fake_request_html)
    validation = SimpleNamespace(original_url='https://picazor.com/fr/someone',
                                 value='someone', url_type=None)

    items, err = picazor.scan(validation)

    assert err is None
    assert len(items) == 140
    assert getattr(items, 'partial', False) is True


def test_profile_scan_of_a_complete_two_page_profile_is_not_partial(monkeypatch):
    """Negative counterpart of the truncation tests above: a profile that
    fully fits in 2 pages (well under MAX_ITEMS and MAX_PAGES), with every
    page loading fine and pagination ending naturally (`page > total_pages`),
    must NOT be flagged partial. Patches `_request_html` (same boundary as
    the other tests here) so the real pagination loop runs to its natural
    end instead of a stubbed scan()."""
    def fake_request_html(url):
        page_num = 2 if url.endswith('/page/2') else 1
        tiles = ''.join(f'"/uploads/a/b/300px_p{page_num}_{i}.jpg"' for i in range(10))
        # max_index=34 => total_pages = ceil(34/24) = 2, so pagination stops
        # naturally after page 2 (page=3 > total_pages=2).
        links = '<a href="/fr/someone/34">34</a>'
        return tiles + links, None

    monkeypatch.setattr(picazor, '_request_html', fake_request_html)
    validation = SimpleNamespace(original_url='https://picazor.com/fr/someone',
                                 value='someone', url_type=None)

    items, err = picazor.scan(validation)

    assert err is None
    assert len(items) == 20
    assert getattr(items, 'partial', False) is False


def test_listing_scan_marks_overflowing_max_items_as_partial(monkeypatch):
    """A single listing page (videos/week, models/...) whose parse yields more
    tiles than MAX_ITEMS in one shot must also be flagged partial, not silently
    sliced to `[:MAX_ITEMS]` with no signal."""
    monkeypatch.setattr(picazor, 'MAX_ITEMS', 5)
    html = ''.join(f'"/uploads/a/b/300px_x{i}.jpg"' for i in range(8))
    monkeypatch.setattr(picazor, '_request_html', lambda url: (html, None))
    validation = SimpleNamespace(original_url='https://picazor.com/fr/models',
                                 value='', url_type=None)

    items, err = picazor.scan(validation)

    assert err is None
    assert len(items) == 5
    assert getattr(items, 'partial', False) is True
