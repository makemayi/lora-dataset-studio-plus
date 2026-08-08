"""RedGifs scan() : un profil/niche légitimement sans vidéo est un résultat vide,
pas un échec (finding #3 — la règle « empty est honoré partout » n'était payée
que pour la famille gallery-dl ; RedGifs répondait 502 sur un profil vide).

Tout est mocké (client RedGifs) : aucun appel réseau."""
from types import SimpleNamespace

import requests

from app.scrape.sources import redgifs
from app.scrape.validators import URLType


def test_scan_returns_empty_not_error_for_a_profile_with_no_videos(monkeypatch):
    monkeypatch.setattr(redgifs.client, 'get_token', lambda: 'tok')
    monkeypatch.setattr(redgifs.client, 'iter_user', lambda username, state=None: iter([]))
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='nobody')

    items, err = redgifs.scan(validation)

    assert err is None
    assert items == []


def test_scan_returns_empty_not_error_for_a_niche_with_no_videos(monkeypatch):
    monkeypatch.setattr(redgifs.client, 'get_token', lambda: 'tok')
    monkeypatch.setattr(redgifs.client, 'iter_niche', lambda niche, state=None: iter([]))
    validation = SimpleNamespace(url_type=URLType.NICHE, value='empty-niche')

    items, err = redgifs.scan(validation)

    assert err is None
    assert items == []


def test_scan_still_returns_items_for_a_populated_profile(monkeypatch):
    monkeypatch.setattr(redgifs.client, 'get_token', lambda: 'tok')
    monkeypatch.setattr(redgifs.client, 'iter_user',
                        lambda username, state=None: iter([{'id': 'abc', 'urls': {}}]))
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='someone')

    items, err = redgifs.scan(validation)

    assert err is None
    assert len(items) == 1
    assert items[0]['url'] == 'https://www.redgifs.com/watch/abc'


def test_a_missing_single_video_stays_a_real_error_not_an_empty_result(monkeypatch):
    """VIDEO (média unique) : l'échec du lookup reste une vraie erreur, pas un
    « rien ici » — non touché par ce finding (cf. rapport)."""
    monkeypatch.setattr(redgifs.client, 'get_token', lambda: 'tok')
    monkeypatch.setattr(redgifs.client, 'get_single_video', lambda video_id: None)
    validation = SimpleNamespace(url_type=URLType.VIDEO, value='deadbeef')

    items, err = redgifs.scan(validation)

    assert items is None
    assert 'not found' in err


def test_scan_reports_a_failure_not_an_empty_success_when_profile_is_rate_limited(monkeypatch):
    """Probe from the reviewer: a 429 on the profile page must not report an
    empty success. Before the fix, `_iter_paged` swallowed EVERY HTTP error
    (429/403/5xx/timeout) and just `return`ed, ending the generator — a
    rate-limited profile then yielded zero items and `scan()` answered
    ([], None): HTTP 200, count 0, "No images found on this page." """
    def _rate_limited(username, state=None):
        raise redgifs.RedGifsAbort("RedGifs: HTTP 429.")
        yield  # pragma: no cover - unreachable, keeps this a generator function

    monkeypatch.setattr(redgifs.client, 'get_token', lambda: 'tok')
    monkeypatch.setattr(redgifs.client, 'iter_user', _rate_limited)
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='throttled')

    items, err = redgifs.scan(validation)

    assert items is None
    assert err is not None
    assert 'rate' in err.lower() or 'blocked' in err.lower()


def test_scan_reports_a_failure_not_an_empty_success_when_niche_is_rate_limited(monkeypatch):
    def _rate_limited(niche, state=None):
        raise redgifs.RedGifsAbort("RedGifs: HTTP 429.")
        yield  # pragma: no cover - unreachable, keeps this a generator function

    monkeypatch.setattr(redgifs.client, 'get_token', lambda: 'tok')
    monkeypatch.setattr(redgifs.client, 'iter_niche', _rate_limited)
    validation = SimpleNamespace(url_type=URLType.NICHE, value='throttled-niche')

    items, err = redgifs.scan(validation)

    assert items is None
    assert err is not None
    assert 'rate' in err.lower() or 'blocked' in err.lower()


def test_iter_paged_turns_an_http_429_into_a_failure_not_an_empty_result(monkeypatch):
    """Unlike the two tests above (which stub `iter_user`/`iter_niche` to raise
    `RedGifsAbort` directly and therefore never execute `_iter_paged` at all),
    this one patches at the HTTP layer `_iter_paged` actually calls —
    `client._get` — so the real `_iter_paged` body runs and must be the thing
    that turns a 429 into `RedGifsAbort`. A reviewer proved the four tests above
    passed even with the fix's three `raise RedGifsAbort` reverted to bare
    `return` in `_iter_paged`, because they never exercised that code path."""
    class _FakeResponse:
        status_code = 429

    def _get_429(url, video_id=None):
        raise requests.HTTPError(response=_FakeResponse())

    monkeypatch.setattr(redgifs.client, 'get_token', lambda: 'tok')
    monkeypatch.setattr(redgifs.client, '_get', _get_429)
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='throttled')

    items, err = redgifs.scan(validation)

    assert items is None
    assert err is not None
    assert 'rate' in err.lower() or 'blocked' in err.lower()


def test_scan_returns_partial_when_rate_limited_after_collecting_some_items(monkeypatch):
    """Pre-existing bug the reviewer also verified: a 429 on page 2 after page 1
    succeeded used to return 1 item with err=None and no truncation signal, while
    dozens of advertised pages were refused. Now it must carry `partial=True`."""
    def _partial(username, state=None):
        yield {'id': 'abc', 'urls': {}}
        raise redgifs.RedGifsAbort("RedGifs: HTTP 429.")

    monkeypatch.setattr(redgifs.client, 'get_token', lambda: 'tok')
    monkeypatch.setattr(redgifs.client, 'iter_user', _partial)
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='throttled')

    items, err = redgifs.scan(validation)

    assert err is None
    assert len(items) == 1
    assert getattr(items, 'partial', False) is True


def test_scan_of_a_complete_profile_is_not_partial(monkeypatch):
    """Negative counterpart of the truncation tests above: a profile whose
    single page exhausts itself naturally (`pages` == 1, no RedGifsAbort,
    well under MAX_ITEMS) must NOT be flagged partial. Patches `client._get`
    (the real HTTP boundary `_iter_paged` calls) so the real pagination loop
    runs to its natural end instead of a stubbed scan()."""
    def _get_single_page(url, video_id=None):
        gifs = [{'id': f'g{i}', 'urls': {}} for i in range(3)]
        return {'gifs': gifs, 'pages': 1}

    monkeypatch.setattr(redgifs.client, 'get_token', lambda: 'tok')
    monkeypatch.setattr(redgifs.client, '_get', _get_single_page)
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='someone')

    items, err = redgifs.scan(validation)

    assert err is None
    assert len(items) == 3
    assert getattr(items, 'partial', False) is False


def test_scan_marks_hitting_max_items_as_partial(monkeypatch):
    """A profile that has more than MAX_ITEMS (100) videos, with every page
    loading fine (no RedGifsAbort at all), must not look complete. Patches
    `client._get` (the real HTTP boundary `_iter_paged` calls) so the actual
    pagination loop runs and really counts up to the cap — the same pitfall
    called out for the four tests that stayed green with the bug reinstated."""
    import re as _re

    def _get_paged(url, video_id=None):
        page = int(_re.search(r'page=(\d+)', url).group(1))
        # 20 items/page, `pages` far beyond MAX_PAGES (10) so the cap that
        # fires is MAX_ITEMS (reached at page 5), not MAX_PAGES exhaustion.
        gifs = [{'id': f'p{page}_{i}', 'urls': {}} for i in range(20)]
        return {'gifs': gifs, 'pages': 50}

    monkeypatch.setattr(redgifs.client, 'get_token', lambda: 'tok')
    monkeypatch.setattr(redgifs.client, '_get', _get_paged)
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='prolific')

    items, err = redgifs.scan(validation)

    assert err is None
    assert len(items) == 100
    assert getattr(items, 'partial', False) is True


def test_scan_marks_exhausting_max_pages_as_partial(monkeypatch):
    """A profile whose pages never fill up fast enough to hit MAX_ITEMS, but
    which has more real pages than MAX_PAGES (10) allows to visit, must still
    be flagged partial. Before this fix, `_iter_paged`'s `while page <=
    MAX_PAGES` simply stopped looping with no signal that pages remained."""
    import re as _re

    def _get_paged(url, video_id=None):
        page = int(_re.search(r'page=(\d+)', url).group(1))
        # Only 5 items/page (10 pages * 5 = 50, well under MAX_ITEMS=100) but
        # `pages` implies far more real pages than MAX_PAGES.
        gifs = [{'id': f'p{page}_{i}', 'urls': {}} for i in range(5)]
        return {'gifs': gifs, 'pages': 40}

    monkeypatch.setattr(redgifs.client, 'get_token', lambda: 'tok')
    monkeypatch.setattr(redgifs.client, '_get', _get_paged)
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='steady')

    items, err = redgifs.scan(validation)

    assert err is None
    assert len(items) == 50
    assert getattr(items, 'partial', False) is True
