"""Instagram scan() : un profil légitimement sans publication est un résultat
vide, pas un échec ; un post unique qui refuse de se convertir en item reste
un vrai échec (finding #3 — verdicts différents pour deux formes de "aucun
média", cf. rapport).

Tout est mocké (`_build_loader`, `instaloader.Profile`/`Post`) : aucun réseau."""
import time
from types import SimpleNamespace

import instaloader

from app.scrape.sources import instagram
from app.scrape.validators import URLType


class _FakeEmptyProfile:
    def get_posts(self):
        return iter([])


class _FakeCarouselPostThatFailsToParse:
    """Carrousel dont l'extraction des slides échoue entièrement — aucun item
    récupérable, MAIS le post existe bel et bien (chargement réussi)."""
    shortcode = 'xyz789'
    typename = 'GraphSidecar'

    def get_sidecar_nodes(self):
        raise RuntimeError('sidecar parse failed')


class _FakeThrottledPost:
    """Ne devrait jamais être converti : le générateur du profil dort avant de
    le céder, assez longtemps pour dépasser `PROFILE_SCAN_TIMEOUT`."""
    shortcode = 'never-reached'


class _FakeThrottledProfile:
    """Simule le rate-controller RÉEL d'instaloader, qui DORT au lieu de lever
    (cf. module docstring / `PROFILE_SCAN_TIMEOUT`) — le test monkeypatche
    `PROFILE_SCAN_TIMEOUT` à une valeur minuscule pour ne pas dormir 60s réelles."""
    def get_posts(self):
        time.sleep(0.05)
        yield _FakeThrottledPost()


class _FakeSimplePost:
    """Post unique (pas carrousel) qui se convertit sans problème."""
    def __init__(self, shortcode):
        self.shortcode = shortcode
        self.typename = 'GraphImage'
        self.is_video = False
        self.url = f'https://example.invalid/{shortcode}.jpg'


class _FakeProfileRateLimitedMidIteration:
    """Un post cède avec succès, puis l'itération paginée lève (rate-limit en
    cours de route) — le cas le plus courant : des items utiles sont déjà là."""
    def get_posts(self):
        yield _FakeSimplePost('collected1')
        raise ConnectionError('429 Too Many Requests')


class _FakeProfileWithManyPosts:
    """A profile with more posts than SCAN_LIMIT, none of which raise or
    time out — a genuine cap-hit, not an abort."""
    def get_posts(self):
        for i in range(instagram.SCAN_LIMIT + 10):
            yield _FakeSimplePost(f'post{i}')


class _FakeProfileWithFewPosts:
    """A profile with fewer posts than SCAN_LIMIT, none of which raise, time
    out or fail conversion — a genuinely complete, uncapped scan."""
    def get_posts(self):
        for i in range(3):
            yield _FakeSimplePost(f'post{i}')


class _FakeCarouselNode:
    def __init__(self, idx):
        self.is_video = False
        self.display_url = f'https://example.invalid/n{idx}.jpg'


class _FakeCarouselPostWithManySlides:
    """One carousel post whose slide count alone exceeds SCAN_LIMIT — the cap
    can be hit mid-post, not just between posts."""
    shortcode = 'carousel1'
    typename = 'GraphSidecar'

    def get_sidecar_nodes(self):
        return [_FakeCarouselNode(i) for i in range(instagram.SCAN_LIMIT + 10)]


class _FakeProfileWithOneBigCarousel:
    def get_posts(self):
        yield _FakeCarouselPostWithManySlides()


class _FakeProfileWhereEveryPostFailsConversion:
    """Deux posts vus, aucun ne survit à la conversion (ex. changement de mise
    en page côté Instagram) — pas la même chose qu'un profil sans publication."""
    def get_posts(self):
        return iter([_FakeCarouselPostThatFailsToParse(),
                     _FakeCarouselPostThatFailsToParse()])


def test_scan_profile_with_no_posts_is_empty_not_an_error(monkeypatch):
    monkeypatch.setattr(instagram, '_build_loader', lambda: SimpleNamespace(context=object()))
    monkeypatch.setattr(instaloader.Profile, 'from_username',
                        staticmethod(lambda context, username: _FakeEmptyProfile()))
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='nobody', original_url=None)

    items, err = instagram.scan(validation)

    assert items is None
    assert getattr(err, 'kind', None) == 'empty'
    assert 'nobody' in err


def test_scan_single_with_no_extractable_media_stays_a_real_failure(monkeypatch):
    monkeypatch.setattr(instagram, '_build_loader', lambda: SimpleNamespace(context=object()))
    monkeypatch.setattr(instaloader.Post, 'from_shortcode',
                        staticmethod(lambda context, shortcode: _FakeCarouselPostThatFailsToParse()))
    validation = SimpleNamespace(url_type=URLType.POST, value='xyz789', original_url=None)

    items, err = instagram.scan(validation)

    assert items is None
    assert getattr(err, 'kind', None) is None
    assert 'usable media' in err


def test_scan_profile_timeout_with_zero_items_is_a_failure_not_empty(monkeypatch):
    """Probe from the reviewer: instaloader's rate controller SLEEPS rather than
    raising, so a throttled profile can hit the PROFILE_SCAN_TIMEOUT cap with
    zero items collected and no exception raised — before this fix that path
    fell into kind='empty', indistinguishable from a profile with no posts."""
    monkeypatch.setattr(instagram, '_build_loader', lambda: SimpleNamespace(context=object()))
    monkeypatch.setattr(instagram, 'PROFILE_SCAN_TIMEOUT', 0.01)
    monkeypatch.setattr(instaloader.Profile, 'from_username',
                        staticmethod(lambda context, username: _FakeThrottledProfile()))
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='throttled', original_url=None)

    items, err = instagram.scan(validation)

    assert items is None
    assert err is not None
    assert getattr(err, 'kind', None) != 'empty'
    assert 'timed out' in err.lower()


def test_scan_profile_marks_a_mid_iteration_rate_limit_as_partial(monkeypatch):
    """Sibling of the RedGifs mid-iteration abort: a rate-limit raised by
    `profile.get_posts()` AFTER at least one post was already collected used to
    return `items[:SCAN_LIMIT], None` — a plain list with no truncation signal,
    presenting a one-item harvest as complete. Must now carry `partial=True`,
    the same convention as `base.ResultList.partial`."""
    monkeypatch.setattr(instagram, '_build_loader', lambda: SimpleNamespace(context=object()))
    monkeypatch.setattr(instaloader.Profile, 'from_username',
                        staticmethod(lambda context, username: _FakeProfileRateLimitedMidIteration()))
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='throttled', original_url=None)

    items, err = instagram.scan(validation)

    assert err is None
    assert len(items) == 1
    assert getattr(items, 'partial', False) is True


def test_scan_profile_marks_hitting_scan_limit_as_partial(monkeypatch):
    """A profile with more than SCAN_LIMIT posts, none of which fail or time
    out, must not look like a complete SCAN_LIMIT-post profile. Patches
    `instaloader.Profile.from_username` — the network boundary `_scan_profile`
    actually iterates over — so the real collection loop runs and really
    counts posts up to the cap, instead of stubbing `scan()`/`_scan_profile`
    themselves out."""
    monkeypatch.setattr(instagram, '_build_loader', lambda: SimpleNamespace(context=object()))
    monkeypatch.setattr(instaloader.Profile, 'from_username',
                        staticmethod(lambda context, username: _FakeProfileWithManyPosts()))
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='prolific', original_url=None)

    items, err = instagram.scan(validation)

    assert err is None
    assert len(items) == instagram.SCAN_LIMIT
    assert getattr(items, 'partial', False) is True


def test_scan_profile_marks_hitting_scan_limit_mid_carousel_as_partial(monkeypatch):
    """Same cap, hit inside a single carousel post's slides rather than between
    posts — the other `break` site guarding SCAN_LIMIT."""
    monkeypatch.setattr(instagram, '_build_loader', lambda: SimpleNamespace(context=object()))
    monkeypatch.setattr(instaloader.Profile, 'from_username',
                        staticmethod(lambda context, username: _FakeProfileWithOneBigCarousel()))
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='bigcarousel', original_url=None)

    items, err = instagram.scan(validation)

    assert err is None
    assert len(items) == instagram.SCAN_LIMIT
    assert getattr(items, 'partial', False) is True


def test_scan_profile_of_a_complete_profile_is_not_partial(monkeypatch):
    """Negative counterpart of the truncation tests above: a profile with
    fewer than SCAN_LIMIT posts, none of which raise, time out or fail
    conversion, must NOT be flagged partial. Patches
    `instaloader.Profile.from_username` (same boundary as the other tests
    here) so the real collection loop runs to its natural end instead of a
    stubbed scan()."""
    monkeypatch.setattr(instagram, '_build_loader', lambda: SimpleNamespace(context=object()))
    monkeypatch.setattr(instaloader.Profile, 'from_username',
                        staticmethod(lambda context, username: _FakeProfileWithFewPosts()))
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='someone', original_url=None)

    items, err = instagram.scan(validation)

    assert err is None
    assert len(items) == 3
    assert getattr(items, 'partial', False) is False


def test_scan_profile_where_every_post_fails_conversion_is_a_failure_not_empty(monkeypatch):
    """Every post fails conversion (systematic layout change) — zero items with
    no exception raised, but posts WERE seen: also fell into kind='empty'
    before the fix."""
    monkeypatch.setattr(instagram, '_build_loader', lambda: SimpleNamespace(context=object()))
    monkeypatch.setattr(instaloader.Profile, 'from_username',
                        staticmethod(lambda context, username: _FakeProfileWhereEveryPostFailsConversion()))
    validation = SimpleNamespace(url_type=URLType.PROFILE, value='brokenlayout', original_url=None)

    items, err = instagram.scan(validation)

    assert items is None
    assert err is not None
    assert getattr(err, 'kind', None) != 'empty'
    assert 'none could be read' in err.lower()
