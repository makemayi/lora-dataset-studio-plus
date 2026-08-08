"""Scans de listings à galeries (PornPics catégorie/tag/recherche) : covers par
défaut, albums complets sur option.

Un scan mot-clé/catégorie remonte par défaut LA VIGNETTE que la page affiche pour
chaque galerie — l'image choisie par le site comme représentative du mot-clé
(jamais la 1re de l'album : _009_, _113_… observé en réel) — via le parse
HTML/AJAX du listing, sans gallery-dl. Le flag `include_albums` (case « Scan
full albums » de l'UI, transmis par /scan) rétablit la plongée intégrale
gallery-dl. Si le parse covers échoue (layout changé), repli gallery-dl borné à
1 image/album. L'URL directe d'une galerie (/galleries/...) n'est pas concernée.

Tout est mocké — aucun appel réseau ni process gallery-dl."""
import time

import pytest

from app.scrape.sources import gdl, image_sites
from app.scrape.sources.base import Match
from app.scrape.sources.image_sites import PornpicsSource, _covers_scan, _full_size


def _mock_gdl_runs(monkeypatch):
    """_run_simulate factice : une « catégorie » de 2 albums, 5 images chacun.
    Retourne la liste des appels (url + image_range) pour inspection."""
    calls = []

    def fake(url, max_items, cookies, extra_opts, image_range=None):
        calls.append({'url': url, 'image_range': image_range})
        if 'category' in url:
            return [[6, 'https://x/album1/', {}], [6, 'https://x/album2/', {}]], None
        n = int((image_range or '1-99').split('-')[1])
        return [[3, f'{url}img{i}.jpg', {'extension': 'jpg'}]
                for i in range(1, 6)][:n], None
    monkeypatch.setattr(gdl, '_run_simulate', fake)
    return calls


def test_enumerate_per_album_1_returns_one_cover_per_album(monkeypatch):
    calls = _mock_gdl_runs(monkeypatch)
    items, err = gdl.enumerate('https://x/category/', per_album=1)
    assert err is None
    assert [it['url'] for it in items] == ['https://x/album1/img1.jpg',
                                           'https://x/album2/img1.jpg']
    # La simulation de chaque album est elle-même bornée (--range 1-1) : gallery-dl
    # ne doit pas énumérer tout l'album pour n'en garder qu'une image.
    assert [c['image_range'] for c in calls[1:]] == ['1-1', '1-1']


def test_enumerate_without_per_album_dives_full_albums(monkeypatch):
    _mock_gdl_runs(monkeypatch)
    items, err = gdl.enumerate('https://x/category/')
    assert err is None
    assert len(items) == 10          # 2 albums × 5 images : comportement historique


# --- Provenance des items (from_albums) — debt "Load more" muet -------------- #

def test_enumerate_flags_album_sourced_items_so_callers_can_disable_pagination(monkeypatch):
    """Ces items viennent EXCLUSIVEMENT de la récursion d'albums (type 6), bornée
    en NOMBRE d'albums (max_albums) — jamais par un offset de page. Un appelant
    qui annoncerait la pagination dessus (UniversalSource) enverrait « Charger
    plus » vers une fenêtre --range que cette récursion ignore complètement :
    silence total. `enumerate()` doit donc porter ce signal sur son retour."""
    _mock_gdl_runs(monkeypatch)
    items, err = gdl.enumerate('https://x/category/')
    assert err is None
    assert getattr(items, 'from_albums', False) is True


def test_enumerate_does_not_flag_top_level_media_as_album_sourced(monkeypatch):
    """Des médias TOP-LEVEL (type 3, pas de récursion) restent normalement
    paginables via image_range — pas de signal from_albums dessus."""
    def fake(url, max_items, cookies, extra_opts, image_range=None):
        return [[3, f'{url}img1.jpg', {'extension': 'jpg'}]], None
    monkeypatch.setattr(gdl, '_run_simulate', fake)

    items, err = gdl.enumerate('https://x/direct-media/')

    assert err is None
    assert getattr(items, 'from_albums', False) is False


# --- Budget de temps global (deadline) — debt requête Flask ~9 min ----------- #

def test_enumerate_stops_the_album_recursion_once_the_deadline_has_passed(monkeypatch):
    """Un deadline déjà expiré (posé directement, sans horloge réelle à faire
    avancer) doit couper la récursion d'albums AVANT le 1er sous-process
    gallery-dl d'album et rendre un résultat vide LÉGITIME (kind='empty',
    même convention que « aucun média trouvé ») — jamais une erreur pour un
    budget épuisé, cf. gdl.enumerate docstring."""
    calls = _mock_gdl_runs(monkeypatch)

    items, err = gdl.enumerate('https://x/category/', per_album=1,
                               deadline=time.monotonic() - 1)   # déjà expiré

    assert items is None
    assert getattr(err, 'kind', None) == 'empty'
    # Le scan top-level (1 appel, trouve les 2 albums) part toujours ; la
    # récursion, elle, s'arrête avant le 1er album — deadline déjà dépassé.
    assert len(calls) == 1


def test_enumerate_applies_a_default_deadline_when_the_caller_passes_none(monkeypatch):
    """Finding #4 : `gdl.enumerate()` doit borner le temps même quand l'appelant
    (image_sites.py, civitai.py, fapello.py, erome.py, sexcom.py — toutes les
    sources gdl-backed SAUF universal.py avant cette vague) ne passe pas
    `deadline` explicitement. Sans ce défaut, ces sources pouvaient lancer
    1 + max_albums sous-process gallery-dl à GDL_TIMEOUT chacun DANS une requête
    Flask synchrone (~9 min pire cas) — la protection n'existait que pour
    l'appelant qui avait pensé à la demander."""
    calls = _mock_gdl_runs(monkeypatch)
    base = 1_000_000.0
    clock = iter([base,                                            # calcul du deadline par défaut
                  base + gdl.DEFAULT_SCAN_BUDGET_SECONDS + 1])      # boucle : avant album1
    monkeypatch.setattr(gdl.time, 'monotonic', lambda: next(clock))

    items, err = gdl.enumerate('https://x/category/', per_album=1)   # PAS de deadline

    assert items is None
    assert getattr(err, 'kind', None) == 'empty'
    assert len(calls) == 1     # top-level seulement ; la récursion n'a jamais démarré


def test_enumerate_deadline_checked_between_albums_lets_the_first_one_through(monkeypatch):
    """Le budget est vérifié en DÉBUT de boucle, jamais pendant un sous-process
    déjà lancé (cf. docstring `enumerate`) : une horloge fictive qui ne dépasse
    le deadline qu'APRÈS le 1er album laisse ce 1er album aboutir et coupe
    seulement le 2e — résultat partiel, pas vide."""
    calls = _mock_gdl_runs(monkeypatch)
    clock = iter([0.0,    # boucle : avant album1 (deadline pas atteint)
                  2.0])   # boucle : avant album2 (deadline dépassé)
    monkeypatch.setattr(gdl.time, 'monotonic', lambda: next(clock))

    items, err = gdl.enumerate('https://x/category/', per_album=1, deadline=1.0)

    assert err is None
    assert [it['url'] for it in items] == ['https://x/album1/img1.jpg']
    assert getattr(items, 'partial', False) is True
    assert len(calls) == 2      # scan top-level (trouve les albums) + album1 seul


def test_enumerate_reports_a_blocked_album_scan_even_when_the_budget_also_expired(monkeypatch):
    """RÉGRESSION (finding #1) : le 1er album renvoie une erreur auth (page
    BLOQUÉE), et le budget de temps expire avant que le 2e album ne soit tenté.
    Avant cette vague, la vérif `timed_out` passait AVANT `album_errors` : le
    kind='empty' du budget épuisé écrasait l'erreur auth déjà collectée, et un
    scan activement bloqué se faisait passer pour une page vide (200/count=0)
    au lieu de remonter le message de l'extracteur en 502. L'erreur auth DOIT
    gagner, que le budget ait aussi expiré ou non."""
    calls = []

    def fake(url, max_items, cookies, extra_opts, image_range=None):
        calls.append(url)
        if 'category' in url:
            return [[6, 'https://x/album1/', {}], [6, 'https://x/album2/', {}]], None
        return None, gdl.GdlError('gallery-dl: auth (429).', 'auth')
    monkeypatch.setattr(gdl, '_run_simulate', fake)

    clock = iter([0.0,    # boucle : avant album1 (deadline pas atteint)
                  2.0])   # boucle : avant album2 (deadline dépassé)
    monkeypatch.setattr(gdl.time, 'monotonic', lambda: next(clock))

    items, err = gdl.enumerate('https://x/category/', per_album=1, deadline=1.0)

    assert items is None
    assert getattr(err, 'kind', None) == 'auth'
    assert '429' in err
    # top-level (trouve les 2 albums) + album1 (erreur) ; album2 jamais tenté
    # (coupé par le deadline) — la vérif porte donc bien sur les DEUX conditions
    # combinées, pas seulement sur album_errors seul.
    assert len(calls) == 2


# --- Mode covers : parse des vignettes du listing -----------------------------
_TILE_HTML = '''
<li><a class="rel-link" href="/galleries/flexible-girl-123/">
  <img src="1px.png" data-src="https://cdni.pornpics.com/460/1/2/123/123_009_ab.jpg" alt="Flexible girl">
</a></li>
<li><a class="rel-link" href="https://www.pornpics.com/galleries/splits-babe-456/">
  <img src='1px.png' data-src='https://cdni.pornpics.com/300/3/4/456/456_113_cd.jpg' alt='Splits babe'>
</a></li>
<li><a class="rel-link" href="/channels/whatever/">nav link sans data-src de tuile</a></li>
'''


def test_full_size_swaps_cdn_size_segment():
    assert _full_size('https://cdni.pornpics.com/460/1/2/123/123_009_ab.jpg') \
        == 'https://cdni.pornpics.com/1280/1/2/123/123_009_ab.jpg'
    assert _full_size('https://cdni.pornpics.com/300/3/4/456/456_113_cd.jpg') \
        == 'https://cdni.pornpics.com/1280/3/4/456/456_113_cd.jpg'


def test_covers_page0_returns_the_listing_thumbnails(monkeypatch):
    monkeypatch.setattr(image_sites, '_listing_html', lambda url: _TILE_HTML)
    items, err = _covers_scan('https://www.pornpics.com/flexible/', 0)
    assert err is None
    # la vignette VISIBLE (image _009_/_113_), pas la 1re image de l'album —
    # et seulement les tuiles galerie (le lien /channels/ est ignoré)
    assert [it['url'] for it in items] == [
        'https://cdni.pornpics.com/1280/1/2/123/123_009_ab.jpg',
        'https://cdni.pornpics.com/1280/3/4/456/456_113_cd.jpg']
    assert items[0]['thumbnail'].startswith('https://cdni.pornpics.com/460/')
    assert items[0]['title'] == 'Flexible girl'


def test_covers_next_pages_use_the_ajax_endpoint(monkeypatch):
    seen = {}

    def fake_json(url, offset):
        seen['offset'] = offset
        return [{'g_url': 'https://www.pornpics.com/galleries/x-789/', 'desc': 'X',
                 't_url_460': 'https://cdni.pornpics.com/460/5/6/789/789_042_ef.jpg'}]
    monkeypatch.setattr(image_sites, '_listing_json', fake_json)
    items, err = _covers_scan('https://www.pornpics.com/flexible/', 2)
    assert err is None and seen['offset'] == 40
    assert items[0]['url'] == 'https://cdni.pornpics.com/1280/5/6/789/789_042_ef.jpg'
    assert items[0]['title'] == 'X'


def test_covers_scan_signals_fallback_never_raises(monkeypatch):
    def boom(url):
        raise RuntimeError('site down')
    monkeypatch.setattr(image_sites, '_listing_html', boom)
    assert _covers_scan('https://www.pornpics.com/flexible/', 0) == (None, None)
    monkeypatch.setattr(image_sites, '_listing_html', lambda url: '<html>layout changé</html>')
    assert _covers_scan('https://www.pornpics.com/flexible/', 0) == (None, None)


# --- Routage PornpicsSource.scan : covers / albums / galerie directe / repli ---
@pytest.fixture()
def _spies(monkeypatch):
    seen = {'enum': None, 'covers': 0}

    def fake_enum(url, **kw):
        seen['enum'] = kw
        return [], None
    monkeypatch.setattr(gdl, 'enumerate', fake_enum)
    monkeypatch.setattr(image_sites, '_listing_html', lambda url: seen.update(covers=seen['covers'] + 1) or _TILE_HTML)
    return seen


def test_pornpics_default_scan_serves_covers_without_gdl(_spies):
    m = Match(url='https://www.pornpics.com/flexible/')
    m.page = 0
    items, err = PornpicsSource().scan(m)
    assert err is None and len(items) == 2
    assert _spies['covers'] == 1 and _spies['enum'] is None   # gallery-dl jamais lancé


def test_pornpics_include_albums_dives_via_gdl(_spies):
    m = Match(url='https://www.pornpics.com/flexible/')
    m.page = 0
    m.include_albums = True
    PornpicsSource().scan(m)
    assert _spies['covers'] == 0                    # pas de parse covers
    assert _spies['enum']['per_album'] is None      # plongée intégrale


def test_pornpics_direct_gallery_url_bypasses_covers(_spies):
    m = Match(url='https://www.pornpics.com/galleries/flexible-girl-123/')
    m.page = 0
    PornpicsSource().scan(m)
    assert _spies['covers'] == 0                    # URL d'album → gallery-dl direct
    assert _spies['enum'] is not None


def test_pornpics_covers_failure_falls_back_to_bounded_gdl(monkeypatch, _spies):
    def boom(url):
        raise RuntimeError('site down')
    monkeypatch.setattr(image_sites, '_listing_html', boom)
    m = Match(url='https://www.pornpics.com/flexible/')
    m.page = 0
    items, err = PornpicsSource().scan(m)
    assert err is None
    assert _spies['enum']['per_album'] == 1         # repli borné : 1 image/album


def test_scan_route_passes_include_albums_to_match(client, monkeypatch):
    seen = {}

    def fake_scan(self, match):
        seen['include_albums'] = getattr(match, 'include_albums', None)
        return [{'url': 'https://cdni.pornpics.com/x.jpg', 'title': '',
                 'thumbnail': None, 'type': 'image', 'platform': 'pornpics'}], None
    monkeypatch.setattr(PornpicsSource, 'scan', fake_scan)
    r = client.post('/api/scrape/scan',
                    json={'url': 'https://www.pornpics.com/flexible/',
                          'include_albums': True})
    assert r.status_code == 200
    assert seen['include_albums'] is True
    r = client.post('/api/scrape/scan',
                    json={'url': 'https://www.pornpics.com/flexible/'})
    assert r.status_code == 200
    assert seen['include_albums'] is False       # défaut : covers seulement
