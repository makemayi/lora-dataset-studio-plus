"""Route /api/scrape/scan : traitement des erreurs de scan() par `kind`.

La règle gouvernante de la vague scraper : un bloc ne doit jamais paraître
vide, un résultat vide ne doit jamais paraître en échec. Ce module vérifie
qu'elle est honorée UNE seule fois, ici — le seul endroit qui voit TOUTES les
sources gdl-backed, pas juste UniversalSource.

Tout est mocké au niveau de `registry.resolve` : aucun appel réseau ni
process gallery-dl."""
from app.scrape.sources import gdl, registry
from app.scrape.sources.base import Match, Source, Capabilities, ResultList


class _FakeSource(Source):
    """Source jetable dont scan() renvoie exactement ce que le test lui donne."""
    name = 'fake'
    priority = 1000       # avant toutes les vraies sources
    capabilities = Capabilities()
    paginated = True
    category = 'image'

    def __init__(self, items=None, err=None):
        self._items = items
        self._err = err

    def match(self, url):
        if url.startswith('https://fake.example.test/'):
            return Match(url=url)
        return None

    def scan(self, match):
        return self._items, self._err


def _use_fake_source(monkeypatch, **kw):
    src = _FakeSource(**kw)

    def fake_resolve(url):
        match = src.match(url)
        if match is not None:
            match.source = src
        return match

    monkeypatch.setattr(registry, 'resolve', fake_resolve)
    return src


def test_scan_empty_kind_is_200_with_zero_items(client, monkeypatch):
    """kind='empty' : gallery-dl (ou équivalent) a tourné sans incident et n'a
    rien trouvé — un scan vide réussi. La route doit répondre 200/count=0,
    jamais 502 : c'est exactement le cas que les 502 anonymes cachaient pour
    onze des quatorze sources avant cette vague."""
    _use_fake_source(monkeypatch, items=None,
                     err=gdl.GdlError('gallery-dl: no media found.', 'empty'))

    r = client.post('/api/scrape/scan',
                    json={'url': 'https://fake.example.test/album/1'})

    assert r.status_code == 200
    body = r.get_json()
    assert body['count'] == 0
    assert body['items'] == []
    assert body['scannable'] is True


def test_scan_auth_kind_still_answers_502_with_its_message(client, monkeypatch):
    """Un vrai blocage (auth/429/DDoS-Guard) ne doit JAMAIS se déguiser en
    résultat vide : seul kind='empty' bascule vers 200, tout le reste garde le
    502 et son message d'origine — cette moitié de la règle ne doit pas
    s'affaiblir en corrigeant l'autre."""
    _use_fake_source(monkeypatch, items=None,
                     err=gdl.GdlError('gallery-dl: auth (429).', 'auth'))

    r = client.post('/api/scrape/scan',
                    json={'url': 'https://fake.example.test/album/1'})

    assert r.status_code == 502
    body = r.get_json()
    assert 'auth' in body['error']


def test_scan_toolerror_kind_still_answers_502(client, monkeypatch):
    """Même garantie pour 'toolerror' (pas seulement 'auth') : n'importe quel
    kind autre que 'empty' reste un échec HTTP explicite."""
    _use_fake_source(monkeypatch, items=None,
                     err=gdl.GdlError('gallery-dl: unreadable response.', 'toolerror'))

    r = client.post('/api/scrape/scan',
                    json={'url': 'https://fake.example.test/album/1'})

    assert r.status_code == 502


def test_scan_surfaces_partial_when_the_time_budget_cut_the_listing_short(client, monkeypatch):
    """`enumerate()` peut renvoyer un `ResultList` (app/scrape/sources/base.py)
    avec `partial=True` (budget de temps épuisé en cours de récursion
    d'albums, items présents mais incomplets, cf. gdl.py). Avant cette vague,
    `partial` s'arrêtait au logger.info de universal.py : la route ne
    l'exposait nulle part, donc l'UI ne pouvait jamais dire à l'utilisateur
    qu'un résultat COMPLET-en-apparence était en réalité tronqué
    (finding #2)."""
    truncated = ResultList([
        {'url': 'https://fake.example.test/a.jpg', 'title': '', 'thumbnail': None,
         'type': 'image', 'platform': 'fake'}])
    truncated.from_albums = True
    truncated.partial = True
    _use_fake_source(monkeypatch, items=truncated, err=None)

    r = client.post('/api/scrape/scan',
                    json={'url': 'https://fake.example.test/album/1'})

    assert r.status_code == 200
    body = r.get_json()
    assert body['partial'] is True
    assert body['count'] == 1


def test_scan_partial_defaults_to_false_for_ordinary_sources(client, monkeypatch):
    """Une source non gdl-backed (liste ordinaire, pas de `ResultList`) ne doit
    jamais faire lever `partial` par accident — `getattr` doit retomber sur False
    plutôt que planter ou renvoyer une valeur truthy inattendue."""
    _use_fake_source(monkeypatch, items=[{'url': 'https://fake.example.test/a.jpg',
                                          'title': '', 'thumbnail': None,
                                          'type': 'image', 'platform': 'fake'}], err=None)

    r = client.post('/api/scrape/scan',
                    json={'url': 'https://fake.example.test/album/1'})

    assert r.status_code == 200
    assert r.get_json()['partial'] is False


def test_scan_a_plain_string_error_without_kind_still_answers_502(client, monkeypatch):
    """Une erreur qui n'est PAS une GdlError (str nu, pas de `.kind`) doit
    rester un 502 — `getattr(err, 'kind', None)` renvoie None, jamais 'empty'
    par accident sur un message sans provenance connue."""
    _use_fake_source(monkeypatch, items=None, err="Fake: something broke.")

    r = client.post('/api/scrape/scan',
                    json={'url': 'https://fake.example.test/album/1'})

    assert r.status_code == 502
