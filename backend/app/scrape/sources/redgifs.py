# app/scrape/sources/redgifs.py
"""Scraper RedGifs — énumération d'un profil / d'une niche / d'une vidéo.

Port autonome (requests, sans `config`/`settings` source) de
`redgifs_downloader/api/redgifs.py`. N'effectue QUE l'énumération (liste des
vidéos + vignettes) ; le téléchargement réel passe par yt-dlp (extracteur
RedGifs) via /api/scrape/download — on renvoie donc l'URL `watch/<id>`.

Hôtes contactés : api.redgifs.com (fixe, public) → pas de risque SSRF.
"""
import logging
import threading

import requests

from ..validators import URLType
from .base import ResultList

logger = logging.getLogger(__name__)


class RedGifsAbort(Exception):
    """Levée par `_iter_paged` quand l'itération s'arrête sur une ERREUR (HTTP
    429/403/5xx, timeout réseau, 401 après épuisement des retries de token)
    plutôt que sur un épuisement propre des pages. Sans ce signal, `scan()` ne
    pouvait pas distinguer « le profil/la niche est vide » d'« une page a été
    refusée » — RedGifs rate-limite couramment (cf. la logique de refresh de
    token juste au-dessus, déjà prévue pour ça) et un `return` silencieux dans
    `_iter_paged` transformait un 429/403/500 en résultat vide légitime."""

REDGIFS_API_BASE = "https://api.redgifs.com/v2"
REDGIFS_AUTH_URL = f"{REDGIFS_API_BASE}/auth/temporary"
# UA FIXE : le token JWT temporaire est lié à l'User-Agent de la requête d'auth.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
TIMEOUT = 30
MAX_PAGES = 10
MAX_ITEMS = 100


class RedGifsClient:
    """Client API RedGifs minimal (token temporaire + énumération)."""

    def __init__(self):
        self._token = None
        self._lock = threading.Lock()
        self._session = requests.Session()

    def _headers(self, video_id=None, auth=True):
        h = {
            "Accept": "application/json, text/plain, */*",
            "User-Agent": USER_AGENT,
            "Referer": "https://www.redgifs.com/",
            "Origin": "https://www.redgifs.com",
            "Content-Type": "application/json",
        }
        if video_id:
            # x-customheader OBLIGATOIRE pour /gifs/{id} (sinon 401 malgré le token).
            h["x-customheader"] = f"https://www.redgifs.com/watch/{video_id}"
        if auth and self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def get_token(self):
        with self._lock:
            if self._token:
                return self._token
        try:
            r = self._session.get(REDGIFS_AUTH_URL, headers=self._headers(auth=False), timeout=TIMEOUT)
            r.raise_for_status()
            token = (r.json() or {}).get('token')
        except Exception as e:
            logger.warning(f"[redgifs] échec obtention token: {e}")
            return None
        if token:
            with self._lock:
                self._token = token
            return token
        return None

    def _reset_token(self):
        with self._lock:
            self._token = None

    def _get(self, url, video_id=None):
        """GET authentifié → JSON. Lève requests.HTTPError sur statut != 2xx."""
        r = self._session.get(url, headers=self._headers(video_id=video_id), timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()

    def get_single_video(self, video_id):
        url = f"{REDGIFS_API_BASE}/gifs/{video_id}"
        try:
            data = self._get(url, video_id=video_id)
        except requests.HTTPError as e:
            if getattr(e.response, 'status_code', None) == 401:
                self._reset_token()
                if self.get_token():
                    try:
                        data = self._get(url, video_id=video_id)
                    except Exception:
                        return None
                else:
                    return None
            else:
                return None
        except Exception:
            return None
        return (data or {}).get('gif')

    def _iter_paged(self, url_for_page, state=None):
        """Itère les pages via une fonction page→url, avec refresh de token sur 401.

        Épuisement propre (plus de gifs sur la page, ou dernière page atteinte) →
        `return` normal, générateur terminé sans lever. Toute page refusée (HTTP
        429/403/5xx, timeout, 401 non récupéré) → lève `RedGifsAbort` au lieu de
        `return` : l'appelant doit pouvoir distinguer « fini » d'« interrompu »
        (cf. `RedGifsAbort`).

        `state` (dict optionnel, posé par l'appelant) : si la boucle épuise ses
        MAX_PAGES itérations SANS avoir atteint `page >= total_pages` (i.e. il
        restait des pages réelles au-delà du garde-fou temps MAX_PAGES),
        `state['capped'] = True` est posé juste avant que le générateur se
        termine — sans ça, ce plafond tronquait silencieusement (même défaut
        que MAX_ITEMS côté `_consume_paged`, cf. son docstring)."""
        page = 1
        token_retries = 0
        while page <= MAX_PAGES:
            url = url_for_page(page)
            try:
                data = self._get(url)
            except requests.HTTPError as e:
                status = getattr(e.response, 'status_code', None)
                if status == 401 and token_retries < 2:
                    token_retries += 1
                    self._reset_token()
                    if self.get_token():
                        continue  # réessaye la même page
                    raise RedGifsAbort("RedGifs: token refresh failed after 401.") from e
                raise RedGifsAbort(f"RedGifs: HTTP {status}.") from e
            except Exception as e:
                raise RedGifsAbort(f"RedGifs: {e}") from e

            gifs = (data or {}).get('gifs') or []
            if not gifs:
                return
            for gif in gifs:
                yield gif

            total_pages = (data or {}).get('pages', 1) or 1
            if page >= total_pages:
                return
            page += 1
        # `while` sorti par épuisement de MAX_PAGES (pas par un `return` ci-dessus
        # donc `page` n'a jamais atteint `total_pages`) : des pages restent.
        if state is not None:
            state['capped'] = True

    def iter_user(self, username, state=None):
        return self._iter_paged(
            lambda p: f"{REDGIFS_API_BASE}/users/{username}/search?order=new&page={p}",
            state=state)

    def iter_niche(self, niche, state=None):
        return self._iter_paged(
            lambda p: f"{REDGIFS_API_BASE}/niches/{niche}/gifs?page={p}",
            state=state)


# Instance globale (feature admin-only mono-utilisateur).
client = RedGifsClient()


def _item_from_gif(gif):
    gid = gif.get('id', '') or ''
    urls = gif.get('urls', {}) or {}
    return {
        'url': f"https://www.redgifs.com/watch/{gid}",
        'title': gid or 'redgif',
        'thumbnail': urls.get('thumbnail') or urls.get('poster'),
        'type': 'video',
        'platform': 'redgifs',
        'duration': gif.get('duration', 0),
    }


def _consume_paged(gen, items, state=None):
    """Draine un générateur `_iter_paged` dans `items` (borné à MAX_ITEMS).

    Retourne True si la récolte est TRONQUÉE pour une raison quelconque :
      - `RedGifsAbort` levée en cours de route (page refusée / réseau) ;
      - plafond MAX_ITEMS atteint AVANT que le générateur ne s'épuise
        proprement — même limite de détection que picazor.py : on ne peut
        pas savoir si le profil s'arrêtait pile là, on choisit le côté
        honnête (cf. son commentaire) ;
      - `state['capped']` posé par `_iter_paged` (plafond MAX_PAGES atteint
        alors que des pages réelles restaient, cf. son docstring).
    Retourne False si l'itération s'est épuisée proprement, sans aucun de ces
    signaux — profil/niche réellement à court de contenu."""
    try:
        for gif in gen:
            items.append(_item_from_gif(gif))
            if len(items) >= MAX_ITEMS:
                return True
    except RedGifsAbort:
        return True
    return bool(state.get('capped')) if state is not None else False


def scan(validation):
    """Énumère les médias d'une URL RedGifs. Retourne (items, error).

    validation : ValidationResult (url_type ∈ {PROFILE, NICHE, VIDEO}, value = username/niche/id).
    Borné à MAX_ITEMS. Ne lève jamais.
    """
    try:
        if not client.get_token():
            return None, "RedGifs: could not obtain a token (Cloudflare / rate-limit?)."

        ut = validation.url_type
        value = validation.value
        items = []
        state = {}

        if ut == URLType.PROFILE:
            truncated = _consume_paged(client.iter_user(value, state=state), items, state)
        elif ut == URLType.NICHE:
            truncated = _consume_paged(client.iter_niche(value, state=state), items, state)
        elif ut == URLType.VIDEO:
            gif = client.get_single_video(value)
            if not gif:
                return None, "RedGifs video not found (or token expired)."
            items.append(_item_from_gif(gif))
            return items, None
        else:
            return None, "Unsupported RedGifs URL type."

        if truncated:
            if items:
                # Des items ont déjà été récoltés avant que la récolte ne soit
                # coupée — page refusée (429/403/5xx/401), plafond MAX_ITEMS, ou
                # plafond MAX_PAGES atteint avec des pages restantes (cf.
                # `_consume_paged`) : résultat PARTIEL, pas un échec — même
                # convention que `base.ResultList.partial` (réutilisée telle
                # quelle, cf. import), lue par `routes/scrape.py` sur l'objet
                # retourné (`getattr(items, 'partial', False)`), sans changement
                # côté route. Corrige au passage un bug préexistant vérifié par le
                # relecteur : un 429 en page 2 après une page 1 réussie renvoyait
                # 1 item avec err=None et aucun signal de troncature — et, plus
                # récemment, un plafond MAX_ITEMS/MAX_PAGES atteint sans incident
                # réseau faisait de même.
                result = ResultList(items[:MAX_ITEMS])
                result.partial = True
                return result, None
            # Zéro item ET troncature signalée : ne peut venir ici que d'un
            # `RedGifsAbort` (le plafond MAX_ITEMS exige au moins 1 item déjà
            # ajouté, et `state['capped']` n'est posé qu'après avoir consommé au
            # moins une page avec des gifs, cf. `_iter_paged`) — une vraie panne
            # (rate-limit/blocage), pas « rien ici » — avant cette correction ce
            # cas se confondait avec le profil/la niche légitimement vide juste
            # en dessous et répondait ([], None), 200, « No images found ».
            return None, "RedGifs: rate-limited or blocked while listing (try again shortly)."

        # PROFILE/NICHE vide (pas VIDEO, cf. branche dédiée ci-dessus qui reste une
        # vraie erreur) après une itération qui s'est épuisée PROPREMENT (aucun
        # `RedGifsAbort`, cf. juste au-dessus) : l'API a répondu sans incident, le
        # compte/la niche n'a juste aucune vidéo publique. Résultat vide LÉGITIME
        # (même convention que gdl.GdlError kind='empty', app/scrape/sources/gdl.py)
        # — pas un échec outil : avant cette vague la route répondait 502 sur un
        # profil vide.
        return items, None
    except Exception as e:  # garde-fou : ne jamais propager
        logger.warning(f"[redgifs] erreur de scan: {e}")
        return None, "RedGifs scan error."


from .base import Source, Capabilities, Match
from . import registry


class RedgifsSource(Source):
    name = 'redgifs'
    priority = 100
    capabilities = Capabilities(can_enumerate_profile=True, own_downloader=False)

    def match(self, url):
        from ..validators import url_validator, Platform
        result = url_validator.validate_url(url)
        if result.is_valid and result.platform == Platform.REDGIFS:
            return Match(url=url, validation=result)
        return None

    def scan(self, match):
        return scan(match.validation)   # délègue au scan(validation) existant


registry.register(RedgifsSource())
