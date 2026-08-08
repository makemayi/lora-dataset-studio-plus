# app/scrape/sources/universal.py
"""Source universelle (catch-all, priorité 0) : hybride gallery-dl → (exit 64) → yt-dlp.
Tente d'abord gallery-dl (extracteurs dédiés de nombreux sites) ; si gallery-dl ne
supporte pas l'URL (exit code & 64), repli sur yt-dlp — mais SEULEMENT pour les hôtes
d'une allowlist vettée (atténuation SSRF interim, cf. spec décision #6)."""
import logging
import os
import time
from urllib.parse import urlparse

from .base import Source, Capabilities, Match
from . import registry, gdl
from .gdl import GdlError
from .. import netfetch

logger = logging.getLogger(__name__)

# Hôtes pour lesquels la branche générique yt-dlp est autorisée (interim SSRF).
# Coomer/Kemono/Cyberdrop/Bunkr n'y figurent plus : ces sources sont retirées et
# validators.py les refuse explicitement avant même d'atteindre cette source (donc
# avant match()) — les garder ici serait du code mort et une fausse impression que
# le repli générique reste possible pour elles.
VETTED_DOMAINS = (
    'x.com', 'twitter.com', 'tiktok.com',
    'youtube.com', 'youtu.be', 'pornhub.com', 'xvideos.com', 'redgifs.com',
    'vimeo.com', 'dailymotion.com',
)

# Fenêtre d'énumération par page : la valeur que le moteur gallery-dl utilise
# déjà par défaut. Pas de réglage supplémentaire à accorder entre les deux.
MAX_ITEMS = gdl.DEFAULT_MAX_ITEMS

# Budget de temps global d'un scan (gdl.enumerate deadline). Cette source est le
# CATCH-ALL des hôtes inconnus (priority=0) — c'est là que vivent les pages
# pathologiques (listings à des centaines d'albums) qui, sans plafond, peuvent
# lancer 1 + DEFAULT_MAX_ALBUMS (8) sous-process gallery-dl à GDL_TIMEOUT (60s)
# chacun dans UNE requête Flask synchrone (~9 min pire cas).
#
# Alias de `gdl.DEFAULT_SCAN_BUDGET_SECONDS` — cette source n'a plus besoin de sa
# propre valeur (gdl.enumerate applique désormais ce budget par défaut à TOUT
# appelant, cf. finding #4/#5), mais la passer explicitement documente ICI, au
# point d'appel le plus exposé (hôtes arbitraires), que la source en dépend.
# ATTENTION : ce budget ne borne le scan top-level (1 sous-process, incompressible,
# jusqu'à GDL_TIMEOUT) qu'INDIRECTEMENT — le pire cas RÉEL est ≈ 2×GDL_TIMEOUT, pas
# cette constante (cf. docstring de `gdl.DEFAULT_SCAN_BUDGET_SECONDS` — ne PAS
# gonfler cette valeur en pensant qu'elle plafonne le temps de réponse total).
SCAN_BUDGET_SECONDS = gdl.DEFAULT_SCAN_BUDGET_SECONDS


def _host_vetted(url):
    host = (urlparse(url).hostname or '').lower()
    if not host:
        return False
    return any(host == d or host.endswith('.' + d) for d in VETTED_DOMAINS)


class UniversalSource(Source):
    name = 'universal'
    priority = 0
    paginated = True
    category = 'image'
    capabilities = Capabilities(is_universal_fallback=True, own_downloader=True,
                                media_kinds=frozenset({'image', 'video'}))

    def match(self, url):
        from ..validators import url_validator, Platform
        # SSRF : le chemin générique est le seul à accepter un hôte arbitraire, et
        # scan() lance gallery-dl dessus. Refuser ICI signifie qu'aucune source ne
        # matche et que la route répond 400 AVANT qu'un sous-process ne parte.
        ok, _err = netfetch._validate_public_http_url(url)
        if not ok:
            return None
        result = url_validator.validate_url(url)
        if result.is_valid and result.platform == Platform.GENERIC:
            return Match(url=url, validation=result)
        return None

    def scan(self, match):
        """Énumération générique via gallery-dl (~300 sites). Défaut : les médias
        directs de la page ; un listing d'albums rend UNE cover par album, la case
        « Scan full albums » (include_albums) rétablit la plongée."""
        url = match.url
        page = max(0, int(getattr(match, 'page', 0) or 0))
        items, err = gdl.enumerate(
            url, platform='generic', max_items=MAX_ITEMS,
            per_album=None if getattr(match, 'include_albums', False) else 1,
            image_range=f'{page * MAX_ITEMS + 1}-{(page + 1) * MAX_ITEMS}',
            deadline=time.monotonic() + SCAN_BUDGET_SECONDS)
        if items:
            if getattr(items, 'from_albums', False):
                # Ces items viennent de la récursion d'albums de gdl.enumerate,
                # bornée en NOMBRE d'albums (max_albums) jamais par un offset de
                # page : `--range` (image_range, calculé ci-dessus depuis `page`)
                # ne borne que les médias TOP-LEVEL, que la récursion d'albums
                # ignore complètement. Annoncer la pagination enverrait « Charger
                # plus » vers une fenêtre que rien ne consomme : silence total,
                # exactement comme le repli `unsupported` juste en dessous coupe
                # déjà `match.paginated` pour la même raison structurelle.
                match.paginated = False
            if getattr(items, 'partial', False):
                # Budget de temps épuisé en cours de récursion : les items présents
                # restent valides, seulement incomplets — un scan tronqué vaut
                # mieux qu'un 502 après plusieurs minutes (cf. gdl.enumerate).
                logger.info("universal scan: budget épuisé, résultat partiel (%s)", url)
            return items, None
        if getattr(err, 'kind', None) == 'unsupported':
            # gallery-dl n'a pas d'extracteur : on restitue le comportement
            # historique (1 média) pour que les hôtes vettés atteignent yt-dlp.
            match.paginated = False
            return ([{'url': url, 'title': url, 'thumbnail': None,
                      'type': 'video', 'platform': 'generic'}], None)
        # Reste (dont kind='empty') : on remonte l'erreur telle quelle. La route
        # (routes/scrape.py) traite désormais TOUT kind='empty' — de N'IMPORTE
        # QUELLE source gdl-backed — comme un scan réussi sans résultat (200,
        # count=0) plutôt qu'un 502 ; dupliquer ce traitement ici serait mort du
        # jour où cette source a arrêté d'être la SEULE à honorer la règle. Auth /
        # 429 / DDoS-Guard / erreur outil restent des erreurs à remonter : ne
        # JAMAIS déguiser un blocage en « aucune image trouvée ».
        #
        # `err or ...` : filet latent (finding #6) — `gdl.enumerate` ne renvoie
        # aujourd'hui jamais `([], None)` (voir sa docstring : chaque branche vide
        # pose un GdlError kind='empty'), mais SI ça arrivait, `err` vaudrait None
        # et le fallback serait un str NU sans `.kind` → la route le traiterait
        # comme un vrai échec (502) sur un résultat pourtant vide. Le repli reste
        # donc lui aussi une GdlError kind='empty', pas une phrase brute.
        return None, err or GdlError("Nothing to scan at this URL.", 'empty')

    def download(self, url, dest_base):
        """Surface héritée sans appelant dans cette app : le vrai chemin de
        téléchargement est `_download_scrape_item` (fetch durci de `item['url']`, cf. `reddit.py`)."""
        # 1) gallery-dl (extracteur dédié) d'abord.
        dest_dir = os.path.dirname(dest_base)
        filename = os.path.basename(dest_base)
        ok, abs_path, err = gdl.download(url, dest_dir, filename)
        if ok and abs_path:
            return True, os.path.basename(abs_path), None
        # 2) gallery-dl ne supporte pas le site → yt-dlp, mais seulement si l'hôte
        #    est vetté (atténuation SSRF, cf. spec décision #6). On teste le KIND,
        #    jamais le texte du message.
        if getattr(err, 'kind', None) == 'unsupported':
            if not _host_vetted(url):
                return False, None, "Site not supported (gallery-dl) and host not vetted for yt-dlp."
            return netfetch.download_via_ytdlp(url, dest_base)
        # 3) auth/réseau (pas 'unsupported') → on remonte l'erreur gallery-dl.
        return False, None, err or "Generic download failed."


registry.register(UniversalSource())
