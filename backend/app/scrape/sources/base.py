# app/scrape/sources/base.py
"""Interface commune des sources de scraping (registry pluggable).

DÉPENDANCE-FREE (abc/dataclasses/typing seulement) pour éviter tout cycle
d'import : les sources concrètes importent `validators` paresseusement dans
leur match(), jamais ce module."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Capabilities:
    """Déclare ce qu'une source sait faire. Lu par les appelants (routes /
    download_service) pour router sans connaître la source concrète."""
    can_enumerate_profile: bool = False         # profil/niche/album vs média unique
    needs_auth: bool = False                     # cookies requis (hint UX, pas un gate dur)
    media_kinds: frozenset = field(default_factory=lambda: frozenset({'video'}))
    own_downloader: bool = False                 # True=source.download() ; False=yt-dlp universel
    polite: bool = False                         # sleep/limit-rate (civitai/x/pornpics/…)
    is_universal_fallback: bool = False          # exactement UNE source (priorité 0)


@dataclass
class Match:
    """Handle de résolution : l'URL + le ValidationResult parsé (ou None pour la
    source universelle) + la Source qui a matché (posée par le registry).

    `page` (0-based) est posé par la route /scan pour la pagination « Charger plus »
    des sources paginables (cf. Source.paginated) ; les autres sources l'ignorent.
    `paginated` permet à scan() d'override ce défaut pour une URL précise (par ex.
    un média unique au sein d'une source qui gère aussi des listings)."""
    url: str
    validation: object = None
    source: object = None
    page: int = 0
    paginated: Optional[bool] = None


class ResultList(list):
    """Items renvoyés par une source de scraping, porteurs de métadonnées de
    provenance (même patron que `gdl.GdlError` sous-classant `str`) : les
    appelants qui traitent le retour comme une simple liste continuent de
    fonctionner sans changement ; ceux qui veulent savoir consultent les
    attributs.

    Foyer public : ce type est un CONTRAT DE SOURCE (au même titre que
    `Source`/`Match`/`Capabilities` ci-dessus), pas un détail d'implémentation
    de gallery-dl — `gdl.py` l'utilise, mais `redgifs.py` et `instagram.py`
    (ports autonomes, sans gallery-dl) le réutilisent aussi pour signaler
    leurs propres troncatures. Vivait avant comme `gdl._ResultList` (nom
    PRIVÉ) : deux modules sans rapport avec gallery-dl l'importaient quand
    même, couplés à un module qu'ils n'utilisent pas ailleurs pour un type qui
    n'a aucun invariant propre à gallery-dl.

    - `from_albums` : True si TOUS les items proviennent de la récursion
      d'albums de `gdl.enumerate` (type 6), pas des médias top-level de la
      page. Ne borne cette récursion QU'au nombre d'albums (`max_albums`),
      jamais par un offset de page — donc une source qui annoncerait la
      pagination sur ces items enverrait « Charger plus » dans le vide (cf.
      le repli `unsupported` qui, lui, coupe déjà `match.paginated` pour la
      même raison). Non pertinent pour redgifs.py/instagram.py (toujours
      False côté leurs producteurs).
    - `partial` : True si le scan s'est arrêté avant d'avoir tout exploré
      (budget de temps gdl.enumerate épuisé en cours de récursion d'albums,
      page RedGifs refusée après une page réussie, itération Instagram
      interrompue en cours de route) — les items présents restent valides,
      il en manque potentiellement. Lu par `routes/scrape.py` via
      `getattr(items, 'partial', False)`, sans connaître la source concrète."""
    from_albums = False
    partial = False


class Source(ABC):
    """Une source de scraping. `match(url)` renvoie un Match si l'URL la concerne,
    sinon None. `scan(match)` énumère les médias. `download(url, dest_base)` n'est
    appelé QUE si capabilities.own_downloader (sinon l'appelant passe par yt-dlp)."""
    name: str = 'source'
    priority: int = 0
    capabilities: Capabilities = Capabilities()
    paginated: bool = False   # scan() honore match.page → la route expose « Charger plus »
    # Catégorie d'accès : 'image' = ouvert aux non-admins (avec la feature scrape),
    # 'video' = RÉSERVÉ à l'admin (scan ET download refusés aux non-admins). Défaut
    # 'video' = fail-closed : une nouvelle source est admin-only tant qu'on ne la classe
    # pas explicitement 'image'.
    category: str = 'video'

    @abstractmethod
    def match(self, url: str) -> Optional[Match]:
        ...

    @abstractmethod
    def scan(self, match: Match) -> tuple[list, Optional[str]]:
        """Retourne (items: list, error: str|None). Ne lève jamais."""
        ...

    def download(self, url: str, dest_base: str) -> tuple[bool, Optional[str], Optional[str]]:
        """Retourne (ok: bool, filename: str|None, error: str|None).
        Défaut : non supporté (les sources own_downloader=False passent par yt-dlp)."""
        raise NotImplementedError(f"{self.name} n'a pas de downloader dédié")
