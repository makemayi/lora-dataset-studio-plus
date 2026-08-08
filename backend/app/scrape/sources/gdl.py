# app/scrape/sources/gdl.py
"""Moteur gallery-dl réutilisable : énumération (--simulate -j) + classification
des codes de sortie. Générique (tag `platform` paramétrable) — hoist du wrapper
ad-hoc d'erome, avec la correction du sentinel d'erreur type -1 (auth/429/DDoS-
Guard étaient silencieusement lus comme « aucun média »).

Sécurité : --ignore-config, shell=False, args en liste, jamais --exec."""
import json
import os
import subprocess
import sys
import time
import logging
from urllib.parse import urlparse

from .base import ResultList

logger = logging.getLogger(__name__)

GDL_TIMEOUT = 60
DOWNLOAD_TIMEOUT = 300
DEFAULT_MAX_ITEMS = 120
DEFAULT_MAX_ALBUMS = 8
_VIDEO_EXTS = ('mp4', 'webm', 'mov', 'm4v')

# Budget de temps par défaut d'un scan (deadline d'`enumerate()`) quand l'appelant
# n'en pose pas explicitement. DÉRIVÉ de GDL_TIMEOUT (pas une constante indépendante) :
# le scan top-level (1 sous-process, incompressible) peut À LUI SEUL consommer
# GDL_TIMEOUT avant même que la boucle d'albums ne regarde le deadline une seule
# fois, et le sous-process d'album déjà lancé au moment où le budget expire va lui
# aussi jusqu'à son terme — le vrai pire cas est donc ≈ 2×GDL_TIMEOUT, PAS cette
# constante (cf. commentaire de `enumerate`). +30s de marge : assez pour qu'un
# petit listing d'albums finisse normalement, trop court pour qu'une page
# pathologique bloque le worker Flask plusieurs minutes.
#
# DÉFAUT (pas juste disponible) : chaque source gdl-backed (image_sites, civitai,
# fapello, erome, sexcom, gdl_source…) lance jusqu'à 1 + max_albums sous-process à
# GDL_TIMEOUT chacun dans UNE requête Flask synchrone ; universal.py était la SEULE
# à passer un deadline explicite, donc la seule protégée. Une protection que chaque
# appelant doit se souvenir de demander est une protection qu'un nouvel appelant
# oubliera — cf. finding #4.
DEFAULT_SCAN_BUDGET_SECONDS = GDL_TIMEOUT + 30

# Codes de sortie gallery-dl (bitmask, gallery_dl/exception.py) — vérifié sur 1.32.3.
EXIT_HTTP = 4
EXIT_NOTFOUND = 8
EXIT_AUTH = 16
EXIT_UNSUPPORTED = 64


def classify_exit(returncode):
    """Code de sortie gallery-dl → failure_kind ('unsupported'|'auth'|'network'|
    'toolerror') ou None si 0. Le bitmask est OR-combiné ; on teste du plus
    spécifique (unsupported) au plus générique.

    Le test de succès est l'ÉGALITÉ à 0, pas la faussité Python (`not returncode`).
    `subprocess.run` ne produit jamais `None`, mais un double de test bâclé le
    peut — `not None` vaut True, ce qui ferait passer ce cas pour un succès (puis
    'empty' au point d'appel, cf. `_run_simulate`) au lieu d'un échec non
    classifié. `None` est explicitement `'toolerror'` : ni un succès, ni un code
    bitmask exploitable."""
    if returncode == 0:
        return None
    if returncode is None:
        return 'toolerror'
    if returncode & EXIT_UNSUPPORTED:
        return 'unsupported'
    if returncode & EXIT_AUTH:
        return 'auth'
    if returncode & (EXIT_NOTFOUND | EXIT_HTTP):
        return 'network'
    return 'toolerror'


class GdlError(str):
    """Message d'erreur gallery-dl porteur de son `kind` classifié.

    Sous-classe de `str` : tous les appelants qui la traitent comme un message
    continuent de fonctionner sans changement. `kind` (cf. classify_exit) est LA
    donnée sur laquelle brancher ; la phrase, elle, n'est que pour l'utilisateur.

    Pourquoi cette classe existe : universal.py comparait l'erreur à
    'gallery-dl : unsupported' — une forme que PERSONNE n'émettait (espace
    parasite) — ce qui a rendu le repli yt-dlp injoignable sans qu'aucun test ne
    rougisse. Un kind transporté comme donnée ne peut plus se perdre ainsi.

    Vocabulaire complet des `kind` (5 valeurs) :
    - 4 dérivées du code de sortie gallery-dl par `classify_exit()` :
      'unsupported' | 'auth' | 'network' | 'toolerror' ;
    - 1 assignée au point d'appel, PAS par `classify_exit()` : 'empty', posée
      par `enumerate()` quand gallery-dl a répondu sans erreur mais qu'aucun
      média n'a été trouvé (page valide, contenu absent). Volontairement
      distincte de 'toolerror' : fusionner les deux rendrait un vrai échec
      outil et un résultat vide légitime indiscernables pour l'appelant."""

    def __new__(cls, message, kind=None):
        obj = super().__new__(cls, message)
        obj.kind = kind
        return obj


def _media_item(entry, platform):
    """Entrée gallery-dl type 3 = [3, media_url, meta] → item schéma commun, ou None."""
    if not isinstance(entry, (list, tuple)) or len(entry) < 3:
        return None
    media_url = entry[1]
    meta = entry[2] if isinstance(entry[2], dict) else {}
    if not isinstance(media_url, str) or not media_url:
        return None
    ext = str(meta.get('extension', '')).lower()
    if not ext:
        ext = os.path.splitext(urlparse(media_url).path)[1].lstrip('.').lower()
    media_type = 'video' if ext in _VIDEO_EXTS else 'image'
    return {
        'url': media_url,
        'title': meta.get('title') or '',
        'thumbnail': meta.get('thumbnail') or (media_url if media_type == 'image' else None),
        'type': media_type,
        'platform': platform,
    }


def _run_simulate(url, max_items, cookies, extra_opts, image_range=None):
    """`gallery-dl --ignore-config --simulate -j` → (entries|None, error|None). Ne lève jamais.

    `image_range` (ex. '101-200') borne la FENÊTRE d'images du listing (image-range) ;
    défaut '1-{max_items}'. Sert la pagination « Charger plus » des sources à images
    directes (Civitai) où `--range` borne le flux (≠ pornpics qui empile des galeries
    → `--chapter-range` dans extra_opts)."""
    cmd = [sys.executable, '-m', 'gallery_dl', '--ignore-config',
           '--simulate', '-j', '--range', image_range or f'1-{max_items}']
    if cookies:
        cmd += ['--cookies', cookies]
    if extra_opts:
        cmd += list(extra_opts)
    cmd += ['--', url]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=GDL_TIMEOUT, shell=False)
    except subprocess.TimeoutExpired:
        return None, GdlError(f"gallery-dl: timed out ({GDL_TIMEOUT}s).", 'network')
    except Exception as e:
        logger.warning("gallery-dl: échec %s: %s", url, e)
        return None, GdlError(f"gallery-dl: failed ({e}).", 'toolerror')

    stdout = (proc.stdout or '').strip()
    if not stdout:
        kind = classify_exit(proc.returncode)
        if kind is None:
            # Code 0 (succès) mais rien sur stdout : gallery-dl a tourné sans
            # incident et n'a juste rien trouvé — un scan vide légitime, PAS un
            # échec outil non classifié (cf. docstring de GdlError : le kind
            # 'empty' existe pour que l'appelant ne le confonde pas avec
            # 'toolerror'). classify_exit() ne renvoie None QUE pour returncode=0.
            kind = 'empty'
        last = ((proc.stderr or '').strip().splitlines() or ['no data'])[-1]
        return None, GdlError(f"gallery-dl: {kind} ({last[:200]}).", kind)
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError) as e:
        return None, GdlError(f"gallery-dl: unreadable response ({e}).", 'toolerror')
    if not isinstance(data, list):
        return None, GdlError("gallery-dl: unexpected format.", 'toolerror')
    return data, None


def _error_sentinel(entries):
    """Si une entrée type -1 (erreur d'extracteur) est présente, renvoie son message."""
    for entry in entries:
        if isinstance(entry, (list, tuple)) and entry and entry[0] == -1:
            meta = entry[1] if len(entry) > 1 and isinstance(entry[1], dict) else {}
            return (meta.get('message') or meta.get('error')
                    or "gallery-dl: the extractor failed.")
    return None


def enumerate(url, *, platform='generic', max_items=DEFAULT_MAX_ITEMS,
              max_albums=DEFAULT_MAX_ALBUMS, cookies=None, extra_opts=None,
              image_range=None, per_album=None, deadline=None):
    """Énumère les médias d'une URL via gallery-dl. Retourne (items, error).

    Gère les types de message : -1 (erreur → remontée), 2 (header, ignoré),
    3 (média), 6 (album enfant → récursion ≤ max_albums). Ne lève jamais.

    `image_range` borne la fenêtre d'images du listing TOP-LEVEL (pagination des
    sources à images directes) ; la récursion d'albums garde le défaut 1-max_items.

    `per_album` borne les images remontées PAR ALBUM lors de la récursion type 6
    (per_album=1 → la cover de chaque album, pas son contenu). Ne touche PAS les
    médias top-level : scanner l'URL d'un album précis rend toujours tout l'album.

    `deadline` : budget de temps global, un timestamp ABSOLU `time.monotonic()`
    (pas une durée) — ça permet à l'appelant comme au test de le poser directement
    (un deadline déjà expiré déclenche la coupure sans attendre). None (défaut) =
    PAS « pas de budget » : `DEFAULT_SCAN_BUDGET_SECONDS` s'applique automatiquement
    (cf. sa docstring — DÉFAUT, pas opt-in, pour que chaque source gdl-backed en
    hérite sans avoir à y penser). Ne borne QUE la récursion d'albums (jusqu'à
    1 + max_albums sous-process gallery-dl, chacun jusqu'à GDL_TIMEOUT) : couper le
    scan top-level n'aurait aucun sens, un seul sous-process y suffit toujours — ce
    qui veut dire que le budget lui-même ne borne PAS le pire cas réel (≈
    2×GDL_TIMEOUT, cf. `DEFAULT_SCAN_BUDGET_SECONDS`). Au dépassement : retourne
    les items déjà collectés (`partial=True` sur le résultat) plutôt qu'une erreur
    — un scan tronqué reste plus utile qu'un 502 après plusieurs minutes."""
    if deadline is None:
        deadline = time.monotonic() + DEFAULT_SCAN_BUDGET_SECONDS
    try:
        entries, err = _run_simulate(url, max_items, cookies, extra_opts,
                                     image_range=image_range)
        if err:
            return None, err
        # CORRECTION clé : remonter le sentinel d'erreur type -1 (auth/429/DDoS-Guard)
        # AVANT de conclure « aucun média » (le bug d'origine d'erome).
        sentinel = _error_sentinel(entries)
        if sentinel:
            return None, GdlError(sentinel, 'toolerror')

        items = []
        for entry in entries:
            if isinstance(entry, (list, tuple)) and entry and entry[0] == 3:
                item = _media_item(entry, platform)
                if item:
                    items.append(item)
                    if len(items) >= max_items:
                        return items[:max_items], None
        if items:
            return items[:max_items], None

        # Aucun média direct → récurser les albums (type 6).
        album_urls = []
        for entry in entries:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[0] == 6:
                if isinstance(entry[1], str) and entry[1]:
                    album_urls.append(entry[1])
                    if len(album_urls) >= max_albums:
                        break
        def _from_albums(collected):
            """Enveloppe les items collectés via la récursion d'albums avec leur
            provenance (`from_albums`, `partial`) — cf. docstring de
            `base.ResultList`."""
            out = ResultList(collected[:max_items])
            out.from_albums = True
            out.partial = timed_out
            return out

        album_errors = []
        timed_out = False
        for album_url in album_urls:
            # Budget global dépassé → on s'arrête là où on en est. Vérifié EN DÉBUT
            # de boucle (jamais pendant un sous-process déjà lancé, cf. docstring) :
            # au moins la 1ère itération part toujours, même deadline déjà expiré.
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                break
            # per_album posé → borner la simulation elle-même (--range 1-N) : gallery-dl
            # s'arrête après N images au lieu d'énumérer tout l'album pour rien.
            sub, sub_err = _run_simulate(album_url, max_items, cookies, extra_opts,
                                         image_range=f'1-{per_album}' if per_album else None)
            if sub_err:
                album_errors.append(sub_err)
                continue
            if not sub:
                continue
            sent = _error_sentinel(sub)
            if sent:
                # Même enveloppe que le sentinel TOP-LEVEL ci-dessus : sans GdlError,
                # ce message brut (un str nu) n'a pas de `.kind` et un appelant qui
                # fait `getattr(err, 'kind', None)` le lit comme None au lieu de
                # 'toolerror' — silencieusement mal aiguillé.
                album_errors.append(GdlError(sent, 'toolerror'))
                continue
            taken = 0
            for entry in sub:
                if isinstance(entry, (list, tuple)) and entry and entry[0] == 3:
                    item = _media_item(entry, platform)
                    if item:
                        items.append(item)
                        taken += 1
                        if len(items) >= max_items:
                            return _from_albums(items), None
                        if per_album and taken >= per_album:
                            break
        if items:
            return _from_albums(items), None
        # RÉGRESSION CORRIGÉE (finding #1) : `album_errors` doit gagner sur
        # `timed_out`, PAS l'inverse. Un scan d'albums BLOQUÉ (auth/429/DDoS-Guard)
        # collecte ses erreurs AVANT que le budget n'expire — si on regardait
        # `timed_out` en premier, un blocage qui a eu le temps de traverser tous
        # les albums (ou une partie) avant le dépassement du deadline se ferait
        # écraser par kind='empty' : la page est activement en train de nous
        # bloquer, pas « vide », et le message de l'extracteur (le seul indice
        # exploitable pour l'utilisateur) disparaissait. Avant cette vague (pas de
        # deadline du tout), le même scan remontait déjà l'erreur d'album — cet
        # ordre restaure ce comportement même quand le budget expire EN PLUS.
        if album_errors:
            return None, album_errors[0]
        if timed_out:
            # Budget épuisé avant le moindre item collecté ET sans qu'aucun album
            # n'ait renvoyé d'erreur (le cas ci-dessus l'aurait absorbé) : même
            # convention que « aucun média trouvé » ci-dessous (kind='empty', PAS
            # une erreur) — un scan tronqué sans résultat reste un résultat vide
            # légitime, pas un échec outil (cf. docstring `deadline`).
            return None, GdlError(
                "gallery-dl: time budget exhausted before any album could be scanned.",
                'empty')
        return None, GdlError("gallery-dl: no media found.", 'empty')
    except Exception as e:  # garde-fou ultime
        logger.exception("gdl.enumerate: erreur inattendue")
        return None, GdlError(f"gallery-dl: unexpected error ({e}).", 'toolerror')


def download(url, dest_dir, filename, *, cookies=None, extra_opts=None):
    """Télécharge RÉELLEMENT via gallery-dl dans `dest_dir` avec un nom déterministe.
    Retourne (ok, abs_path|None, error|None). Ne lève jamais. Sécurité : --ignore-config,
    shell=False, args en liste, séparateur -- avant l'URL."""
    cmd = [sys.executable, '-m', 'gallery_dl', '--ignore-config',
           '-D', dest_dir, '-o', f'filename={filename}_{{num}}.{{extension}}',
           '--no-part', '--no-mtime']
    if cookies:
        cmd += ['--cookies', cookies]
    if extra_opts:
        cmd += list(extra_opts)
    cmd += ['--', url]
    try:
        os.makedirs(dest_dir, exist_ok=True)
        before = set(os.listdir(dest_dir))
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=DOWNLOAD_TIMEOUT, shell=False)
    except subprocess.TimeoutExpired:
        # NB : un éventuel fichier partiel n'est pas nettoyé ici (hors périmètre).
        return False, None, GdlError("gallery-dl: download timed out.", 'network')
    except Exception as e:
        logger.warning("gallery-dl download: échec %s: %s", url, e)
        return False, None, GdlError(f"gallery-dl: failed ({e}).", 'toolerror')

    if proc.returncode:
        kind = classify_exit(proc.returncode)
        last = ((proc.stderr or '').strip().splitlines() or [''])[-1]
        return False, None, GdlError(f"gallery-dl: {kind or 'failed'} ({last[:200]}).", kind)

    # Chemin produit : 1) parser le stdout (gallery-dl imprime les chemins écrits) ;
    # 2) repli = le fichier le plus récent apparu dans dest_dir.
    for line in reversed((proc.stdout or '').splitlines()):
        line = line.strip()
        if line and os.path.isfile(line):
            return True, line, None
    after = set(os.listdir(dest_dir)) - before
    if after:
        newest = max((os.path.join(dest_dir, f) for f in after), key=os.path.getmtime)
        return True, newest, None
    return False, None, GdlError("gallery-dl: no file produced.", 'toolerror')
