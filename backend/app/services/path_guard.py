"""A dataset and a bank must never share bytes — they only ever TRANSIT.

Images move between a bank and a dataset by COPY, in both directions
(``start_dataset_import`` one way, ``_promote_job`` the other), so each side
owns its files and curating one can never mutate the other. That is an
invariant, not a feature: the moment a bank's source folder IS a dataset's
storage folder, 🗑 Delete rejected stops deleting the bank's rejects and starts
deleting the dataset's images.

Nothing enforced it. ``overlapping_banks`` only ever compared banks against
banks, and ``POST /bank/create`` accepted any readable folder — including the
one a dataset lives in, which the UI made people go looking for by hand because
it displayed the path nowhere. This module is the comparison half of the fix; it
is deliberately free of Flask, of the DB and of the models so it can be tested
directly and reused wherever a folder is accepted.

Why paths are compared through ``realpath`` + ``normcase`` and not as strings:
the same folder has many spellings. Different separators, a different case (on
Windows), a relative path, a `..` hop, and — the sharp one — a symlink or an NTFS
junction whose path looks completely unrelated. ``realpath`` collapses all of
those, including links, on both platforms; ``normcase`` folds case and separators
where the filesystem does (Windows) and is a no-op where it does not (POSIX), so
the guarantee is stated correctly on each rather than assumed on both.
"""
from __future__ import annotations

import os

from .. import config as cfg

# What a refused folder should make the user do INSTEAD. A refusal that does not
# name the alternative turns a trap into a wall — and the alternative already
# exists and already does the right thing (it copies).
ALTERNATIVE = ('To re-triage a dataset\'s images in a bank, open the dataset and use '
               '🗃 Import to bank — it COPIES them into a bank of their own, so '
               'curating the bank can never touch the dataset.')


def norm(path) -> str | None:
    """One canonical spelling of a folder, or None when there is nothing to
    compare. Resolves links/junctions and folds case + separators where the
    filesystem does. Trailing separators are dropped so ``x`` and ``x/`` match,
    while a drive/filesystem root keeps its own."""
    raw = str(path or '').strip().strip('"\'')
    if not raw:
        return None
    try:
        resolved = os.path.normcase(os.path.realpath(raw))
    except (OSError, ValueError):
        return None
    trimmed = resolved.rstrip('\\/')
    return trimmed or resolved


def relation(a, b) -> str | None:
    """How folder ``a`` stands to folder ``b``: 'same', 'inside' (a is under b),
    'contains' (a holds b), or None when the two trees are disjoint.

    Both directions matter here. A bank INSIDE the datasets tree lists a
    dataset's files; a bank that CONTAINS it walks recursively and lists every
    dataset's files. Only the disjoint case is safe."""
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return None
    if na == nb:
        return 'same'
    # Compare on a separator boundary, never with a bare startswith: `.../data2`
    # must not read as being inside `.../data`.
    if na.startswith(nb + os.sep):
        return 'inside'
    if nb.startswith(na + os.sep):
        return 'contains'
    return None


def dataset_folder_conflict(folder, *, datasets_root=None) -> dict | None:
    """The refusal payload for a folder that would make a bank share a dataset's
    files, or None when the folder is the user's own and shares nothing.

    Every dataset stores its images at ``dataset_images_root()/<dataset id>``
    (see ``services.dataset_storage``), so ONE comparison against that root
    settles every dataset at once — no per-dataset walk, and datasets created
    after this check still land on the right side of it.

    Returns {'relation', 'scope', 'dataset_id', 'path', 'message'} where scope is
    'dataset' (this folder is one dataset's storage, or sits inside it) or 'root'
    (it is the datasets root itself, or a folder containing it)."""
    try:
        root = str(datasets_root) if datasets_root else str(cfg.dataset_images_root())
    except OSError:
        return None
    rel = relation(folder, root)
    if rel is None:
        return None
    dataset_id = None
    if rel == 'inside':
        nfolder, nroot = norm(folder), norm(root)
        head = os.path.relpath(nfolder, nroot).split(os.sep)[0]
        if head.isdigit():
            dataset_id = int(head)
    return {
        'relation': rel,
        'scope': 'dataset' if rel == 'inside' else 'root',
        'dataset_id': dataset_id,
        'path': root,
        'message': _message(rel, dataset_id),
    }


def _message(rel, dataset_id) -> str:
    """English, and it says three things: what this folder is, why that is
    refused, and what to do instead."""
    if rel == 'inside':
        who = f'dataset #{dataset_id}' if dataset_id is not None else 'a dataset'
        what = f'That folder belongs to {who} — it is where the app stores its images.'
    elif rel == 'same':
        what = ('That folder is the datasets folder itself — it holds the images of '
                'every dataset.')
    else:
        what = ('That folder contains the datasets folder, so a bank there would '
                'list every dataset\'s images.')
    return (f'{what} A bank and a dataset must never share files on disk: a bank '
            'points at a LIVE folder, so 🗑 Delete rejected in the bank would delete '
            f'images out of the dataset. {ALTERNATIVE}')
