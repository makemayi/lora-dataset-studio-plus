"""WHO wrote a caption — the one vocabulary, and the one place that stamps it.

A caption is a single Text column on both image tables, and until this module a
caption typed by hand and a caption produced by a model were the same string in
the same column.  A forced caption pass ("Re-caption") therefore could not spare
the first without also sparing the second, and said so on screen: it warned that
it was about to destroy hand-written work it had no way to recognise.

NO NEW VOCABULARY.  Both values are lifted from columns that already exist on
the SAME tables, so a reader who knows one knows this one:

  * ``ASSERTED`` ('asserted') is ``BankImage.face_cluster_origin``'s value, with
    the same meaning and the same consequence — a human declared this, so the
    pass SKIPS the row instead of overwriting the user's word.
  * ``ENGINES`` ('joycaption' | 'ollama') is ``BankImage.watermark_source``'s
    idea: record WHICH engine decided, because a bank is captioned over weeks
    and can hold both (the 'auto' backend even writes with both engines inside a
    single run).

  * NULL is the third state and it is the one that matters most, because it is
    what every row that predates these columns carries.  It does NOT mean
    "machine": it means the origin was never recorded.  Those rows CAN be
    re-captioned — the alternative would make Re-caption inert on every bank
    that exists today — but the screen never folds them into the machine-written
    ones: "keeping the 3 you wrote" and "47 whose origin was never recorded" are
    two different sentences about two different things.

THE VALUES ARE STORED IN USER DATABASES.  They are frozen: renaming one needs an
alias table, exactly like the variation labels and the What's-new ids.

PROTECTION IS ABOUT TEXT, NOT ABOUT A MARKER.  ``is_protected`` requires a
NON-EMPTY caption as well as the 'asserted' stamp, so a row whose caption was
cleared can never stay protected on an empty string — that would be a row the
pass skips forever, and it would never be visible from the screen.
"""

from sqlalchemy import and_, or_

# A human wrote or corrected this text.  Same token, same contract as
# BankImage.face_cluster_origin: the pass skips it.
ASSERTED = 'asserted'

# Which engine produced the text.  Same idea as BankImage.watermark_source.
JOYCAPTION = 'joycaption'
OLLAMA = 'ollama'
ENGINES = (JOYCAPTION, OLLAMA)

VALUES = (ASSERTED,) + ENGINES


def engine_origin(engine):
    """The origin value for a resolved captioning engine, or None.

    'auto' is deliberately NOT a value: it is a CHAIN (JoyCaption first, Ollama
    for what it missed), so recording it would name a policy instead of the
    engine that actually wrote the sentence — the very thing watermark_source
    exists to avoid.  Callers that cannot tell which half ran pass None and the
    row stays NULL, which reads as "never recorded" rather than as a wrong
    attribution.
    """
    name = (engine or '').strip().lower()
    return name if name in ENGINES else None


def stamp(row, text, origin, *, field='caption'):
    """Write a caption AND its origin together, in one gesture.

    Every writer goes through here rather than assigning ``row.caption`` and
    remembering a second line: the second line is exactly what the next writer
    added six months later would forget, and a forgotten stamp is not a missing
    feature — it is a WRONG label, on the side that loses the user's work.

    An empty text clears the origin too (see the module header).  The TEXT is
    stored exactly as handed over — '' stays '' and None stays None — because
    each writer already decided which of the two blanks it means, and this helper
    is not the place to quietly change one into the other.
    """
    setattr(row, field, text)
    setattr(row, f'{field}_origin', origin if (text or '').strip() else None)
    return row


def is_protected(row, *, field='caption'):
    """True when this row carries text a human asserted — what the pass skips."""
    return (bool((getattr(row, field, None) or '').strip())
            and getattr(row, f'{field}_origin', None) == ASSERTED)


def protected_clause(model, field='caption'):
    """SQL for :func:`is_protected` — the rows a forced pass must leave alone."""
    column = getattr(model, field)
    return and_(column.isnot(None), column != '',
                getattr(model, f'{field}_origin') == ASSERTED)


def unprotected_clause(model, field='caption'):
    """SQL for its complement, spelled out rather than negated.

    ``not_(protected_clause(...))`` would be WRONG: in SQL, NOT(true AND NULL)
    is NULL, so every row whose origin was never recorded — i.e. every row that
    predates this column — would silently fall out of a forced pass and never be
    re-captioned at all.  The three-valued logic is the bug; naming the three
    cases is the fix.
    """
    column = getattr(model, field)
    origin = getattr(model, f'{field}_origin')
    return or_(column.is_(None), column == '',
               origin.is_(None), origin != ASSERTED)


def unrecorded_clause(model, field='caption'):
    """SQL for "has text, and nobody recorded who wrote it".

    Its own count, never folded into the machine-written ones: the pass rewrites
    both, but only one of the two is a risk the user should be told about.
    """
    column = getattr(model, field)
    return and_(column.isnot(None), column != '',
                getattr(model, f'{field}_origin').is_(None))
