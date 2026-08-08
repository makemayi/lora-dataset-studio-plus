"""🗃️ Bank grid: ordering by ANY measured quantity, and hiding by word.

Two halves of one surface (the listing query), pinned together because they are
used together — "sharpest first, minus what I already captioned".

SORT. The grid used to order only by resolution, then by three quantities; it now
offers every figure a pass persists on bank_image, both ways. What must hold:
  * every advertised id (GRID_SORTS) is actually accepted and actually orders —
    a menu entry the server ignores is worse than a missing one;
  * rows the pass never reached (NULL) sink to the END in BOTH directions;
  * an unknown id degrades to the default order instead of 500 — the value comes
    from a user's localStorage and may predate/outlive any given build.

EXCLUDE. The inverse of the search bar, for working a captioned bank as a
checklist. What must hold:
  * it hides what mentions the word, in the caption OR the file name;
  * an image with NO caption is NEVER hidden — that is the whole point ("what
    have I not done yet"), and it is also the SQL trap: NOT (NULL LIKE x) is
    NULL, which drops the row unless the caption is coalesced;
  * several terms in one field, and composition with the search and with `total`
    (a filter that narrows the page but not the count is a lie);
  * the curation pool sees it too — hiding an image in the grid must keep it out
    of "pick 60 diverse".
"""
from app.extensions import db
from app.models import BankImage
from app.services import image_bank_service as banks
from test_image_bank import _mkbank, flat


# One column per sortable id, with the value written straight into the row: this
# is about the ORDER BY, not about re-testing each pass's arithmetic.
COLUMN_OF = {
    'res': 'width', 'size': 'file_size', 'aesthetic': 'aesthetic_score',
    'nsfw': 'nsfw_score', 'sharp': 'blur_score', 'noise': 'noise_score',
    'flat': 'uniformity_score', 'detail': 'detail_ratio', 'bars': 'bars_ratio',
    'jpeg': 'jpeg_quality', 'face': 'face_det',
}


def _mk3(client, tmp_path):
    """Three images, deterministic id order a.png < b.png < c.png."""
    bank_id, _src = _mkbank(client, tmp_path, {
        'a.png': flat(), 'b.png': flat(), 'c.png': flat()})
    return bank_id


def _rows_by_name(bank_id):
    return {r.relpath.replace('\\', '/').rsplit('/', 1)[-1]: r
            for r in BankImage.query.filter_by(bank_id=bank_id).all()}


def _set(app, bank_id, values):
    """values = {'a.png': {'caption': …, 'blur_score': …}, …}"""
    with app.app_context():
        rows = _rows_by_name(bank_id)
        for name, cols in values.items():
            for col, val in cols.items():
                setattr(rows[name], col, val)
        db.session.commit()


def _names(client, bank_id, **params):
    r = client.get(f'/api/bank/{bank_id}/images', query_string=params)
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    return ([i['name'] for i in body['images']], body['total'])


# --- sort -------------------------------------------------------------------

def test_every_advertised_sort_orders_both_ways_and_sinks_the_unmeasured(
        client, tmp_path, app):
    """The central assertion, run over EVERY id the menu offers: b > a on the
    measure, c never measured. Descending reads b, a, c — ascending reads a, b,
    c. `c` is last in BOTH, never first: a "worst first" order that opens on the
    pile nobody measured is worse than no order at all."""
    bank_id = _mk3(client, tmp_path)
    for key, column in COLUMN_OF.items():
        # Clear every sortable column first, so the previous loop turn cannot
        # tie-break this one.
        _set(app, bank_id, {n: {c: None for c in COLUMN_OF.values()}
                            for n in ('a.png', 'b.png', 'c.png')})
        # 'res' ranks on width*height, so height must not be NULL for a and b.
        extra = {'height': 10} if key == 'res' else {}
        _set(app, bank_id, {'a.png': {column: 2, **extra},
                            'b.png': {column: 9, **extra},
                            'c.png': {column: None}})
        desc, total = _names(client, bank_id, sort=f'{key}_desc')
        assert desc == ['b.png', 'a.png', 'c.png'], f'{key}_desc gave {desc}'
        assert total == 3, f'{key}: sorting must never change membership'
        asc, _ = _names(client, bank_id, sort=f'{key}_asc')
        assert asc == ['a.png', 'b.png', 'c.png'], f'{key}_asc gave {asc}'


def test_the_menu_and_the_server_advertise_the_same_ids():
    """GRID_SORTS is what the UI menu is built from; every entry must round-trip
    through _sort_order, or the menu offers an order the server silently ignores."""
    assert len(set(banks.GRID_SORTS)) == len(banks.GRID_SORTS)
    for sort_id in banks.GRID_SORTS:
        assert banks._sort_order(sort_id) is not None, sort_id


def test_an_unknown_or_stale_sort_id_degrades_to_the_default_order(
        client, tmp_path, app):
    """The value arrives from localStorage — from any build, ever."""
    bank_id = _mk3(client, tmp_path)
    _set(app, bank_id, {'a.png': {'blur_score': 1}, 'b.png': {'blur_score': 9}})
    for bogus in ('', 'sharp', 'sharp_sideways', 'face_score_desc', 'drop table'):
        names, total = _names(client, bank_id, sort=bogus)
        assert names == ['a.png', 'b.png', 'c.png'], bogus
        assert total == 3


# --- exclude ----------------------------------------------------------------

def test_exclude_hides_the_word_and_never_hides_an_uncaptioned_image(
        client, tmp_path, app):
    """The SQL trap: with a bare `NOT (caption LIKE …)`, c.png — caption NULL —
    would vanish, which is the exact opposite of what a checklist asks for."""
    bank_id = _mk3(client, tmp_path)
    _set(app, bank_id, {'a.png': {'caption': 'a woman in a red dress'},
                        'b.png': {'caption': 'a blue sky'},
                        'c.png': {'caption': None}})
    names, total = _names(client, bank_id, exclude='red')
    assert names == ['b.png', 'c.png']
    # The count follows the filter — a page that narrows while the total keeps
    # advertising 3 images sends the user hunting through empty pages.
    assert total == 2
    # Case-insensitive, like the search it mirrors.
    assert _names(client, bank_id, exclude='RED')[0] == ['b.png', 'c.png']


def test_exclude_matches_the_file_name_too_and_takes_several_terms(
        client, tmp_path, app):
    """Same two fields as the search bar (caption + path), and one field holds a
    comma-separated list, because hiding 'logo, watermark' is one gesture."""
    bank_id, _src = _mkbank(client, tmp_path, {
        'shot.png': flat(), 'logo_overlay.png': flat(), 'clean.png': flat()})
    _set(app, bank_id, {'shot.png': {'caption': 'a watermark in the corner'}})
    assert _names(client, bank_id, exclude='logo')[0] == ['clean.png', 'shot.png']
    assert _names(client, bank_id, exclude='logo, watermark')[0] == ['clean.png']
    # Blank and comma-only fields filter nothing (a mid-keystroke value).
    assert len(_names(client, bank_id, exclude=' , ')[0]) == 3


def test_exclude_composes_with_search_and_with_the_other_facets(
        client, tmp_path, app):
    """'dress, but not the red one' is a legitimate question, and the answer must
    also survive a status facet."""
    bank_id = _mk3(client, tmp_path)
    _set(app, bank_id, {'a.png': {'caption': 'red dress'},
                        'b.png': {'caption': 'blue dress'},
                        'c.png': {'caption': 'red car'}})
    assert _names(client, bank_id, search='dress', exclude='red')[0] == ['b.png']
    with app.app_context():
        _rows_by_name(bank_id)['b.png'].status = 'reject'
        db.session.commit()
    assert _names(client, bank_id, search='dress', exclude='red',
                  status='pending')[0] == []


def test_exclude_escapes_like_metacharacters(client, tmp_path, app):
    """A literal % or _ in the term matches itself — it is not a wildcard that
    silently empties the grid."""
    bank_id = _mk3(client, tmp_path)
    _set(app, bank_id, {'a.png': {'caption': '100% cotton'},
                        'b.png': {'caption': 'plain cotton'},
                        'c.png': {'caption': 'wool'}})
    assert _names(client, bank_id, exclude='100%')[0] == ['b.png', 'c.png']
    # '%' alone would match everything if it were passed through as a wildcard.
    assert _names(client, bank_id, exclude='%')[0] == ['b.png', 'c.png']


def test_the_curation_pool_hides_what_the_grid_hides(client, tmp_path, app):
    """Otherwise 'pick 60 diverse' hands back the very images the user just
    declared done."""
    bank_id = _mk3(client, tmp_path)
    _set(app, bank_id, {'a.png': {'caption': 'red dress'},
                        'b.png': {'caption': 'blue dress'}})
    with app.app_context():
        pool = banks._pool_query(bank_id, banks.thresholds(), exclude='red')
        assert sorted(r.relpath.replace('\\', '/').rsplit('/', 1)[-1]
                      for r in pool.all()) == ['b.png', 'c.png']


# --- the lean id snapshot (▶ Review / Select all in filter) ------------------

def test_ids_only_answers_the_whole_filter_in_the_grid_order(client, tmp_path, app):
    """▶ Review needs a SNAPSHOT of ids, and used to get it by walking the grid
    500 rows at a time — 46 requests and 16 MB of image payloads on a 22 940-image
    bank, to keep 23 000 integers. The lean answer must be the SAME list, in the
    SAME order, or the two paths would disagree about what the filter contains."""
    bank_id = _mk3(client, tmp_path)
    _set(app, bank_id, {'a.png': {'aesthetic_score': 2, 'caption': 'a red logo'},
                        'b.png': {'aesthetic_score': 9, 'caption': 'a blue sky'},
                        'c.png': {'aesthetic_score': None, 'caption': None}})

    def lean(**params):
        r = client.get(f'/api/bank/{bank_id}/images',
                       query_string={**params, 'ids_only': '1'})
        assert r.status_code == 200, r.get_json()
        body = r.get_json()
        # The lean answer carries ids and NOTHING else — no image payload sneaking
        # back in, which is the whole point of the path.
        assert set(body) == {'ids'}, body
        return body['ids']

    def paged(**params):
        r = client.get(f'/api/bank/{bank_id}/images', query_string=params)
        return [i['id'] for i in r.get_json()['images']]

    for params in ({}, {'sort': 'aesthetic_desc'}, {'sort': 'aesthetic_asc'},
                   {'exclude': 'logo'}, {'sort': 'res_desc', 'status': 'pending'}):
        assert lean(**params) == paged(**params), params
    # It really is the WHOLE filter, not one page: the unscored row is in there,
    # last, exactly as the sort promises.
    ordered = lean(sort='aesthetic_desc')
    assert len(ordered) == 3
    assert ordered[-1] == paged(sort='aesthetic_desc')[-1]


# --- 🏷️ tag chips (attributes picked off one image's caption) ----------------

def test_tags_are_ANDed_and_matched_as_whole_words(client, tmp_path, app):
    """The 🏷️ chips narrow: each one ticked must SHRINK the result, never grow
    it. And a chip comes from a caption's own tokens, so it matches as a WORD —
    'car' must not find 'scarf', which is exactly the confusion that would make
    the feature lie about what it found."""
    bank_id, _src = _mkbank(client, tmp_path, {
        'a.png': flat(), 'b.png': flat(), 'c.png': flat(), 'd.png': flat()})
    _set(app, bank_id, {
        'a.png': {'caption': 'a woman in a red dress on a balcony'},
        'b.png': {'caption': 'a woman in a blue dress'},
        'c.png': {'caption': 'a red car in a scarf shop'},
        'd.png': {'caption': None}})

    def names(**params):
        r = client.get(f'/api/bank/{bank_id}/images', query_string=params)
        assert r.status_code == 200, r.get_json()
        return [i['name'] for i in r.get_json()['images']], r.get_json()['total']

    assert names(tags='dress')[0] == ['a.png', 'b.png']
    assert names(tags='red')[0] == ['a.png', 'c.png']
    # AND, not OR: 'red' + 'dress' is the intersection, and it is SMALLER than
    # either chip alone. An OR here would grow the set with every click.
    assert names(tags='red,dress')[0] == ['a.png']
    assert names(tags='red,dress,balcony')[0] == ['a.png']
    # Whole words: 'car' does not match 'scarf'; 'scarf' does not match 'car'.
    assert names(tags='car')[0] == ['c.png']
    assert names(tags='scarf')[0] == ['c.png']
    assert names(tags='ca')[0] == []
    # The count follows the filter, like every other facet.
    assert names(tags='red,dress')[1] == 1


def test_tags_compose_with_the_other_text_filters_without_sharing_their_key(
        client, tmp_path, app):
    """Three text filters, three parameters, no overlap: search narrows, tags
    narrow as words, exclude hides. A shared key is how one feature silently ate
    another's field once already."""
    bank_id, _src = _mkbank(client, tmp_path, {
        'a.png': flat(), 'b.png': flat(), 'c.png': flat()})
    _set(app, bank_id, {'a.png': {'caption': 'a red dress on a balcony'},
                        'b.png': {'caption': 'a red dress in a studio'},
                        'c.png': {'caption': 'a blue dress on a balcony'}})

    def names(**params):
        r = client.get(f'/api/bank/{bank_id}/images', query_string=params)
        return [i['name'] for i in r.get_json()['images']]

    assert names(tags='dress', exclude='studio') == ['a.png', 'c.png']
    assert names(tags='dress,balcony', exclude='blue') == ['a.png']
    assert names(search='balcony', tags='red') == ['a.png']
    # Each parameter keeps its own meaning when they all travel together.
    assert names(search='dress', tags='balcony', exclude='blue') == ['a.png']


def test_a_tag_filter_reaches_the_curation_pool_too(client, tmp_path, app):
    """Hiding images in the grid must keep them out of a curation pick."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat(), 'b.png': flat()})
    _set(app, bank_id, {'a.png': {'caption': 'a red dress'},
                        'b.png': {'caption': 'a blue hat'}})
    with app.app_context():
        pool = banks._pool_query(bank_id, banks.thresholds(), tags='dress')
        assert [r.relpath for r in pool.all()] == ['a.png']


def test_tag_matching_survives_punctuation_and_case(client, tmp_path, app):
    """Captioners end sentences, use semicolons and capitalise. A word next to a
    full stop is still that word."""
    bank_id, _src = _mkbank(client, tmp_path, {'a.png': flat(), 'b.png': flat()})
    _set(app, bank_id, {'a.png': {'caption': 'A woman on a Balcony.'},
                        'b.png': {'caption': 'indoors; no balcony visible'}})

    def names(**params):
        r = client.get(f'/api/bank/{bank_id}/images', query_string=params)
        return [i['name'] for i in r.get_json()['images']]

    assert names(tags='balcony') == ['a.png', 'b.png']
    assert names(tags='woman') == ['a.png']
    assert names(tags='BALCONY') == ['a.png', 'b.png']
    # The file name is part of the haystack, like it is for the search box.
    assert names(tags='a') == ['a.png']
