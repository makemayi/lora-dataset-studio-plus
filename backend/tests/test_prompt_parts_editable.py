"""The five prompt parts that used to be hardcoded, and the composed preview.

Klein and Krea assemble ~1000 characters from six sources. Four of them (the
identity locks) were already editable; these tests cover the five that were not —
the markings hold order, the outfit/expression directives baked into every human
shot, the concrete-garment palette, the rendering tail and the per-framing detail
block — plus the preview endpoint that finally SHOWS the assembled result.

The load-bearing test is the first one: with nothing overridden, every wrapper
must still emit the exact bytes it emitted when those five parts were literals.
That is the invariant the whole override mechanism rests on, and it is checked
against strings rebuilt from the constants, NOT against the wrapper's own output.
"""
import pytest

from app.services import face_variations as fv


@pytest.fixture
def no_config(monkeypatch):
    """No saved override for anything — the shipped path."""
    monkeypatch.setattr(fv, 'get_identity_prompt', fv.get_identity_prompt)
    from app import config as cfg
    monkeypatch.setattr(cfg, 'get', lambda key, default=None: default)
    return None


# --- The invariant: no override -> byte-identical to the hardcoded era --------

def _expected_klein(prompt, st, framing, nsfw, markings):
    """The prompt rebuilt from the raw CONSTANTS, the way the wrappers inlined
    them before any of this was configurable."""
    noun = fv._KLEIN_SUBJECT_NOUN.get(st, 'subject')
    detail = fv._KLEIN_FRAMING_DETAIL_BY_SUBJECT.get(
        st, fv._KLEIN_FRAMING_DETAIL).get(framing or '', '')
    if st == 'human' and framing in ('body', 'back') and not nsfw:
        detail = f'{detail}{fv._BOTTOMS_ANTI_LEAK}' if detail else fv._BOTTOMS_ANTI_LEAK.strip()
    medium = fv._KLEIN_MEDIUM.get(st, fv._KLEIN_MEDIUM_DEFAULT)
    nsfw_tail, sfw_tail = fv._KLEIN_RENDER_TAIL.get(st, fv._KLEIN_RENDER_TAIL_DEFAULT)
    ending = nsfw_tail if nsfw else sfw_tail
    return (f"Create a new {medium} of the same {noun} as the reference image: {prompt}. "
            + (f"{detail} " if detail else "")
            + (fv.KREA_MARKINGS_LOCK if markings else "")
            + f"{fv.identity_prompt_default('klein_identity', st)} {ending}")


@pytest.mark.parametrize('st', ['human', 'animal', 'object', 'creature', 'other', 'anime'])
@pytest.mark.parametrize('framing', ['face', 'bust', 'body', 'back', None, 'nonsense'])
@pytest.mark.parametrize('nsfw', [False, True])
def test_default_path_is_byte_identical_for_klein(no_config, st, framing, nsfw):
    # markings=True: Klein carries the skin-hold order too since 2026-07-27 (the
    # fix was born on Krea, then MEASURED on Klein — a forehead tattoo vanished
    # without it, same seed). "Byte-identical" means identical to what Klein
    # shipped, not to what it shipped before that measurement.
    assert fv.wrap_variation_klein('a shot', nsfw=nsfw, framing=framing,
                                   subject_type=st) == \
        _expected_klein('a shot', st, framing, nsfw, markings=True)


@pytest.mark.parametrize('st', ['human', 'anime', 'animal'])
@pytest.mark.parametrize('framing', ['face', 'body', None])
def test_default_path_is_byte_identical_for_krea(no_config, st, framing):
    # A catalog prompt, so the baked outfit/expression directives are really in it.
    raw = fv.VARIATION_CATALOG[0]['prompt']
    body = fv.krea_outfit_directive(raw, 'Face front, neutral')
    assert fv.wrap_variation_krea(raw, framing=framing, subject_type=st,
                                  label='Face front, neutral') == \
        _expected_klein(body, st, framing, False, markings=True)


def test_default_path_is_byte_identical_for_api_engines(no_config):
    raw = fv.VARIATION_CATALOG[0]['prompt']
    assert fv.wrap_variation(raw) == f'{fv.IDENTITY_GUARD} {raw}'
    assert fv.wrap_variation(raw, ref_count=2) == f'{fv.IDENTITY_GUARD_MULTI} {raw}'


def test_every_new_part_ships_a_real_default():
    for st in fv._IDENTITY_DEFAULTS_BY_SUBJECT:
        for kind in fv.PROMPT_PART_KINDS:
            assert fv.identity_prompt_default(kind, st).strip(), (st, kind)


# --- Each part actually reaches the prompt when overridden -------------------

def _preview(**over):
    """Compose Klein+Krea with an unsaved override tree (flat / human keys)."""
    return over


def test_markings_lock_override_reaches_the_krea_prompt():
    with fv.preview_prompt_overrides({'markings_lock': 'HOLD THE SKIN.'}):
        out = fv.wrap_variation_krea('a shot', framing='bust', label='L')
    assert 'HOLD THE SKIN. ' in out
    assert fv.KREA_MARKINGS_LOCK.strip() not in out


def test_your_own_outfit_directive_beats_the_concrete_garment(no_config):
    """The order of two rewrites, and the only order under which the Settings box
    is honest.

    Both engines that name a concrete garment do it by REPLACING the shipped
    outfit directive. Run that before the overrides and it eats the very text the
    override keys on, so a user who wrote their own directive got the palette
    anyway and no error — the box looked applied and did nothing. Overrides first:
    the garment substitution then finds nothing to replace and yields.
    """
    stored = 'close-up portrait, ' + fv.OUTFIT_VARY
    with fv.preview_prompt_overrides({'outfit_vary': 'wearing a red raincoat'}):
        for out in (fv.wrap_variation_krea(stored, framing='face', label='Face front'),
                    fv.wrap_variation_klein(stored, framing='face', label='Face front')):
            assert 'wearing a red raincoat' in out
            assert fv.OUTFIT_VARY not in out
    # ...and with NO override the palette still dresses the shot, unchanged.
    plain = fv.wrap_variation_krea(stored, framing='face', label='Face front')
    assert fv.OUTFIT_VARY not in plain and 'wearing ' in plain
    assert 'red raincoat' not in plain


def test_render_tail_override_is_per_subject_and_per_rating():
    over = {'render_tail_sfw': 'MY SFW TAIL.', 'render_tail_nsfw': 'MY NSFW TAIL.'}
    with fv.preview_prompt_overrides(over):
        assert fv.wrap_variation_klein('p', framing='bust').endswith('MY SFW TAIL.')
        assert fv.wrap_variation_klein('p', framing='bust', nsfw=True).endswith('MY NSFW TAIL.')
        # Written on the HUMAN screen -> it must not ride on an anime dataset.
        anime = fv.wrap_variation_klein('p', framing='bust', subject_type='anime')
    assert 'MY SFW TAIL.' not in anime
    assert anime.endswith(fv.identity_prompt_default('render_tail_sfw', 'anime'))


def test_framing_detail_override_is_per_framing_and_per_subject():
    over = {'framing_body': 'SHOW THE WHOLE THING.',
            'by_subject': {'anime': {'framing_body': 'DRAWN FULL BODY.'}}}
    with fv.preview_prompt_overrides(over):
        assert 'SHOW THE WHOLE THING. ' in fv.wrap_variation_klein('p', framing='body')
        # another framing is untouched
        assert 'SHOW THE WHOLE THING.' not in fv.wrap_variation_klein('p', framing='face')
        an = fv.wrap_variation_klein('p', framing='body', subject_type='anime')
    assert 'DRAWN FULL BODY. ' in an
    assert 'SHOW THE WHOLE THING.' not in an


def test_outfit_and_expression_overrides_rewrite_a_stored_prompt():
    """The directives are baked into the catalog at import time and PERSISTED in
    variation_prompt, so the override has to be applied at wrap time — which is
    also what makes it reach datasets built before the edit."""
    stored = 'close-up portrait, ' + fv.OUTFIT_VARY + ', ' + fv.EXPRESSION_NEUTRAL
    with fv.preview_prompt_overrides({'outfit_vary': 'wearing anything but a suit',
                                      'expression_neutral': 'looking bored'}):
        api = fv.wrap_variation(stored)
        klein = fv.wrap_variation_klein(stored, framing='face')
    for out in (api, klein):
        assert 'wearing anything but a suit' in out
        assert 'looking bored' in out
        assert fv.OUTFIT_VARY not in out
        assert fv.EXPRESSION_NEUTRAL not in out


def test_outfit_palette_override_is_used_and_is_deterministic():
    import zlib
    palette = 'a red poncho\n\n  a blue kilt  \na green kimono\n'
    with fv.preview_prompt_overrides({'outfit_palette': palette}):
        assert fv.outfit_palette() == ('a red poncho', 'a blue kilt', 'a green kimono')
        got = fv.krea_outfit_for('Bust, front')
        assert got == fv.outfit_palette()[zlib.crc32(b'Bust, front') % 3]
        # ...and the same label gives the same garment on a second call.
        assert fv.krea_outfit_for('Bust, front') == got


def test_emptied_palette_degrades_to_the_shipped_one():
    """"The app must work everywhere": a user who clears the list, or leaves only
    blank lines, gets the shipped garments — never a prompt with no outfit."""
    for junk in ('', '   ', '\n\n\t\n'):
        with fv.preview_prompt_overrides({'outfit_palette': junk}):
            assert fv.outfit_palette() == fv.KREA_OUTFIT_PALETTE


def test_a_corrupt_override_tree_degrades_to_defaults():
    """A hand-edited config (or a truncated request body) must not raise."""
    for junk in ({'by_subject': 'not-a-dict'}, {'by_subject': {'anime': 42}},
                 {'markings_lock': 123}, {'render_tail_sfw': None}):
        with fv.preview_prompt_overrides(junk):
            out = fv.wrap_variation_krea('p', framing='bust', subject_type='anime', label='L')
        assert out and fv.identity_prompt_default('render_tail_sfw', 'anime') in out


def test_blank_override_still_means_follow_the_default():
    with fv.preview_prompt_overrides({'markings_lock': '   ', 'render_tail_sfw': ''}):
        out = fv.wrap_variation_krea('p', framing='bust', label='L')
    assert fv.KREA_MARKINGS_LOCK.strip() in out
    assert out.endswith(fv.identity_prompt_default('render_tail_sfw'))


def test_preview_overrides_do_not_leak_outside_the_block(no_config):
    with fv.preview_prompt_overrides({'render_tail_sfw': 'LEAK.'}):
        pass
    assert 'LEAK.' not in fv.wrap_variation_klein('p', framing='bust')


# --- The composed preview ----------------------------------------------------

def test_preview_returns_the_same_text_the_wrapper_would(no_config):
    p = fv.compose_preview('klein', framing='body')
    assert p['prompt'] == fv.wrap_variation_klein(
        p['shot_prompt'], nsfw=False, framing='body', subject_type='human')
    assert p['length'] == len(p['prompt'])
    assert p['shot_id'] and p['shot_label']


def test_preview_uses_a_real_catalog_shot_of_the_asked_framing():
    for st in ('human', 'animal', 'anime'):
        for fr in fv.PROMPT_FRAMINGS:
            e = fv.preview_shot(st, fr)
            assert e['prompt']
            # every shipped catalog has all four framings; if one ever loses a
            # framing the fallback keeps the preview non-empty rather than blank.
            assert e.get('framing') in (fr, None) or e['prompt']


def test_preview_reflects_unsaved_overrides_not_the_saved_config():
    p = fv.compose_preview('krea', framing='bust',
                           overrides={'markings_lock': 'UNSAVED HOLD.'})
    assert 'UNSAVED HOLD.' in p['prompt']


def test_preview_engine_fallbacks_and_bad_input():
    assert fv.compose_preview('nanobanana')['engine'] == 'nanobanana'
    # unknown / legacy engine id -> Klein, the same rule the client uses
    assert fv.compose_preview('some-legacy-id')['engine'] == 'klein'
    assert fv.compose_preview(None)['engine'] == 'klein'
    # a bogus framing falls back to a real one instead of dropping the block
    assert fv.compose_preview('klein', framing='sideways')['framing'] == 'bust'
    assert fv.compose_preview('klein', subject_type='martian')['subject_type'] == 'human'


def test_preview_route_composes_without_saving_anything(client):
    r = client.post('/api/settings/prompt-preview', json={
        'engine': 'krea', 'framing': 'body', 'subject_type': 'human',
        'identity_prompts': {'markings_lock': 'PREVIEW ONLY.'}})
    assert r.status_code == 200
    assert 'PREVIEW ONLY.' in r.get_json()['prompt']
    # ...and nothing was written: the saved config still follows the default.
    from app import config as cfg
    assert not (cfg.get('identity_prompts.markings_lock') or '')


def test_preview_route_survives_a_malformed_body(client):
    for body in ({}, {'engine': 123}, {'identity_prompts': 'nope'},
                 {'nsfw': 'yes', 'framing': None}):
        r = client.post('/api/settings/prompt-preview', json=body)
        assert r.status_code == 200, body
        assert r.get_json()['prompt']


# --- The markings lock is editable, and that is a sharp edge -----------------

def test_an_override_can_reintroduce_a_summoned_word_which_the_ui_must_warn_about():
    """Making this field editable means a user CAN put `tattoos` back in it — the
    exact wording that had the model painting tattoos on subjects who have none.
    The three anti-summon tests pin the SHIPPED text; nothing can pin a user's.
    This test states the residual risk explicitly so it is never mistaken for
    covered, and the warning that carries it lives in the field's help text
    (pinned by the frontend contract test)."""
    with fv.preview_prompt_overrides({'markings_lock': 'keep the same tattoos'}):
        out = fv.wrap_variation_krea('p', framing='bust', label='L')
    assert 'tattoos' in out
    # the shipped default remains clean whatever a user writes
    assert 'tattoo' not in fv.identity_prompt_default('markings_lock').lower()
