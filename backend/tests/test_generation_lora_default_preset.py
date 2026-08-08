"""The two halves the generation-LoRA preset feature was missing.

(1) A configured preset could not apply on its own. The run panel opened on
    "None" on EVERY visit, so the request carried no preset name and the finished
    image's metadata showed no LoRA at all — which reads, from the outside, as an
    app ignoring its own settings. The fix is a per-engine DEFAULT preset key.
    What is pinned here is the part that lives in the backend: the keys exist,
    ship EMPTY (so an existing install sees nothing change), survive a partial
    save, and stay INDEPENDENT per engine — klein.generation_lora_presets and
    krea.generation_lora_presets are separate lists where one name can designate
    two different chains.

(2) The elected Krea base is now published so a screen can name it. Nothing in
    the election changes; `resolve_krea_unet()` stays the one authority, and the
    probe simply reports what it says.

Nothing here renders anything: no GPU second, no paid call.
"""


# --- (1) the default-preset keys --------------------------------------------

def test_both_engines_ship_the_key_empty(app):
    """An install that never touches this must behave exactly as before."""
    from app import config as cfg
    conf = cfg.load_config()
    for engine in ('klein', 'krea'):
        assert conf[engine]['default_generation_lora_preset'] == '', engine


def test_the_two_engines_carry_SEPARATE_defaults(app):
    """One name, two chains: a shared key would apply Klein's pick to Krea."""
    from app import config as cfg
    cfg.save_config({'klein': {'default_generation_lora_preset': 'Skin'},
                     'krea': {'default_generation_lora_preset': 'Bypass'}})
    conf = cfg.load_config()
    assert conf['klein']['default_generation_lora_preset'] == 'Skin'
    assert conf['krea']['default_generation_lora_preset'] == 'Bypass'


def test_a_default_survives_an_unrelated_save(app):
    """Settings saves are partial. A grounding tweak must not clear the default."""
    from app import config as cfg
    cfg.save_config({'krea': {'default_generation_lora_preset': 'Bypass'}})
    cfg.save_config({'krea': {'grounding_px': 768}})
    conf = cfg.load_config()
    assert conf['krea']['default_generation_lora_preset'] == 'Bypass'
    assert conf['krea']['grounding_px'] == 768


def test_a_default_survives_the_klein_lora_migration(app):
    """_migrate_klein_loras rewrites the klein block on every load and on save.
    The default lives in that same block, so it is exactly where a migration
    could silently drop it."""
    from app import config as cfg
    cfg.save_config({'klein': {'default_generation_lora_preset': 'Skin',
                               'generation_loras': [   # the legacy flat format
                                   {'file': 'klein/a.safetensors', 'strength': 0.6}]}})
    conf = cfg.load_config()
    assert conf['klein']['default_generation_lora_preset'] == 'Skin'
    # the migration still did its job
    assert [p['name'] for p in conf['klein']['generation_lora_presets']] \
        == [cfg.MIGRATED_LORA_PRESET_NAME]
    assert 'generation_loras' not in conf['klein']


def test_the_default_names_a_preset_but_never_injects_one(app):
    """The default is a UI STARTING POINT, not a server-side injection: a request
    that names no preset still resolves to no extra LoRAs. Otherwise a run whose
    picker was set to None for that run would silently get the default anyway —
    the exact opposite of giving the user the last word."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    from app.services import krea_edit_helper as krh
    presets = [{'name': 'Skin', 'loras': [{'file': 'klein/a.safetensors', 'strength': 0.6}]}]
    cfg.save_config({'klein': {'generation_lora_presets': presets,
                               'default_generation_lora_preset': 'Skin'},
                     'krea': {'generation_lora_presets': presets,
                              'default_generation_lora_preset': 'Skin'}})
    assert keh.resolve_generation_lora_preset('') == []
    assert keh.resolve_generation_lora_preset(None) == []
    assert krh.resolve_generation_lora_preset('') == []
    # …and naming it still resolves the chain, unchanged.
    assert keh.resolve_generation_lora_preset('Skin') == \
        [{'file': 'klein/a.safetensors', 'strength': 0.6}]


def test_an_unknown_default_is_the_callers_problem_not_a_crash(app):
    """Fail-closed all the way down: the key is free-form (config.json is hand
    editable), so a stale name must survive load and resolve to nothing."""
    from app import config as cfg
    from app.services import klein_edit_helper as keh
    cfg.save_config({'klein': {'default_generation_lora_preset': 'Gone'}})
    assert cfg.load_config()['klein']['default_generation_lora_preset'] == 'Gone'
    assert keh.resolve_generation_lora_preset('Gone') == []


# --- (2) the elected Krea base is published ---------------------------------

_TURBO = 'Krea' + chr(92) + 'krea2_turbo_fp8_scaled.safetensors'
_PIN = 'Krea' + chr(92) + 'my_pin.safetensors'


def _probe(app, monkeypatch, resolved):
    """probe() with the Krea resolver stubbed. It needs an app context, and the
    30 s cache has to be cleared or a probe from an earlier test answers here."""
    from app import capabilities
    from app.services import krea_edit_helper as krh
    monkeypatch.setattr(krh, 'resolve_krea_unet', lambda selected=None: resolved)
    with app.app_context():
        capabilities._cache = None
        capabilities._cache_ts = 0.0
        return capabilities.probe(force=True)


def test_the_probe_publishes_the_base_resolve_krea_unet_elects(app, monkeypatch):
    """The SAME resolver the generation path calls, not a second ranking."""
    caps = _probe(app, monkeypatch, _TURBO)
    assert caps['comfyui']['krea_base_resolved'] == _TURBO


def test_no_base_on_disk_publishes_an_empty_string_not_None(app, monkeypatch):
    """The front renders this straight into a sentence; None would print 'null'."""
    caps = _probe(app, monkeypatch, None)
    assert caps['comfyui']['krea_base_resolved'] == ''


def test_the_published_base_is_whatever_the_resolver_says_pin_included(app, monkeypatch):
    """A user who pinned a file must see THAT file named, not the election's —
    the whole point is that the line answers 'what will this run load?'. The pin
    is honoured inside resolve_krea_unet (covered by its own tests); what this
    pins is that the probe reports its answer verbatim rather than re-deriving
    one, which is how the two surfaces would eventually name different files."""
    caps = _probe(app, monkeypatch, _PIN)
    assert caps['comfyui']['krea_base_resolved'] == _PIN
