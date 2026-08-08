"""The Krea 2 default base is ELECTED from what is on disk, not a frozen filename.

WHAT WAS WRONG
--------------
`krea2_turbo.json` node 20 carried the literal `Krea\\krea2_turbo_fp8.safetensors`,
and `lora_test_studio` repeated that basename as a constant. Two separate defects
hid behind one string:

  * it is NOT the file Setup installs — `setup_installer` fetches Comfy-Org's
    `krea2_turbo_fp8_scaled.safetensors`. ComfyUI validates a loader widget by
    exact string match against the list it publishes, so on an install that
    simply followed Setup the Studio's default named a file that is not there and
    the prompt was refused before a step ran;
  * the file it does name is a community repack that carries tensors this family
    never declares. Measured on the real header (12.90 GB): 432 tensors where the
    family's own full-precision checkpoint has 430, the two extras being
    `last.down.weight` / `last.up.weight` `[6144, 6144]`, which the file's OWN
    `__metadata__` (`egg_format = chw_m1p1_flat`, `egg_w = egg_h = 6144`,
    `egg_c = 1`) describes as an embedded image — about 75 MB of picture inside a
    base model. Two other community repacks from a different repacker carry the
    same four keys and the same two tensors.

WHAT THESE TESTS PIN
--------------------
The ELECTION, not the outcome on one machine: the ranking, and the two places it
is allowed to be surprising — the sampling regime outranks file quality (the
graph is guidance-distilled, so a pristine Raw checkpoint is a worse default than
a quantized Turbo one), and a payload-carrying file is ranked LAST rather than
removed, so an install where it is the only Krea base still gets a default.
"""
import importlib
import json
import os
import struct

import pytest


# --- fixtures ---------------------------------------------------------------

def _fresh_config(monkeypatch, tmp_path):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as config
    importlib.reload(config)
    return config


@pytest.fixture
def krea(monkeypatch, tmp_path):
    """krea_edit_helper + a throwaway ComfyUI tree, with the header caches clear."""
    config = _fresh_config(monkeypatch, tmp_path)
    base = tmp_path / 'Comfy'
    for sub in ('diffusion_models', 'unet', 'loras', 'text_encoders', 'vae'):
        (base / 'models' / sub).mkdir(parents=True, exist_ok=True)
    config.save_config({'comfyui': {'base_dir': str(base)}})
    from app.services import comfy_model_paths, model_integrity
    comfy_model_paths.clear_cache()
    model_integrity.clear_cache()
    import app.services.krea_edit_helper as keh
    importlib.reload(keh)
    yield keh, base / 'models' / 'diffusion_models' / 'Krea'
    comfy_model_paths.clear_cache()
    model_integrity.clear_cache()


def _header(meta=None, tensors=None):
    """Bytes of a safetensors file that carries only a header — the only part any
    of this reads. `tensors` is {name: dtype}."""
    obj = {}
    if meta:
        obj['__metadata__'] = {k: str(v) for k, v in meta.items()}
    for name, dtype in (tensors or {}).items():
        obj[name] = {'dtype': dtype, 'shape': [2, 2], 'data_offsets': [0, 0]}
    blob = json.dumps(obj).encode('utf-8')
    return struct.pack('<Q', len(blob)) + blob


# The four keys the community repacks declare, measured. `egg_format` names the
# raster layout; the three others are its dimensions.
_EGG = {'egg_format': 'chw_m1p1_flat', 'egg_w': 6144, 'egg_h': 6144, 'egg_c': 1}
_WEIGHTS = {'last.linear.weight': 'BF16', 'last.norm.scale': 'F32'}
_CAST = {'last.linear.weight': 'F8_E4M3', 'last.norm.scale': 'F8_E4M3'}
_PACKED = {'last.linear.weight': 'F8_E4M3', 'last.linear.scale_weight': 'F32'}


def _put(folder, name, data):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(data)
    return folder / name


# --- the detector -----------------------------------------------------------

def test_the_dimensions_alone_are_not_enough_to_call_it_a_payload(krea, tmp_path):
    """The MECHANISM is `egg_format` — the key that names a raster layout. Three
    numbers on their own could be anything, and a detector that fired on them
    would start refusing files for carrying a stray metadata key."""
    from app.services import model_integrity as mi
    dims = _put(tmp_path / 'x', 'dims.safetensors',
                _header({'egg_w': 6144, 'egg_h': 6144, 'egg_c': 1}, _WEIGHTS))
    assert mi.foreign_payload_report(dims)['present'] is False
    full = _put(tmp_path / 'x', 'full.safetensors', _header(_EGG, _WEIGHTS))
    report = mi.foreign_payload_report(full)
    assert report['present'] is True
    assert report['kind'] == mi.PAYLOAD_EMBEDDED_RASTER
    # The signals are the keys that decided it — not a restatement of the verdict.
    assert 'egg_format' in report['signals']
    assert report['note'] and 'full.safetensors' in report['note']


def test_a_clean_checkpoint_announces_nothing(krea, tmp_path):
    from app.services import model_integrity as mi
    clean = _put(tmp_path / 'x', 'clean.safetensors', _header(None, _WEIGHTS))
    assert mi.foreign_payload_report(clean) == {
        'present': False, 'kind': '', 'signals': [], 'note': None}


def test_an_unreadable_header_is_not_a_clean_bill_of_health(krea, tmp_path):
    """`present=False` means 'nothing announced', and the docstring says so. This
    pins that an HTML gate page — which announces nothing because it is not a
    model at all — does not silently become a preferred base."""
    from app.services import model_integrity as mi
    gate = _put(tmp_path / 'x', 'gate.safetensors', b'<!doctype html><html>')
    assert mi.foreign_payload_report(gate)['present'] is False
    # …and the structural validator is the one that catches it, unchanged.
    assert mi.validate_model_file(str(gate))['blocking'] is True


# --- the ranking ------------------------------------------------------------

def test_the_ranks_are_ordered_worst_last(krea):
    """A reshuffle of these constants would silently invert the election."""
    from app.services import model_integrity as mi
    assert (mi.HEALTH_FULL_PRECISION < mi.HEALTH_BARE_CAST
            < mi.HEALTH_PACKED_EXPORT < mi.HEALTH_FOREIGN_PAYLOAD)


def test_a_full_precision_file_that_carries_a_payload_still_ranks_last(krea, tmp_path):
    """The interesting half of the rule: precision does NOT redeem a file that
    says it carries something other than weights. The measured repack is fp8, so
    a test using an fp8 fixture would pass for the wrong reason — the dtype — and
    prove nothing about the payload."""
    from app.services import model_integrity as mi
    bf16_with_payload = _put(tmp_path / 'x', 'a.safetensors', _header(_EGG, _WEIGHTS))
    plain_bf16 = _put(tmp_path / 'x', 'b.safetensors', _header(None, _WEIGHTS))
    assert mi.base_health(bf16_with_payload)['rank'] == mi.HEALTH_FOREIGN_PAYLOAD
    assert mi.base_health(plain_bf16)['rank'] == mi.HEALTH_FULL_PRECISION


def test_a_packed_export_ranks_between_a_cast_and_a_payload(krea, tmp_path):
    from app.services import model_integrity as mi
    cast = _put(tmp_path / 'x', 'cast.safetensors', _header(None, _CAST))
    packed = _put(tmp_path / 'x', 'packed.safetensors', _header(None, _PACKED))
    assert mi.base_health(cast)['rank'] == mi.HEALTH_BARE_CAST
    assert mi.base_health(packed)['rank'] == mi.HEALTH_PACKED_EXPORT


# --- the election -----------------------------------------------------------

def test_the_file_setup_installs_wins_over_any_community_turbo(krea):
    keh, kdir = krea
    _put(kdir, keh.KREA_CANONICAL_UNET, _header(None, _CAST))
    _put(kdir, 'someTURBOrepack.safetensors', _header(None, _WEIGHTS))
    assert keh.resolve_krea_unet() == os.path.join('Krea', keh.KREA_CANONICAL_UNET)


def test_the_sampling_regime_outranks_the_file_quality(krea):
    """A pristine bf16 RAW build loses to a quantized TURBO one. Not a preference:
    the Krea graphs pin cfg 1 and a few steps, which is what a guidance-distilled
    build IS — a Raw base renders mush at those settings."""
    keh, kdir = krea
    _put(kdir, 'aaa_raw_bf16.safetensors', _header(None, _WEIGHTS))
    _put(kdir, 'zzz_turbo_fp8.safetensors', _header(None, _CAST))
    assert keh.resolve_krea_unet() == os.path.join('Krea', 'zzz_turbo_fp8.safetensors')


def test_a_payload_carrier_never_wins_against_a_sibling_of_its_own_tier(krea):
    """Both are 'turbo' builds, so the regime cannot separate them; the header
    does. Name order would have picked the payload carrier, which is exactly how
    the old default was chosen."""
    keh, kdir = krea
    _put(kdir, 'aaa_turbo.safetensors', _header(_EGG, _WEIGHTS))
    _put(kdir, 'bbb_turbo.safetensors', _header(None, _PACKED))
    assert keh.resolve_krea_unet() == os.path.join('Krea', 'bbb_turbo.safetensors')


def test_the_only_krea_base_on_disk_stays_the_default_even_carrying_a_payload(krea):
    """The install this must not break: one Krea file, and it is the booby-trapped
    one. Ranked last, never removed — a missing default is a dead engine."""
    keh, kdir = krea
    _put(kdir, 'krea2_turbo_fp8.safetensors', _header(_EGG, _CAST))
    assert keh.resolve_krea_unet() == os.path.join('Krea', 'krea2_turbo_fp8.safetensors')


def test_a_base_the_family_knows_renders_noise_is_never_elected(krea):
    """BigLove* carries 'krea' and renders pure noise under this pipeline —
    already measured, already excluded from the Generate resolver's folders. The
    elector applies it too, so a list that happens to include one cannot promote
    it to default."""
    keh, _kdir = krea
    assert keh.elect_krea_base(['Krea\\BigLoveKreaEdit1_fp8mixed.safetensors']) is None
    assert keh.elect_krea_base([
        'Krea\\BigLoveKreaEdit1_fp8mixed.safetensors',
        'Krea\\plain.safetensors']) == 'Krea\\plain.safetensors'


def test_a_file_no_loader_can_open_is_never_elected(krea):
    keh, _kdir = krea
    assert keh.elect_krea_base(['Krea\\notes_about_krea.txt']) is None


def test_an_explicit_pick_is_never_second_guessed(krea):
    """The user's own `krea.base_model` outranks the whole ranking — including the
    payload rule. Choosing is not a mistake to protect people from."""
    keh, kdir = krea
    _put(kdir, 'krea2_turbo_fp8.safetensors', _header(_EGG, _CAST))
    _put(kdir, 'clean_turbo.safetensors', _header(None, _WEIGHTS))
    assert keh.resolve_krea_unet(selected='krea2_turbo_fp8.safetensors') == (
        os.path.join('Krea', 'krea2_turbo_fp8.safetensors'))


def test_setup_installs_exactly_the_name_the_resolver_calls_canonical():
    """The drift guard. The canonical name is not a guess about the community's
    filenames — it is this app's own installation contract, and if the download
    changes without this constant, the elected default silently stops being the
    file Setup puts on disk."""
    from app.services import krea_edit_helper as keh
    from app import setup_installer
    dest = setup_installer._KREA_DOWNLOADS['krea_model']['dest']
    assert dest[-1] == keh.KREA_CANONICAL_UNET
    assert setup_installer._KREA_DOWNLOADS['krea_model']['url'].endswith(
        keh.KREA_CANONICAL_UNET)


# --- what the Studio says and loads -----------------------------------------

def test_the_official_label_is_kept_for_the_file_setup_installs(app, krea, monkeypatch):
    from app.services import lora_test_studio as lts
    keh, _kdir = krea
    with app.app_context():
        monkeypatch.setattr(lts, 'get_krea_models',
                            lambda: ['Krea\\' + keh.KREA_CANONICAL_UNET])
        entry = lts.krea_default_base_entry()
        assert entry['value'] == ''          # the id never moves — it is persisted
        assert entry['label'] == 'Official – Krea 2 Turbo'
        assert entry['note'] is None


def test_a_local_build_is_named_for_what_it_is_not_called_official(app, krea, monkeypatch):
    from app.services import lora_test_studio as lts
    keh, _kdir = krea
    with app.app_context():
        monkeypatch.setattr(lts, 'get_krea_models', lambda: ['Krea\\myTurboMerge.safetensors'])
        entry = lts.krea_default_base_entry()
        assert entry['value'] == ''
        assert 'Official' not in entry['label']
        assert 'myTurboMerge' in entry['label']
        # …and the note names the file Setup would have installed AND the gesture
        # that fixes it. A note that only diagnoses leaves the reader with a fact
        # and nothing to do with it.
        assert keh.KREA_CANONICAL_UNET in entry['note']
        assert 'Setup ▸ Install' in entry['note']
        assert keh.KREA_ASSETS['krea_model']['path'] in entry['note']


def test_the_note_says_it_when_the_only_base_carries_a_payload(app, krea, monkeypatch):
    """« s'il n'y a que lui, il reste le défaut, mais dis-le »."""
    from app.services import lora_test_studio as lts
    keh, kdir = krea
    _put(kdir, 'krea2_turbo_fp8.safetensors', _header(_EGG, _CAST))
    with app.app_context():
        monkeypatch.setattr(lts, 'get_krea_models',
                            lambda: ['Krea\\krea2_turbo_fp8.safetensors'])
        note = lts.krea_default_base_entry()['note']
        assert note and 'egg_format' in note
        # No alternative to offer, so no picker — the note must not depend on one.
        assert lts.krea_alt_base_models() == []


def test_the_official_entry_loads_the_elected_base_into_node_20(app, krea, monkeypatch):
    from app.services import lora_test_studio as lts
    with app.app_context():
        monkeypatch.setattr(lts, 'get_krea_models', lambda: ['Krea\\elected_turbo.safetensors'])
        wf = {'20': {'inputs': {'unet_name': 'Krea\\krea2_turbo_fp8.safetensors'}}}
        lts.apply_krea_lora_test_settings(
            wf, lora_name=None, strength=1.0, prompt='p', seed=1, width=8, height=8,
            base_model=None)
        assert wf['20']['inputs']['unet_name'] == 'Krea\\elected_turbo.safetensors'


def test_nothing_on_disk_leaves_the_workflow_value_alone(app, krea, monkeypatch):
    """An unconfigured ComfyUI must not blank the node — the missing-asset
    preflight owns that case and says something actionable about it."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        monkeypatch.setattr(lts, 'get_krea_models', lambda: [])
        wf = {'20': {'inputs': {'unet_name': 'Krea\\krea2_turbo_fp8.safetensors'}}}
        lts.apply_krea_lora_test_settings(
            wf, lora_name=None, strength=1.0, prompt='p', seed=1, width=8, height=8,
            base_model=None)
        assert wf['20']['inputs']['unet_name'] == 'Krea\\krea2_turbo_fp8.safetensors'


def test_the_alternatives_never_repeat_the_elected_default(app, krea, monkeypatch):
    """Same base copied at a root AND in a subfolder is ONE model: the picker must
    not offer it twice under two labels."""
    from app.services import lora_test_studio as lts
    with app.app_context():
        monkeypatch.setattr(lts, 'get_krea_models', lambda: [
            'Krea\\my_turbo.safetensors', 'my_turbo.safetensors',
            'Krea\\other.safetensors'])
        assert lts.krea_alt_base_models() == ['Krea\\other.safetensors']
