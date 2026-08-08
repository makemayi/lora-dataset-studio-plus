"""Merging a LoRA into a base checkpoint: the arithmetic, and what it refuses.

The interesting halves are (1) that every refusal is decided from HEADERS, so the
button can be disabled with its reason before anyone commits to a 26 GB write,
and (2) that a tensor the merge does not touch is never judged — it is copied
through and DISCLOSED, which is how the ~75 MB of image hidden in a real
community Krea 2 file becomes a line in the plan instead of a silent passenger.

Key layouts here are the MEASURED ones, read off real files:
    base  blocks.0.attn.wk.weight                        BF16 [1536, 6144]
    LoRA  diffusion_model.blocks.0.attn.wk.lora_A.weight      [32, 6144]
          diffusion_model.blocks.0.attn.wk.lora_B.weight      [1536, 32]
(scaled down here, same shape relationship). Krea 2 keeps q/k/v as separate
matrices, so the mapping is direct — there is no fused qkv to slice.
"""
import json
import os
import struct

import pytest

from app.services import lora_merge as lm
from app.services import lora_merge_job as job


# --- fixtures -----------------------------------------------------------------

def _header(entries, metadata=None):
    """A safetensors HEADER dict, the way the readers see one. No file, no bytes."""
    out = {}
    if metadata:
        out['__metadata__'] = metadata
    offset = 0
    for name, (dtype, shape) in entries.items():
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * lm._DTYPE_BYTES[dtype]
        out[name] = {'dtype': dtype, 'shape': list(shape),
                     'data_offsets': [offset, offset + nbytes]}
        offset += nbytes
    return out


def _write_stub(path, entries, metadata=None):
    """A structurally VALID .safetensors whose data block is zeros — enough for
    every header-only reader, and it needs no torch."""
    header = _header(entries, metadata)
    blob = json.dumps(header).encode('utf-8')
    total = max((v['data_offsets'][1] for k, v in header.items()
                 if k != '__metadata__'), default=0)
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(blob)))
        fh.write(blob)
        fh.write(b'\x00' * total)
    return str(path)


# A miniature of the real Krea 2 layout: two blocks, the same key spellings.
_BASE_ENTRIES = {
    'blocks.0.attn.wk.weight': ('BF16', (48, 64)),
    'blocks.0.attn.wq.weight': ('BF16', (64, 64)),
    'blocks.0.prenorm.scale': ('F32', (64,)),
    'blocks.1.attn.wk.weight': ('BF16', (48, 64)),
    'first.weight': ('F32', (64, 8)),
    'last.linear.weight': ('F32', (8, 64)),
}
_LORA_ENTRIES = {
    'diffusion_model.blocks.0.attn.wk.lora_A.weight': ('F16', (4, 64)),
    'diffusion_model.blocks.0.attn.wk.lora_B.weight': ('F16', (48, 4)),
    'diffusion_model.blocks.0.attn.wq.lora_A.weight': ('F16', (4, 64)),
    'diffusion_model.blocks.0.attn.wq.lora_B.weight': ('F16', (64, 4)),
}


def _base_header(extra=None):
    entries = dict(_BASE_ENTRIES)
    entries.update(extra or {})
    return _header(entries)


def _lora_header(entries=None, metadata=None):
    return _header(entries or _LORA_ENTRIES, metadata)


# --- key mapping --------------------------------------------------------------

def test_normalise_key_replaces_only_whole_numeric_segments():
    assert lm.normalise_key('blocks.7.attn.wk.weight') == 'blocks.{i}.attn.wk.weight'
    assert lm.normalise_key('txtfusion.refiner_blocks.1.mlp.up.weight') \
        == 'txtfusion.refiner_blocks.{i}.mlp.up.weight'
    assert lm.normalise_key('last.linear.weight') == 'last.linear.weight'
    # a digit INSIDE a segment is not an index
    assert lm.normalise_key('blocks.0.fc1.weight') == 'blocks.{i}.fc1.weight'


@pytest.mark.parametrize('key,expected,which', [
    ('diffusion_model.blocks.0.attn.wk.lora_A.weight', 'blocks.0.attn.wk.weight', 'A'),
    ('diffusion_model.blocks.0.attn.wk.lora_B.weight', 'blocks.0.attn.wk.weight', 'B'),
    # the older kohya spelling of the same two factors
    ('diffusion_model.blocks.0.mlp.up.lora_down.weight', 'blocks.0.mlp.up.weight', 'A'),
    ('diffusion_model.blocks.0.mlp.up.lora_up.weight', 'blocks.0.mlp.up.weight', 'B'),
    # no prefix at all is still resolvable
    ('blocks.0.attn.wq.lora_A.weight', 'blocks.0.attn.wq.weight', 'A'),
])
def test_base_key_for_maps_both_spellings(key, expected, which):
    base_key, _module, got = lm.base_key_for(key)
    assert (base_key, got) == (expected, which)


def test_base_key_for_ignores_anything_that_is_not_a_factor():
    assert lm.base_key_for('blocks.0.attn.wk.weight') is None
    assert lm.base_key_for('diffusion_model.blocks.0.attn.wk.alpha') is None


def test_lora_modules_pairs_factors_and_drops_a_lonely_one():
    header = _lora_header({
        'diffusion_model.blocks.0.attn.wk.lora_A.weight': ('F16', (4, 64)),
        'diffusion_model.blocks.0.attn.wk.lora_B.weight': ('F16', (48, 4)),
        # B with no A: unmergeable, and it must not reach the writer
        'diffusion_model.blocks.1.attn.wk.lora_B.weight': ('F16', (48, 4)),
    })
    modules = lm.lora_modules(header)
    assert set(modules) == {'blocks.0.attn.wk.weight'}
    slot = modules['blocks.0.attn.wk.weight']
    assert slot['A_key'].endswith('lora_A.weight')
    assert slot['B_key'].endswith('lora_B.weight')
    assert slot['alpha_key'] is None


def test_lora_modules_finds_the_alpha_sibling():
    header = _lora_header({
        'diffusion_model.blocks.0.attn.wk.lora_A.weight': ('F16', (4, 64)),
        'diffusion_model.blocks.0.attn.wk.lora_B.weight': ('F16', (48, 4)),
        'diffusion_model.blocks.0.attn.wk.alpha': ('F32', ()),
    })
    slot = lm.lora_modules(header)['blocks.0.attn.wk.weight']
    assert slot['alpha_key'] == 'diffusion_model.blocks.0.attn.wk.alpha'


# --- scaling ------------------------------------------------------------------

def test_module_scale_without_alpha_is_one_not_one_over_rank():
    """The LoRAs this app trains carry no alpha. Treating that as 1/rank would
    divide every delta by 32 and produce a merge that looks like a no-op."""
    assert lm.module_scale(32, None) == 1.0


@pytest.mark.parametrize('rank,alpha,expected', [
    (32, 32.0, 1.0),
    (32, 16.0, 0.5),
    (8, 4.0, 0.5),
])
def test_module_scale_applies_alpha_over_rank(rank, alpha, expected):
    assert lm.module_scale(rank, alpha) == pytest.approx(expected)


@pytest.mark.parametrize('rank,alpha', [(0, 8.0), (32, 0.0), (32, -1.0),
                                        (32, float('nan')), ('x', 'y')])
def test_module_scale_never_returns_a_nonsense_multiplier(rank, alpha):
    assert lm.module_scale(rank, alpha) == 1.0


def test_weight_for_is_the_single_number_when_nothing_overrides_it():
    assert lm.weight_for({'weight': 0.8}, 'blocks.0.attn.wk.weight') == 0.8
    assert lm.weight_for({'weight': 0.8, 'key_weights': {}}, 'blocks.0.attn.wk.weight') == 0.8


def test_weight_for_accepts_a_per_key_table_most_specific_first():
    """V1's UI never sends this. The seam exists so that the day a per-block ratio
    is MEASURED to help, it is a table handed to this function rather than a
    rewrite of the merge loop."""
    spec = {'weight': 1.0, 'key_weights': {
        'blocks.0.attn.wk.weight': 0.25,          # exact key
        'blocks.{i}.attn.wq.weight': 0.5,         # index-normalised pattern
        'txtfusion.': 0.75,                       # prefix
    }}
    assert lm.weight_for(spec, 'blocks.0.attn.wk.weight') == 0.25
    assert lm.weight_for(spec, 'blocks.3.attn.wq.weight') == 0.5
    assert lm.weight_for(spec, 'txtfusion.refiner_blocks.0.mlp.up.weight') == 0.75
    assert lm.weight_for(spec, 'last.linear.weight') == 1.0      # falls back


def test_weight_for_prefers_the_longest_matching_prefix():
    spec = {'weight': 1.0, 'key_weights': {'blocks.': 0.2,
                                           'blocks.0.attn.': 0.9}}
    assert lm.weight_for(spec, 'blocks.0.attn.wk.weight') == 0.9
    assert lm.weight_for(spec, 'blocks.0.mlp.up.weight') == 0.2


# --- plan_merge: what it accepts ----------------------------------------------

def test_plan_merge_reports_exactly_the_tensors_it_would_touch():
    plan = lm.plan_merge(_base_header(), [('demo.safetensors', _lora_header())])
    assert set(plan['targets']) == {'blocks.0.attn.wk.weight', 'blocks.0.attn.wq.weight'}
    assert plan['touched_tensors'] == 2
    assert plan['base_tensors'] == len(_BASE_ENTRIES)
    assert plan['loras'][0]['rank'] == 4
    assert plan['loras'][0]['has_alpha'] is False


def test_two_loras_touching_the_same_tensor_both_register_on_it():
    plan = lm.plan_merge(_base_header(), [('a.safetensors', _lora_header()),
                                          ('b.safetensors', _lora_header())])
    assert plan['targets']['blocks.0.attn.wk.weight'] == [0, 1]
    assert len(plan['loras']) == 2


def test_output_bytes_is_arithmetic_on_the_header_not_an_estimate():
    """The output keeps every key, shape and dtype of the base, so its size is a
    fact before anything is read — which is what lets the disk check be exact."""
    expected = 0
    for dtype, shape in _BASE_ENTRIES.values():
        numel = 1
        for dim in shape:
            numel *= dim
        expected += numel * lm._DTYPE_BYTES[dtype]
    assert lm.output_bytes(_base_header()) == expected
    # 48*64 bf16 + 64*64 bf16 + 64 f32 + 48*64 bf16 + 64*8 f32 + 8*64 f32
    assert expected == (48 * 64 * 2) + (64 * 64 * 2) + (64 * 4) + (48 * 64 * 2) \
        + (64 * 8 * 4) + (8 * 64 * 4)


# --- plan_merge: what it refuses, and how it says so --------------------------

def test_a_lora_for_another_model_is_refused_by_name():
    foreign = _lora_header({
        'diffusion_model.blocks.9.attn.wk.lora_A.weight': ('F16', (4, 64)),
        'diffusion_model.blocks.9.attn.wk.lora_B.weight': ('F16', (48, 4)),
    })
    with pytest.raises(lm.MergeError) as ei:
        lm.plan_merge(_base_header(), [('other.safetensors', foreign)])
    assert 'blocks.9.attn.wk.weight' in str(ei.value)
    assert 'trained on a different model' in str(ei.value)


def test_factors_that_do_not_multiply_back_to_the_weight_are_refused():
    wrong = _lora_header({
        'diffusion_model.blocks.0.attn.wk.lora_A.weight': ('F16', (4, 32)),   # in=32, base=64
        'diffusion_model.blocks.0.attn.wk.lora_B.weight': ('F16', (48, 4)),
    })
    with pytest.raises(lm.MergeError) as ei:
        lm.plan_merge(_base_header(), [('wrong.safetensors', wrong)])
    assert 'do not multiply back' in str(ei.value)
    assert 'blocks.0.attn.wk.weight' in str(ei.value)


def test_a_quantized_target_tensor_is_refused_with_the_route_out():
    quantized = _base_header()
    quantized['blocks.0.attn.wk.weight']['dtype'] = 'F8_E4M3'
    with pytest.raises(lm.MergeError) as ei:
        lm.plan_merge(quantized, [('demo.safetensors', _lora_header())])
    assert 'F8_E4M3' in str(ei.value)
    assert 'quantize the result' in str(ei.value)


def test_a_file_with_no_factor_pairs_is_not_a_lora():
    with pytest.raises(lm.MergeError) as ei:
        lm.plan_merge(_base_header(), [('base.safetensors', _base_header())])
    assert 'not a LoRA file' in str(ei.value)


def test_an_empty_base_is_refused():
    with pytest.raises(lm.MergeError):
        lm.plan_merge({}, [('demo.safetensors', _lora_header())])


# --- the passenger: disclosed, not refused, not dropped -----------------------

# The two tensors a real community Krea 2 file carries that are not weights:
# [6144, 6144] each, ~75 MB of an image, hiding under the legitimate `last.`
# prefix and declared by egg_* metadata. Scaled down here; the NAMES are real.
_EGG = {'last.down.weight': ('BF16', (64, 64)),
        'last.up.weight': ('BF16', (64, 64))}


def test_a_tensor_the_family_does_not_declare_is_reported_not_refused():
    """The first design refused these outright. That was wrong: a legitimate
    future Krea variant with one extra key would have been refused too, from a
    manifest read off one file on one machine. We copy them and say so."""
    plan = lm.plan_merge(_base_header(_EGG),
                         [('demo.safetensors', _lora_header())], family='krea')
    names = [row['name'] for row in plan['carried_over']]
    assert names == ['last.down.weight', 'last.up.weight']
    assert plan['carried_over_bytes'] == 64 * 64 * 2 * 2
    # and it is still a perfectly good merge
    assert plan['touched_tensors'] == 2


def test_nothing_is_reported_when_the_base_is_only_the_declared_layout():
    plan = lm.plan_merge(_base_header(), [('demo.safetensors', _lora_header())],
                         family='krea')
    assert plan['carried_over'] == [] and plan['carried_over_bytes'] == 0


def test_no_family_means_no_claim_rather_than_a_guess():
    plan = lm.plan_merge(_base_header(_EGG), [('demo.safetensors', _lora_header())],
                         family=None)
    assert plan['carried_over'] == []


def test_the_layout_description_matches_on_name_only_never_dtype():
    """A dense master LDS trained itself has the SAME 430 keys as the reference
    base with different dtypes (BF16 where the reference keeps F32). A
    dtype-sensitive description would call every one of them foreign and report
    a 26 GB 'passenger'."""
    all_bf16 = {name: ('BF16', shape) for name, (_dtype, shape) in _BASE_ENTRIES.items()}
    assert lm.foreign_tensors(_header(all_bf16), 'krea') == []
    all_f32 = {name: ('F32', shape) for name, (_dtype, shape) in _BASE_ENTRIES.items()}
    assert lm.foreign_tensors(_header(all_f32), 'krea') == []


def test_the_real_krea_layout_is_described_by_the_pattern_set():
    """Spot-check the manifest against key spellings taken from a real 430-tensor
    Krea 2 header, including the two txtfusion sub-trees that are easy to miss."""
    for key in ('blocks.27.attn.wo.weight', 'blocks.0.mod.lin', 'first.bias',
                'tproj.0.weight', 'txtmlp.1.scale',
                'txtfusion.layerwise_blocks.1.attn.qknorm.knorm.scale',
                'txtfusion.refiner_blocks.0.mlp.down.weight',
                'txtfusion.projector.weight', 'last.modulation.lin'):
        assert lm.normalise_key(key) in lm.KREA2_KEY_PATTERNS, key
    for intruder in ('last.down.weight', 'last.up.weight'):
        assert lm.normalise_key(intruder) not in lm.KREA2_KEY_PATTERNS


# --- naming and traceability --------------------------------------------------

def test_the_merged_name_keeps_the_stem_and_carries_a_timestamp():
    from datetime import datetime, timezone
    when = datetime(2026, 8, 4, 14, 32, 5, tzinfo=timezone.utc)
    assert lm.merged_name_for('krea2_raw_bf16.safetensors', when=when) \
        == 'krea2_raw_bf16_merged_20260804-143205.safetensors'
    # the Krea prefix the licence expects, and the delivery checks match on
    assert lm.merged_name_for('Krea_lds146_x.safetensors', when=when).startswith('Krea')


def test_the_header_says_it_is_a_merge_and_not_a_trained_model():
    """The one line that has to survive the file being renamed and re-uploaded.
    On the model sites 'finetune' routinely means exactly this object; the point
    of writing it into the header is that the file travels without its UI."""
    meta = lm.merge_metadata('/models/krea2_raw_bf16.safetensors',
                             [{'path': '/loras/mine.safetensors', 'weight': 0.8},
                              {'path': '/loras/turbo.safetensors', 'weight': 1.0}])
    assert meta['lds_merge'] == 'lora_into_base'
    assert meta['lds_merge_base'] == 'krea2_raw_bf16.safetensors'
    assert json.loads(meta['lds_merge_loras']) == [
        {'name': 'mine.safetensors', 'weight': 0.8},
        {'name': 'turbo.safetensors', 'weight': 1.0}]
    assert meta['lds_merge_date'].startswith('20')
    assert 'NOT trained' in meta['lds_merge_note']
    # no machine path leaks into a file that gets published
    assert '/models/' not in json.dumps(meta)


def test_the_duration_estimate_grows_with_the_file():
    assert lm.estimate_seconds(0) >= 1
    small, big = lm.estimate_seconds(10 ** 9), lm.estimate_seconds(26 * 10 ** 9)
    assert big > small > 0


# --- the job: refusals a UI can show ------------------------------------------

@pytest.fixture
def ready(monkeypatch):
    """Pin the worker probe so these tests assert THIS module's refusals, not
    whether the machine running them happens to have torch."""
    monkeypatch.setattr(job.fp8_quantize, 'interpreter',
                        lambda: {'python': 'python', 'ready': True,
                                 'missing': [], 'reason': None})


@pytest.fixture
def base_and_lora(tmp_path):
    base = _write_stub(tmp_path / 'krea2_raw_bf16.safetensors', _BASE_ENTRIES)
    lora = _write_stub(tmp_path / 'mine.safetensors', _LORA_ENTRIES,
                       {'ss_base_model_version': 'krea2'})
    return base, lora


def test_plan_describes_the_whole_operation_without_writing_anything(
        app, tmp_path, ready, base_and_lora):
    base, lora = base_and_lora
    before = sorted(os.listdir(tmp_path))
    with app.app_context():
        info = job.plan(base, [{'path': lora, 'weight': 0.8}])
    assert info['base_name'] == 'krea2_raw_bf16.safetensors'
    assert info['destination_name'].startswith('krea2_raw_bf16_merged_')
    assert info['destination_exists'] is False
    assert info['merged_tensors'] == 2
    assert info['output_bytes'] > 0
    assert info['required_bytes'] == info['output_bytes'] + job.WRITE_HEADROOM_BYTES
    assert info['estimated_seconds'] >= 1
    assert 'never modified' in info['on_failure']
    assert info['loras'][0]['weight'] == 0.8
    assert sorted(os.listdir(tmp_path)) == before, 'plan must write nothing'


def test_an_already_quantized_base_is_refused_and_told_the_way_round(
        app, tmp_path, ready):
    """Not refused because it is impossible — refused because merging into the
    bf16 and quantizing after gives the SAME final file without the double loss.
    A button whose best case is worse than the other button is a trap."""
    # Shaped like the real thing: the community turbo file is 266 F8 tensors to
    # 166 F32, and carries no marker key at all — the majority-of-dtypes signal
    # is what identifies it. That makes it the BARE CAST form.
    entries = {name: ('F32' if name.endswith('.scale') else 'F8_E4M3', shape)
               for name, (_dtype, shape) in _BASE_ENTRIES.items()}
    base = _write_stub(tmp_path / 'krea2_turbo_fp8.safetensors', entries)
    lora = _write_stub(tmp_path / 'mine.safetensors', _LORA_ENTRIES)
    with app.app_context():
        with pytest.raises(job.MergeJobError) as ei:
            job.plan(base, [{'path': lora, 'weight': 1.0}])
    message = str(ei.value)
    assert 'already a quantized export' in message
    assert 'round the numbers twice' in message      # the bare-cast wall
    assert 'same final file without the double loss' in message
    assert 'fp8 tool' in message


def test_the_structured_form_is_refused_for_its_own_reason(app, tmp_path, ready):
    """A scaled export stores W/scale beside a separate scale tensor: there is no
    full-precision weight in the file to add a delta to. Different wall from the
    bare cast, same way out — and naming it is what makes the refusal an
    instruction instead of a prohibition."""
    from app.services import model_integrity
    entries = {name: ('F8_E4M3' if dtype == 'BF16' else dtype, shape)
               for name, (dtype, shape) in _BASE_ENTRIES.items()}
    entries['scaled_fp8'] = ('F8_E4M3', (2,))            # the legacy marker key
    base = _write_stub(tmp_path / 'scaled.safetensors', entries)
    lora = _write_stub(tmp_path / 'mine.safetensors', _LORA_ENTRIES)
    report = model_integrity.quantization_report(base)
    assert report['form'] == model_integrity.FORM_STRUCTURED, report
    with app.app_context():
        with pytest.raises(job.MergeJobError) as ei:
            job.plan(base, [{'path': lora, 'weight': 1.0}])
    message = str(ei.value)
    assert 'no full-precision weight in it to add a LoRA to' in message
    assert 'same final file without the double loss' in message


def test_the_dtype_guard_refuses_a_quantized_base_on_its_own(app, tmp_path):
    """Defence in depth: plan_merge refuses the same files without consulting
    model_integrity at all, so the guard that speaks well can change without the
    safety depending on it."""
    quantized = _base_header()
    quantized['blocks.0.attn.wk.weight']['dtype'] = 'F8_E4M3'
    with pytest.raises(lm.MergeError, match='F8_E4M3'):
        lm.plan_merge(quantized, [('demo.safetensors', _lora_header())])


def test_an_existing_destination_is_refused_rather_than_overwritten(
        app, tmp_path, ready, base_and_lora):
    base, lora = base_and_lora
    with app.app_context():
        info = job.plan(base, [{'path': lora, 'weight': 1.0}])
        open(info['destination'], 'wb').close()
        with pytest.raises(job.MergeJobError, match='already exists'):
            job.plan(base, [{'path': lora, 'weight': 1.0}],
                     destination=info['destination'])
        # and it is allowed when the caller says so on purpose
        again = job.plan(base, [{'path': lora, 'weight': 1.0}],
                         destination=info['destination'], overwrite=True)
        assert again['destination_exists'] is True


def test_a_full_drive_is_refused_with_its_own_arithmetic(
        app, tmp_path, ready, base_and_lora, monkeypatch):
    base, lora = base_and_lora
    monkeypatch.setattr(job, '_free_bytes', lambda _p: 1000)
    with app.app_context():
        with pytest.raises(job.MergeJobError) as ei:
            job.plan(base, [{'path': lora, 'weight': 1.0}])
    message = str(ei.value)
    assert 'not enough disk space' in message
    assert 'GB free' in message and 'working headroom' in message


def test_space_error_states_the_numbers_it_used():
    assert job.space_error(None, 10 ** 12) is None      # unmeasurable never blocks
    assert job.space_error(10 ** 12, 10 ** 9) is None
    message = job.space_error(3 * 10 ** 9, 26 * 10 ** 9)
    assert '3.0 GB free' in message and '26.0 GB' in message


@pytest.mark.parametrize('weight,fragment', [
    (5.0, 'merge weights run from'),
    (-9.0, 'merge weights run from'),
    (0, 'contribute nothing'),
    ('', 'not a number'),                  # the field was left empty
    (None, 'not a number'),                # JSON null (Number('') -> NaN -> null)
    ('abc', 'not a number'),
])
def test_a_weight_that_cannot_be_meant_is_refused_before_the_write(
        app, tmp_path, ready, base_and_lora, weight, fragment):
    """Every one of these is a readable refusal in the plan, never an exception
    that reaches the user as a 500 after they committed to a 26 GB write."""
    base, lora = base_and_lora
    with app.app_context():
        with pytest.raises(job.MergeJobError, match=fragment):
            job.plan(base, [{'path': lora, 'weight': weight}])


@pytest.mark.parametrize('weight', ['0,8', '1,0', ' 0,5 '])
def test_a_decimal_comma_is_named_rather_than_called_not_a_number(
        app, tmp_path, ready, base_and_lora, weight):
    """This machine is fr-FR: the weight field DISPLAYS "0,8". It cannot send one
    — <input type="number"> hands JavaScript a dot-decimal string by
    specification, and the captured request body carried "weight":0.8 while the
    field read 0,8 on screen. But a script, a curl, or a future text field can,
    and "the weight is not a number" is a baffling thing to read after typing a
    perfectly ordinary French number. Name it and say what to type instead."""
    base, lora = base_and_lora
    with app.app_context():
        with pytest.raises(job.MergeJobError, match='use a dot for the decimal'):
            job.plan(base, [{'path': lora, 'weight': weight}])


def test_the_same_lora_twice_is_a_mistake_worth_naming(
        app, tmp_path, ready, base_and_lora):
    base, lora = base_and_lora
    with app.app_context():
        with pytest.raises(job.MergeJobError, match='in the list twice'):
            job.plan(base, [{'path': lora, 'weight': 0.5},
                            {'path': lora, 'weight': 0.5}])


def test_no_lora_at_all_is_refused(app, tmp_path, ready, base_and_lora):
    base, _lora = base_and_lora
    with app.app_context():
        with pytest.raises(job.MergeJobError, match='at least one LoRA'):
            job.plan(base, [])


def test_a_missing_file_says_which_one(app, tmp_path, ready, base_and_lora):
    base, lora = base_and_lora
    with app.app_context():
        with pytest.raises(job.MergeJobError, match='nowhere.safetensors'):
            job.plan(base, [{'path': str(tmp_path / 'nowhere.safetensors'),
                             'weight': 1.0}])


def test_a_worker_without_torch_is_a_plan_refusal_not_a_surprise(
        app, tmp_path, monkeypatch, base_and_lora):
    """'Can this machine do it at all' is a planning question. Discovering it
    thirty seconds after the click is the defect this whole lane was built to
    stop repeating."""
    base, lora = base_and_lora
    monkeypatch.setattr(job.fp8_quantize, 'interpreter',
                        lambda: {'python': 'python', 'ready': False,
                                 'missing': ['torch'],
                                 'reason': 'the Python that would do the conversion '
                                           'is missing torch. Quantizing needs them, '
                                           'and pip install torch'})
    with app.app_context():
        with pytest.raises(job.MergeJobError) as ei:
            job.plan(base, [{'path': lora, 'weight': 1.0}])
    assert 'Merging needs them' in str(ei.value)
    assert 'pip install torch' in str(ei.value)


def test_the_probe_asks_for_torch_and_nothing_else():
    """Both lanes that use this probe read and write safetensors by hand, so
    torch is the only module either of them needs. A probe demanding more would
    refuse an environment that works — and nobody re-reads a probe."""
    assert job.fp8_quantize.DEP_MODULES == ('torch',)


def test_describe_never_raises_so_the_button_can_carry_the_reason(app, tmp_path):
    with app.app_context():
        out = job.describe(str(tmp_path / 'nope.safetensors'), [])
    assert out['ok'] is False and out['error']


def test_a_merge_stranded_by_a_restart_does_not_block_the_next_one(app):
    """The job state outlives the tab on purpose (a merge is minutes long and a
    reload must find it again). The price is a 'running' left behind when the app
    is restarted mid-merge: a progress bar that can never move, and six hours of
    "a merge is already running". Nobody would connect that to the restart."""
    with app.app_context():
        job._worker_thread = None                      # nothing running HERE
        job.queue_manager._set_system_state(job._STATE_KEY, {
            'status': 'running', 'base_name': 'b.safetensors',
            'destination_name': 'out.safetensors', 'destination_dir': 'D:\\x',
            'output_bytes': 1, 'loras': [], 'done': 12, 'total': 430,
        }, ttl_seconds=60)
        assert job.status()['status'] == 'running'
        assert job.reconcile() is True
        after = job.status()
        assert after['status'] == 'error'
        assert 'restarted' in after['error']
        assert 'Nothing was overwritten' in after['error']
        # idempotent, and it never touches a state that is not a ghost
        assert job.reconcile() is False


def test_reconcile_never_disturbs_a_merge_that_is_really_running(app):
    import threading as _threading
    stop = _threading.Event()
    thread = _threading.Thread(target=stop.wait, daemon=True)
    thread.start()
    with app.app_context():
        job._worker_thread = thread
        job.queue_manager._set_system_state(job._STATE_KEY, {
            'status': 'running', 'base_name': 'b.safetensors',
            'destination_name': 'out.safetensors', 'destination_dir': 'D:\\x',
            'output_bytes': 1, 'loras': [],
        }, ttl_seconds=60)
        try:
            assert job.reconcile() is False
            assert job.status()['status'] == 'running'
        finally:
            stop.set()
            thread.join(timeout=5)
            job._worker_thread = None


def test_the_status_route_reports_what_it_reconciled(app, client):
    with app.app_context():
        job._worker_thread = None
        job.queue_manager._set_system_state(job._STATE_KEY, {
            'status': 'running', 'base_name': 'b.safetensors',
            'destination_name': 'out.safetensors', 'destination_dir': 'D:\\x',
            'output_bytes': 1, 'loras': [],
        }, ttl_seconds=60)
    body = client.get('/api/tools/lora-merge/status').get_json()
    assert body['ok'] is True and body['reconciled'] is True
    assert body['status'] == 'error'


def test_the_worker_is_invoked_on_the_merge_module_itself():
    command = job.worker_command('py', '/tmp/spec.json', budget_seconds=60)
    assert command[0] == 'py'
    assert command[1].endswith('lora_merge.py')
    assert '--spec' in command and '/tmp/spec.json' in command
    assert '--progress' in command


def test_the_spec_handed_to_the_worker_carries_the_traceability_block(
        app, tmp_path, ready, base_and_lora):
    base, lora = base_and_lora
    with app.app_context():
        info = job.plan(base, [{'path': lora, 'weight': 0.8}])
        spec_path = job.write_spec(info)
    try:
        with open(spec_path, encoding='utf-8') as fh:
            spec = json.load(fh)
    finally:
        os.remove(spec_path)
    assert spec['base'] == base
    assert spec['loras'] == [{'path': lora, 'weight': 0.8}]
    assert spec['metadata']['lds_merge'] == 'lora_into_base'
    assert 'NOT trained' in spec['metadata']['lds_merge_note']


# --- routes -------------------------------------------------------------------

def test_the_plan_route_answers_200_even_when_it_refuses(client):
    """A refusal is a disabled button carrying its reason, never a toast after
    the user already committed to a 26 GB write."""
    r = client.post('/api/tools/lora-merge/plan',
                    json={'base': '', 'loras': []})
    assert r.status_code == 200
    body = r.get_json()
    assert body['ok'] is False and body['error']


def test_the_status_route_is_pollable_before_anything_ran(client):
    r = client.get('/api/tools/lora-merge/status')
    assert r.status_code == 200 and r.get_json()['ok'] is True


def test_cancelling_when_nothing_runs_is_not_an_error(client):
    r = client.post('/api/tools/lora-merge/cancel', json={})
    assert r.status_code == 200
    assert r.get_json()['cancelled'] is False


# --- the real thing -----------------------------------------------------------

torch = pytest.importorskip('torch', reason='merging needs torch')
safetensors_torch = pytest.importorskip('safetensors.torch')


def _real_base(tmp_path, name='krea2_raw_bf16.safetensors', extra=None):
    torch.manual_seed(7)
    tensors = {
        'blocks.0.attn.wk.weight': torch.randn(48, 64).bfloat16(),
        'blocks.0.attn.wq.weight': torch.randn(64, 64).bfloat16(),
        'blocks.0.prenorm.scale': torch.ones(64),
        'blocks.1.attn.wk.weight': torch.randn(48, 64).bfloat16(),
        'first.weight': torch.randn(64, 8),
        'last.linear.weight': torch.randn(8, 64),
    }
    tensors.update(extra or {})
    path = tmp_path / name
    safetensors_torch.save_file(tensors, str(path))
    return path


def _real_lora(tmp_path, name='mine.safetensors', rank=4, alpha=None):
    """Targets one BF16 tensor and one F32 tensor, on purpose.

    The proportionality assertions below (half the weight = half the delta, two
    LoRAs add up, alpha/rank) are read off the F32 one. On a bf16 weight around
    1.0 the representable step is ~0.008, which is the same order as the delta a
    small LoRA applies — so a bf16 comparison measures rounding, not arithmetic,
    and would fail for a merge that is perfectly correct. That is not a reason to
    loosen the tolerance until it passes: it is a reason to assert the arithmetic
    where it is exact, and to check the bf16 path separately against the same
    formula computed the same way (test_merging_writes_the_expected_arithmetic).
    """
    torch.manual_seed(11)
    tensors = {
        'diffusion_model.blocks.0.attn.wk.lora_A.weight': torch.randn(rank, 64) * 0.05,
        'diffusion_model.blocks.0.attn.wk.lora_B.weight': torch.randn(48, rank) * 0.05,
        'diffusion_model.last.linear.lora_A.weight': torch.randn(rank, 64) * 0.05,
        'diffusion_model.last.linear.lora_B.weight': torch.randn(8, rank) * 0.05,
    }
    if alpha is not None:
        tensors['diffusion_model.blocks.0.attn.wk.alpha'] = torch.tensor(float(alpha))
        tensors['diffusion_model.last.linear.alpha'] = torch.tensor(float(alpha))
    path = tmp_path / name
    safetensors_torch.save_file(tensors, str(path))
    return path


# The F32 tensor the arithmetic assertions are read off (see _real_lora).
_EXACT = 'last.linear.weight'


def _delta(merged_path, base_path, key=_EXACT):
    return (safetensors_torch.load_file(str(merged_path))[key].float()
            - safetensors_torch.load_file(str(base_path))[key].float())


def test_merging_writes_the_expected_arithmetic_and_never_touches_the_base(tmp_path):
    base = _real_base(tmp_path)
    lora = _real_lora(tmp_path)
    before = base.read_bytes()
    out = tmp_path / 'merged.safetensors'

    original = safetensors_torch.load_file(str(base))
    factors = safetensors_torch.load_file(str(lora))
    expected = (original['blocks.0.attn.wk.weight'].float()
                + 0.8 * (factors['diffusion_model.blocks.0.attn.wk.lora_B.weight'].float()
                         @ factors['diffusion_model.blocks.0.attn.wk.lora_A.weight'].float())
                ).bfloat16()

    result = lm.merge_into_base(str(base), str(out),
                                [{'path': str(lora), 'weight': 0.8}],
                                metadata=lm.merge_metadata(str(base),
                                                           [{'path': str(lora),
                                                             'weight': 0.8}]))
    assert result['merged_tensors'] == 2          # one bf16 weight, one f32 weight
    assert result['tensors'] == 6
    merged = safetensors_torch.load_file(str(out))
    assert torch.equal(merged['blocks.0.attn.wk.weight'], expected)
    # every untouched tensor is byte-identical
    for key in ('blocks.0.attn.wq.weight', 'blocks.1.attn.wk.weight',
                'first.weight', 'blocks.0.prenorm.scale'):
        assert torch.equal(merged[key], original[key]), key
    assert base.read_bytes() == before, 'the base must never be rewritten'
    assert not (tmp_path / 'merged.safetensors.part').exists()


def test_the_output_is_verified_and_carries_its_provenance(tmp_path):
    base = _real_base(tmp_path)
    lora = _real_lora(tmp_path)
    out = tmp_path / 'merged.safetensors'
    lm.merge_into_base(str(base), str(out), [{'path': str(lora), 'weight': 1.0}],
                       metadata=lm.merge_metadata(str(base),
                                                  [{'path': str(lora), 'weight': 1.0}]))
    check = lm.verify_merge(str(out), str(base))
    assert check['verified'] is True, check['verify_error']
    header = lm.read_header(str(out))['__metadata__']
    assert header['lds_merge_base'] == 'krea2_raw_bf16.safetensors'
    assert json.loads(header['lds_merge_loras'])[0]['name'] == 'mine.safetensors'
    assert 'NOT trained' in header['lds_merge_note']


def test_merging_onto_a_merged_model_keeps_the_first_merge_on_record(tmp_path):
    """Merging in two rounds is a route the app actively suggests, so the second
    round must not overwrite the first round's provenance — a file claiming a
    one-step lineage while being two steps deep is exactly the lie this metadata
    exists to prevent."""
    base = _real_base(tmp_path)
    one = _real_lora(tmp_path, 'first.safetensors')
    two = _real_lora(tmp_path, 'second.safetensors')
    round1, round2 = tmp_path / 'r1.safetensors', tmp_path / 'r2.safetensors'
    lm.merge_into_base(str(base), str(round1), [{'path': str(one), 'weight': 1.0}],
                       metadata=lm.merge_metadata(str(base),
                                                  [{'path': str(one), 'weight': 1.0}]))
    lm.merge_into_base(str(round1), str(round2), [{'path': str(two), 'weight': 0.5}],
                       metadata=lm.merge_metadata(str(round1),
                                                  [{'path': str(two), 'weight': 0.5}]))
    meta = lm.read_header(str(round2))['__metadata__']
    # the newest merge is the headline...
    assert meta['lds_merge_base'] == 'r1.safetensors'
    assert json.loads(meta['lds_merge_loras'])[0]['name'] == 'second.safetensors'
    # ...and the one it was built on is still there, in full
    previous = json.loads(meta['lds_merge_previous'])
    assert previous['lds_merge_base'] == 'krea2_raw_bf16.safetensors'
    assert json.loads(previous['lds_merge_loras'])[0]['name'] == 'first.safetensors'


def test_a_first_merge_carries_no_empty_lineage_field(tmp_path):
    base = _real_base(tmp_path)
    lora = _real_lora(tmp_path)
    out = tmp_path / 'merged.safetensors'
    lm.merge_into_base(str(base), str(out), [{'path': str(lora), 'weight': 1.0}],
                       metadata=lm.merge_metadata(str(base),
                                                  [{'path': str(lora), 'weight': 1.0}]))
    assert 'lds_merge_previous' not in lm.read_header(str(out))['__metadata__']


def test_the_merged_file_keeps_the_bases_dtypes_and_shapes(tmp_path):
    """A merge that silently promoted bf16 to fp32 would double the file and
    stop being the thing ComfyUI expects to load."""
    base = _real_base(tmp_path)
    lora = _real_lora(tmp_path)
    out = tmp_path / 'merged.safetensors'
    lm.merge_into_base(str(base), str(out), [{'path': str(lora), 'weight': 1.0}],
                       metadata={'lds_merge': 'lora_into_base'})
    produced = lm.tensor_entries(lm.read_header(str(out)))
    expected = lm.tensor_entries(lm.read_header(str(base)))
    assert {k: (v['dtype'], v['shape']) for k, v in produced.items()} \
        == {k: (v['dtype'], v['shape']) for k, v in expected.items()}
    assert out.stat().st_size == pytest.approx(base.stat().st_size, rel=0.01)


def test_weight_zero_point_five_applies_exactly_half_the_delta(tmp_path):
    base = _real_base(tmp_path)
    lora = _real_lora(tmp_path)
    full, half = tmp_path / 'full.safetensors', tmp_path / 'half.safetensors'
    meta = {'lds_merge': 'lora_into_base'}
    lm.merge_into_base(str(base), str(full), [{'path': str(lora), 'weight': 1.0}],
                       metadata=meta)
    lm.merge_into_base(str(base), str(half), [{'path': str(lora), 'weight': 0.5}],
                       metadata=meta)
    assert torch.allclose(_delta(half, base) * 2, _delta(full, base), atol=1e-6)


def test_two_loras_accumulate_into_the_same_tensor(tmp_path):
    base = _real_base(tmp_path)
    one = _real_lora(tmp_path, 'one.safetensors')
    two = _real_lora(tmp_path, 'two.safetensors')
    out = tmp_path / 'merged.safetensors'
    result = lm.merge_into_base(
        str(base), str(out),
        [{'path': str(one), 'weight': 1.0}, {'path': str(two), 'weight': 1.0}],
        metadata={'lds_merge': 'lora_into_base'})
    assert result['merged_tensors'] == 2
    factors = safetensors_torch.load_file(str(one))
    single = (factors['diffusion_model.last.linear.lora_B.weight'].float()
              @ factors['diffusion_model.last.linear.lora_A.weight'].float())
    # the same LoRA twice at 1.0 == once at 2.0
    assert torch.allclose(_delta(out, base), single * 2, atol=1e-6)


def test_alpha_over_rank_is_honoured_when_the_lora_records_one(tmp_path):
    base = _real_base(tmp_path)
    plain = _real_lora(tmp_path, 'plain.safetensors', rank=4)
    scaled = _real_lora(tmp_path, 'scaled.safetensors', rank=4, alpha=2.0)  # 2/4 = 0.5
    a_out, b_out = tmp_path / 'a.safetensors', tmp_path / 'b.safetensors'
    meta = {'lds_merge': 'lora_into_base'}
    lm.merge_into_base(str(base), str(a_out), [{'path': str(plain), 'weight': 1.0}],
                       metadata=meta)
    lm.merge_into_base(str(base), str(b_out), [{'path': str(scaled), 'weight': 1.0}],
                       metadata=meta)
    assert torch.allclose(_delta(b_out, base) * 2, _delta(a_out, base), atol=1e-6)


def test_a_tensor_no_lora_touches_is_carried_over_byte_identical(tmp_path):
    """Including the ones that are not weights at all. We do not drop someone's
    bytes without saying so, and we do not refuse the whole merge over them."""
    torch.manual_seed(5)
    egg = {'last.down.weight': torch.randn(64, 64).bfloat16(),
           'last.up.weight': torch.randn(64, 64).bfloat16()}
    base = _real_base(tmp_path, extra=egg)
    lora = _real_lora(tmp_path)
    out = tmp_path / 'merged.safetensors'
    result = lm.merge_into_base(str(base), str(out),
                                [{'path': str(lora), 'weight': 1.0}],
                                metadata={'lds_merge': 'lora_into_base'}, family='krea')
    assert result['carried_over'] == 2
    merged = safetensors_torch.load_file(str(out))
    for key, value in egg.items():
        assert torch.equal(merged[key], value), key


def test_a_merge_that_fails_leaves_no_half_written_checkpoint(tmp_path):
    """Out of disk, killed, power cut: the user must not be left with a file that
    looks loadable and is not."""
    base = _real_base(tmp_path)
    lora = _real_lora(tmp_path)
    out = tmp_path / 'merged.safetensors'
    with pytest.raises(lm.MergeError):
        lm.merge_into_base(str(base), str(out), [{'path': str(lora), 'weight': 1.0}],
                           metadata={'lds_merge': 'lora_into_base'},
                           budget_seconds=-1)          # expired before the first tensor
    assert not out.exists()
    assert not (tmp_path / 'merged.safetensors.part').exists()


def test_verify_catches_a_file_that_is_not_the_model_it_claims(tmp_path):
    base = _real_base(tmp_path)
    other = _real_base(tmp_path, 'other.safetensors',
                       extra={'extra.weight': torch.zeros(4, 4)})
    check = lm.verify_merge(str(other), str(base))
    assert check['verified'] is False
    assert 'key set changed' in check['verify_error']


def test_the_worker_cli_round_trips_through_its_spec_file(tmp_path):
    """The exact path the app takes: a JSON spec in, one RESULT line out."""
    base = _real_base(tmp_path)
    lora = _real_lora(tmp_path)
    out = tmp_path / 'merged.safetensors'
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({
        'base': str(base), 'destination': str(out), 'family': 'krea',
        'loras': [{'path': str(lora), 'weight': 0.9}],
        'metadata': lm.merge_metadata(str(base), [{'path': str(lora), 'weight': 0.9}]),
    }), encoding='utf-8')
    assert lm.main(['--spec', str(spec)]) == 0
    assert out.is_file()
    assert lm.verify_merge(str(out), str(base))['verified'] is True


def test_the_worker_cli_reports_a_refusal_instead_of_crashing(tmp_path, capsys):
    spec = tmp_path / 'spec.json'
    spec.write_text(json.dumps({
        'base': str(tmp_path / 'missing.safetensors'),
        'destination': str(tmp_path / 'out.safetensors'), 'loras': [],
    }), encoding='utf-8')
    assert lm.main(['--spec', str(spec)]) == 1
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if ln.startswith(lm.RESULT_PREFIX)][-1]
    payload = json.loads(line[len(lm.RESULT_PREFIX):])
    assert payload['ok'] is False and payload['error']
