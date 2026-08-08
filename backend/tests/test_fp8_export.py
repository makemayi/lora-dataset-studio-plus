"""fp8 export: the file we write must be the one ComfyUI knows how to read.

The expensive half of this feature runs on a rented pod on a 26 GB checkpoint,
so the only honest proof available here is a SYNTHETIC checkpoint round-tripped
through the real writer with the real torch: correct dtypes, correct scale
tensors, the legacy marker ComfyUI keys on, and a dequantization error small
enough to be worth shipping.
"""
import json
import struct

import pytest

from app.services import fp8_export as fx

torch = pytest.importorskip('torch', reason='fp8 export needs torch')
safetensors_torch = pytest.importorskip('safetensors.torch')


# --- plan (pure, no weights) ----------------------------------------------------

def _spec(dtype, shape):
    return {'dtype': dtype, 'shape': list(shape), 'data_offsets': [0, 0]}


def test_plan_quantizes_only_big_2d_weights():
    header = {
        'blocks.0.attn.wq.weight': _spec('BF16', [3072, 3072]),
        'blocks.0.attn.wq.bias': _spec('BF16', [3072]),
        'blocks.0.prenorm.scale': _spec('F32', [3072]),
        'blocks.0.attn.qnorm.weight': _spec('BF16', [3072, 3072]),   # 'norm'
        'small.proj.weight': _spec('BF16', [64, 64]),                # too small
        'x_embedder.weight': _spec('BF16', [3072, 3072]),            # 'embed'
        '__metadata__': {'a': 'b'},
    }
    plan = fx.plan_quantization(header)
    assert plan['quantize'] == ['blocks.0.attn.wq.weight']
    assert 'blocks.0.attn.qnorm.weight' in plan['keep']
    assert 'x_embedder.weight' in plan['keep']
    assert 'small.proj.weight' in plan['keep']
    # One bf16 matrix halved is the only saving here, so the forecast must be
    # smaller but not fantastically so.
    assert plan['bytes_after'] < plan['bytes_before']


def test_scale_key_is_a_sibling_of_the_weight_not_a_child():
    """ComfyUI derives the MODULE from ``key[:-len('.scale_weight')]`` and then
    looks for that module's ``.weight``. ``…wq.weight.scale_weight`` would
    register a phantom module and leave the real weight unquantized — the bug
    this asserts against was live until the loader source was read."""
    assert fx.scale_key_for('blocks.0.attn.wq.weight') == 'blocks.0.attn.wq.scale_weight'
    assert fx.scale_key_for('odd_name') == 'odd_name.scale_weight'


def test_plan_skips_a_weight_whose_scale_key_is_already_taken():
    header = {
        'blocks.0.attn.wq.weight': _spec('BF16', [3072, 3072]),
        'blocks.0.attn.wq.scale_weight': _spec('F32', []),
    }
    assert fx.plan_quantization(header)['quantize'] == []


def test_plan_is_header_only_and_never_reads_weights(tmp_path):
    path = tmp_path / 'm.safetensors'
    safetensors_torch.save_file(
        {'blocks.0.mlp.up.weight': torch.zeros(1024, 1024, dtype=torch.bfloat16)},
        str(path))
    header = fx.read_header(path)
    assert 'blocks.0.mlp.up.weight' in header
    assert fx.plan_quantization(header)['quantize'] == ['blocks.0.mlp.up.weight']


# --- the written file -----------------------------------------------------------

def _source(tmp_path, name='Krea_full_demo_000000250.safetensors'):
    torch.manual_seed(7)
    state = {
        'blocks.0.attn.wq.weight': torch.randn(1024, 1024, dtype=torch.float32).bfloat16(),
        'blocks.0.mlp.up.weight': (torch.randn(1024, 2048) * 0.02).bfloat16(),
        'blocks.0.attn.wq.bias': torch.randn(1024).bfloat16(),
        'blocks.0.prenorm.scale': torch.ones(1024, dtype=torch.float32),
    }
    path = tmp_path / name
    safetensors_torch.save_file(state, str(path))
    return path, state


def _read_written_header(path):
    with open(path, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        return json.loads(fh.read(n).decode('utf-8'))


def test_export_writes_the_comfyui_scaled_fp8_layout(tmp_path):
    src, state = _source(tmp_path)
    dst = tmp_path / fx.fp8_name_for(src.name)
    summary = fx.export_scaled_fp8(src, dst)

    assert dst.name == 'Krea_full_demo_000000250_fp8.safetensors'
    assert summary['quantized'] == 2
    header = _read_written_header(dst)

    # The two big weights became fp8 + an F32 per-tensor scale scalar.
    for name in ('blocks.0.attn.wq.weight', 'blocks.0.mlp.up.weight'):
        assert header[name]['dtype'] == 'F8_E4M3'
        scale = header[fx.scale_key_for(name)]
        assert scale['dtype'] == 'F32' and scale['shape'] == []
    # Everything else kept its source dtype, bit for bit.
    assert header['blocks.0.attn.wq.bias']['dtype'] == 'BF16'
    assert header['blocks.0.prenorm.scale']['dtype'] == 'F32'
    # The legacy marker ComfyUI keys the whole file on: fp8 dtype selects the
    # flavour, 2 elements ask for the full-precision matmul path.
    assert header[fx.MARKER_KEY]['dtype'] == 'F8_E4M3'
    assert header[fx.MARKER_KEY]['shape'] == [2]
    # Writing _quantization_metadata would make current ComfyUI SKIP the legacy
    # conversion this file depends on.
    assert '_quantization_metadata' not in (header.get('__metadata__') or {})
    assert header['__metadata__']['lds_quantization'] == 'comfyui_scaled_fp8'


def test_export_reloads_in_torch_and_dequantizes_close_to_the_source(tmp_path):
    src, state = _source(tmp_path)
    dst = tmp_path / fx.fp8_name_for(src.name)
    fx.export_scaled_fp8(src, dst)

    loaded = safetensors_torch.load_file(str(dst))
    assert loaded[fx.MARKER_KEY].dtype is torch.float8_e4m3fn
    assert loaded[fx.MARKER_KEY].nelement() == 2

    for name in ('blocks.0.attn.wq.weight', 'blocks.0.mlp.up.weight'):
        payload = loaded[name]
        scale = loaded[fx.scale_key_for(name)]
        assert payload.dtype is torch.float8_e4m3fn
        assert payload.shape == state[name].shape
        assert scale.dtype is torch.float32 and scale.ndim == 0
        original = state[name].float()
        # This is ComfyUI's own load-time dequantization.
        restored = fx.dequantize_weight(payload, scale)
        error = (restored - original).abs().max().item()
        span = original.abs().max().item()
        # fp8_e4m3 keeps 3 mantissa bits: ~6 % worst-case relative step. A
        # BARE cast (no scale) on the 0.02-scaled tensor would be far worse —
        # that comparison is the whole point of choosing the scaled format.
        assert error <= 0.07 * span, f'{name}: {error} vs span {span}'

    # Untouched tensors survive bit for bit.
    assert torch.equal(loaded['blocks.0.attn.wq.bias'], state['blocks.0.attn.wq.bias'])
    assert torch.equal(loaded['blocks.0.prenorm.scale'], state['blocks.0.prenorm.scale'])


def test_scaled_export_beats_a_bare_fp8_cast_on_a_small_magnitude_tensor(tmp_path):
    """The reason we do not simply cast to fp8: without a scale, a tensor whose
    values live far below 448 loses most of the representable range."""
    src, state = _source(tmp_path)
    dst = tmp_path / 'out.safetensors'
    fx.export_scaled_fp8(src, dst)
    loaded = safetensors_torch.load_file(str(dst))

    original = state['blocks.0.mlp.up.weight'].float()
    scaled_err = (fx.dequantize_weight(
        loaded['blocks.0.mlp.up.weight'],
        loaded[fx.scale_key_for('blocks.0.mlp.up.weight')]) - original).abs().max().item()
    bare_err = (original.to(torch.float8_e4m3fn).float() - original).abs().max().item()
    assert scaled_err < bare_err


def test_export_respects_its_time_budget_and_leaves_no_partial_file(tmp_path):
    src, _ = _source(tmp_path)
    dst = tmp_path / 'out.safetensors'
    clock = iter([0.0, 0.0, 9999.0, 9999.0, 9999.0, 9999.0])
    with pytest.raises(fx.Fp8ExportError, match='budget'):
        fx.export_scaled_fp8(src, dst, budget_seconds=5, _now=lambda: next(clock))
    assert not dst.exists()
    assert not (tmp_path / 'out.safetensors.part').exists()


def test_export_refuses_a_checkpoint_with_nothing_to_quantize(tmp_path):
    path = tmp_path / 'tiny.safetensors'
    safetensors_torch.save_file({'a.weight': torch.zeros(8, 8)}, str(path))
    with pytest.raises(fx.Fp8ExportError, match='qualifies'):
        fx.export_scaled_fp8(path, tmp_path / 'out.safetensors')


def test_all_zero_weight_does_not_divide_by_zero(tmp_path):
    path = tmp_path / 'z.safetensors'
    safetensors_torch.save_file(
        {'blocks.0.mlp.up.weight': torch.zeros(1024, 1024, dtype=torch.bfloat16)},
        str(path))
    dst = tmp_path / 'out.safetensors'
    fx.export_scaled_fp8(path, dst)
    loaded = safetensors_torch.load_file(str(dst))
    assert float(loaded[fx.scale_key_for('blocks.0.mlp.up.weight')]) == 1.0
    assert not torch.isnan(loaded['blocks.0.mlp.up.weight'].float()).any()


def test_estimate_and_name_helpers():
    assert fx.fp8_name_for('Krea_full_x_000002500.safetensors') == \
        'Krea_full_x_000002500_fp8.safetensors'
    # ~26 GB bf16 -> ~11 GB planning figure, never zero, never larger.
    est = fx.estimate_fp8_bytes(26 * 1000 ** 3)
    assert 0 < est < 26 * 1000 ** 3


# --- the reader: nothing is memory-mapped, in either direction ------------------
# Root incident: safetensors' safe_open maps the WHOLE container, and Windows
# reserves that against the system commit charge when the mapping is created.
# Measured on this project's own 25.64 GB dense master while the machine had
# 5.7-10.4 GB of commit available: safe_open raised OSError 1455 ("the paging
# file is too small") and, on one attempt, took the process down outright, while
# the reader below opened the same file and pulled a tensor out of it for 0.30 GB.
# The box where this was NOT reproducible has a 119 GB pagefile — which is
# exactly how a defect like this ships.

def test_reader_returns_exactly_what_safetensors_would(tmp_path):
    """The replacement has to be boring: same tensors, same dtypes, same values."""
    torch.manual_seed(5)
    tensors = {
        'blocks.0.attn.wq.weight': torch.randn(64, 32).bfloat16(),
        'blocks.0.prenorm.scale': torch.ones(64),
        'blocks.0.mlp.up.weight': torch.randn(48, 16).half(),
        'scalar': torch.tensor(1.5),
    }
    path = tmp_path / 'm.safetensors'
    safetensors_torch.save_file(tensors, str(path))
    reference = safetensors_torch.load_file(str(path))
    with fx._Reader(path) as reader:
        assert set(reader.keys()) == set(tensors)
        for name, expected in reference.items():
            got = reader.get_tensor(name)
            assert got.dtype is expected.dtype, name
            assert tuple(got.shape) == tuple(expected.shape), name
            assert torch.equal(got, expected), name


def test_reader_says_so_when_the_file_is_truncated(tmp_path):
    path = tmp_path / 'cut.safetensors'
    safetensors_torch.save_file(
        {'blocks.0.mlp.up.weight': torch.randn(64, 64).bfloat16()}, str(path))
    whole = path.read_bytes()
    path.write_bytes(whole[:-2048])            # lose the tail of the data block
    with fx._Reader(path) as reader:
        with pytest.raises(fx.Fp8ExportError, match='truncated'):
            reader.get_tensor('blocks.0.mlp.up.weight')


def test_reader_refuses_a_file_that_is_not_safetensors(tmp_path):
    junk = tmp_path / 'nope.safetensors'
    junk.write_bytes(b'this is not a safetensors container at all')
    with pytest.raises(fx.Fp8ExportError, match='not a readable'):
        fx._Reader(junk)


def test_no_code_path_in_this_module_memory_maps_a_checkpoint():
    """A HEURISTIC, and the only guard available — be clear about which.

    WHAT IT CATCHES: somebody reaching for ``safe_open`` or importing
    ``safetensors`` in this module again, for the source, for the read-back
    verification, or for anything else.

    WHAT IT DOES NOT CATCH: memory-mapping by any other spelling — ``mmap``,
    ``np.memmap``, a helper in another module that maps on our behalf, or a
    future safetensors API with a different name. It matches text, so it cannot
    know what a call actually does.

    It exists in this weak form because the real defect CANNOT be unit tested:
    it needs a multi-GB file AND a machine whose commit charge is not generous,
    which is never the machine the code is written on. Measured, it took a
    25.6 GB file and a box down to 5.7 GB of free commit. A grep that catches
    the one spelling we know about is worth more than nothing and much less than
    a proof — treat a green here as "the known mistake was not repeated", not as
    "nothing is mapped".
    """
    import inspect
    code = '\n'.join(line for line in inspect.getsource(fx).splitlines()
                     if not line.lstrip().startswith('#'))
    # Docstrings name safe_open on purpose (they explain why it is not used);
    # what must not exist is a CALL or an import.
    assert 'safe_open(str(' not in code and 'safe_open(path' not in code, \
        'fp8_export must not memory-map a checkpoint — use _Reader'
    assert 'from safetensors import' not in code
    assert 'import safetensors' not in code
