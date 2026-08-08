"""Bake one or more LoRAs INTO a base checkpoint and write a full model.

WHY THIS EXISTS
---------------
This is how the community actually makes the checkpoints it publishes. Of the
Krea 2 checkpoints on the model sites whose authors describe their method, the
ones that explain themselves describe a MERGE, not a training run: a LoRA trained
on Raw, folded into a base, re-quantized, uploaded. LDS could train the LoRA and
could quantize the result, but the step in the middle — the one that turns "my
LoRA" into "my model" — did not exist, so a user could not reproduce what they
saw other people doing.

Two uses, in order of value:

1. **My LoRA, as a full model.** Someone trained an adapter here and wants the
   complete checkpoint: to publish it, or to train again on top of it.
2. **The Turbo transplant.** Krea publishes a re-distillation LoRA. Merged at
   ~0.8-1.0 into a fine-tune of the *Raw* model, it restores the few-step
   behaviour, which is how the same model gets published in both a Raw and a
   Turbo flavour. It matters here because a dense run on Raw — the only dense
   lane LDS opens — otherwise leaves the user with a slow model and no route to
   a fast one. We have NOT verified this ourselves, and the UI says so: a
   transplant is an approximation, not an identity.

WHAT THIS PRODUCES IS NOT A TRAINED MODEL, AND SAYS SO
------------------------------------------------------
On the model sites "finetune" is routinely used for a checkpoint that is really a
LoRA merged into someone else's base — including by authors who describe the
merge themselves in the next sentence. Copying that vocabulary would make LDS
lie in the one place a file can still be identified six months later. So the
output's ``__metadata__`` records that it came from a merge, which base it used,
which LoRAs at which weights, and when (``merge_metadata``). Names lie; headers
are what a checkpoint can be identified by.

THE ARITHMETIC
--------------
For every LoRA module the base has a matching weight matrix::

    delta = (alpha / rank) * (B @ A)
    W'    = W + weight * delta

``A`` is ``[rank, in]``, ``B`` is ``[out, rank]``, so ``B @ A`` is ``[out, in]``
— the base tensor's exact shape, which is asserted per tensor before any byte is
written. Accumulation is done in **float32** and cast back to the base's own
dtype at the end: a bf16 base accumulated in bf16 would lose most of a small
delta to rounding, and several stacked LoRAs would compound it.

We refuse to merge into an already-quantized base (see ``lora_merge_job``): fp8
weights would be dequantized, modified and re-quantized, which is lossy every
cycle. Merge into the bf16 master and quantize afterwards — LDS already has that
button.

``weight`` is resolved PER BASE KEY (``weight_for``), not as one scalar handed to
the loop. Today every key gets the same number and the UI offers exactly one
field, on purpose: per-block ratios are a tuning surface with no published recipe
behind them, and eight sliders nobody can advise on are worse than one number we
can explain. But the day someone MEASURES that a per-block ratio helps, it is a
lookup table handed to this function rather than a rewrite of the merge loop.

THE KEY LAYOUT IS MEASURED, NOT ASSUMED
---------------------------------------
Read off the real files rather than inferred from a name:

    base  krea2_raw_bf16.safetensors            430 tensors, 256 BF16 + 174 F32
          blocks.0.attn.wk.weight               BF16 [1536, 6144]
    LoRA  (ai-toolkit, ss_base_model_version=krea2)   512 tensors
          diffusion_model.blocks.0.attn.wk.lora_A.weight  [32, 6144]
          diffusion_model.blocks.0.attn.wk.lora_B.weight  [1536, 32]

Krea 2 stores q/k/v as SEPARATE matrices (``wq``/``wk``/``wv``/``wo``/``gate``),
so the mapping is a direct one — strip ``diffusion_model.``, append ``.weight``.
There is no fused-qkv slicing to do. (Some synthetic test fixtures in this repo
spell a ``blocks.0.attn.qkv.weight`` that does not exist in any real Krea 2
checkpoint; they only ever feed a name sniff, but do not learn the layout from
them.)

WHAT WE VALIDATE, AND WHY IT IS NOT A CONFORMITY CHECK
------------------------------------------------------
The first design of this module carried a manifest of the 51 index-normalised key
patterns of a reference Krea 2 checkpoint and REFUSED any base holding a tensor
outside it. That was the wrong instinct, and it is worth writing down why: it
would refuse a legitimate future Krea variant that adds one key, and the user
would have no recourse — a refusal derived from one file on one machine,
presented as a fact about the family.

So validation asks only what THIS merge actually requires:

  * every LoRA key has a counterpart in the base,
  * the factors multiply back to that tensor's exact shape,
  * that tensor is in a dtype we can accumulate into.

A tensor in the base that no LoRA touches cannot be got wrong, because we copy
its bytes through untouched. That includes the strange ones: a community Krea 2
file on the machine this was written on carries ``last.down.weight`` and
``last.up.weight``, [6144, 6144] each — ~75 MB of an image smuggled under a
legitimate prefix (``last.`` is real, so a prefix check never sees it) and
declared by ``egg_*`` keys in ``__metadata__``. No LoRA targets those, so the
merge does not touch them; they are copied over unchanged, and the plan SAYS SO
with their names and their byte cost. Silently dropping someone's bytes would be
its own kind of lie, and refusing the whole merge over them helps nobody.

``KREA2_KEY_PATTERNS`` therefore survives only as a DESCRIPTION used to write
that sentence — never as a gate. A stale manifest costs a wrong note, not a
blocked user.

HOW IT RUNS
-----------
Same shape as the fp8 lane, for the same reason: the arithmetic needs ``torch``,
and the app's own environment deliberately does not have it (torch is gigabytes,
and LDS installs and runs without it). This file is BOTH an importable module
(unit tested — every function above the writer is pure header maths and needs no
torch at all) and a self-contained CLI that a configured interpreter runs as a
subprocess.

torch is the ONLY thing the worker needs. Reading and writing safetensors is done
here, by hand, against the format's own 8-byte length + JSON header — see
``Reader`` for why that is not laziness but the fix for a measured crash.

MEMORY
------
Streaming, one tensor at a time, with no memory mapping anywhere. The output
header is computed from the SOURCE header — same keys, same shapes, same dtypes,
so the output is the same size as the base and the layout is knowable before a
single weight is read. Peak resident memory is one tensor in float32 plus its
delta (~1.2 GB on the largest Krea 2 matrix), never one checkpoint.
"""
from __future__ import annotations

import json
import os
import re
import struct
import time
from datetime import datetime, timezone

# LoRA tensors live under this prefix in every ai-toolkit export we ship and in
# the community LoRAs for this family. The base has no prefix at all.
LORA_PREFIX = 'diffusion_model.'

# (down/A, up/B) suffix spellings. ai-toolkit/PEFT writes lora_A/lora_B; the
# older kohya convention writes lora_down/lora_up. Both mean the same factors.
LORA_SUFFIX_PAIRS = (
    ('.lora_A.weight', '.lora_B.weight'),
    ('.lora_down.weight', '.lora_up.weight'),
)
ALPHA_SUFFIX = '.alpha'
WEIGHT_SUFFIX = '.weight'

# A real safetensors header is far below this; past it the leading 8 bytes are
# not a header length at all.
_HEADER_LEN_MAX = 512 * 1024 * 1024

_DTYPE_BYTES = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
    'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
    'I32': 4, 'U32': 4, 'F32': 4,
    'I64': 8, 'U64': 8, 'F64': 8,
}

# Dtypes we can merge INTO. Anything else is either quantized (refused upstream,
# with a better sentence) or not a float we can accumulate in.
MERGEABLE_DTYPES = ('BF16', 'F16', 'F32')


def torch_dtypes() -> dict:
    """safetensors dtype spelling -> torch dtype.

    A function, not a module constant: importing torch at module scope would
    make this file un-importable on the app's own environment, which is the
    whole reason the merge runs in a subprocess.
    """
    import torch
    return {
        'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32,
        'F64': torch.float64, 'F8_E4M3': torch.float8_e4m3fn,
        'F8_E5M2': torch.float8_e5m2, 'I64': torch.int64, 'I32': torch.int32,
        'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8,
        'BOOL': torch.bool,
    }

PROGRESS_PREFIX = 'LDS_MERGE_PROGRESS'
RESULT_PREFIX = 'LDS_MERGE_RESULT'


class MergeError(RuntimeError):
    """A refusal with a sentence for the user. Never a stack trace."""


# --- family key descriptions ------------------------------------------------------
# Index-normalised tensor names ("blocks.7.attn.wk.weight" -> "blocks.{i}.attn.wk.weight"),
# read off a real 430-tensor krea2_raw_bf16.safetensors and verified identical on a
# dense model LDS trained itself.
#
# THIS IS A DESCRIPTION, NOT A GATE. Nothing is refused for being absent from it
# and nothing is refused for being present and unlisted; see the module note. Its
# only job is to let the plan say "these N tensors are not part of the Krea 2
# layout and will be carried over unchanged" — which is how the 75 MB of image
# hidden in one community file becomes a visible line instead of a surprise.
#
# It matches on NAME ONLY, never dtype: the same 430 keys appear as BF16 in a
# dense master where the reference base keeps F32, and a dtype-sensitive
# description would call every one of them foreign.
KREA2_KEY_PATTERNS = frozenset({
    'blocks.{i}.attn.gate.weight',
    'blocks.{i}.attn.qknorm.knorm.scale',
    'blocks.{i}.attn.qknorm.qnorm.scale',
    'blocks.{i}.attn.wk.weight',
    'blocks.{i}.attn.wo.weight',
    'blocks.{i}.attn.wq.weight',
    'blocks.{i}.attn.wv.weight',
    'blocks.{i}.mlp.down.weight',
    'blocks.{i}.mlp.gate.weight',
    'blocks.{i}.mlp.up.weight',
    'blocks.{i}.mod.lin',
    'blocks.{i}.postnorm.scale',
    'blocks.{i}.prenorm.scale',
    'first.bias',
    'first.weight',
    'last.linear.bias',
    'last.linear.weight',
    'last.modulation.lin',
    'last.norm.scale',
    'tmlp.{i}.bias',
    'tmlp.{i}.weight',
    'tproj.{i}.bias',
    'tproj.{i}.weight',
    'txtfusion.layerwise_blocks.{i}.attn.gate.weight',
    'txtfusion.layerwise_blocks.{i}.attn.qknorm.knorm.scale',
    'txtfusion.layerwise_blocks.{i}.attn.qknorm.qnorm.scale',
    'txtfusion.layerwise_blocks.{i}.attn.wk.weight',
    'txtfusion.layerwise_blocks.{i}.attn.wo.weight',
    'txtfusion.layerwise_blocks.{i}.attn.wq.weight',
    'txtfusion.layerwise_blocks.{i}.attn.wv.weight',
    'txtfusion.layerwise_blocks.{i}.mlp.down.weight',
    'txtfusion.layerwise_blocks.{i}.mlp.gate.weight',
    'txtfusion.layerwise_blocks.{i}.mlp.up.weight',
    'txtfusion.layerwise_blocks.{i}.postnorm.scale',
    'txtfusion.layerwise_blocks.{i}.prenorm.scale',
    'txtfusion.projector.weight',
    'txtfusion.refiner_blocks.{i}.attn.gate.weight',
    'txtfusion.refiner_blocks.{i}.attn.qknorm.knorm.scale',
    'txtfusion.refiner_blocks.{i}.attn.qknorm.qnorm.scale',
    'txtfusion.refiner_blocks.{i}.attn.wk.weight',
    'txtfusion.refiner_blocks.{i}.attn.wo.weight',
    'txtfusion.refiner_blocks.{i}.attn.wq.weight',
    'txtfusion.refiner_blocks.{i}.attn.wv.weight',
    'txtfusion.refiner_blocks.{i}.mlp.down.weight',
    'txtfusion.refiner_blocks.{i}.mlp.gate.weight',
    'txtfusion.refiner_blocks.{i}.mlp.up.weight',
    'txtfusion.refiner_blocks.{i}.postnorm.scale',
    'txtfusion.refiner_blocks.{i}.prenorm.scale',
    'txtmlp.{i}.bias',
    'txtmlp.{i}.scale',
    'txtmlp.{i}.weight',
})

KNOWN_LAYOUTS = {'krea': KREA2_KEY_PATTERNS}
FAMILY_LABELS = {'krea': 'Krea 2', 'zimage': 'Z-Image', 'sdxl': 'SDXL',
                 'flux': 'FLUX.1', 'flux2klein': 'FLUX.2 Klein', 'anima': 'Anima'}

_INDEX_SEGMENT = re.compile(r'(?<=\.)\d+(?=\.)')


def normalise_key(name: str) -> str:
    """``blocks.7.attn.wk.weight`` -> ``blocks.{i}.attn.wk.weight``.

    Only whole dotted segments made of digits are replaced, so a name that
    merely contains a digit (``fc1.weight``) is left alone.
    """
    return _INDEX_SEGMENT.sub('{i}', str(name))


def tensor_bytes(spec) -> int:
    width = _DTYPE_BYTES.get(str((spec or {}).get('dtype') or '').upper())
    if not width:
        return 0
    numel = 1
    for dim in _shape_of(spec):
        numel *= dim
    return numel * width


def foreign_tensors(header: dict, family) -> list:
    """Tensors the family's known layout does not describe, for DISCLOSURE.

    ``[{'name', 'bytes'}]``, sorted, or ``[]`` when we have no description of
    that family (then we say nothing rather than guess). These are never
    refused and never modified — they are copied through, and the plan names
    them so a 75 MB passenger is a line the user reads before clicking.
    """
    patterns = KNOWN_LAYOUTS.get(family)
    if not patterns:
        return []
    return sorted(
        ({'name': name, 'bytes': tensor_bytes(spec)}
         for name, spec in tensor_entries(header).items()
         if normalise_key(name) not in patterns),
        key=lambda row: row['name'])


# --- header reading ---------------------------------------------------------------

def read_header(path) -> dict:
    """The safetensors header of ``path``. Header ONLY — no weight byte is read."""
    try:
        with open(path, 'rb') as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ValueError('file too short to be a safetensors container')
            n = struct.unpack('<Q', raw)[0]
            if n <= 0 or n > _HEADER_LEN_MAX:
                raise ValueError('implausible safetensors header length')
            blob = fh.read(n)
            if len(blob) != n:
                raise ValueError('truncated safetensors header')
            obj = json.loads(blob.decode('utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        raise MergeError(f'not a readable .safetensors file ({e})') from e
    if not isinstance(obj, dict):
        raise MergeError('safetensors header is not an object')
    return obj


def tensor_entries(header: dict) -> dict:
    return {k: v for k, v in header.items()
            if k != '__metadata__' and isinstance(v, dict)}


class Reader:
    """One tensor at a time out of a .safetensors, by ordinary file I/O.

    DELIBERATELY NOT ``safetensors.safe_open``. That memory-maps the whole
    container, and on Windows a 26 GB mapping is charged against the pagefile the
    moment it is created — so opening the real Krea 2 base fails outright with
    ``OSError 1455`` ("the paging file is too small for this operation") on a
    machine with plenty of free RAM and 246 GB of free disk. Measured here, on
    the real file: the first end-to-end merge died on that line before reading a
    single tensor.

    A merge is a strictly sequential pass — read tensor, transform, write, drop —
    so a mapping buys nothing anyway. Seeking to the offset the header already
    gives us has the same peak memory (one tensor), no address-space cost, and no
    dependency on how the user happens to have configured their pagefile. The
    offsets are relative to the end of the header, which is where the data
    section starts.
    """

    def __init__(self, path):
        self.path = str(path)
        self._fh = open(self.path, 'rb')
        try:
            raw = self._fh.read(8)
            if len(raw) != 8:
                raise ValueError('file too short to be a safetensors container')
            n = struct.unpack('<Q', raw)[0]
            if n <= 0 or n > _HEADER_LEN_MAX:
                raise ValueError('implausible safetensors header length')
            self.header = json.loads(self._fh.read(n).decode('utf-8'))
        except Exception as e:
            self._fh.close()
            raise MergeError(f'not a readable .safetensors file ({e})') from e
        self._start = 8 + n
        self.entries = tensor_entries(self.header)

    def keys(self):
        return list(self.entries)

    def get_tensor(self, name):
        import torch
        spec = self.entries.get(name)
        if spec is None:
            raise MergeError(f'{name} is not in {os.path.basename(self.path)}')
        dtype = torch_dtypes().get(str(spec.get('dtype') or '').upper())
        if dtype is None:
            raise MergeError(f'unsupported dtype {spec.get("dtype")!r} on {name}')
        begin, end = int(spec['data_offsets'][0]), int(spec['data_offsets'][1])
        nbytes = end - begin
        self._fh.seek(self._start + begin)
        # readinto a bytearray, not `bytes(fh.read(...))`: torch.frombuffer warns
        # (once per tensor, so hundreds of times) about a read-only buffer, and
        # converting after the fact would hold two copies of a 200 MB tensor at
        # the peak for no reason.
        raw = bytearray(nbytes)
        got = self._fh.readinto(raw)
        if got != nbytes:
            raise MergeError(
                f'{os.path.basename(self.path)} is truncated: {name} claims '
                f'{nbytes} bytes and only {got} are there.')
        return torch.frombuffer(raw, dtype=dtype).reshape(_shape_of(spec))

    def close(self):
        try:
            self._fh.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _shape_of(spec) -> list:
    try:
        return [int(d) for d in (spec.get('shape') or [])]
    except (TypeError, ValueError, AttributeError):
        return []


# --- LoRA structure ---------------------------------------------------------------

def base_key_for(lora_key: str):
    """``diffusion_model.blocks.0.attn.wk.lora_A.weight`` -> ``blocks.0.attn.wk.weight``.

    Returns ``(base_key, module, which)`` where ``which`` is ``'A'`` or ``'B'``,
    or None when the key is not one half of a LoRA factor pair.
    """
    name = str(lora_key)
    for down, up in LORA_SUFFIX_PAIRS:
        for suffix, which in ((down, 'A'), (up, 'B')):
            if name.endswith(suffix):
                module = name[:-len(suffix)]
                if module.startswith(LORA_PREFIX):
                    module = module[len(LORA_PREFIX):]
                if not module:
                    return None
                return module + WEIGHT_SUFFIX, module, which
    return None


def lora_modules(header: dict) -> dict:
    """``{base_key: {'module', 'A', 'B', 'A_key', 'B_key', 'alpha_key'}}``.

    Pure function of the LoRA's header, so "what would this merge touch" is
    answerable — and assertable — without reading a weight.
    """
    entries = tensor_entries(header)
    alphas = {}
    for name in entries:
        if name.endswith(ALPHA_SUFFIX):
            module = name[:-len(ALPHA_SUFFIX)]
            if module.startswith(LORA_PREFIX):
                module = module[len(LORA_PREFIX):]
            alphas[module] = name
    out = {}
    for name, spec in entries.items():
        parsed = base_key_for(name)
        if not parsed:
            continue
        base_key, module, which = parsed
        slot = out.setdefault(base_key, {'module': module, 'A': None, 'B': None,
                                         'alpha_key': alphas.get(module),
                                         'A_key': None, 'B_key': None})
        slot[which] = spec
        slot[which + '_key'] = name
    # A factor without its twin cannot make a delta; carrying it forward would
    # crash mid-write instead of refusing in the plan.
    return {k: v for k, v in out.items() if v['A'] and v['B']}


def module_scale(rank, alpha) -> float:
    """The ``alpha / rank`` multiplier LoRA math applies before the user weight.

    No alpha recorded (the case for every ai-toolkit LoRA this app trains, and
    for the community Krea 2 LoRAs measured here) means the factors are already
    at their intended scale: 1.0, NOT ``1/rank``, which would silently divide
    every delta by the rank and produce a merge that looks like it did nothing.
    """
    if alpha is None:
        return 1.0
    try:
        rank = int(rank)
        alpha = float(alpha)
    except (TypeError, ValueError):
        return 1.0
    if rank <= 0 or alpha != alpha or alpha <= 0:      # 0, NaN, negative
        return 1.0
    return alpha / rank


def weight_for(spec, base_key) -> float:
    """The multiplier for ONE base tensor from ONE LoRA entry.

    V1 always answers ``spec['weight']`` — the UI offers a single number per
    LoRA and nothing calls the other branch. The seam exists because the
    published per-block merge nodes for this family DO vary the ratio by block
    (``first.`` / ``tmlp.`` / ``blocks.N`` / ``last.``), and when somebody
    measures that it helps, this is where that table plugs in — instead of
    reopening the merge loop, which is the part that must not churn.

    ``key_weights`` accepts an exact key, an index-normalised pattern, or a
    prefix, most specific first.
    """
    table = (spec or {}).get('key_weights')
    default = (spec or {}).get('weight')
    try:
        default = float(default or 0.0)
    except (TypeError, ValueError):
        default = 0.0
    if not isinstance(table, dict) or not table:
        return default
    for candidate in (str(base_key), normalise_key(base_key)):
        if candidate in table:
            try:
                return float(table[candidate])
            except (TypeError, ValueError):
                return default
    best = None
    for prefix, value in table.items():
        if str(base_key).startswith(str(prefix)) \
                and (best is None or len(str(prefix)) > len(str(best[0]))):
            best = (prefix, value)
    if best is None:
        return default
    try:
        return float(best[1])
    except (TypeError, ValueError):
        return default


def plan_merge(base_header: dict, lora_headers: list, *, family=None) -> dict:
    """What merging these LoRAs into this base would touch — header maths only.

    ``lora_headers`` is a list of ``(label, header)``. Raises ``MergeError`` on
    anything that would make the WRITE fail, and on nothing else: a LoRA key with
    no matching base tensor, a factor pair whose product is not the base tensor's
    shape, a base tensor in a dtype we cannot accumulate into, or a LoRA that
    overlaps nothing.

    Every one of those is a header question. That is deliberate: a refusal that
    can only be discovered while writing a 26 GB file is a refusal the UI cannot
    show before the click. Tensors we merely copy are NOT validated against
    anything — see the module note — they are reported in ``carried_over``.
    """
    base_entries = tensor_entries(base_header)
    if not base_entries:
        raise MergeError('the base checkpoint declares no tensors')
    targets: dict = {}
    per_lora = []
    for index, (label, header) in enumerate(lora_headers):
        modules = lora_modules(header)
        if not modules:
            raise MergeError(
                f'{label} carries no LoRA weights — no lora_A/lora_B (or '
                'lora_down/lora_up) pair is present. This is not a LoRA file.')
        missing, mismatched, ranks = [], [], []
        for base_key, slot in sorted(modules.items()):
            spec = base_entries.get(base_key)
            if spec is None:
                missing.append(base_key)
                continue
            a_shape, b_shape = _shape_of(slot['A']), _shape_of(slot['B'])
            w_shape = _shape_of(spec)
            if len(a_shape) != 2 or len(b_shape) != 2 or len(w_shape) != 2:
                mismatched.append(base_key)
                continue
            # A is [rank, in], B is [out, rank]; B @ A must BE the base tensor.
            if a_shape[0] != b_shape[1] or b_shape[0] != w_shape[0] \
                    or a_shape[1] != w_shape[1]:
                mismatched.append(base_key)
                continue
            dtype = str(spec.get('dtype') or '').upper()
            if dtype not in MERGEABLE_DTYPES:
                raise MergeError(
                    f'{base_key} is stored as {dtype or "an unknown dtype"} in this '
                    'base — a merge can only add to bf16/fp16/fp32 weights. Merge '
                    'into the full-precision master, then quantize the result.')
            ranks.append(a_shape[0])
            targets.setdefault(base_key, []).append(index)
        if missing:
            shown = ', '.join(missing[:4])
            more = f' (and {len(missing) - 4} more)' if len(missing) > 4 else ''
            raise MergeError(
                f'{label} targets {len(missing)} weight(s) this base does not have: '
                f'{shown}{more}. It was trained on a different model.')
        if mismatched:
            shown = ', '.join(mismatched[:4])
            more = f' (and {len(mismatched) - 4} more)' if len(mismatched) > 4 else ''
            raise MergeError(
                f'{label} does not fit this base: the factors for {shown}{more} do '
                'not multiply back to the shape of the base weight. Same family, '
                'different size or variant.')
        per_lora.append({
            'label': label,
            'modules': len(modules),
            'rank': max(ranks) if ranks else 0,
            'has_alpha': any(slot['alpha_key'] for slot in modules.values()),
        })
    if not targets:
        raise MergeError('none of these LoRAs touches a weight of this base')
    carried = [row for row in foreign_tensors(base_header, family)
               if row['name'] not in targets]
    return {
        'targets': targets,
        'loras': per_lora,
        'touched_tensors': len(targets),
        'base_tensors': len(base_entries),
        'carried_over': carried,
        'carried_over_bytes': sum(row['bytes'] for row in carried),
        'output_bytes': output_bytes(base_header),
    }


def output_bytes(base_header: dict) -> int:
    """Exact size of the weight body we will write.

    Not an estimate: the output keeps every key, shape and dtype of the base —
    only values change — so the byte count is arithmetic on the source header.
    """
    return sum(tensor_bytes(spec) for spec in tensor_entries(base_header).values())


# Effective bytes/second for the whole streaming pass (read the base, matmul,
# write the output), used to put a duration in the plan.
#
# MEASURED, not guessed: one real merge of the 26.28 GB Krea 2 Raw bf16 base with
# a rank-32 LDS-trained LoRA (256 of 430 tensors rewritten) took 119.5 s for
# 52.6 GB moved = 440 MB/s, on a machine that was busy with other work at the
# time. The run is bound by disk, not by arithmetic — 256 small matmuls are
# nothing next to reading and writing 26 GB each way.
#
# It stays labelled an estimate in the UI, because it is one: a slower drive, a
# network folder or a concurrent job can move this a lot, and a number presented
# as a promise is worse than one presented as a guide.
MERGE_THROUGHPUT_BPS = 440 * 1000 ** 2


def estimate_seconds(base_bytes, out_bytes=None) -> int:
    """Rough wall-clock for a merge of this size. Honest about being rough."""
    try:
        base_bytes = int(base_bytes or 0)
        total = base_bytes + int(base_bytes if out_bytes is None else (out_bytes or 0))
    except (TypeError, ValueError):
        return 0
    return int(max(1, total / MERGE_THROUGHPUT_BPS))


# --- output naming ----------------------------------------------------------------

def merged_name_for(base_name, when=None) -> str:
    """``krea2_raw_bf16.safetensors`` -> ``krea2_raw_bf16_merged_20260804-143205.safetensors``.

    The timestamp is not decoration. Two runs of the same name have already
    overwritten each other on this project once; a merge writes ~26 GB and the
    thing it would overwrite may be the only copy of an earlier merge.

    It is a second-resolution stamp, so it is not a uniqueness PROOF — two merges
    of the same base started inside the same second would propose the same name.
    What guarantees nothing is destroyed is the pair: a name that practically
    never repeats, AND ``lora_merge_job.plan`` refusing outright when the
    destination already exists. Neither half is load-bearing alone.

    The base's own stem stays in front so the ``Krea`` prefix the Community
    Licence expects — and that the delivery checks match on — survives the merge.
    """
    base = os.path.basename(str(base_name or 'model.safetensors'))
    stem = base[:-len('.safetensors')] if base.endswith('.safetensors') else base
    stamp = (when or datetime.now(timezone.utc)).strftime('%Y%m%d-%H%M%S')
    return f'{stem}_merged_{stamp}.safetensors'


def merge_metadata(base_path, loras, *, when=None) -> dict:
    """The traceability block stamped into the output's ``__metadata__``.

    File names lie — a checkpoint restore on this project once trusted one and
    picked the wrong run. Six months from now this header is the only thing that
    can still say what this file is, so it records the base, every LoRA with the
    weight it was merged at, and the date. It also states, in words, that the
    result is a MERGE and not a training run: the model sites use "finetune" for
    both, and this app does not.
    """
    stamp = (when or datetime.now(timezone.utc)).replace(microsecond=0).isoformat()
    rows = []
    for item in (loras or []):
        try:
            value = round(float(item.get('weight') or 0), 4)
        except (TypeError, ValueError):
            value = 0.0
        rows.append({'name': os.path.basename(str(item.get('path') or '')),
                     'weight': value})
    return {
        'lds_merge': 'lora_into_base',
        'lds_merge_base': os.path.basename(str(base_path)),
        'lds_merge_loras': json.dumps(rows, separators=(',', ':')),
        'lds_merge_date': stamp,
        'lds_merge_note': ('weights merged into a base checkpoint - this model was '
                           'NOT trained as a whole; it is a base plus one or more '
                           'LoRAs folded into its weights'),
    }


# --- streaming writer -------------------------------------------------------------

def _raw_bytes(tensor) -> bytes:
    """Contiguous little-endian payload of any tensor, bf16 and fp8 included.

    ``numpy()`` has neither float8 nor bfloat16, so those buffers are
    reinterpreted through an integer view of the SAME itemsize: exact bytes, no
    conversion of the values.
    """
    import torch
    flat = tensor.detach().to('cpu').contiguous().reshape(-1)
    view = {torch.float8_e4m3fn: torch.uint8, torch.float8_e5m2: torch.uint8,
            torch.bfloat16: torch.int16}.get(flat.dtype)
    if view is not None:
        flat = flat.view(view)
    return flat.numpy().tobytes()


def _pack_header(index: dict, metadata: dict | None) -> bytes:
    obj = dict(index)
    if metadata:
        obj['__metadata__'] = {str(k): str(v) for k, v in metadata.items()}
    blob = json.dumps(obj, separators=(',', ':')).encode('utf-8')
    blob += b' ' * ((-len(blob)) % 8)          # keep the data section 8-byte aligned
    return struct.pack('<Q', len(blob)) + blob


def merge_delta(base_tensor, a_tensor, b_tensor, *, weight, scale):
    """``W + weight * scale * (B @ A)``, accumulated in float32.

    Returns a tensor in the BASE's dtype. float32 throughout is the whole point:
    a bf16 accumulator has ~8 bits of mantissa, and a delta two orders of
    magnitude below the weight it corrects would round away to nothing —
    silently, and worse with every LoRA stacked on top.
    """
    import torch
    work = base_tensor.detach().to(torch.float32)
    delta = torch.matmul(b_tensor.detach().to(torch.float32),
                         a_tensor.detach().to(torch.float32))
    if tuple(delta.shape) != tuple(work.shape):   # guarded in plan_merge; belt and braces
        raise MergeError(
            f'internal shape mismatch: delta {tuple(delta.shape)} vs weight '
            f'{tuple(work.shape)}')
    work += float(weight) * float(scale) * delta
    return work.to(base_tensor.dtype)


def merge_into_base(base_path, dst_path, loras, *, metadata=None, family=None,
                    progress=None, budget_seconds=None, _now=time.monotonic) -> dict:
    """Write ``base_path`` with ``loras`` folded in, to ``dst_path``.

    ``loras`` is a list of ``{'path': str, 'weight': float}`` (optionally
    ``'key_weights'``). Streaming: the output header is computed from the source
    header (identical keys/shapes/dtypes), then each tensor is read, updated if
    any LoRA targets it, and appended once.

    Writes to ``<dst>.part`` and renames only on success, so a merge that dies
    half way — out of disk, killed, power cut — leaves no truncated checkpoint
    that looks loadable, and never touches the base.
    """
    base_header = read_header(base_path)
    lora_headers = [(os.path.basename(str(item['path'])), read_header(item['path']))
                    for item in loras]
    plan = plan_merge(base_header, lora_headers, family=family)
    entries = tensor_entries(base_header)
    targets = plan['targets']

    order = sorted(entries)
    index, offset = {}, 0
    for name in order:
        spec = entries[name]
        dtype = str(spec.get('dtype') or '').upper()
        if dtype not in _DTYPE_BYTES:
            raise MergeError(f'unsupported dtype {dtype!r} on {name}')
        nbytes = tensor_bytes(spec)
        index[name] = {'dtype': dtype, 'shape': _shape_of(spec),
                       'data_offsets': [offset, offset + nbytes]}
        offset += nbytes

    out_meta = {k: v for k, v in (base_header.get('__metadata__') or {}).items()
                if isinstance(k, str)}
    # Merging ONTO a merged model is a route we actively suggest (the refusal for
    # too many LoRAs says "merge in two rounds", and the output of one merge is a
    # valid base for the next). Without this, the second merge's metadata would
    # overwrite the first one's and the file would claim a lineage one step deep
    # while being two — the precise failure this metadata exists to prevent.
    prior = {k: out_meta[k] for k in ('lds_merge_base', 'lds_merge_loras',
                                      'lds_merge_date') if k in out_meta}
    if prior:
        out_meta['lds_merge_previous'] = json.dumps(prior, separators=(',', ':'))
    out_meta.update({str(k): str(v) for k, v in (metadata or {}).items()})

    # Per-LoRA module maps, resolved once (header maths, no weights).
    module_maps = [lora_modules(header) for _label, header in lora_headers]

    started = _now()
    tmp = str(dst_path) + '.part'
    merged_count = 0
    readers = []
    try:
        try:
            for item in loras:
                readers.append(Reader(item['path']))
            with Reader(base_path) as base_reader, open(tmp, 'wb') as out:
                out.write(_pack_header(index, out_meta))
                for done, name in enumerate(order, start=1):
                    if budget_seconds and (_now() - started) > budget_seconds:
                        raise MergeError(
                            f'the merge exceeded its {int(budget_seconds)}s budget '
                            f'after {done}/{len(order)} tensors')
                    tensor = base_reader.get_tensor(name)
                    for lora_index in targets.get(name, []):
                        slot = module_maps[lora_index][name]
                        reader = readers[lora_index]
                        a_tensor = reader.get_tensor(slot['A_key'])
                        b_tensor = reader.get_tensor(slot['B_key'])
                        alpha = None
                        if slot['alpha_key']:
                            alpha = float(reader.get_tensor(slot['alpha_key']).item())
                        tensor = merge_delta(
                            tensor, a_tensor, b_tensor,
                            weight=weight_for(loras[lora_index], name),
                            scale=module_scale(a_tensor.shape[0], alpha))
                        del a_tensor, b_tensor
                    if name in targets:
                        merged_count += 1
                    out.write(_raw_bytes(tensor))
                    del tensor
                    if progress:
                        try:
                            progress(done, len(order))
                        except Exception:      # noqa: BLE001 — never fatal
                            progress = None
        finally:
            for reader in readers:
                reader.close()
        os.replace(tmp, dst_path)
    except MergeError:
        _unlink(tmp)
        raise
    except Exception as e:                     # noqa: BLE001
        _unlink(tmp)
        raise MergeError(f'the merge failed: {e}') from e
    return {'path': str(dst_path), 'tensors': len(order),
            'merged_tensors': merged_count,
            'carried_over': len(plan['carried_over']),
            'loras': [{'name': os.path.basename(str(i['path'])),
                       'weight': float(i.get('weight') or 0)} for i in loras],
            'bytes_after': os.path.getsize(dst_path),
            'seconds': round(_now() - started, 1)}


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def verify_merge(path, base_path) -> dict:
    """Re-open what we just wrote and prove it is still the model it claims to be.

    A merge that produced an unloadable file must say so HERE, not at generation
    time days later. Checks the key set and every shape against the base, that
    the traceability metadata survived, and that a real weight reads back finite
    — a NaN from a pathological LoRA would otherwise only show up as black images.
    """
    out = {'verified': False, 'verify_error': None}
    try:
        header = read_header(path)
        produced = tensor_entries(header)
        expected = tensor_entries(read_header(base_path))
        if set(produced) != set(expected):
            missing = sorted(set(expected) - set(produced))[:3]
            extra = sorted(set(produced) - set(expected))[:3]
            raise ValueError(f'the key set changed (missing {missing}, extra {extra})')
        for name, spec in produced.items():
            if _shape_of(spec) != _shape_of(expected[name]):
                raise ValueError(f'{name} changed shape')
            if str(spec.get('dtype')) != str(expected[name].get('dtype')):
                raise ValueError(f'{name} changed dtype')
        if not (header.get('__metadata__') or {}).get('lds_merge'):
            raise ValueError('the merge metadata was not written')
        import torch
        with Reader(path) as fh:
            probe = next((k for k in sorted(fh.keys()) if k.endswith('.weight')), None)
            if probe is None:
                raise ValueError('no weight tensor in the output')
            sample = fh.get_tensor(probe)
        if not bool(torch.isfinite(sample.to(torch.float32)).all().item()):
            raise ValueError(f'{probe} contains NaN or infinity after the merge')
        # NOT 'tensors': the CLI merges this dict into the writer's summary, which
        # already reports a tensor count. Two keys of the same name there is a
        # TypeError that surfaces as "the merge failed" AFTER a 26 GB write has
        # succeeded — the worst possible place to lose a result.
        out.update(verified=True, verified_tensors=len(produced))
    except Exception as e:                     # noqa: BLE001 — reported, not raised
        out['verify_error'] = str(e)[:300]
    return out


# --- CLI ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Worker entry point. Reads a JSON spec file, prints one result line.

    The spec arrives in a FILE rather than as arguments: a command line is
    visible to every process on the machine and the paths involved are the
    user's own folders. It also sidesteps quoting a Windows path through a
    subprocess argument list, which is where this kind of worker usually breaks.
    """
    import argparse
    parser = argparse.ArgumentParser(prog='lds-lora-merge')
    parser.add_argument('--spec', required=True)
    parser.add_argument('--progress', action='store_true')
    parser.add_argument('--budget-seconds', type=int, default=0)
    args = parser.parse_args(argv)

    result = {'ok': False, 'error': None, 'path': None}
    try:
        with open(args.spec, encoding='utf-8') as fh:
            spec = json.load(fh)

        def report(done, total):
            print(f'{PROGRESS_PREFIX} {done} {total}', flush=True)

        summary = merge_into_base(
            spec['base'], spec['destination'], spec['loras'],
            metadata=spec.get('metadata'), family=spec.get('family'),
            budget_seconds=args.budget_seconds or None,
            progress=report if args.progress else None)
        result.update(ok=True, **summary,
                      **verify_merge(spec['destination'], spec['base']))
    except Exception as e:                     # noqa: BLE001 — reported, not raised
        result['error'] = str(e)[:500]
    print(RESULT_PREFIX + ' ' + json.dumps(result))
    return 0 if result['ok'] else 1


if __name__ == '__main__':                     # pragma: no cover
    raise SystemExit(main())
