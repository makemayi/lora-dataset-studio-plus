"""Krea 2 checkpoints installed on this machine are choosable TRAINING bases.

The defect: `/train/base-info` returned ONE hardcoded Krea entry
(``[{'value': '', 'label': 'Official - Krea 2'}]``) while the same endpoint
listed Z-Image through `get_zimage_models()` and SDXL through
`get_checkpoint_models()`. A Krea 2 model the user had just trained — or any
community Krea build on disk — was therefore not offerable as a base, while a
Z-Image merge was.

The half that is easy to get wrong is the ADDRESS, and it is why these tests
exist as much as the list itself. The Studio and the trainer speak different
vocabularies on purpose:

  * the Studio picker sends a ComfyUI-relative loader name (``Krea/x.safetensors``)
    to a ComfyUI running on this machine;
  * the trainer reads an ABSOLUTE path as "custom weights, load this file"
    (`lora_training._is_custom_weights`) and a RELATIVE name as another family's
    base — `foreign_base_reason` classifies it as foreign and the builder falls
    back to the official weights, silently. It can also run on a remote pod that
    must RECEIVE the file, which a ComfyUI folder name cannot express.

So the list is built with the Studio's scanner and translated to the trainer's
currency. `test_the_listed_value_reaches_the_job_config` is the one that fails if
that translation is ever "simplified" away.
"""
import json
import os
import struct

import pytest

from app.config import LOCAL_USER


@pytest.fixture(autouse=True)
def _clear_caches():
    from app.services import comfy_model_paths as cmp
    from app.services import model_integrity as mi
    from app.utils import comfyui as cu
    cmp.clear_cache()
    cu.clear_model_caches()
    mi.clear_cache()
    yield
    cmp.clear_cache()
    cu.clear_model_caches()
    mi.clear_cache()


@pytest.fixture()
def aitoolkit(tmp_path, app):
    """base-info is gated on a valid ai-toolkit install; fake one (never executed)
    so the route answers instead of 409-ing."""
    from app import config as cfg
    root = tmp_path / 'aitoolkit'
    (root / 'venv' / 'Scripts').mkdir(parents=True)
    (root / 'venv' / 'Scripts' / 'python.exe').write_text('fake')
    (root / 'run.py').write_text('fake')
    with app.app_context():
        cfg.save_config({'aitoolkit': {'dir': str(root)}})   # save_config deep-merges
    return root


def _safetensors(path, tensors, metadata=None):
    """A header-only fixture: the declared body is never written, which is the
    point — every check here reads the header and nothing else."""
    index = {}
    offset = 0
    for name, (dtype, nbytes) in tensors.items():
        index[name] = {'dtype': dtype, 'shape': [1],
                       'data_offsets': [offset, offset + nbytes]}
        offset += nbytes
    if metadata:
        index['__metadata__'] = metadata
    blob = json.dumps(index).encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<Q', len(blob)))
        fh.write(blob)
        fh.write(b'\0' * 16)
    return str(path)


def _bf16(path, n=20):
    return _safetensors(path, {f'blocks.{i}.attn.wq.weight': ('BF16', 128)
                               for i in range(n)})


def _bare_fp8(path):
    """The shape MEASURED on the checkpoint this app itself ships as the Krea 2
    Turbo default: an fp8 payload, F32 norms, the SAME tensor names as the bf16
    master, no marker and no scale sibling."""
    tensors = {f'blocks.{i}.attn.wq.weight': ('F8_E4M3', 128) for i in range(20)}
    tensors.update({f'blocks.{i}.norm.scale': ('F32', 12) for i in range(12)})
    return _safetensors(path, tensors)


def _scaled_fp8(path):
    """A ComfyUI scaled export: the payload PLUS per-tensor scales and the legacy
    marker — extra KEYS, which is what a strict state-dict load cannot survive."""
    tensors = {f'blocks.{i}.attn.wq.weight': ('F8_E4M3', 128) for i in range(20)}
    tensors.update({f'blocks.{i}.attn.wq.scale_weight': ('F32', 4) for i in range(20)})
    tensors['scaled_fp8'] = ('F8_E4M3', 2)
    return _safetensors(path, tensors)


@pytest.fixture()
def comfy(tmp_path, app):
    """A ComfyUI tree with a 'Krea' UNET folder — the layout both
    `get_krea_models()` and `comfy_model_paths.search_roots` resolve from."""
    from app import config as cfg
    base = tmp_path / 'ComfyUI'
    (base / 'output').mkdir(parents=True)
    (base / 'models' / 'unet' / 'Krea').mkdir(parents=True)
    with app.app_context():
        cfg.save_config({'comfyui': {'base_dir': str(base)}})
    return base


@pytest.fixture()
def krea_ds(app):
    from app.services import face_dataset_service as svc
    with app.app_context():
        ds = svc.create_dataset(LOCAL_USER, 'Krea subject', 'ktrig',
                                train_type='krea')
        return ds.id


def _source_without_docstring(source: str) -> str:
    """`source` with its leading docstring removed, by LINE RANGE from the parse
    tree rather than by string value.

    Value-based removal is the trap this exists to avoid: ``fn.__doc__`` is the
    unescaped text, so any docstring containing a backslash escape differs from
    the bytes in the file and the removal quietly does nothing."""
    import ast
    import textwrap
    tree = ast.parse(textwrap.dedent(source))
    fn = tree.body[0]
    first = fn.body[0] if fn.body else None
    if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return source
    lines = source.splitlines()
    return '\n'.join(lines[:first.lineno - 1] + lines[first.end_lineno:])


def _krea_bases(client, ds_id):
    body = client.get(f'/api/dataset/{ds_id}/train/base-info').get_json()
    return body['bases_by_type']['krea']


# --- 1) the list exists at all -------------------------------------------------

def test_an_installed_krea_checkpoint_is_offered_as_a_base(
        app, client, aitoolkit, comfy, krea_ds):
    """RED before the fix: the Krea list was one hardcoded official entry."""
    _bf16(comfy / 'models' / 'unet' / 'Krea' / 'my_krea_merge.safetensors')
    bases = _krea_bases(client, krea_ds)
    labels = [b['label'] for b in bases]
    assert bases[0]['value'] == '', 'the official base must stay first'
    assert 'my_krea_merge' in labels, f'installed Krea checkpoint missing: {labels}'


def test_the_official_entry_survives_an_install_with_no_krea_file(
        app, client, aitoolkit, comfy, krea_ds):
    assert [b['value'] for b in _krea_bases(client, krea_ds)] == ['']


# --- 2) the ADDRESS: the trainer's vocabulary, not the Studio's -----------------

def test_every_listed_value_is_what_the_trainer_consumes(
        app, client, aitoolkit, comfy, krea_ds):
    """A relative name here is not a cosmetic detail: `foreign_base_reason` reads
    it as another family's base and the run trains on the OFFICIAL weights while
    the panel claims otherwise."""
    from app.services import lora_training as lt
    _bf16(comfy / 'models' / 'unet' / 'Krea' / 'my_krea_merge.safetensors')
    picked = [b for b in _krea_bases(client, krea_ds) if b['value']]
    assert picked, 'nothing to check — the list regressed to official-only'
    for entry in picked:
        assert os.path.isabs(entry['value']), entry
        assert lt._is_custom_weights(entry['value']), entry
        assert lt.foreign_base_reason('krea', entry['value']) is None, entry
        assert os.path.isfile(entry['value']), entry


def test_the_listed_value_reaches_the_job_config(
        app, client, aitoolkit, comfy, krea_ds):
    """End of the chain: what the picker offers is what ai-toolkit is told to
    load. The Anima twin of this test pins the opposite direction (a foreign
    relative name must NOT reach it)."""
    from app.services import face_dataset_service as svc
    from app.services import lora_training as lt
    _bf16(comfy / 'models' / 'unet' / 'Krea' / 'my_krea_merge.safetensors')
    value = [b['value'] for b in _krea_bases(client, krea_ds) if b['value']][0]
    with app.app_context():
        ds = svc.get_dataset(LOCAL_USER, krea_ds)
        ds.train_base_model = value
        svc.db.session.commit()
        cfg = lt._build_job_config_krea(ds, 'folder', 100, training_folder='out')
    assert cfg['config']['process'][0]['model']['name_or_path'] == value


def test_a_name_that_resolves_to_no_file_is_dropped_not_emitted_relative(
        app, monkeypatch, client, aitoolkit, comfy, krea_ds):
    """Offering a value the trainer would silently ignore is worse than a shorter
    list — the run would report a base it never loaded."""
    from app.routes import training as tr
    monkeypatch.setattr(tr, 'get_krea_models',
                        lambda: ['Krea' + os.sep + 'ghost.safetensors'])
    assert [b['value'] for b in _krea_bases(client, krea_ds)] == ['']


def test_the_scanner_is_the_studios_one_not_a_fifth_copy(app, comfy):
    """Four scanners already disagree with each other on this exact question. The
    route must call `get_krea_models`, so a fix to it lands everywhere at once.

    Asserted by PROVENANCE, not by object identity. `tr.get_krea_models is
    cu.get_krea_models` was the obvious spelling and it is wrong: another module
    in this suite calls ``importlib.reload`` on app.utils.comfyui, which rebuilds
    every function in it while the route keeps the binding its own `from … import`
    took. Same source, same behaviour, two objects — an identity check turns that
    into a red that says nothing about the code. Whether the route grew a private
    scan is what we actually care about, and `__module__` answers it across a
    reload."""
    import inspect

    from app.routes import training as tr
    assert tr.get_krea_models.__module__ == 'app.utils.comfyui'
    assert tr.get_krea_models.__name__ == 'get_krea_models'
    # The other half of "no fifth scanner": the route must not walk the model
    # tree itself. It resolves names the scanner returned, nothing more.
    #
    # HEURISTIC, and it says so rather than implying otherwise: this catches the
    # COMMON spellings of a directory scan, not every possible one. A fifth
    # scanner could still be born through an indirection none of these names
    # appear in (a helper in another module, os.fdopen games, a comprehension over
    # something already listed). Read a red here as "someone started scanning",
    # never a green as "no scan can exist" — the guarantee that actually holds is
    # the `__module__` assertion above.
    #
    # Matched with the trailing "(" so both `os.walk(` and a bare `walk(` from
    # `from os import walk` are caught, and so are the pathlib forms
    # (`p.glob(`, `p.rglob(`, `p.iterdir(`) that a `glob.glob` check misses.
    scan_calls = ('walk(', 'scandir(', 'listdir(', 'glob(', 'iglob(',
                  'rglob(', 'iterdir(')
    source = inspect.getsource(tr._krea_installed_bases)
    body = _source_without_docstring(source)
    # The strip must actually strip. `source.replace(fn.__doc__, '')` was the
    # first spelling and it silently did NOTHING: __doc__ holds the UNESCAPED
    # text, so a docstring containing `Krea\\x.safetensors` (two characters in
    # the file, one in __doc__) never matches. It passed anyway — because the
    # docstring happens to name no scan call today — which is a check attesting
    # more than it verifies. Pin the mechanism, not just the outcome.
    assert len(body) < len(source), 'the docstring strip is a no-op again'
    assert 'Studio picker speaks' not in body, 'the docstring survived the strip'
    for forbidden in scan_calls:
        assert forbidden not in body, f'the route scans the disk itself ({forbidden})'


# --- 3) what the picker says about a quantized file ----------------------------

def test_a_bare_fp8_checkpoint_is_listed_trainable_with_a_quantified_note(
        app, client, aitoolkit, comfy, krea_ds):
    """The app's own default Krea 2 Turbo checkpoint has exactly this shape. It
    loads (the trainer up-casts it) — so it is offered, with the cost stated."""
    _bare_fp8(comfy / 'models' / 'unet' / 'Krea' / 'krea2_turbo_fp8.safetensors')
    entry = [b for b in _krea_bases(client, krea_ds) if b['value']][0]
    assert entry['trainable'] is True
    assert entry['quantization'] == 'bare_cast'
    note = entry['note']
    assert note and 'F8_E4M3' in note
    assert '20 of its 32 tensors' in note, note      # the count, not an adjective
    assert 'bf16' in note


def test_a_scaled_fp8_checkpoint_is_listed_as_not_trainable(
        app, client, aitoolkit, comfy, krea_ds):
    _scaled_fp8(comfy / 'models' / 'unet' / 'Krea' / 'merge_fp8.safetensors')
    entry = [b for b in _krea_bases(client, krea_ds) if b['value']][0]
    assert entry['trainable'] is False
    assert entry['quantization'] == 'structured'
    assert entry['note']


def test_a_plain_bf16_checkpoint_is_listed_with_nothing_to_say(
        app, client, aitoolkit, comfy, krea_ds):
    _bf16(comfy / 'models' / 'unet' / 'Krea' / 'clean.safetensors')
    entry = [b for b in _krea_bases(client, krea_ds) if b['value']][0]
    assert entry['trainable'] is True
    assert not entry['quantization']
    assert entry['note'] is None


# --- 4) selecting one of them, end to end --------------------------------------

def test_selecting_a_listed_bare_fp8_base_is_accepted(
        app, client, aitoolkit, comfy, krea_ds):
    path = _bare_fp8(comfy / 'models' / 'unet' / 'Krea' / 'krea2_turbo_fp8.safetensors')
    r = client.post(f'/api/dataset/{krea_ds}/train/settings',
                    json={'base_model': path})
    assert r.status_code == 200, r.get_json()
    from app.services import face_dataset_service as svc
    with app.app_context():
        assert svc.get_dataset(LOCAL_USER, krea_ds).train_base_model == path


def test_selecting_a_listed_scaled_fp8_base_is_refused_with_the_reason(
        app, client, aitoolkit, comfy, krea_ds):
    path = _scaled_fp8(comfy / 'models' / 'unet' / 'Krea' / 'merge_fp8.safetensors')
    r = client.post(f'/api/dataset/{krea_ds}/train/settings',
                    json={'base_model': path})
    assert r.status_code == 400
    message = json.dumps(r.get_json()).lower()
    # Semantics, not wording: it must name the FORMAT obstacle and point at the
    # way out. Pinning the sentence would freeze a text we want free to improve.
    assert 'scale_weight' in message or 'scaled_fp8' in message
    assert 'bf16' in message
    from app.services import face_dataset_service as svc
    with app.app_context():
        assert not svc.get_dataset(LOCAL_USER, krea_ds).train_base_model


# --- 5) a path TYPED into « Custom weights… » ----------------------------------
# The one base that cannot be picked from a list — a checkpoint downloaded five
# minutes ago, which is the whole reason the field exists — was also the only one
# whose "the trainer cannot load this" arrived after the dataset had been
# exported and, on the cloud lane, after a GPU had been rented.

def _advisory(client, path):
    return client.get('/api/train/base-file-advisory',
                      query_string={'path': str(path)}).get_json()


def test_a_typed_packed_export_is_refused_before_the_launch(client, tmp_path):
    body = _advisory(client, _scaled_fp8(tmp_path / 'downloads' / 'packed.safetensors'))
    assert body['status'] == 'ok'
    assert body['trainable'] is False
    assert body['level'] == 'error'
    assert body['quantization'] == 'structured'
    # Same sentence the picker shows for a LISTED base — one source, so the two
    # surfaces cannot come to disagree about the same file.
    from app.services import model_integrity
    assert body['note'] == model_integrity.QUANT_REFUSAL


def test_a_typed_bare_cast_trains_and_says_what_it_costs(client, tmp_path):
    body = _advisory(client, _bare_fp8(tmp_path / 'downloads' / 'cast.safetensors'))
    assert body['trainable'] is True
    assert body['level'] == 'warning'
    assert body['note'] and 'tensors' in body['note']


def test_a_typed_clean_base_says_nothing(client, tmp_path):
    body = _advisory(client, _bf16(tmp_path / 'downloads' / 'clean.safetensors'))
    assert body['trainable'] is True
    assert body['level'] == ''
    assert body['note'] is None


def test_a_path_that_is_not_there_says_so_instead_of_guessing(client, tmp_path):
    body = _advisory(client, tmp_path / 'downloads' / 'absent.safetensors')
    assert body['status'] == 'missing'
    assert body['trainable'] is False
    assert 'absent.safetensors' in body['note']


def test_a_file_that_is_not_a_single_file_checkpoint_is_named_as_such(client, tmp_path):
    body = _advisory(client, tmp_path / 'downloads' / 'model.gguf')
    assert body['status'] == 'not_a_model'
    assert body['trainable'] is False


def test_the_reply_never_carries_the_path_it_was_given(client, tmp_path):
    """These payloads end up in pasted diagnostics, and a Windows path is a
    username. The BASENAME is what identifies the file to its owner; the folder
    it sits in is the caller's own input and it comes back to nobody."""
    path = _bf16(tmp_path / 'downloads' / 'private' / 'clean.safetensors')
    body = _advisory(client, path)
    blob = json.dumps(body)
    assert 'clean.safetensors' in blob
    assert str(tmp_path) not in blob
    assert 'private' not in blob


def test_an_empty_path_is_a_question_with_no_answer_not_a_refusal(client):
    body = _advisory(client, '')
    assert body['trainable'] is True and body['note'] is None
