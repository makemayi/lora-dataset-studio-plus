"""The shipped 'face swap.json' must keep the exact node IDs face_swap_helper
rewires, and must not carry display-only leaves.

REPLACED 2026-08-09. The graph was rebuilt and nothing about its shape carried
over: 28 nodes became 52 and every id the helper touches changed. That is
precisely why these assertions exist — the helper writes into nodes BY ID, so a
graph swap that nobody notices does not crash, it silently produces the wrong
picture (or drops the seed, or saves under ComfyUI's own prefix and makes every
tile show the same file).

The two ids most worth understanding, because getting them backwards is the one
mistake that still renders something plausible:

    423  the TARGET  — masked by three SAM3Segment passes and repainted
    221  the SOURCE  — the identity, rides in as a reference latent

Display-only leaves are dropped from the shipped copy the same way the previous
graph's rgthree Image Comparer was: they pull custom-node packs into a headless
job for no output.
"""
import json
from pathlib import Path

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / 'workflows' / 'face swap.json'

NODE_TARGET_IMAGE = '423'
NODE_REF_IMAGE = '221'
_REQUIRED_NODES = (NODE_TARGET_IMAGE, NODE_REF_IMAGE, '424:251', '424:104',
                   '424:175', '424:246', '424:253', '261', '425:211', '156')


def _load():
    with open(WORKFLOW_PATH, encoding='utf-8') as f:
        return json.load(f)


def test_required_nodes_present():
    wf = _load()
    for node in _REQUIRED_NODES:
        assert node in wf, f'required node {node} missing from face swap.json'


def test_the_helper_and_the_workflow_agree_on_every_id():
    """The list above is a copy of the helper's. A graph swap that updates one
    and not the other is exactly the failure this file is here to catch."""
    from app.services import face_swap_helper as fsh
    assert set(fsh._REQUIRED_NODES) == set(_REQUIRED_NODES)


def test_no_display_only_leaves():
    wf = _load()
    consumed = {v[0] for n in wf.values() for v in n.get('inputs', {}).values()
                if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)}
    for nid, node in wf.items():
        if node.get('class_type') in ('PreviewImage', 'MaskPreview',
                                      'Image Comparer (rgthree)'):
            assert nid in consumed, f'{nid} is a display-only leaf; drop it'


def test_target_and_reference_load_image_roles():
    """Backwards would swap which face survives — and still produce an image."""
    wf = _load()
    assert wf[NODE_TARGET_IMAGE]['class_type'] == 'LoadImage'
    assert wf[NODE_TARGET_IMAGE]['_meta']['title'] == '目标图'
    assert wf[NODE_REF_IMAGE]['class_type'] == 'LoadImage'
    # The target is what the segmentation masks; the reference never is.
    seg_sources = {tuple(n['inputs']['image']) for n in wf.values()
                   if n.get('class_type') == 'SAM3Segment'
                   and isinstance(n.get('inputs', {}).get('image'), list)}
    assert seg_sources, 'the new graph masks with SAM3Segment'
    reachable = _upstream_load_images(wf, seg_sources)
    assert NODE_TARGET_IMAGE in reachable
    assert NODE_REF_IMAGE not in reachable


def _upstream_load_images(wf, starts):
    """Every LoadImage id reachable by walking inputs backwards from `starts`."""
    seen, stack, found = set(), [s[0] for s in starts], set()
    while stack:
        nid = stack.pop()
        if nid in seen or nid not in wf:
            continue
        seen.add(nid)
        if wf[nid].get('class_type') == 'LoadImage':
            found.add(nid)
        for v in wf[nid].get('inputs', {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                stack.append(v[0])
    return found


def test_both_seed_nodes_exist():
    """The graph samples TWICE. Randomising one seed and not the other makes a
    batch share a stage, which reads to a user as "the seed does nothing"."""
    wf = _load()
    assert 'seed' in wf['261']['inputs']
    assert 'noise_seed' in wf['425:211']['inputs']


def test_every_link_resolves_to_an_existing_node():
    wf = _load()
    ids = set(wf)
    for nid, node in wf.items():
        for v in node.get('inputs', {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in ids, f'node {nid} references missing node {v[0]}'


def test_swap_lora_node_names_the_fixed_lora():
    wf = _load()
    assert wf['424:246']['class_type'] == 'LoraLoaderModelOnly'
    assert wf['424:246']['inputs']['lora_name'] == \
        'klein\\Klein2-9B-SmartCharacterSwap.safetensors'


def test_the_shipped_copy_carries_no_maintainer_filenames():
    """The helper overwrites these on every run, so whatever is committed is
    pure noise — and the source graph had a personal reference photo name and a
    pasted-clipboard path in them."""
    wf = _load()
    assert wf[NODE_TARGET_IMAGE]['inputs']['image'] == 'target.png'
    assert wf[NODE_REF_IMAGE]['inputs']['image'] == 'reference.png'
    assert wf['261']['inputs']['seed'] == 0
    assert wf['425:211']['inputs']['noise_seed'] == 0
