"""The shipped 'minimax h3 swap.json' must keep the exact node IDs
minimax_h3_swap_helper rewires, and must not carry display-only leaves.

Same reasoning as test_face_swap_workflow_shape.py, and the same failure mode:
the helper writes into nodes BY ID, so a graph swap nobody notices does not
crash — it silently produces the wrong picture, drops the seed, or saves under
ComfyUI's own prefix so every tile of a batch displays the same file.

The two ids most worth understanding, because getting them backwards is the one
mistake that still renders something plausible:

    313  the TARGET — its head is masked out and repainted
    114  the SOURCE — the identity that gets grafted on

The three OPTIONAL stages ship WIRED. That direction is deliberate: the helper
can only ever remove a stage (repointing its consumers at what the maintainer's
own bypassed export fell back to), and a test that lets the file ship with a
stage already unwired would let that fallback wiring rot unnoticed.
"""
import json
from pathlib import Path

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / 'workflows' / 'minimax h3 swap.json'

NODE_TARGET_IMAGE = '313'
NODE_REF_IMAGE = '114'
_REQUIRED_NODES = (NODE_TARGET_IMAGE, NODE_REF_IMAGE, '426:312', '426:130',
                   '426:121', '426:172', '426:305', '426:170', '426:139',
                   '426:131', '412')
_STAGE_TAILS = ('426:423:65', '426:411', '426:396:370')


def _load():
    with open(WORKFLOW_PATH, encoding='utf-8') as f:
        return json.load(f)


def _consumed(wf):
    return {v[0] for n in wf.values() for v in n.get('inputs', {}).values()
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)}


def test_required_nodes_present():
    wf = _load()
    for node in _REQUIRED_NODES:
        assert node in wf, f'required node {node} missing from minimax h3 swap.json'


def test_the_helper_and_the_workflow_agree_on_every_id():
    """The list above is a copy of the helper's. A graph swap that updates one
    and not the other is exactly the failure this file is here to catch."""
    from app.services import minimax_h3_swap_helper as sh
    assert set(sh._REQUIRED_NODES) == set(_REQUIRED_NODES)
    assert {tail for tail, _fallback in sh.STAGES.values()} == set(_STAGE_TAILS)


def test_no_display_only_leaves():
    wf = _load()
    consumed = _consumed(wf)
    for nid, node in wf.items():
        if node.get('class_type') in ('PreviewImage', 'MaskPreview', 'MaskPreview+',
                                      'Image Comparer (rgthree)'):
            assert nid in consumed, f'{nid} is a display-only leaf; drop it'


def test_the_only_output_node_is_the_save():
    """Every OUTPUT node executes, so a second one is a second render nobody
    asked for — and the pruner keeps SaveImage's dependencies alone."""
    wf = _load()
    outputs = [nid for nid, n in wf.items() if n.get('class_type') == 'SaveImage']
    assert outputs == ['412']


def test_every_link_resolves_to_an_existing_node():
    wf = _load()
    ids = set(wf)
    for nid, node in wf.items():
        for v in node.get('inputs', {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in ids, f'node {nid} references missing node {v[0]}'


def test_no_orphan_nodes():
    """A node that reaches neither the save nor anything else is dead weight the
    export left behind — and it can still fail validation for the whole prompt."""
    wf = _load()
    consumed = _consumed(wf)
    orphans = [nid for nid, n in wf.items()
               if nid not in consumed and n.get('class_type') != 'SaveImage']
    assert orphans == [], f'unreachable nodes shipped: {orphans}'


def test_target_and_reference_load_image_roles():
    """Backwards would swap which face survives — and still produce an image."""
    wf = _load()
    assert wf[NODE_TARGET_IMAGE]['class_type'] == 'LoadImage'
    assert wf[NODE_TARGET_IMAGE]['_meta']['title'] == '目标图'
    assert wf[NODE_REF_IMAGE]['class_type'] == 'LoadImage'
    # The target is what gets masked; the identity photo never is.
    mask_sources = {tuple(n['inputs']['images']) for n in wf.values()
                    if n.get('class_type') == 'LayerMask: PersonMaskUltra'
                    and isinstance(n.get('inputs', {}).get('images'), list)}
    assert mask_sources, 'the graph masks the head with PersonMaskUltra'
    reachable = _upstream_load_images(wf, mask_sources)
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


def test_the_identity_reaches_h3_as_a_reference():
    """H3 takes TWO references: the identity photo and the masked target crop.
    Losing the first one silently turns a swap into a re-render of the target."""
    wf = _load()
    h3 = wf['426:170']['inputs']
    assert 'ref_images.ref_image_0' in h3 and 'ref_images.ref_image_1' in h3
    assert NODE_REF_IMAGE in _upstream_load_images(
        wf, {tuple(h3['ref_images.ref_image_0'])})
    assert NODE_TARGET_IMAGE in _upstream_load_images(
        wf, {tuple(h3['ref_images.ref_image_1'])})


def test_all_three_optional_stages_ship_wired():
    consumed = _consumed(_load())
    for tail in _STAGE_TAILS:
        assert tail in consumed, (
            f'{tail} ships unwired — the helper only ever REMOVES a stage, so a '
            'stage that is already bypassed in the file can never be switched on')


def test_the_shipped_copy_carries_no_maintainer_filenames():
    """The helper overwrites these on every run, so whatever is committed is
    pure noise — and the source graph had a pasted-clipboard filename in one."""
    wf = _load()
    assert wf[NODE_TARGET_IMAGE]['inputs']['image'] == 'target.png'
    assert wf[NODE_REF_IMAGE]['inputs']['image'] == 'reference.png'
    assert wf['426:131']['inputs']['noise_seed'] == 0
    assert wf['426:423:73']['inputs']['noise_seed'] == 0
    assert wf['426:396:370']['inputs']['seed'] == 0


def test_the_frame_selector_keeps_exactly_one_frame():
    """H3 samples a packet. Without select_count 1 this is a video model writing
    several tiles per swap."""
    wf = _load()
    assert wf['426:304']['class_type'] == 'H3FrameSelect'
    assert wf['426:304']['inputs']['select_count'] == 1
