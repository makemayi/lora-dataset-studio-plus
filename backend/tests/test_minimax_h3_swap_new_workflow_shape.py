"""The shipped 'minimax h3 swap new.json' must keep the exact node IDs
minimax_h3_swap_new_helper rewires, and must not carry display-only leaves.

Same reasoning and the same failure mode as
test_minimax_h3_swap_workflow_shape.py for the older graph: the helper writes
into nodes BY ID, so a graph swap nobody notices does not crash — it silently
produces the wrong picture, drops the seed, or saves under ComfyUI's own prefix
so every tile of a batch displays the same file.

The two ids most worth understanding, because getting them backwards is the one
mistake that still renders something plausible:

    1003  the TARGET — its head is erased and re-rendered
     114  the SOURCE — the identity that gets grafted on

The two OPTIONAL stages ship WIRED. That direction is deliberate: the helper can
only ever remove a stage (repointing its consumers at what a bypass of that node
falls back to), and a file that shipped with a stage already unwired would let
that fallback wiring rot unnoticed.
"""
import json
from pathlib import Path

WORKFLOW_PATH = (Path(__file__).resolve().parents[1] / 'workflows'
                 / 'minimax h3 swap new.json')

NODE_TARGET_IMAGE = '1003'
NODE_REF_IMAGE = '114'
_REQUIRED_NODES = (NODE_TARGET_IMAGE, NODE_REF_IMAGE, '928', '957', '170',
                   '990', '991', '988', '131', '139', '304', '305',
                   '925:427', '925:922', '925:921', '925:926', '983:1002',
                   '983:973', '983:962', '983:963', '983:966', '983:969', '165',
                   # The Klein pass's own instruction and its negative: the
                   # helper writes both from config on every job.
                   '983:964', '983:965')
_STAGE_TAILS = ('983:1002', '991')


def _load():
    with open(WORKFLOW_PATH, encoding='utf-8') as f:
        return json.load(f)


def _consumed(wf):
    return {v[0] for n in wf.values() for v in n.get('inputs', {}).values()
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)}


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


def test_required_nodes_present():
    wf = _load()
    for node in _REQUIRED_NODES:
        assert node in wf, f'required node {node} missing from the new swap graph'


def test_the_helper_and_the_workflow_agree_on_every_id():
    """The list above is a copy of the helper's. A graph swap that updates one
    and not the other is exactly the failure this file is here to catch."""
    from app.services import minimax_h3_swap_new_helper as sh
    assert set(sh._REQUIRED_NODES) == set(_REQUIRED_NODES)
    assert {tail for tail, _fallback in sh.STAGES.values()} == set(_STAGE_TAILS)


def test_no_display_only_leaves():
    wf = _load()
    consumed = _consumed(wf)
    for nid, node in wf.items():
        if node.get('class_type') in ('PreviewImage', 'MaskPreview', 'MaskPreview+',
                                      'Image Comparer (rgthree)',
                                      'ClipboardImageNode'):
            assert nid in consumed, f'{nid} is a display-only leaf; drop it'


def test_the_only_output_node_is_the_save():
    wf = _load()
    assert [nid for nid, n in wf.items() if n.get('class_type') == 'SaveImage'] == ['165']


def test_every_link_resolves_to_an_existing_node():
    wf = _load()
    ids = set(wf)
    for nid, node in wf.items():
        for v in node.get('inputs', {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in ids, f'node {nid} references missing node {v[0]}'


def test_no_orphan_nodes():
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
    # The target is what gets segmented; the identity photo never is.
    seg = {tuple(n['inputs']['images']) for n in wf.values()
           if n.get('class_type') == 'ClothesSegment'
           and isinstance(n.get('inputs', {}).get('images'), list)}
    assert seg, 'the graph masks the head with ClothesSegment'
    reachable = _upstream_load_images(wf, seg)
    assert NODE_TARGET_IMAGE in reachable
    assert NODE_REF_IMAGE not in reachable


def test_the_mask_covers_the_head_and_nothing_else():
    """The mask decides what Klein erases. Anything else switched on here is a
    region the graph deletes and then asks H3 to invent."""
    mask = _load()['957']
    assert mask['class_type'] == 'ClothesSegment'
    assert mask['inputs']['Face'] is True
    assert mask['inputs']['Hair'] is True
    for part in ('Upper-clothes', 'Dress', 'Pants', 'Background'):
        assert mask['inputs'][part] is False, f'{part} must stay out of the mask'


def test_the_identity_reaches_h3_as_a_reference():
    """H3 takes TWO references: the identity photo and the head-removed target.
    Losing the first one silently turns a swap into a re-render of the target."""
    wf = _load()
    h3 = wf['170']['inputs']
    assert 'ref_images.ref_image_0' in h3 and 'ref_images.ref_image_1' in h3
    assert NODE_REF_IMAGE in _upstream_load_images(
        wf, {tuple(h3['ref_images.ref_image_0'])})
    assert NODE_TARGET_IMAGE in _upstream_load_images(
        wf, {tuple(h3['ref_images.ref_image_1'])})


def test_the_prompt_arrives_as_a_link_from_the_text_node():
    """On this graph the H3 prompt is WIRED, not typed. Anything written to the
    H3 node's own `prompt` input is discarded at execution time, which is why
    the helper writes 990 — and why a future edit that turns this back into a
    literal string has to notice."""
    wf = _load()
    assert isinstance(wf['170']['inputs']['prompt'], list)
    assert wf['990']['class_type'] == 'Text Multiline'
    assert wf['991']['class_type'] == 'Text Concatenate'
    assert wf['991']['inputs']['text_a'] == ['990', 0]
    assert wf['991']['inputs']['text_b'] == ['988', 0]


def test_both_optional_stages_ship_wired():
    consumed = _consumed(_load())
    for tail in _STAGE_TAILS:
        assert tail in consumed, (
            f'{tail} ships unwired — the helper only ever REMOVES a stage, so a '
            'stage that is already bypassed in the file can never be switched on')


def test_the_shipped_copy_carries_no_maintainer_filenames_or_seeds():
    """The helper overwrites these on every run, so whatever is committed is
    pure noise — and the source graph carried the maintainer's own filenames."""
    wf = _load()
    assert wf[NODE_TARGET_IMAGE]['inputs']['image'] == 'target.png'
    assert wf[NODE_REF_IMAGE]['inputs']['image'] == 'reference.png'
    assert wf['131']['inputs']['noise_seed'] == 0
    assert wf['983:969']['inputs']['seed'] == 0
    assert wf['988']['inputs']['seed'] == 0


def test_the_frame_selector_keeps_exactly_one_frame():
    """H3 samples a packet. Without select_count 1 this is a video model writing
    several tiles per swap."""
    wf = _load()
    assert wf['304']['class_type'] == 'H3FrameSelect'
    assert wf['304']['inputs']['select_count'] == 1


def test_the_hybrid_loader_takes_two_models():
    """Both halves are rewritten by the helper, and the BASE is the Fl2VA file
    every other H3 path here refuses to pick."""
    loader = _load()['925:427']
    assert loader['class_type'] == 'MiniMaxH3HybridLoader'
    assert 'base_model' in loader['inputs'] and 'overlay_model' in loader['inputs']


def test_the_klein_instruction_matches_the_config_default():
    """Two copies of one string, deliberately: the config value is what RUNS
    (the helper writes it on every job), the node's own text is what someone
    sees when they open the graph in ComfyUI. This pins them together so the
    file cannot start describing a pass it no longer performs."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.config import DEFAULTS
    wf = _load()
    fs = DEFAULTS['face_swap']
    assert wf['983:964']['inputs']['text'] == fs['h3_head_removal_prompt']
    assert wf['983:965']['inputs']['text'] == fs['h3_head_removal_negative']


def test_the_klein_instruction_replaces_the_head_rather_than_deleting_it():
    """The stand-in Klein leaves behind is the ONLY thing telling H3 how big the
    head was and which way it faced — delete instead of replace and the swap is
    guessing, which is where a doll-sized head comes from.

    The maintainer's original led with three deletion verbs and hung the
    stand-in off the end as a transform of the head it had just erased; the
    model followed the repeated intent and stopped after erasing."""
    text = _load()['983:964']['inputs']['text']
    head, _, rest = text.partition('\n')
    # The FIRST clause is the replacement, and no deletion verb precedes it.
    assert '替换' in head
    assert head.index('替换') < min([head.index(v) for v in ('移除', '抹除', '消除')
                                     if v in head] or [len(head)])
    # The geometry is pinned by name: it is what the rest of the graph reads.
    for word in ('大小', '位置', '朝向', '透视'):
        assert word in text, word
    # ...and the removal is a CONSTRAINT, not the instruction.
    assert '没有任何身份特征' in rest


def test_the_swap_instruction_knows_the_head_is_a_mannequin_now():
    """The Klein pass no longer erases the head, it leaves a grey mannequin. An
    instruction that does not mention it lets H3 treat the grey head as real
    scene content — painting around it, or leaving part of it in the result."""
    text = _load()['990']['inputs']['text']
    assert '灰色素模头' in text
    assert '不得残留任何灰色素模' in text
    # ...and it must carry the geometry over from the stand-in, which is the
    # whole reason the stand-in exists.
    for word in ('大小', '位置', '朝向', '透视'):
        assert word in text, word
