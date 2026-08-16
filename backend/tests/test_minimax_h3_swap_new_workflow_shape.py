"""The shipped 'minimax h3 swap new.json' must keep the exact node IDs
minimax_h3_swap_new_helper rewires, and must not carry display-only leaves.

Same reasoning and the same failure mode as
test_minimax_h3_swap_workflow_shape.py for the older graph: the helper writes
into nodes BY ID, so a graph swap nobody notices does not crash — it silently
produces the wrong picture, drops the seed, or saves under ComfyUI's own prefix
so every tile of a batch displays the same file.

The two ids most worth understanding, because getting them backwards is the one
mistake that still renders something plausible:

    1010  the TARGET — its head is erased and re-rendered
     114  the SOURCE — the identity that gets grafted on

The remaining OPTIONAL stage ships WIRED. That direction is deliberate: the
helper can only ever remove a stage (repointing its consumers at what a bypass
of that node falls back to), and a file that shipped with a stage already
unwired would let that fallback wiring rot unnoticed.

Re-pointed at the 2026-08-16 graph. What moved, and why the ids are all new:
the H3 loader group was a subgraph (`925:*`) and is now `1009:*`; the head is
masked by SAM3 (`1001:75`) instead of RMBG's ClothesSegment; `H3FrameSelect`
became a plain `ImageFromBatch`; the frame packet length moved off its own
PrimitiveInt onto the H3 node; and `AILab_MaskOverlay` — the `mask_overlay`
stage — has no node in this graph at all, so that stage is gone rather than
kept-but-inert.
"""
import json
from pathlib import Path

WORKFLOW_PATH = (Path(__file__).resolve().parents[1] / 'workflows'
                 / 'minimax h3 swap new.json')

NODE_TARGET_IMAGE = '1010'
NODE_REF_IMAGE = '114'
_REQUIRED_NODES = (NODE_TARGET_IMAGE, NODE_REF_IMAGE, '928', '1001:75', '170',
                   '990', '991', '131', '999',
                   '1009:1005', '1009:1002', '1009:1008', '1009:1007',
                   '983:973', '983:962', '983:963', '983:966', '983:969', '165',
                   # The Klein pass's own instruction and its negative: the
                   # helper writes both from config on every job.
                   '983:964', '983:965',
                   # BasicGuider — the anchor extra LoRAs chain onto; and
                   # BasicScheduler, whose 8 steps an accelerator LoRA has to
                   # be able to override.
                   '128', '126')
_STAGE_TAILS = ('991',)


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
    seg = {tuple(n['inputs']['image']) for n in wf.values()
           if n.get('class_type') == 'SAM3_Detect'
           and isinstance(n.get('inputs', {}).get('image'), list)}
    assert seg, 'the graph masks the head with SAM3_Detect'
    reachable = _upstream_load_images(wf, seg)
    assert NODE_TARGET_IMAGE in reachable
    assert NODE_REF_IMAGE not in reachable


def test_the_mask_covers_the_head_and_nothing_else():
    """The mask decides what Klein erases. Anything else named here is a region
    the graph deletes and then asks H3 to invent.

    ClothesSegment's per-part booleans became SAM3's open-vocabulary phrase, so
    the check moved from "these switches are off" to "no phrase names anything
    below the neck". The job itself never runs this node — the app's own mask is
    substituted in — but the shipped graph still has to open in ComfyUI as the
    thing it claims to be."""
    wf = _load()
    mask = wf['1001:75']
    assert mask['class_type'] == 'SAM3_Detect'
    phrase = wf[mask['inputs']['conditioning'][0]]['inputs']['text']
    assert 'face' in phrase and 'hair' in phrase
    for part in ('clothes', 'dress', 'pants', 'background', 'body'):
        assert part not in phrase.lower(), f'{part} must stay out of the mask'


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
    # No `text_b`: the head analysis moved into the app, which appends it to 990
    # directly, so the concat node has one writer and one input.
    assert 'text_b' not in wf['991']['inputs']


def test_every_optional_stage_ships_wired():
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
    # The export's two accelerator LoRAs named files under the maintainer's own
    # `minimax H3\` folder. The helper drops both nodes on every run, so a path
    # here is noise that only ever misleads someone reading the file.
    for nid in ('1009:938', '1009:939'):
        assert wf[nid]['inputs']['lora_name'] == ''


def test_the_frame_selector_keeps_exactly_one_frame():
    """H3 samples a packet. Without a single-frame take this is a video model
    writing several tiles per swap.

    `H3FrameSelect` scored the packet and kept a winner; this graph pins the
    packet to one frame on the H3 node and takes index 0, so the guarantee is
    now `length == 1` on both ends rather than `select_count`."""
    wf = _load()
    assert wf['999']['class_type'] == 'ImageFromBatch'
    assert wf['999']['inputs']['length'] == 1
    assert wf['999']['inputs']['batch_index'] == 0
    assert wf['170']['inputs']['length'] == 1


def test_the_hybrid_loader_takes_two_models():
    """Both halves are rewritten by the helper, and the BASE is the Fl2VA file
    every other H3 path here refuses to pick."""
    loader = _load()['1009:1005']
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
    model followed the repeated intent and stopped after erasing.

    2026-08-16 added one deletion that is NOT the head — the hair goes first, so
    the stand-in is not a wig on a mannequin. So the rule is no longer "no
    deletion verb at all", it is "no deletion verb aimed at the HEAD before the
    replacement"."""
    text = _load()['983:964']['inputs']['text']
    head, _, rest = text.partition('\n')
    # The replacement is in the first clause, and nothing before it deletes the
    # HEAD — removing the hair is allowed and is what this graph asks for.
    assert '替换' in head
    for verb in ('移除', '抹除', '消除'):
        start = 0
        while (at := head.find(verb, start)) != -1:
            if at < head.index('替换'):
                target = head[at + len(verb):at + len(verb) + 6]
                assert '头发' in target, (
                    f'{verb} before 替换 must target the hair, not {target!r}')
            start = at + len(verb)
    # The geometry is pinned by name: it is what the rest of the graph reads.
    # SIZE is deliberately not in that list any more — the stand-in is smaller
    # than the head it replaces, so it is stated as its own clause instead, with
    # the reason attached. Both halves are required: a bare "缩小五分之一" reads
    # as an arbitrary shrink and invites the model to skip it.
    for word in ('位置', '朝向', '透视'):
        assert word in text, word
    assert '颅骨' in text and '五分之一' in text
    assert '不要把头发的体积算进素模头' in text
    # ...and the removal is a CONSTRAINT, not the instruction.
    assert '没有任何身份特征' in rest


def test_the_swap_instruction_knows_the_head_is_a_mannequin_now():
    """The Klein pass no longer erases the head, it leaves a grey mannequin. An
    instruction that does not mention it lets H3 treat the grey head as real
    scene content — painting around it, or leaving part of it in the result.

    The stand-in is 灰白色 since 2026-08-16 and models the SKULL: the Klein pass
    strips the hair and shrinks the head by a fifth, because hair thickness read
    as head size is what produced the doll head. So this instruction has to say
    which of the two outlines the stand-in is, or H3 grows the skull to fill it
    and then puts hair on top of THAT — the same doll head by another route.

    Both halves are pinned: the skull matches the stand-in, and the finished
    outline is larger than it."""
    text = _load()['990']['inputs']['text']
    assert '灰白色素模头' in text
    assert '不得残留任何灰白色素模' in text
    # The geometry the stand-in carries — size included, but named as the SKULL.
    for word in ('位置', '朝向', '透视', '颅骨'):
        assert word in text, word
    assert '去掉头发厚度' in text
    assert '头发生长在颅骨之外' in text
    # The failure this wording exists to prevent, said out loud.
    assert '不要为了填满素模头而放大颅骨' in text
