"""The shipped 'face swap.json' must keep the exact node IDs face_swap_helper
rewires, and must NOT carry the rgthree Image Comparer (node '117' in the
user-provided 'klein swap.json') — it's a display-only leaf that pulls in the
rgthree-comfy custom-node pack for no benefit in a headless job."""
import json

WORKFLOW_PATH = 'backend/workflows/face swap.json'

_REQUIRED_NODES = ('151', '121', '165:126', '165:102', '165:146',
                   '165:161', '165:156', '9')


def _load():
    with open(WORKFLOW_PATH, encoding='utf-8') as f:
        return json.load(f)


def test_required_nodes_present():
    wf = _load()
    for node in _REQUIRED_NODES:
        assert node in wf, f'required node {node} missing from face swap.json'


def test_rgthree_comparer_node_removed():
    wf = _load()
    assert '117' not in wf
    assert not any(n.get('class_type') == 'Image Comparer (rgthree)' for n in wf.values())


def test_target_and_reference_load_image_titles():
    wf = _load()
    assert wf['151']['class_type'] == 'LoadImage'
    assert wf['151']['_meta']['title'] == '目标图'
    assert wf['121']['class_type'] == 'LoadImage'
    assert wf['121']['_meta']['title'] == '参考图'


def test_every_link_resolves_to_an_existing_node():
    wf = _load()
    ids = set(wf)
    for nid, node in wf.items():
        for v in node.get('inputs', {}).values():
            if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str):
                assert v[0] in ids, f'node {nid} references missing node {v[0]}'


def test_swap_lora_node_names_the_fixed_lora():
    wf = _load()
    assert wf['165:161']['class_type'] == 'LoraLoaderModelOnly'
    assert wf['165:161']['inputs']['lora_name'] == 'klein\\Klein2-9B-SmartCharacterSwap.safetensors'
