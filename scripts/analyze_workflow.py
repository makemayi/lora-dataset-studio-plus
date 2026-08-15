"""Flatten a ComfyUI API-format workflow so a diff or a reader can see it.

A workflow JSON is a graph keyed by node id, and reading one in a text editor
means jumping between a node and the ids it links to. This prints one node per
block with its class and every input, in file order, which is enough to answer
the questions that actually come up: which model file a graph loads, what the
sampler settings are, and whether an input is a literal or a link.

    python scripts/analyze_workflow.py backend/workflows/*.json

A LINK is printed as `[node_id, slot]`, which is exactly how it is stored — the
point is to see that the value comes from somewhere else, not to follow it.

API format only. A UI-format export (the one with a top-level "nodes" list) has
no `class_type` anywhere, so it prints zero nodes rather than guessing.
"""

import json
import sys


def analyze(path):
    """Print every node in one workflow. Returns the (id, class, inputs) list."""
    with open(path, encoding='utf-8') as handle:
        graph = json.load(handle)
    nodes = [(key, value['class_type'], value.get('inputs', {}))
             for key, value in graph.items()
             if isinstance(value, dict) and 'class_type' in value]
    print(f"\n{'=' * 70}\nFILE: {path}  ({len(nodes)} nodes)\n{'=' * 70}")
    if not nodes:
        print('  no class_type anywhere - this is probably a UI-format export,')
        print('  not the API format ComfyUI queues.')
    for key, class_type, inputs in nodes:
        print(f"[{key}] {class_type}")
        for name, value in inputs.items():
            if isinstance(value, (str, int, float, bool, list)):
                print(f"      {name} = {value}")
    return nodes


def main(argv):
    if not argv:
        print(__doc__.splitlines()[0])
        print('usage: python scripts/analyze_workflow.py <workflow.json> ...')
        return 1
    for path in argv:
        analyze(path)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
