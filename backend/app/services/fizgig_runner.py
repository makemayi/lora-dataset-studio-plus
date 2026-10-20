"""Runs one Fizgig training plan: cache latents, cache text, train — in order.

Executed by FIZGIG's own interpreter (not this app's), so it imports nothing
from the app. One process owns the whole run, which is what lets the app's
existing Stop (`taskkill /T` on the recorded pid) end whichever stage is
running, and lets a Flask restart lose nothing: the chain does not live in a
thread of the server.

    python fizgig_runner.py <plan.json>

plan.json = {"cwd": str, "env": {str: str}, "stages": [{"name": str, "args": [str]}]}
Output of every stage goes to this process's stdout (the run log). The first
stage that fails ends the run with its exit code.
"""
import json
import os
import subprocess
import sys
import time


def main(argv) -> int:
    if len(argv) != 2:
        print('usage: fizgig_runner.py <plan.json>', flush=True)
        return 2
    with open(argv[1], encoding='utf-8') as fh:
        plan = json.load(fh)
    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in (plan.get('env') or {}).items()})
    stages = plan.get('stages') or []
    for i, stage in enumerate(stages, 1):
        name = stage.get('name') or f'stage {i}'
        print(f'[lds-fizgig] [{i}/{len(stages)}] {name} — {time.strftime("%Y-%m-%d %H:%M:%S")}',
              flush=True)
        rc = subprocess.call([str(a) for a in stage['args']], cwd=plan.get('cwd') or None,
                             env=env, stdout=sys.stdout, stderr=subprocess.STDOUT)
        if rc != 0:
            print(f'[lds-fizgig] {name} failed (exit {rc})', flush=True)
            return rc or 1
    print(f'[lds-fizgig] done — {time.strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
