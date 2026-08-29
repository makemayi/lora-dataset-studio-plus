"""Run OneTrainer's `scripts/train.py` so that a stop request SAVES the LoRA.

WHY THIS FILE EXISTS
--------------------
OneTrainer already does the right thing on a cancel. Its headless entry point
ends like this::

    canceled = False
    try:
        trainer.train()
    except KeyboardInterrupt:
        canceled = True

    if not canceled or train_config.backup_before_save:
        trainer.end()

and `GenericTrainer.end()` writes the backup and then saves the model to
`output_model_destination`. With `backup_before_save` true — which both shipped
Krea 2 presets set — a CANCELLED run still saves everything it has trained.

The app never got any of that, because its Stop button ran `taskkill /F`: the
process is destroyed, `KeyboardInterrupt` is never raised, and hours of training
go with it. Three stops on 2026-08-29 threw away a run each.

The missing piece is Windows-specific. A console control event can be delivered
to a child spawned with CREATE_NEW_PROCESS_GROUP, but CTRL_BREAK arrives as
SIGBREAK, whose default action terminates the process outright — it does NOT
become a KeyboardInterrupt the way Ctrl-C does. So the child has to install a
handler that turns it into one, and `train.py` cannot: it is OneTrainer's file,
not ours. This shim is the smallest thing that can — it installs the handler and
then runs their script, unmodified, in the same process.

Stdlib only, on purpose: it executes inside OneTrainer's own venv.
"""
import os
import runpy
import signal
import sys


def _interrupt(_signum, _frame):
    """Hand the trainer the one exception its own cancel path is written for."""
    raise KeyboardInterrupt()


def main() -> None:
    # SIGBREAK is what CTRL_BREAK_EVENT delivers on Windows; SIGINT keeps a
    # plain Ctrl-C behaving identically when someone runs this by hand.
    for name in ('SIGBREAK', 'SIGINT'):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _interrupt)
            except (ValueError, OSError):        # not the main thread / unsupported
                pass

    # The launcher runs us with cwd = the OneTrainer install, the same way it
    # ran train.py directly. `scripts/` has to lead sys.path because train.py's
    # first line is `from util.import_util import script_imports`, which is
    # `scripts/util/import_util.py` — resolvable only when Python is started
    # from inside that directory, which runpy does not reproduce on its own.
    root = os.getcwd()
    scripts_dir = os.path.join(root, 'scripts')
    sys.path.insert(0, scripts_dir)
    sys.argv = ['train.py', *sys.argv[1:]]
    runpy.run_path(os.path.join(scripts_dir, 'train.py'), run_name='__main__')


if __name__ == '__main__':
    main()
