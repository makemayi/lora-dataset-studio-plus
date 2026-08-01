"""Opt-in bootstrap imported automatically by Python's ``site`` module.

The upstream ``run.py`` deliberately loads ``.env``, sets hub/telemetry process
variables, imports torch, and creates its Accelerator in that order.  Importing
the bridge runtime directly from sitecustomize would move torch to the very
front of the process and silently change CUDA/device initialisation.

This stdlib-only shim therefore waits for the *return* of the source-probed
``toolkit.accelerator.get_accelerator`` call.  Only then may the torch-using
runtime be imported and patch trainer classes.
"""

from __future__ import annotations

import os
import sys

try:
    from lds_aitk_bridge_contract import atomic_json_nofollow
except BaseException:
    atomic_json_nofollow = None

try:
    from lds_aitk_bridge_contract import (
        load_identity,
        verify_identity_model_artifacts,
    )
except BaseException:
    load_identity = None
    verify_identity_model_artifacts = None


_PREVIOUS_PROFILE = None
_BOOTSTRAP_FINISHED = False


def _failure_status(message: str) -> None:
    target = os.environ.get("LDS_AITK_BRIDGE_STATUS_FILE")
    if not target or atomic_json_nofollow is None:
        return
    try:
        atomic_json_nofollow(
            target,
            {
                "bridge": "lds-aitoolkit-state-bridge",
                "status": "bootstrap_error",
                "exact_supported": False,
                "reasons": [message],
            },
        )
    except (OSError, TypeError, ValueError):
        pass


def _waiting_status() -> None:
    target = os.environ.get("LDS_AITK_BRIDGE_STATUS_FILE")
    if not target or atomic_json_nofollow is None:
        return
    try:
        atomic_json_nofollow(
            target,
            {
                "bridge": "lds-aitoolkit-state-bridge",
                "status": "bootstrap_waiting",
                "exact_supported": None,
                "reasons": [
                    "waiting for upstream environment, torch, and Accelerator setup"
                ],
            },
        )
    except (OSError, TypeError, ValueError):
        pass


def _must_fail_closed() -> bool:
    return bool(os.environ.get("LDS_AITK_STATE_RESTORE_DIR")) or (
        os.environ.get("LDS_AITK_BRIDGE_STRICT") == "1"
    )


def _bootstrap_failure(exc: BaseException) -> None:
    message = f"{type(exc).__name__}: {exc}"
    _failure_status(message)
    print(f"[LDS bridge] bootstrap failed: {message}", file=sys.stderr)
    if _must_fail_closed():
        os._exit(78)


def _verify_child_model_artifacts() -> None:
    identity_path = os.environ.get("LDS_AITK_IDENTITY_FILE")
    if not identity_path:
        raise RuntimeError("exact-state child identity is unavailable")
    if load_identity is None or verify_identity_model_artifacts is None:
        raise RuntimeError("exact-state child artifact verifier is unavailable")
    identity = load_identity(identity_path)
    verify_identity_model_artifacts(identity)


def _after_accelerator_profile(frame, event, argument):
    global _BOOTSTRAP_FINISHED
    if _PREVIOUS_PROFILE is not None:
        _PREVIOUS_PROFILE(frame, event, argument)
    if _BOOTSTRAP_FINISHED or event != "return" or argument is None:
        return
    if (
        frame.f_globals.get("__name__") != "toolkit.accelerator"
        or frame.f_code.co_name != "get_accelerator"
    ):
        return
    _BOOTSTRAP_FINISHED = True
    # Stop profiling before importing the runtime; imports and torch internals
    # must not recursively re-enter this hook.
    sys.setprofile(_PREVIOUS_PROFILE)
    try:
        # This is the last stdlib-only boundary before importing the runtime and
        # ai-toolkit model classes. Re-open and fully hash every lexical model
        # target in the child so a cache/blob mutation after parent preflight
        # cannot silently resume against different weights.
        _verify_child_model_artifacts()
        import lds_aitk_bridge_runtime

        lds_aitk_bridge_runtime.install_from_environment()
    except BaseException as exc:
        _bootstrap_failure(exc)


if os.environ.get("LDS_AITK_BRIDGE_ENABLE") == "1":
    if "torch" in sys.modules:
        # A manually late import of sitecustomize cannot prove which environment
        # existed when torch initialised.  Never guess that exact state is safe.
        _bootstrap_failure(
            RuntimeError("torch was imported before the deferred bridge bootstrap")
        )
    else:
        _PREVIOUS_PROFILE = sys.getprofile()
        _waiting_status()
        sys.setprofile(_after_accelerator_profile)
