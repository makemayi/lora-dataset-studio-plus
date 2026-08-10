"""Cancelling talks to a ComfyUI that is BUSY — that is the whole point of it.

Twice reported as "a dead task I cannot clear", and the log named the cause both
times:

    Could not cancel ComfyUI prompt 5242c1e0…: Read timed out. (read timeout=3)

A ComfyUI mid-generation does not answer HTTP in three seconds. The cancel never
landed, so the prompt stayed queued; the durable barrier stays with the prompt,
and every pending tile behind it became unclearable — by design, because a
barrier whose remote work might still exist must not be dropped.

So these pin the two halves of the timeout: patient on READ (the busy case) and
still impatient on CONNECT (a ComfyUI that is down must fail fast, not hang the
click for half a minute).
"""
import app.utils.comfyui as comfyui


def test_the_cancel_path_waits_long_enough_for_a_busy_comfyui():
    connect, read = comfyui._CANCEL_TIMEOUT
    assert read >= 20, 'a sampling ComfyUI routinely needs more than 3 s to answer'
    assert connect <= 5, 'a ComfyUI that is DOWN must still fail fast'


def test_every_cancel_call_uses_it(monkeypatch):
    """Not just one of the three: the queue read, the delete and the absence
    probe all talk to the same busy server, and one of them left short is enough
    to lose the cancel."""
    seen = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {'queue_pending': [], 'queue_running': []}

    def _record(url, **kw):
        seen.append(kw.get('timeout'))
        return _Resp()

    monkeypatch.setattr(comfyui.requests, 'get', _record)
    monkeypatch.setattr(comfyui.requests, 'post', _record)
    comfyui.comfyui_prompt_is_absent('p1')
    comfyui.cancel_comfyui_prompt_state('p1', 'c1')
    assert seen, 'no request was made'
    assert all(t == comfyui._CANCEL_TIMEOUT for t in seen), seen


def test_a_timeout_is_still_UNKNOWN_not_success(monkeypatch):
    """The patience is not permission: a cancel that timed out has proved
    nothing, and reporting it as done is what would strand real GPU work."""
    import requests

    def _boom(*a, **kw):
        raise requests.exceptions.ReadTimeout('too slow')

    monkeypatch.setattr(comfyui.requests, 'get', _boom)
    assert comfyui.comfyui_prompt_is_absent('p1') is None
    assert comfyui.cancel_comfyui_prompt_state('p1', 'c1') is comfyui.ComfyPromptState.UNKNOWN
