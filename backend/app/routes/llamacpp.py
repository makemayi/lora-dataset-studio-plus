"""llama.cpp (llama-server) API.

llama.cpp is BYO-server: it is started by the user (it needs its model file at
launch), so — unlike Ollama — there is deliberately NO start route here. What
the app needs is the loaded-model list for the pickers; reachability and the
Settings Test live in capabilities (`probe_llamacpp_connection`).
"""
from flask import Blueprint, jsonify

from ..services import vision_llamacpp

bp = Blueprint('llamacpp', __name__, url_prefix='/api/llamacpp')


@bp.get('/models')
def list_models():
    """Loaded models for the picker. Always 200 — {ok, reachable, models:[...]}.
    An unreachable server is a handled outcome (reachable:False, empty list),
    not a server fault. llama.cpp lists only what is LOADED; there is no pull."""
    return jsonify(vision_llamacpp.list_models()), 200
