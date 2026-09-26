"""Vision-server routes — the model list of whichever local engine
`vision.backend` currently selects (ollama | llamacpp).

One endpoint so the frontend model pickers (caption options, H3 head analysis)
never hardcode a backend: pickers consume {engine, can_pull, reachable, models}
and hide the pull UI when the active engine has no pull (llama.cpp lists only
what is loaded; models are chosen at llama-server launch).
"""
from flask import Blueprint, jsonify

from .. import config as cfg
from ..services import ollama_control, vision_llamacpp

bp = Blueprint('vision', __name__, url_prefix='/api/vision')


@bp.get('/models')
def list_models():
    backend = (cfg.get('vision.backend') or 'ollama').strip().lower()
    if backend == 'llamacpp':
        out = vision_llamacpp.list_models()
        return jsonify({**out, 'engine': 'llamacpp', 'can_pull': False}), 200
    out = ollama_control.list_models()
    return jsonify({**out, 'engine': 'ollama', 'can_pull': True}), 200
