import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest

@pytest.fixture(autouse=True)
def _restore_secret_env():
    """set_secrets() writes os.environ directly; snapshot & restore the secret keys.

    Also CLEAR them at setup: config.py runs load_dotenv(ENV_PATH) at import, and at
    collection time (before LDS_ENV is pointed at a tmp file) ENV_PATH is the real
    repo .env — so a developer who saved a real Gemini/OpenAI key via the app would
    leak it into os.environ and make "unconfigured" probes see a key. Starting each
    test with the keys unset makes the suite independent of the local .env; tests
    that need a key set it themselves via monkeypatch.setenv."""
    import os
    from app.config import SECRET_KEYS as keys   # stays in sync as keys are added
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

@pytest.fixture(autouse=True)
def _reset_inmemory_registries():
    """dataset_activity is a process-global in-memory store (a batch dies with the
    process, not the request). With :memory: DBs each test restarts dataset ids at
    1, so a batch a PRIOR test began on 'dataset 1' would look live to the next
    test's fresh 'dataset 1' — enough to make the kind-switch guard 409 spuriously.
    Clear it around every test so in-memory activity never leaks across cases.

    The bank folder-sync cooldowns are process-global for the same reason: bank
    id 1 of a prior test would make the next test's first walk a no-op.

    The vision keep-warm LEASE is the same kind of global, and the nastiest: any
    test that crops/uploads through detect_head_bbox grants a 120 s lease, and
    every later test whose code path is "about to take the GPU" then calls
    revoke() -> a REAL HTTP POST to whatever answers on the Ollama URL. That is
    how test_training_queue_atomic failed once in a full suite and never alone:
    the launch under test paid for a live unload of a machine's actual Ollama,
    which can take tens of seconds. The suite must never depend on a lease left
    behind by an earlier test, nor talk to a live Ollama by accident.

    The 🔤 text-search query cache is a third one, and it bites the same way in
    reverse: it is keyed by PHRASE only (a CLIP vector depends on the checkpoint,
    not on the bank), so a query encoded by one test would be served from memory
    to the next — whose LDS_DATA_DIR is a different empty tmp dir. A test proving
    "the encoder is invoked exactly once" would then see zero calls and fail, and
    one proving "no ML python ⇒ 503" would silently get a cache hit and a 200.
    Both did, before this line existed."""
    from app.services import bank_jobs, bank_undo, clip_text_encoder
    from app.services import dataset_activity
    from app.services import image_bank_service, vision_keepalive
    dataset_activity.reset()
    bank_jobs.reset()
    bank_undo.reset()
    image_bank_service.reset_folder_sync()
    vision_keepalive.forget_lease()
    clip_text_encoder.forget_memory_cache()
    yield
    dataset_activity.reset()
    bank_jobs.reset()
    bank_undo.reset()
    image_bank_service.reset_folder_sync()
    vision_keepalive.forget_lease()
    clip_text_encoder.forget_memory_cache()
    clip_text_encoder.release()

@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv('LDS_DATA_DIR', str(tmp_path / 'data'))
    monkeypatch.setenv('LDS_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('LDS_ENV', str(tmp_path / '.env'))
    import app.config as _cfg
    monkeypatch.setattr(_cfg, 'ENV_PATH', tmp_path / '.env')   # never touch the real .env in tests
    # config.py caches load_config() in a module-level global keyed on nothing but
    # "has it been loaded before" -- it isn't tied to LDS_CONFIG. Without resetting it
    # here, a test that calls save_config() with a real comfyui.base_dir leaks that
    # value into every later test's "fresh" app (same process, stale cache), even
    # though each test gets its own tmp_path/env vars. Task 14 (Klein path) hit this:
    # a test asserting "ComfyUI unconfigured -> RuntimeError" silently inherited a
    # previous test's real base_dir and passed for the wrong reason.
    monkeypatch.setattr(_cfg, '_cache', None)
    from app import create_app
    application = create_app({'TESTING': True, 'WTF_CSRF_ENABLED': False,
                              'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:'})
    yield application

@pytest.fixture()
def client(app):
    return app.test_client()
