"""Config core: layered config.json over DEFAULTS, secrets in .env."""
import copy, json, os, secrets as _secrets, threading, unicodedata
from pathlib import Path
from dotenv import load_dotenv

LOCAL_USER = 'local'

BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
REPO_ROOT = BACKEND_DIR.parent

def _data_dir() -> Path:
    return Path(os.environ.get('LDS_DATA_DIR', str(REPO_ROOT / 'data')))

def data_dir() -> Path:
    """Public accessor for the app's writable data directory (created on demand).
    Where app-managed artefacts live that aren't user datasets — e.g. the dedicated
    Python env the watermark-inpainting installer auto-provisions (data/envs/…)."""
    d = _data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d

def _config_path() -> Path:
    return Path(os.environ.get('LDS_CONFIG', str(REPO_ROOT / 'config.json')))

ENV_PATH = Path(os.environ.get('LDS_ENV', str(REPO_ROOT / '.env')))
load_dotenv(ENV_PATH)

# REDDIT_CLIENT_ID / CIVITAI_API_KEY / PEXELS_API_KEY: scraping credentials
# (Settings > Scraping & sources). Sources read their env var at request time,
# and set_secrets() stamps os.environ on save, so changes apply without restart.
SECRET_KEYS = ('GEMINI_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY', 'HF_TOKEN',
               'HF_CLOUD_TOKEN', 'VAST_API_KEY', 'REDDIT_CLIENT_ID', 'CIVITAI_API_KEY',
               'PEXELS_API_KEY', 'QWEN_API_KEY',
               # comfy.org key for ComfyUI's API nodes (the ChatGPT 'comfyui'
               # lane). It rides in each prompt's extra_data, so it is a
               # credential like the rest: .env, never config.json.
               'COMFY_ORG_API_KEY')

# A Krea install saved by a previous release can carry its old *defaults* in
# config.json, so a changed DEFAULTS value alone would never reach it. This
# marker lets us distinguish that one-time profile migration from settings the
# user changes after the new profile is available.
KREA_CALIBRATION_VERSION = 4
_LEGACY_KREA_GROUNDING_PX = 1024.0
_LEGACY_KREA_REF_BOOST = 4.0
_PREVIOUS_KREA_GROUNDING_PX = 512.0
_PREVIOUS_KREA_REF_BOOST = 1.0
_PREVIOUS_KREA_STEPS = 10
# v3's shipped ref_boost (0.25) turned out too weak for a character-LoRA-
# augmented setup; v4 raises the floor back to 4 and adds the per-framing
# split. grounding_px/steps are unchanged from v3, so only ref_boost is
# checked here — a v3 install that already customised it keeps its choice.
_V3_KREA_REF_BOOST = 0.25

DEFAULTS = {
    # host: '127.0.0.1' = this machine only ; '0.0.0.0' = reachable from the LAN
    # (phone, tablet, another PC) — the Settings "Server" card's LAN toggle just
    # flips this. Port defaults to 5050 to match start.bat's default bind (so the
    # Settings port field shows what's actually running, not a phantom mismatch).
    # require_token (default OFF): a home LAN is trusted, so LAN access is open by
    # default — no token to type on a phone. Turn it ON to demand a token from
    # remote devices (access_token is then generated + persisted here so it
    # survives restarts and is copyable from Settings). Loopback never needs it.
    'server': {'host': '127.0.0.1', 'port': 5050, 'require_token': False, 'access_token': ''},
    # Every path here means "'' = the default under DATA_DIR". Storing the
    # resolved path instead would freeze today's disk into config.json and make
    # a later DATA_DIR move silently wrong, so blank stays blank.
    #   cloud_runs_dir     working area of a cloud run (dataset copy, samples,
    #                      logs) — big, and safe to throw away once a run ended.
    #   checkpoints_dir    the DURABLE checkpoint store. Deliberately NOT under
    #                      cloud_runs_dir: the staging cleanup used to be the
    #                      only copy of a never-deployed .safetensors, and
    #                      emptying the trash after it destroyed weights.
    'paths': {'dataset_images_root': '',                       # '' -> DATA_DIR/datasets
              'cloud_runs_dir': '',                            # '' -> DATA_DIR/cloud_runs
              'checkpoints_dir': '',                           # '' -> DATA_DIR/checkpoints
              'video_datasets_dir': ''},                       # '' -> DATA_DIR/video_datasets
    'comfyui': {'api_url': 'http://127.0.0.1:8188', 'base_dir': '',
                'output_dir': '', 'input_dir': '', 'models_dir': '', 'loras_dir': '',
                # setup_skipped (default False): the user consciously chose "continue
                # without ComfyUI" in the Setup wizard. It ONLY makes the Setup step
                # render a neutral "skipped" instead of nagging; it never gates a
                # capability. Setting base_dir annuls it (see settings.put_settings and
                # the DERIVED comfyui.skipped in capabilities.probe), so it can never
                # mask a real error of a configured ComfyUI.
                'setup_skipped': False,
                # Seconds ComfyUI is allowed to spend ANSWERING the /object_info
                # enumeration (the heaviest probe in the app). It is a READ budget
                # only: the connection itself still has to be accepted in
                # `utils.comfyui._OBJECT_INFO_CONNECT_TIMEOUT` seconds, so a ComfyUI
                # that is genuinely OFF never costs this. This has to be a setting
                # rather than a constant because the /object_info payload grows with
                # the number of custom nodes and model files INSTALLED — the richer
                # the install, the longer it takes, which is exactly why the old
                # hardcoded 8 s broke the people who had invested the most in their
                # ComfyUI (reported by j_o_e_l. on Discord, who measured ~15 s on his
                # own install). Clamped to 5-300 by utils.comfyui.object_info_timeout().
                'object_info_timeout_s': 45},
    'ollama': {'url': 'http://127.0.0.1:11434', 'vision_model': 'huihui_ai/qwen3-vl-abliterated:8b-instruct',  # -instruct, NOT ':8b' (=thinking): see get_vision_model()
               # How many vision calls a bank pass keeps in flight. 4 is the
               # measured knee; see services/vision_pool.py for the numbers.
               'vision_concurrency': 4,
               # Seconds an ISOLATED vision call may keep the model resident when
               # nothing else wants the GPU (0 = always unload, the old
               # behaviour). See services/vision_keepalive.py.
               'vision_keep_warm_seconds': 120},
    'aitoolkit': {'dir': '', 'datasets_dir': '', 'output_dir': '', 'hf_home': '',
                  # Explicit interpreter for installs without venv/.venv
                  # (conda, uv, system python). Empty = auto-detect.
                  'python': ''},
    # OneTrainer — second local training backend (Krea 2 first slice, see
    # docs/superpowers/specs/2026-07-30-onetrainer-backend-design.md, a
    # local-only file — this repo's docs/superpowers is gitignored).
    # Deliberately as thin as aitoolkit's own block was before any install
    # wizard existed: this slice has no Setup-wizard integration yet.
    'onetrainer': {'dir': '',
                   # Explicit interpreter override (conda/uv/system python).
                   # Empty = derive venv/Scripts/python.exe (or venv/bin/python)
                   # under `dir`.
                   'python': '',
                   # The PEFT adapter OneTrainer trains: 'LORA' (default, what
                   # this app's own Krea 2 Edit inference graph loads via
                   # LoraLoaderModelOnly) or 'OFT_2' (Orthogonal Finetuning —
                   # a different adapter algorithm, user-confirmed to train
                   # and load correctly on 2026-07-31). Not validated against
                   # OneTrainer's own supported set here — an unrecognised
                   # value degrades to LORA at launch time rather than
                   # failing the whole run.
                   'peft_type': 'LORA'},
    # `enabled` is the ENGINE CATALOG as well as the default selection: adding an
    # engine here is what makes it reach existing installs (see _merge_new_engines
    # and LEGACY_KNOWN_ENGINES below). `known` is not a setting — it is the ledger
    # of which engines the app offered the last time the user picked, written by
    # save_config; [] means "no ledger yet".
    'engines': {'default': 'chatgpt',
                'enabled': ['nanobanana', 'chatgpt', 'openrouter', 'qwen', 'klein',
                            'krea', 'minimax_h3'],
                'known': [],
                # chatgpt_auth: 'auto' = subscription when connected, else API key.
                # 'comfyui' is the third lane: the SAME shot through ComfyUI's
                # OpenAI API node, billed in comfy.org credits instead of this
                # OpenAI account (services/chatgpt_comfy.py). Still an API
                # engine — the picture is made on a third-party server — so the
                # NSFW fail-closed rule is unchanged.
                'chatgpt_auth': 'auto',            # auto|api|subscription|comfyui
                # Quality asked of the ComfyUI lane's node: low|medium|high.
                # THE cost dial of that lane — it moves both the credits and the
                # wall clock, and nothing else there does (measured: three shots
                # at `high` took 1'55", 2'11", 2'12", all of it OpenAI's time).
                # `low` rather than the `high` this shipped with on 2026-08-09
                # (maintainer's call): high is the expensive end of a per-image
                # bill and a dataset is a hundred images, so the cheap end is the
                # honest default and the dial is one click away. Ignored by the
                # two direct lanes.
                'chatgpt_comfy_quality': 'low',
                # The Codex ROUTER model of the subscription lane — NOT the image
                # model. The image model of the API-key lane is
                # engines.chatgpt_image_model (below); the subscription lane
                # renders on whatever OpenAI's image_generation tool serves that
                # plan (gpt-image-2 today) and takes no model from us. Two
                # different settings — never merged.
                'chatgpt_subscription_model': 'gpt-5.4-mini',
                # OpenRouter reaches the SAME upstream models as the two other
                # API engines through one account, so the slug is free text: its
                # catalogue moves fast and a renamed/retired model must never
                # require a new release. Default = the Nano Banana weights.
                'openrouter_model': 'google/gemini-3-pro-image',
                # Nano Banana / ChatGPT / Qwen image models. Free text for the same
                # reason as OpenRouter's, but DEFAULTED TO BLANK, unlike it:
                # environment variables (NANOBANANA_MODEL, CHATGPT_IMAGE_MODEL, QWEN_MODEL)
                # already existed and some installs set them. A literal default
                # here would be a non-blank cfg.get() that silently outranked
                # that env var. Blank means "not chosen", so the documented
                # order holds: setting > environment variable > built-in default
                # (services/nanobanana.DEFAULT_MODEL, services/chatgpt_image.DEFAULT_IMAGE_MODEL,
                # services/qwen_image.DEFAULT_MODEL).
                'nanobanana_model': '',
                # Twin of chatgpt_base_url below, same reasoning and same
                # blank-means-Google default: non-blank points the Nano Banana
                # engine at a Gemini-compatible gateway, and sends your
                # reference PHOTOS there instead. A GEMINI_BASE_URL env var is
                # honoured underneath it.
                'nanobanana_base_url': '',
                'chatgpt_image_model': '',
                # Where the ChatGPT API-KEY lane sends its requests. Blank =
                # OpenAI itself, and that is the only value most people should
                # ever have. Non-blank points the lane at an OpenAI-compatible
                # gateway/reseller — which means your reference PHOTOS and
                # prompts go to that operator instead, under their retention
                # policy, with a key they issued. It buys nothing but price.
                # Same blank-default reasoning as the model slugs above: an
                # OPENAI_BASE_URL env var is honoured underneath it.
                # NEVER read by the subscription lane or the ComfyUI lane —
                # those two are not OpenAI-key traffic at all.
                'chatgpt_base_url': '',
                'qwen_model': '',
                # DashScope region configuration: 'sg' (Singapore, default), 'cn', or 'us'.
                'qwen_region': 'sg'},
    'captioning': {'backend': 'auto'},                         # auto|joycaption|ollama|none
    # 📥 What happens to a photo the moment it enters a dataset. Until now this
    # was two hardcoded numbers with no sentence anywhere saying they existed
    # (reported by Qeeyana on Reddit: "images added to dataset are automatically
    # normalized to 1024. Why? Let me choose not to.").
    #
    # max_side: longest side kept, in px, when a WebP normalization mode is
    #   explicitly selected. 0 = store at the original size. Whatever the value,
    #   the ceiling below applies: it is a format limit, not a preference.
    # encoding: 'preserve' (the shipped default) keeps supported static source
    #   bytes exactly as supplied: JPEG, PNG, WebP or BMP. 'standard' (WebP q92),
    #   'high' (WebP q100) and 'lossless' (lossless WebP) remain opt-in legacy
    #   normalization modes. `max_side` deliberately does not affect `preserve`:
    #   training creates its own disposable PNG staging copy at launch, so an
    #   import must not throw away the user's master first.
    #
    # Applies to the dataset INGEST lanes only (photo import, kohya ZIP/folder
    # merge, scrape-to-dataset). It does not touch generated images, the ≤2048
    # copies handed to a generation API, or an image the user already curated.
    'dataset_import': {'max_side': 1024, 'encoding': 'preserve'},
    # 🛡️ The shared image INPUT budget — how big a source file any lane is
    # allowed to decode. Not a dataset-import preference: dataset import, ZIP
    # and scrape ingest, Bank scan and thumbnails, edits, ComfyUI staging and
    # Ollama vision all read these two numbers, so an image that can be
    # imported can also be looked at.
    #
    # It is a MEMORY guard, so it is reasoned in decoded bytes: 3 B per RGB
    # pixel, 4 B per RGBA pixel, and an edit or analysis pass can hold a second
    # copy at once. The shipped 64 Mi-pixels is ~192 MiB for one RGB decode
    # (~256 MiB RGBA) and ~384-512 MiB with a working copy — room for every
    # current phone/35 mm master (61 MP = 57 Mi-pixels) and for panoramas,
    # which the previous hardcoded 16 Mi-pixels / 8192 px refused.
    #
    # 0 on either key = NO limit for that dimension. The app then also stops
    # capping Pillow's own decompression-bomb threshold, so a malformed or
    # hostile file can be decoded until it exhausts memory. That is a real
    # trade, offered rather than imposed; the Settings card says so.
    'image_input': {'max_side': 16384, 'max_pixels': 64 * 1024 * 1024},
    'training': {'default_family': 'zimage'},
    # Concept face masking (opt-in per dataset, Advanced training options). Both
    # knobs are exposed because NOBODY has measured the right value: no public A/B
    # of a concept LoRA trained with vs without face masking exists, so shipping a
    # frozen number would be a guess dressed as a default.
    #
    # `expand`: how far the detected FACE box is grown to become a HEAD box.
    # InsightFace returns eyes-to-chin; untouched it leaks jaw, hair and neck.
    # 2.0 is the only published default for this exact chain (ai-toolkit-perceptual's
    # face_suppression_expand, documented "1.8-2.0 = full head coverage").
    #
    # `min_weight`: the loss weight left INSIDE the mask (ai-toolkit maps black ->
    # mask_min_value). NOT zero, on purpose, and the floor below is not cosmetic:
    #   - a zero-weight region is not "ignored", it is unpenalised — the model may
    #     put anything there at no cost (OneTrainer discussion #347: phantom limbs,
    #     edge artefacts), and the only published sweep of this knob (SECourses, 9
    #     runs) reports "anatomically disproportional" output below 0.1;
    #   - ai-toolkit divides the mask by its own mean (SDTrainer), so an image
    #     masked edge to edge at exactly 0.0 divides by zero -> NaN loss -> dead run.
    'face_mask': {'expand': 2.0, 'min_weight': 0.1},
    # Cloud GPU training (vast.ai). Everything has a sane default: the only
    # required user input is the VAST_API_KEY secret. Values here are knobs
    # for power users / for adjusting after the real-world smoke test.
    'cloud': {
        # Official vast.ai "Ostris AI Toolkit" template (smoke-validated
        # 2026-07-12): publishes the UI behind the pod's Caddy proxy on 18675
        # and generates the per-instance auth token. Clearing this falls back
        # to a raw-image launch using `image`/`onstart` below.
        'template_hash': '471ed5903d8cdb8e63b0d0e50f6cd519',
        'ui_port': 18675,              # container port the UI is reachable on (Caddy proxy)
        # Raw-image fallback only — BUT the tag also names the ai-toolkit commit
        # the dense (full-transformer) recipe's supported/refused verdicts were
        # read against. Bumping it means a different trainer: re-read the lever
        # comments in services/lora_training.py first (a test enforces the pin).
        'image': 'vastai/ostris-ai-toolkit:4625406-2026-07-12-cuda-12.9',
        'max_price_per_hour': 0.80,    # background safety cap on offer price, $/h
        'offer_scan_limit': 100,       # offers fetched when listing GPU speed tiers
        'pod_overhead_minutes': 35,    # boot+model download+quantize (measured ~40 min live), in cost estimates
        'max_concurrent_runs': 1,      # simultaneous cloud pods; raise in Settings
        'min_inet_down_mbps': 400,     # skip hosts too slow to pull the 7 GB image
        'min_disk_bw_mbps': 500,       # skip hosts too slow to EXTRACT it (frozen 'loading')
        'min_reliability': 0.98,       # vast reliability floor (0.95 let a dead host through)
        # Offer trust filters. verified_only=True preserves the historical
        # behaviour; Secure Cloud is Vast's `datacenter` tier and is opt-in
        # because it usually narrows the marketplace and raises the price.
        'verified_only': True,
        'secure_cloud_only': False,
        'host_blacklist_days': 3,      # skip hosts whose pod never became ready
        # A host killed while it was still VISIBLY booting (see boot_budget below)
        # was slow, not broken: it is skipped for hours, not days, so a bad night
        # on one uplink does not silently shrink the marketplace for three days.
        'slow_boot_blacklist_hours': 6,
        # Boot is guarded by the same two clocks as the pre-step-1 phase below.
        # ready_timeout is IDLE time — rearmed every time the pod shows a boot
        # fact it had never shown before (a new vast status, the UI port getting
        # published, a moving host progress line), so a pod honestly pulling a
        # 26 GB image is never cut; boot_budget is the ABSOLUTE ceiling on the
        # phase, so a host too slow to ever finish still dies fast (0 = none).
        'ready_timeout_minutes': 25,   # no boot progress at all past this -> kill
        'boot_budget_minutes': 90,     # hard ceiling on the whole boot phase
        'max_runtime_minutes': 480,    # safety net (stall watchdog is the first line): hard stop past this
        'stall_timeout_minutes': 30,   # no step progress past this -> rescue + kill
        # Before step 1 the pod is fetching the base model. Two clocks guard it:
        # first_step_timeout is IDLE time — rearmed every time the pod's log
        # reports more downloaded bytes, so an honestly slow download is never
        # cut; download_budget is the ABSOLUTE ceiling on that phase, so a host
        # too slow to ever finish still dies well before the runtime cap
        # (0 = no ceiling, runtime cap is then the only backstop).
        'first_step_timeout_minutes': 45,  # no step 1 AND no new bytes past this -> kill
        'first_step_download_budget_minutes': 180,  # hard ceiling on the pre-step-1 phase
        # Out-of-monitor freeze watchdog: a training run whose own monitor stopped
        # reporting for this long is terminated by the supervisor (0 = only warn
        # in the UI, never cut). Slow-by-design phases (boot/download) are
        # never judged on this value -- they get a fixed 2 h floor.
        'freeze_watchdog_minutes': 45,
        # ... and the dataset upload's own version of it. Not a budget for the
        # transfer (a 24 GB dataset may legitimately take hours) but the time
        # allowed with NO byte at all reaching the pod. 0 = never cut.
        'upload_stall_minutes': 25,
        'unreachable_grace_minutes': 6,  # tolerated mid-run network blackout before giving up on the pod
        'monthly_budget_usd': 0,       # 0 = unlimited; launches blocked past this
        'disk_gb': 60,                 # instance disk (base model + dataset + checkpoints)
        # min_vram_gb est PAR FAMILLE (pas par variante) : pour flux2klein on prend
        # 32 — le 9B (32-48 GB) est la voie cloud principale de cette famille, et un
        # pod 32 GB entraîne aussi le 4B sans problème (l'inverse serait faux).
        'min_vram_gb': {'zimage': 24, 'sdxl': 16, 'krea': 24, 'flux2klein': 32},
        # Dedicated dense Krea 2 lane.  A full-transformer checkpoint is ~26 GB
        # and training keeps the official base, working weights/caches and the
        # save side by side; it must never inherit the 24 GB / 60 GB LoRA lane.
        # Price and runtime deliberately keep using the regular cloud knobs so
        # operators can tune those two policy limits in one place.
        'full_transformer': {
            'min_vram_gb': 80,
            'disk_gb': 200,
            # HF is eventually consistent and ai-toolkit may finish just before
            # the uploaded files become visible through the Hub listing API.
            # Verification is bounded: exhaustion keeps the paid pod for manual
            # recovery instead of declaring success or destroying the only copy.
            'verification_attempts': 3,
            'verification_retry_seconds': 5,
            # Private Hugging Face storage the delivery needs, checked BEFORE a
            # pod is rented (run #146 died at step 2750/3000 on a 403 "private
            # repository storage limit reached"). 0 = infer the allowance from
            # the plan documented by Hugging Face (100 GB free / 1 TB PRO) —
            # an ESTIMATE: the Hub publishes no quota endpoint, and the observed
            # refusal came well below the documented free figure. Set the real
            # number here to make the pre-check exact. Never a hard lock: the
            # refusal is confirmable ("Train anyway").
            'private_storage_limit_gb': 0,
            # Headroom on top of checkpoint × saves kept (model card, licence,
            # a push that lands exactly on the ceiling still fails).
            'storage_margin_gb': 20,
            # 0 = size one dense checkpoint from what past runs really delivered
            # (their persisted Hub integrity proof), else ~26 GB.
            'checkpoint_size_gb': 0,
            # Where a full model is delivered: 'both' (default) downloads it to
            # this computer FIRST, proves it, and only then backs the master up
            # to the private Hugging Face repository; 'local' skips the backup
            # (and with it the ability to continue that run later); 'hub' is the
            # historical Hugging-Face-only delivery. The order is the point: a
            # full private quota can no longer end a training, because nothing
            # is pushed while it trains.
            'delivery': 'both',
            # Free space to leave on the checkpoint volume on top of the
            # delivery itself, checked before the pod is rented.
            'local_disk_margin_gb': 15,
            # Ceilings for the two pod-side Hugging Face transfers (backing the
            # master up, and pulling it back when continuing a run).
            'hub_push_budget_seconds': 3600,
            'hub_fetch_budget_seconds': 3600,
        },
        'onstart': '',                 # raw-image fallback: optional startup command
    },
    'face_scoring': {'python': '', 'models_root': '', 'green': 0.50, 'orange': 0.45},
    # 🗃️ Image bank triage thresholds. Raw scores are persisted per image;
    # these thresholds only drive the FLAGS computed at read time — so tuning
    # them re-sorts an already-scanned bank instantly, no rescan needed.
    # sharpness_min: Laplacian variance below this = flagged blurry (the classic
    #   ~100 rule of thumb). noise_max: residual std above this = flagged noisy.
    # uniformity_min: grayscale std below this = flagged flat/uniform (solid
    #   colors, empty screenshots). dup_distance: dHash Hamming distance (same
    #   64-bit hash as dataset imports) at or under which two images group as
    #   near-duplicates. min_side: smaller side under this = flagged small
    #   (mirrors the dataset import guard: trainers only downscale).
    # face_threshold: cosine similarity at or above which two faces are the
    #   same person when clustering the bank by subject.
    # aesthetic_min: LAION aesthetic score (~1..10) below which an image is flagged
    #   'low_aesthetic' — the "keep the nice ones" cut of a mixed dump.
    # nsfw_max: NSFW probability (0..1) above which is_nsfw is flagged, to split a
    #   mixed SFW/NSFW dump.
    # style_threshold: cosine similarity on the CLIP image embeddings at or above
    #   which two images share a visual STYLE when clustering by style.
    # semantic_dup_threshold: cosine similarity on the SAME CLIP embeddings at or
    #   above which two scored images are flagged a SEMANTIC near-duplicate (stage 2:
    #   crops / re-compressed variants of the same shot a dHash misses). Higher than
    #   style_threshold on purpose — a crop is far closer than merely "same style".
    'bank': {'sharpness_min': 100.0, 'noise_max': 15.0, 'uniformity_min': 12.0,
             'dup_distance': 8, 'min_side': 768, 'face_threshold': 0.45,
             'aesthetic_min': 5.0, 'nsfw_max': 0.5, 'style_threshold': 0.6,
             'semantic_dup_threshold': 0.96,
             # detail_min: effective resolution (0..1 of the stored size) below
             #   which an image is flagged 'soft_detail' — its pixels promise more
             #   picture than they deliver. 0.72 was picked on a real 36 000-image
             #   bank: it selects the softest ~3%, and sits below the 10th
             #   percentile of images measured to be genuinely full-resolution, so
             #   a sharp photo does not trip it. Raise it to be pickier.
             'detail_min': 0.72,
             # bars_max: fraction of the frame allowed to be flat black letterbox
             #   before the 'bars' flag. 0.04 ~ a thin band; it caught ~4% of the
             #   reference bank (screenshots of videos, padded stills).
             'bars_max': 0.04},
    'masks': {'python': ''},
    # Bank ✨ Score pass interpreter (CLIP aesthetic/NSFW stack). Auto-provisioned
    # by the bank_scoring installer into its own venv — declared here so a
    # full-config Save round-trips it instead of failing "unknown config section".
    # text_search_idle_minutes: how long the 🔤 text-search encoder stays warm
    #   after its last query. Loading CLIP costs ~8 s; encoding a phrase costs
    #   ~20 ms — so the worker is kept alive to make a refine-and-retry session
    #   instant, and reaped afterwards because it holds ~2.4 GB of RAM. 0 means
    #   "never stay warm": every distinct query pays the ~8 s load, which is the
    #   right trade on a memory-tight machine.
    'bank_scoring': {'python': '', 'text_search_idle_minutes': 10},
    # 🗣 Which checkpoint writes the video captions. A SETTING rather than a
    # constant because the choice is not a preference: a model that describes
    # what it sees in evasive terms produces captions that are about something
    # slightly other than the footage, and a LoRA trained on those learns to look
    # away too — with nothing in the output to reveal it. Empty = the shipped
    # default, so an install that sets nothing captions exactly as before.
    # Any checkpoint of the same architecture works; see settings-reference.
    # style: which PROMPT writes the captions — 'standard' (default, the shipped
    #   wording) or 'plain', which grants explicit permission to name what is on
    #   screen. Measured to matter MORE than the checkpoint: asked the standard
    #   way, even an uncensored model describes around explicit footage, and the
    #   base model asked plainly outperformed it. A caption that talks around its
    #   subject teaches the trained model to look away. Empty = 'standard'.
    'video_caption': {'model': '', 'style': ''},
    # Optional second semantic space for Image Bank. Its interpreter is recorded
    # separately so ✨ Score may borrow a user's CUDA Python without making the
    # SigLIP2 installer mutate that environment. Existing configs without this
    # key retain the historical bank_scoring.python fallback at runtime. The
    # aesthetic MLP is CLIP-specific, while SigLIP2 powers only semantic
    # search, selection, coverage and near-duplicate grouping. Weights are
    # installed explicitly in Setup and inference is local-files-only, so
    # selecting it can never trigger a surprise 1.5 GB download.
    #
    # The cosine distribution is not CLIP's. Keep its duplicate calibration in
    # this separate section so SigLIP2 cannot retune historical CLIP Banks.
    'bank_semantic': {
        'python': '', 'models_root': '', 'device': 'auto',
        'siglip2_semantic_dup_threshold': 0.97,
    },
    # fp8 quantization runs `fp8_export.py` in a SUBPROCESS, because it needs
    # torch + safetensors and this app deliberately installs without them
    # (gigabytes). Empty -> the same interpreter ✨ Score uses, then ai-toolkit's,
    # then the app's own. Never imported in-process: doing so shipped a feature
    # that could not run at all on a real install.
    'quantize': {'python': ''},
    # Watermark inpainting (simple-lama-inpainting, extra ML). Dedicated key so a
    # user can override it, but defaults empty -> reuse the same ML interpreter as
    # rembg/insightface (masks.python) then sys.executable. Never imported in-process.
    # allow_crop (default True = the shipped behaviour): when False the auto-routing
    # NEVER crops a border mark — it repaints it instead (LaMa/Klein per the chosen
    # engine). A persisted user preference (Settings ▸ Watermark inpainting AND the
    # batch Clean bar both edit it); the review lightbox can still override it per image.
    'watermark': {'python': '', 'device': 'auto', 'allow_crop': True},  # auto|cuda|cpu
    # 🚩 Dedicated watermark DETECTOR (optional extra: a SigLIP2 classifier that
    # ranks + a Grounding DINO pass that locates). When installed, the Find pass
    # uses it instead of asking the vision model image by image; when not, nothing
    # changes and the vision model still does the work.
    # python: its interpreter. Empty = reuse the bank-scoring environment (which
    #   already has torch + transformers) and then the app's own — so a normal
    #   install simply probes ✗ and keeps the vision-model path.
    # models_root: where the ~0.9 GB of weights live. Empty = data/models/watermark_detect.
    # threshold: the classifier score at or above which an image is FLAGGED.
    #   0.94 is MEASURED, not a guess, and it is nowhere near the 0.5 a
    #   probability normally implies: this model's scores are compressed hard
    #   against 1, so on a 110-image hand-labelled sample of a real 29 759-image
    #   bank, 0.5 flagged 52 of the 55 CLEAN images while 0.94 flagged none of
    #   them and still caught 54 of the 55 marked ones. Raise it toward 0.96 to
    #   miss more rather than crop anything by mistake; lower it toward 0.92 to
    #   catch the faintest marks and hand-check a few clean images.
    # device: auto|cuda|cpu, same meaning as the inpainting device.
    # locate: run the second (localisation) model on flagged images. Off = images
    #   are flagged with NO box, which the crop/inpaint levels cannot route on —
    #   only worth it to save time on a bank you intend to filter, not clean.
    # backend: WHICH route 🧽 Find watermarks takes, on BOTH surfaces (bank and
    #   dataset). 'auto' = the detector when its extra is installed, the vision
    #   model otherwise — which is exactly what the bank has always done, so
    #   'auto' changes nothing anywhere. 'detector' / 'vision' pin one route; a
    #   pinned 'detector' with no extra installed does NOT fail, it runs the
    #   vision route and SAYS so (see watermark_detector.resolve_backend).
    'watermark_detect': {'python': '', 'models_root': '', 'threshold': 0.94,
                         'device': 'auto', 'locate': True, 'backend': 'auto'},
    # 🎬 Shot-boundary detection for the video bank (TransNetV2). Declared here so
    # a full-config Save round-trips these keys instead of failing "unknown config
    # section" — the same reason bank_scoring is declared above.
    # python: its interpreter. Empty = reuse the bank-scoring environment, which
    #   already carries torch; a second copy would cost the user ~2.5 GB for
    #   nothing. Then the app's own, which simply probes unavailable.
    # threshold: the detector's cut probability at or above which a frame is a
    #   boundary. 0.5 is the reference implementation's own default and is NOT a
    #   measured value for this app's material — lower it to cut more finely on
    #   soft transitions, raise it if dissolves are being split into fragments.
    # min_shot_frames: shots shorter than this are DROPPED, not merged into a
    #   neighbour — merging would silently move that neighbour's boundary, and a
    #   boundary is the one thing the whole lane is built to get right. 5 rejects
    #   a stray flash cut while leaving real rapid montages intact. Also not a
    #   measured constant; no labelled sample of "too short" exists yet.
    # device: auto|cuda|cpu. The network runs on 48x27 frames, so it is never the
    #   bottleneck — decoding is. CPU is a perfectly reasonable choice here, and
    #   it leaves the GPU free for captioning and training.
    'shot_detect': {'python': '', 'threshold': 0.5, 'min_shot_frames': 5,
                    'device': 'auto'},
    # 🎬 Video bank quality cuts (wave 2). ALL None by default — a cut that has
    # not been chosen filters NOTHING. That is a decision, not an omission: the
    # published thresholds measurably do not transfer between corpora (the public
    # motion floor lands at the 7th percentile of this machine's own test bank),
    # so shipping one as a default would silently gut some users' banks. The
    # dry-run endpoint exists precisely so a user picks cuts against their OWN
    # distribution. Raw scores persist; flags are recomputed at read time, so
    # changing any of these re-sorts every bank instantly, no rescan.
    # Quality cuts of the 🎬 video bank — all None, because published thresholds
    # measurably do not transfer between corpora. See video_metrics.THRESHOLD_KEYS
    # for the canonical list; anything missing here still reads as None.
    # watermark_max is the ONE cut here that ships with a number, and the reason
    # is that it is not a corpus statistic. Motion and sharpness are properties
    # of someone's footage (which is why the published defaults land at the 7th
    # percentile of this machine's bank); a watermark score is a CLASSIFIER's
    # probability, calibrated with the model itself — so the image lane's
    # measurement transfers where a motion floor does not. 0.94 is that
    # measurement (see watermark_detect.threshold above: 110 hand-labelled images
    # of a 29 759-image bank; 0.94 flagged none of the 55 clean ones and still
    # caught 54 of the 55 marked ones). Set it to null to flag nothing.
    #
    # duplicate_threshold is a COMPUTE-time setting, not a read-time cut, which is
    # why it is not in video_metrics.THRESHOLD_KEYS: changing it means re-running
    # the ✂ Duplicates pass (instant — it re-reads the vectors 🔎 Search cached,
    # no GPU). 0.96 is inherited from the image lane's semantic near-duplicate cut
    # over the SAME CLIP space (bank.semantic_dup_threshold); no video-pair
    # calibration exists yet, and video_clip_dedup says so out loud.
    'video_bank': {'min_duration_s': None,
                   'motion_floor': None, 'motion_ceiling': None,
                   'luma_floor': None, 'freeze_max': None,
                   'sharpness_floor': None,
                   'watermark_max': 0.94,
                   'duplicate_threshold': 0.96},
    # consistency_strength: the dx8152 LoRA anchors STRUCTURE (composition/
    # background), not the face — its own guide says start at 0.5 and that
    # 0.8-1.0 "can prevent edits from applying". 0.9 made every variation a
    # near-copy of the reference. 0 disables the LoRA entirely.
    'klein': {'consistency_lora': 'klein/Flux2-Klein-9B-consistency-V2.safetensors',
              # Optional user-pinned model files for the three required Klein
              # slots. Each accepts a ComfyUI-relative loader name (e.g.
              # 'klein/flux-2-klein-9b-fp8.safetensors' under models/unet or
              # models/diffusion_models; a bare name for a file at a root) OR an
              # ABSOLUTE path — a path under any registered model root (including
              # extra_model_paths.yaml roots) is converted to the relative name a
              # loader node needs. klein.consistency_lora and the
              # generation_lora_presets rows take a path the same way.
              # Empty = auto-detect (canonical download name, then the narrow
              # token scan) — the historical behaviour, byte for byte.
              #
              # The scan is deliberately narrow (wrong model >> missing model),
              # which means it DECLINES anything it cannot name: a UNET outside a
              # 'klein'-named folder, an encoder whose filename carries no known
              # token. Those files are on disk and get reported as MISSING, and
              # no amount of re-downloading fixes that. A pin removes the
              # resolver's discretion — the named file resolves, so the integrity
              # verdict (klein_invalid_assets) finally gets to say "present but
              # unreadable" when that is the truth.
              #
              # A pinned file that cannot be resolved falls back to auto-detection
              # with a visible badge in Settings — it never blocks generation. A
              # file genuinely outside every ComfyUI root cannot be loaded by
              # ComfyUI at all: register its folder in extra_model_paths.yaml (the
              # app parses it identically) — the badge says exactly that.
              'unet': '', 'text_encoder': '', 'vae': '',
              'consistency_strength': 0.5,
              # Optional generation-LoRA PRESETS (Idea by @waltm — Discord
              # feature request): named combinations the user picks per run.
              # Each preset: {name, loras: [{file, strength}]} — loras is an
              # ORDERED list (list order = chain order after the consistency
              # LoRA on the local Klein edit graph), file is a loras-relative
              # name (like consistency_lora; the app never hardcodes one).
              # There is deliberately NO automatic per-LoRA gating: the chosen
              # preset carries the intent (make an "NSFW full" preset if you
              # want one). Caps: 8 LoRAs/preset, 12 presets
              # (klein_edit_helper.MAX_GENERATION_LORAS / _PRESETS). The older
              # generation_loras flat list and the very old ultra_real_lora /
              # nsfw_lora keys are migrated in by _migrate_klein_loras() and
              # then dropped.
              'generation_lora_presets': [],
              # LoRAs chained onto the FACE SWAP graph, after its own swap LoRA
              # and the optional style LoRA the shipped graph names. A FLAT
              # [{file, strength}] list, not the named presets above: face swap
              # is one fixed action with no per-run picker, so a preset would be
              # a name nobody ever selects. Cap 8
              # (face_swap_helper.MAX_FACE_SWAP_LORAS); a row whose file is not
              # on disk is skipped with a log line rather than failing the whole
              # batch on a ComfyUI validation 400.
              'face_swap_loras': [],
              # Which of the presets above the run panel STARTS on. Empty = none,
              # which is byte-for-byte the behaviour every install had before this
              # key existed (the picker opened on "None" on every single visit —
              # so a carefully configured preset applied only if you remembered to
              # re-pick it, and the PNG metadata of a run that forgot showed no
              # LoRA at all). It is a STARTING POINT, not a lock: the picker still
              # offers None and every other preset for that run, and choosing
              # differently there never rewrites this setting.
              # Fail-closed like the rest of the preset chain: a name matching no
              # configured preset falls back to "none", never to a blocked run.
              # Per ENGINE on purpose — klein.generation_lora_presets and
              # krea.generation_lora_presets are independent lists and the same
              # name can mean two different chains.
              'default_generation_lora_preset': '',
              # Optional instruction for small scraped-image rescue only.
              # Empty is intentional: never invent a restoration prompt for the user.
              'small_image_prompt': '',
              # Sampler steps for Klein GENERATION (variations, regenerate, small-image
              # rescue). 5 = the value hardcoded in the shipped workflow (node 77), so an
              # untouched install renders exactly as before. More steps = slower, usually
              # a cleaner render; clamped to 50 (face_dataset_service._IMPROVE_MAX_STEPS).
              # Raised on request by ashish.sinha (Discord). Separate from improve_steps,
              # which drives the manual "Upscale & improve" pass only.
              'generation_steps': 5,
              # Enhancement LoRA strength for every Klein EDIT/generation lane
              # (reference edit, variations, regenerate, small-image rescue) —
              # node 139 of improve skin.json, klein/realistic.safetensors.
              # The workflow pins that node at 0.8 and NO lane except "Upscale &
              # improve" ever overrode it, which did not matter while the file
              # shipped with nobody: the node was bypassed on every install. Once
              # Setup started downloading it (klein_enhancement_lora), a detail/
              # style LoRA at 0.8 quietly joined every edit and pulled results
              # away from the instruction — the "edits are not conformant" report.
              # 0.0 = the behaviour every install had before the LoRA existed
              # locally. Raise it to let the LoRA add detail on purpose.
              # Mirror of improve_base_lora_strength, which already defaults to 0.
              'edit_base_lora_strength': 0.0,
              # Manual "Upscale & improve" quality profile. Its INSTRUCTION was
              # already editable (identity_prompts.klein_improve) but the knobs
              # deciding how much the pass actually changes were hardcoded at the
              # call site — including BOTH LoRA strengths pinned to 0, which meant
              # the workflow's own realistic LoRA (0.8 in improve skin.json) never
              # applied at all. These defaults reproduce that exact historical
              # behaviour, so an untouched install renders byte-identically; raise
              # improve_base_lora_strength to actually let that LoRA work.
              'improve_steps': 4,
              'improve_base_lora_strength': 0.0,
              # Overrides klein.consistency_strength for THIS pass only. It is the
              # dx8152 consistency LoRA (anchors composition/background), NOT an
              # identity LoRA — clamped [0, 1.5] by enqueue_klein_edit.
              # 1.0 where generation defaults to 0.5: dx8152 warns that 0.8-1.0 "can
              # prevent edits from applying", which is a problem for a restaging and
              # exactly the point here — an improve pass must add detail WITHOUT
              # redrawing the composition. Tuned on real runs, not from the guide.
              'improve_consistency_strength': 1.0,
              # Total pixel budget the source is rescaled to before sampling, so it
              # is the output resolution. 2 = the value hardcoded in the workflow.
              'improve_megapixels': 2.0},
    # Krea 2 Identity Edit — the second LOCAL generation engine (services/
    # krea_edit_helper.py). Every value here is a RESOLUTION HINT or a sampler
    # knob, never a hardcoded machine path: blank/absent means "find it yourself"
    # (canonical filename first, then a narrow token match, across every
    # extra_model_paths root), which is what makes the engine work on installs
    # that look nothing like the developer's.
    'krea': {
        # Internal marker for the calibrated dataset-restaging defaults below.
        # It is intentionally not a user-facing setting.
        'calibration_version': KREA_CALIBRATION_VERSION,
        # Blank = auto-resolve a Krea 2 base under any 'krea'-named model folder,
        # preferring a Turbo then a Raw build. Set it to a filename to pin one.
        'base_model': '',
        # The edit LoRA the whole engine hangs on. Not found under this name ->
        # the resolver scans the loras roots for a krea2_identity_edit* file, so a
        # renamed download still works.
        'identity_lora': 'krea/krea2_identity_edit_v1_2.safetensors',
        # Optional always-on generation LoRAs, as NAMED presets (mirrors
        # klein.generation_lora_presets — the mechanism is @waltm's idea). Pure
        # user data: [{name, loras: [{file, strength}]}], empty by default, and
        # the ONLY source of truth for which files may chain and in what order.
        # Krea never had the legacy single-slot LoRA keys Klein carries, so there
        # is no migration and no save carve-out — _deep_merge preserves a list
        # the incoming partial doesn't mention.
        'generation_lora_presets': [],
        # Krea's own starting preset for the run panel — the twin of
        # klein.default_generation_lora_preset, and deliberately a SEPARATE key:
        # the two preset lists are independent, so one name can name two
        # different chains. Empty = none = the historical behaviour.
        'default_generation_lora_preset': '',
        # THE consistency <-> prompt-adherence dial, in pixels: the resolution the
        # reference is shown to the vision text-encoder at. LOW = follows the
        # PROMPT (more variety, weaker likeness); HIGH = RESEMBLES the reference
        # (stronger likeness, but it starts copying the pose and the outfit you
        # asked it to change). 512 is the measured default for dataset poses: it
        # keeps identity while giving the prompt room to move the pose.
        'grounding_px': 512,
        # Per-framing override (community request): the SAME grounding_px pixel
        # budget is not equally "strong" on every framing. On a face close-up
        # the reference IS almost entirely face, so the whole budget goes into
        # holding it -- which also means pose/background copy more. On a full
        # body shot the face is a small fraction of the same reference, so the
        # same pixel budget encodes it in far less detail -- identity is the
        # one that needs MORE help there, not less. Blank/0 for a framing =
        # fall back to grounding_px above (today's behaviour, unchanged for
        # anyone who never touches this). Keys mirror identity_prompts'
        # framing_face/framing_bust/framing_body/framing_back naming.
        'grounding_px_by_framing': {'face': '', 'bust': '', 'body': '', 'back': ''},
        # Pack reference workflow values, measured working. cfg is pinned at 1.0
        # in code (guidance-distilled model) and is deliberately NOT a setting.
        'steps': 8,
        'identity_lora_strength': 1.0,
        # How hard the source latent is pushed back into the model each step —
        # the SAME consistency dial as grounding_px, applied at a different
        # stage of the pipeline. Higher = stronger identity retention (but
        # less room for the prompt to change pose/outfit/scene); lower =
        # more freedom, weaker likeness.
        'ref_boost': 4,
        # Per-framing override (same rationale as grounding_px_by_framing): a
        # face close-up already has the reference's identity signal at full
        # strength, but a bust/body/back shot dilutes it across more of the
        # frame, so a HIGHER boost is what it takes to hold the same identity
        # there. Non-blank shipped defaults (unlike grounding_px_by_framing) —
        # measured against a real dataset's own character-LoRA-augmented setup.
        'ref_boost_by_framing': {'face': 4, 'bust': 6, 'body': 8, 'back': 8},
        # OPTIONAL extra LoRAs: the user's own character/person LoRAs (e.g.
        # trained on ai-toolkit), chained AFTER the identity-edit LoRA in LIST
        # ORDER for extra likeness on top of Krea's baseline consistency. Up to
        # 5 slots (mirrors Klein's generation-LoRA chain cap); a blank `file`
        # row is simply skipped when the graph is built, so 5 empty rows is a
        # complete no-op — the graph stays byte-identical to before this
        # existed. Unlike identity_lora there is no auto-resolution: each names
        # ONE specific file the user trained, so a typo should surface as
        # ComfyUI's own "file not found" rather than silently falling back to a
        # guess. Pure user data (like klein.generation_lora_presets) — empty
        # default, nothing to merge; see _migrate_krea_character_lora for the
        # one-time upgrade from the earlier single-slot krea.character_lora.
        'character_loras': [],
        # OPTIONAL alternative sampler path: KreaTwoStageSampler +
        # KreaDualResolutionSelector (a low-res stage1 handed off to an
        # upscaled stage2, plus a stage2-only accelerator LoRA) instead of the
        # single KSampler above. Off by default — both custom node classes
        # must be present in the target ComfyUI, checked the same way as the
        # base pack.
        'two_stage': False,
        # The 5 knobs below only matter when two_stage (above) is on, except
        # max_output_mp -- that one caps the SINGLE-stage path's output size
        # too (fit_output_size's shared default), so it lives here rather
        # than under a two_stage_-only umbrella. Defaults match the
        # calibrated export this path used to hardcode -- unchanged
        # behaviour for anyone who never touches these.
        'two_stage_stage1_steps': 52,
        'two_stage_stage2_steps': 12,
        # Where stage 1 hands off to stage 2, as a percentage of the total
        # denoising schedule (0 = stage2 only, 100 = stage1 only).
        'two_stage_handoff_percent': 16.67,
        'two_stage_base_megapixels': 1.0,
        # Final output size, both paths. 2.0 is the calibrated default;
        # raise for more detail (slower), lower to speed either path up.
        'max_output_mp': 2.0,
    },
    # MiniMax H3 — the THIRD local generation engine (services/minimax_h3_helper.py).
    # An identity engine like Krea, reached through a VIDEO model: it samples a
    # short frame packet from one reference photo and keeps the best single frame.
    # Blank/absent file keys mean "find it yourself", same contract as `krea`.
    'minimax_h3': {
        # Blank = auto-resolve the Ref2VA model under any diffusion-model root
        # (any quantisation). The fl2va sibling is excluded by name: it loads and
        # then does a different job. Set a filename to pin one build.
        'base_model': '',
        # Frames sampled per shot. Two legal shapes, not one range: 1 (a single
        # frame) or the packet grid 5, 22, 39 ... We keep ONE frame either way,
        # so a packet of 5 pays full sampling cost for four frames nobody reads;
        # more frames only give the frame selector more candidates, which is the
        # only reason to raise it.
        #
        # 1 REQUIRES A PATCHED ComfyUI. Stock `comfy_extras/nodes_minimax_h3.py`
        # declares `length` as min=5, and an off-grid value is a validation error
        # at queue time — i.e. a whole batch of dead tiles, not one bad image.
        # A ComfyUI update reverts the patch silently. If tiles start failing to
        # queue after an update, re-apply it or set this back to 5.
        'length': 1,
        'steps': 25,
        # THE identity <-> speed dial, H3's answer to krea.grounding_px:
        # 'match' scales the reference to the generation's pixel area, 'max' uses
        # the 2048px reference pipeline for the best likeness and is, per the
        # node's own tooltip, several times slower — reference tokens ride
        # through every sampling step.
        'ref_image_size': 'match',
        # The reference is downscaled to this longer edge BEFORE it reaches the
        # 32B vision encoder. Every reference pixel is paid for on each prompt
        # change; 1024 was the measured working point.
        'ref_longer_edge': 1024,
        # How much "looks like the reference photo" counts when picking which
        # frame of the packet to keep, against sharpness and exposure. Non-zero
        # because a dataset wants likeness — but UNVERIFIED: on a 5-frame packet
        # raising it changed nothing, since the sharpest frame was also the most
        # reference-like. It earns its keep at higher `length`.
        'frame_weight_reference': 1.0,
        # Two DEGRADATION switches, not features. On means "use these when the
        # ComfyUI actually has them": the speed nodes (Spectrum forecasting +
        # Sage attention) do not change the image, and the RTX upscaler is
        # NVIDIA-only, so a card without it loses the 2x rather than the engine.
        'use_speed_nodes': True,
        'use_rtx_upscale': True,
        # Measured at 1 MP. Larger canvases are untested here and cost five
        # times more than they look, because the model samples a packet.
        'max_output_mp': 1.0,
        # The instruction the H3 HEAD-SWAP graph sends — BOTH graphs read this
        # key, but they carry different instructions and different pictures, so
        # one wording rarely suits both. On 'minimax_h3_old' it replaces the
        # English prompt on the H3 node; on 'minimax_h3' it replaces the text
        # node the graph concatenates from (the H3 node's own prompt is a link
        # there, so writing to it would be discarded), and the shipped text is
        # the maintainer's Chinese head-transplant instruction, not the wording
        # described below.
        # Blank = the prompt the shipped graph carries: subject-neutral, and
        # explicit about the four things that fail silently (keep <Picture 1>'s
        # identity, repaint nothing outside the white mask, match <Picture 2>'s
        # head angle and lighting, no seam at hairline and neck). Set this to
        # A/B a wording without editing the workflow file, which an app update
        # replaces. NOTE <Picture 2> is the masked inpaint CROP, not the tile,
        # and the white area is the HEAD (face + hair).
        'swap_prompt': '',
        # Extra LoRAs chained onto H3's model for the SWAP graph
        # (`minimax_h3` — the new engine). [{file, strength}], empty by default.
        #
        # It exists because the accelerator LoRAs that make an H3 swap bearable
        # cannot be shipped: they are community distills and re-quantisations
        # that differ per install, so the graph carries none and this is the
        # only place one can come from. `file` is a path relative to a ComfyUI
        # loras root, exactly as the picker writes it.
        #
        # Chained onto whatever feeds the guider, i.e. AFTER the optional speed
        # patches — switching `use_speed_nodes` off must not move where these
        # land. A row whose file is not on disk is SKIPPED with a log line
        # rather than failing the job: losing every tile of a batch to one
        # stale filename in a settings list is the worse trade.
        'swap_loras': [],
        # Sampler steps for the SWAP graph. 0 = whatever the graph carries (25).
        #
        # Paired with `swap_loras` on purpose, in the config and in the UI: the
        # reason to add a LoRA up there is almost always a step-distill, and a
        # 4-step distill run for 25 steps is BOTH slower than the stock model
        # and visibly worse. Someone who adds the LoRA and not the steps has
        # bought nothing and lost quality, which is the sort of thing nobody
        # attributes to the right setting.
        #
        # Clamped to 1..100 at write time. It applies to `minimax_h3` (the new
        # swap engine) only — the generation lane's own `steps` is a separate
        # key because the two graphs are sampled differently.
        'swap_steps': 0,
    },
    # Which engine the 🎭↔ face/head swap runs on. Its own namespace, exactly
    # like `improve` above and for the same reason: the action is no longer
    # Klein-only.
    #   'klein'           repaints the head with a swap LoRA (fast, one graph,
    #                     needs Klein2-9B-SmartCharacterSwap on disk).
    #   'minimax_h3'      the CURRENT H3 graph (2026-08-14 redesign): a Klein
    #                     pass erases the head and renders a depth map of where
    #                     it was, then H3 re-renders the whole frame around the
    #                     new head. No crop and no stitch — the body, clothing
    #                     and background are re-rendered too, which is why the
    #                     older graph is kept rather than retired.
    #   'minimax_h3_old'  the ORIGINAL H3 graph: mask the head, crop around it,
    #                     send only that crop through H3, composite it back, so
    #                     every pixel outside the mask survives untouched.
    # 'klein' is the default because it is what every swap did before this
    # setting existed. The redesign kept the id 'minimax_h3' so that anyone who
    # had already picked H3 gets it without touching a setting.
    #
    # WHICH KEYS BELOW APPLY TO WHICH ENGINE:
    #   both H3 graphs  h3_mask_source, h3_mask_prompt, h3_mask_opacity,
    #                   h3_pose_hint
    #   'minimax_h3'    h3_new_stages
    #   'minimax_h3_old' h3_stages, h3_context_factor, h3_blend_pixels,
    #                   h3_lama_model — all four are crop/stitch or stage
    #                   parameters that the new graph has no node for.
    'face_swap': {
        'engine': 'klein',
        # The two nodes the NEW H3 graph ships bypassed, each a step the graph
        # carries WIRED and this switch removes when off (the helper only ever
        # subtracts — the fallback wiring is what ComfyUI's own bypass does to
        # that node, read off the maintainer's export):
        #   mask_overlay  paints the head region a flat blue over the
        #                 head-removed image before H3 sees it. Off, H3 gets the
        #                 Klein output as it is — the head gone, a depth map of
        #                 where it was — which is the point of that pass.
        #                 `h3_mask_opacity` applies to this node when it is on.
        #   ollama        an Ollama vision call describes how the head sits in
        #                 THIS photo (angle, occlusion, lighting) and appends it
        #                 to the instruction. Runs on `h3_new_ollama_model`, or
        #                 on `ollama.vision_model` when that is blank — never on
        #                 the tag the graph carries. It costs a second model
        #                 on the same GPU as 40 GB of H3, and it makes
        #                 `h3_pose_hint` redundant — with this on the hint is not
        #                 sent, because the two describe the same thing and only
        #                 one of them is looking at the actual picture.
        'h3_new_stages': {'mask_overlay': False, 'ollama': False},
        # THE KLEIN PASS'S OWN INSTRUCTION — `minimax_h3` (new) only.
        #
        # This is the step that decides whether the swap comes back in
        # proportion. Klein does not merely delete the head: it replaces it with
        # a featureless grey MANNEQUIN head, and that stand-in is the only thing
        # left in the frame telling H3 how big the head was, where it sat and
        # which way it faced. Delete instead of replace and H3 is guessing —
        # which is where a doll-sized head comes from.
        #
        # The shipped text was rewritten on 2026-08-14. The maintainer's
        # original led with three deletion verbs (移除 / 抹除 / 消除) and hung
        # "转换为脸部深度图" off the end, phrased as a transform of the head it
        # had just said to erase; an edit model follows the dominant, repeated
        # intent and stops after erasing. Three rules came out of that:
        #   * the FIRST verb is 替换 (replace), never 移除;
        #   * the geometry is pinned explicitly (size, position, orientation,
        #     tilt, perspective) — it is what the rest of the graph reads;
        #   * removal is written as a CONSTRAINT ("no identity features"), not
        #     as the instruction.
        # A mannequin rather than a depth map on purpose: a mannequin head is an
        # object the model has seen, a depth render is a stylised output edit
        # models routinely botch — and the mannequin carries shading, so it says
        # MORE about the volume than a flat depth pass would.
        #
        # Kept in sync with the shipped graph's own node text by
        # test_minimax_h3_swap_new_workflow_shape, so the workflow still opens
        # sensibly in ComfyUI. The value here is the one that runs.
        # 2026-08-16: the stand-in now models the SKULL, not the silhouette.
        # Removing the hair and shrinking the head by a fifth are one change with
        # one purpose — hair thickness is not part of the head, and a stand-in
        # sized to the hairline makes H3 grow a skull to fill it and then put
        # hair on TOP of that, which is where the doll head came from.
        #
        # So "identical to the original" had to stop covering SIZE: the stand-in
        # is deliberately smaller, and saying both in one instruction asks for
        # two different heads. Size is still pinned — to the skull — and the swap
        # instruction (node 990) states the other half of the same contract:
        # hair grows outside the stand-in, so the finished outline is larger.
        # Change one, read the other.
        'h3_head_removal_prompt':
            '移除人物的头发，然后把人物的头部替换为一个无五官的灰白色素模头'
            '（假人头模）。\n'
            '素模头表示的是去掉头发厚度之后的颅骨，因此比原图中带头发的头部'
            '小约五分之一；不要把头发的体积算进素模头。\n'
            '头部的位置、朝向、倾斜角度与透视关系必须与原图完全一致，'
            '只保留头部的立体形状。\n'
            '表面为均匀的哑光浅灰色，有自然的明暗起伏体现立体感，'
            '没有五官、没有头发、没有任何配饰、没有任何身份特征。\n'
            '颈部、肩膀、身体、服装、背景与画面光照完全保持不变。',
        # The Klein pass's negative. Shipped EMPTY by the maintainer; the three
        # entries that matter are the failure modes of the instruction above —
        # a hole, a headless body, a background showing through — plus the
        # identity features that must not survive the replacement.
        'h3_head_removal_negative':
            '原本的脸, 五官, 头发, 眼睛, 鼻子, 嘴, 皮肤纹理, '
            '空洞, 无头, 头部消失, 背景穿透',
        # Which pulled Ollama model the stage above runs on. Blank = whatever
        # `ollama.vision_model` is set to, which is the captioning model and the
        # only behaviour that existed before this key.
        #
        # It is a SEPARATE key rather than a reuse of `ollama.vision_model`
        # because the two jobs pull in opposite directions: captioning runs on
        # every image in a dataset, so it wants the small 8B; this stage runs
        # once per swap next to 40 GB of H3 already on the card, and the answer
        # it writes goes straight into the instruction — so on a big card a
        # heavier model is worth it here and ruinous there. Tying them together
        # would make one of the two choices wrong.
        'h3_new_ollama_model': '',
        # Three optional stages of the OLD H3 swap graph ('minimax_h3_old'),
        # each a step the graph
        # ships WIRED and this switch removes when off (minimax_h3_swap_helper
        # only ever subtracts — the fallback wiring is read off the maintainer's
        # own bypassed export, never reconstructed). All three cost a second
        # model family in a job that already loads 40 GB of H3, which is why
        # they default to off:
        #   hair_removal  a Klein 9B edit pass ("remove the hair, change
        #                 nothing else") on the target before the head is
        #                 masked. With it ON, a missing Klein asset blocks the
        #                 swap — the stage is a full second engine.
        #   lama          LaMa inpainting wipes the masked head region before
        #                 H3 sees it, so the model is not reading the old head
        #                 through a translucent mask. See `h3_lama_model`: the
        #                 graph's own choice (zits) fails on this lane.
        #   face_detail   a Z-Image Turbo detailer over the eyes and mouth of
        #                 the chosen frame. The ONLY stage whose model
        #                 filenames are not re-resolved: it names a Z-Image
        #                 checkpoint, a lumina2 encoder and a private LoRA
        #                 stack, and this app has no Z-Image resolver — so on a
        #                 ComfyUI without those exact files it answers a
        #                 validation error naming the file.
        'h3_stages': {'hair_removal': False, 'lama': False, 'face_detail': False},
        # OLD ENGINE ONLY ('minimax_h3_old' — the new graph has no crop node).
        # How much of the shot around the head the H3 swap actually looks at —
        # InpaintCropImproved's `context_from_mask_extend_factor` on the swap
        # graph. The node grows the crop from the MASK box, then clamps it to
        # the image:
        #     grow_per_side = head_box * (factor - 1) / 2, clamped to the frame
        # so ONE number adapts by itself. At 3.0 a full-body shot crops to head
        # plus chest, while a bust or a portrait — where the head already fills
        # much of the frame — hits the edges and is not cropped at all.
        #
        # The graph shipped at 1.3, i.e. head-only everywhere. That is the most
        # pixels per head, and it is also why the model had nothing to size the
        # head against: the prompt asks it to match the shoulders, and at 1.3
        # there are no shoulders in the picture.
        #
        # The cost is real and worth knowing: H3 renders one ~1 MP canvas
        # whatever the crop, so doubling the crop's side quarters the pixels the
        # head gets. Raising `minimax_h3.max_output_mp` buys them back at the
        # price of sampling time. A bigger crop does NOT risk repainting the
        # body — InpaintStitchImproved composites only the masked region back.
        'h3_context_factor': 3.0,
        # How solidly the head is painted out before H3 is asked to redraw it —
        # AILab_MaskOverlay's `mask_opacity`. On the old graph that node is
        # always in the job; on the new one it is the `mask_overlay` stage, so
        # this dial does nothing there until that stage is switched on.
        #
        # At 1.0 (the shipped graph's value) the head becomes a flat white slab
        # with no structure at all, and a generative model asked to fill a flat
        # slab sometimes paints the slab back: that is the "white face" result.
        #
        # LOWERING IT IS NOT THE FIX, MEASURED 2026-08-11. At 0.75 the swap
        # returned the ORIGINAL face: the ghost showing through is the very face
        # being replaced, so the model reconstructs it instead of the reference.
        # There is no useful middle — the structure a partial mask leaks IS the
        # identity. Leave this at 1.0 and reach for `h3_lama_model` (fills the
        # hole with plausible NON-face content) or a less copyable placeholder
        # colour instead. Kept configurable because it is the dial the failure
        # was diagnosed with, not because 0.75 is a setting worth using.
        #
        # It is also what makes the `lama` stage mean anything: LaMa wipes the
        # masked region, and at opacity 1.0 the overlay paints straight over its
        # work, so that stage cannot change a single pixel until this is lowered.
        'h3_mask_opacity': 1.0,
        # WHERE the head mask comes from (BOTH H3 engines):
        #   'graph' — the segmenter inside the workflow (PersonMaskUltra on the
        #             old graph, which needs ComfyUI_LayerStyle; ClothesSegment
        #             from ComfyUI-RMBG on the new one). Either way the app
        #             never sees the mask it is about to repaint through;
        #   'app'   — services/auto_mask (SAM 3 in the app's own interpreter),
        #             which makes the mask visible, cacheable and testable here,
        #             and drops the LayerStyle dependency with it.
        # 'graph' stays the default: the app lane needs the automask environment
        # and sam3.pt, and a swap must not start failing on an install that has
        # neither.
        'h3_mask_source': 'graph',
        # What the app lane masks, comma-separated. Open-vocabulary, so this is
        # where the masked REGION is decided, and the list is a list for one
        # reason: a head is not one object to a segmenter. Asked for as 'head'
        # alone it returns the head with a HOLE where the glasses are, and the
        # swap then paints a new face around the old pair. Everything worn on the
        # head belongs here; the masks are unioned, and a phrase that matches
        # nothing in a photo costs one cheap grounding pass (the image is encoded
        # once for all of them) and adds nothing to the mask.
        'h3_mask_prompt': 'head, glasses, sunglasses, hat, headband, earrings',
        'h3_lama_model': 'lama',
        # OLD ENGINE ONLY ('minimax_h3_old' — the new graph has no stitch node,
        # because it re-renders the whole frame instead of compositing a crop).
        # How wide the band is over which the swapped head is blended back into
        # the untouched photo — InpaintStitchImproved's `mask_blend_pixels`, 0-64.
        #
        # This is the MECHANICAL half of "it does not blend". The prompt can ask
        # the model to match the picture's light, colour, grain and focus, and it
        # helps, but the composite itself is a feather this wide: a hard-ish edge
        # at 32 px is visible on a large head, and no wording changes that. The
        # trade is the usual one — a wider feather hides the join and also lets
        # more of the OLD head's edge pixels survive at the boundary, so past
        # ~48 px a hairline can start to ghost.
        'h3_blend_pixels': 40,
        # Tell the swap HOW THE HEAD SITS in the tile it is repainting, taken
        # from the catalog prompt that generated that tile ("three-quarter
        # view", "a calm neutral facial expression" — see face_swap_pose).
        #
        # Without it the model has to infer orientation and expression from a
        # cropped photo, and the expression is the one it most often invents: a
        # laughing body under a calm face reads as a paste-up as loudly as a bad
        # seam does. The sentence is APPENDED to the instruction, never replaces
        # it, and a tile whose row says nothing usable (an imported photo) keeps
        # the generic "match the shoulders" wording rather than being given a
        # guess. False turns it off.
        'h3_pose_hint': True,
        # Which reference pipeline the H3 SWAP runs its H3 node on (BOTH H3
        # engines). Deliberately SEPARATE from `minimax_h3.ref_image_size`, which
        # is the GENERATION lane's dial: the swap used to inherit that key, and
        # the inheritance was a silent defect — the generation default is 'match',
        # so every swap overwrote the new graph's shipped 'max' (the node's own
        # "best likeness" pipeline, 2048px reference, several times slower) and
        # identity fidelity was quietly the LOWER one. A swap is the one job
        # where likeness IS the product.
        #
        # Blank = leave whatever the shipped graph carries ('max' on the new
        # graph, 'match' on the old), the same contract as `swap_steps`' 0 — an
        # untouched install keeps the maintainer's tuning for its engine. Set it
        # to 'match' or 'max' to take the decision away from the graph. Anything
        # else is ignored the same way.
        'h3_ref_image_size': '',
        # Which inpainting model the `lama` stage runs. The maintainer's graph
        # asked for 'zits', and on this lane that CRASHES: ZITS pads to a
        # multiple of 32 and drives a 256->512 structure upsampler, while the
        # inpaint crop here is an arbitrary size (output_resize_to_target_size is
        # off), so it dies inside TorchScript at `upsample_bilinear2d`. Plain
        # 'lama' pads to 8, has no such upsampler, and is iopaint's own default —
        # hence the default here. The node also offers ldm / mat / fcf / manga /
        # spread; they differ in look, not in this constraint.
    },
    # Automatic masking (services/auto_mask.py) — SAM 3, open-vocabulary: the
    # region is named in words ("head", "hands", "watermark", "clothing"), so one
    # mechanism serves the face swap, image repair and whatever asks next.
    'auto_mask': {
        # The app-managed interpreter Setup builds (data/envs/automask). Blank =
        # not installed yet; the service then names the Setup action instead of
        # failing obscurely. NEVER the Flask venv: SAM 3 wants torch >= 2.7 and
        # CUDA >= 12.6 and documents no CPU path, which is not something to force
        # into the app's own Python.
        'python': '',
        # Meta's own `sam3.pt`, BORROWED from whatever ComfyUI install has it —
        # any segmentation pack fetches it from facebook/sam3. Blank = resolve it
        # (models/sam3, then the other model roots, extra_model_paths.yaml
        # included). An absolute path anywhere is honoured too: this checkpoint
        # is not loaded by ComfyUI, so it need not live under one.
        # NOT the Comfy-Org `sam3.1_multiplex_fp16.safetensors`: that build is
        # remapped for ComfyUI's own loader and will not load here.
        'checkpoint': '',
        # Blank = CUDA when torch sees a card, else CPU. Pin 'cpu' to keep the
        # GPU free for a generation that is running; expect it to be slow.
        'device': '',
        # Detection confidence. Lower finds more (and more of the wrong thing);
        # the service refuses an answer covering the whole frame either way.
        'threshold': 0.5,
    },
    # The ✨ Upscale & improve pass — which engine runs it. Its own namespace
    # rather than a key under `klein`, because the whole point of the setting is
    # that the pass is no longer Klein-only: 'klein' rewrites detail, 'seedvr2'
    # restores it without reinterpreting. 'klein' is the default because it is
    # what every improve did before this setting existed.
    'improve': {'engine': 'klein'},
    # SeedVR2 — the FIDELITY upscaler (services/seedvr2_helper.py, issue #32 by
    # SurpassHR). Not a generation engine: it restores detail and leaves the
    # content alone, which is the opposite trade from Klein's ✨ improve. Same
    # discipline as every other engine block: blank means "find it yourself",
    # never a machine path.
    'seedvr2': {
        # Blank = auto-resolve: the canonical 3B FP8 build when present, else the
        # first build in the SEEDVR2 folder. Set it to a filename to pin one (a
        # 7B build you dropped in yourself resolves exactly the same way).
        'model': '',
        # Same contract for the VAE: blank = the canonical ema_vae_fp16, else the
        # first file in the folder whose name says VAE. Set it to a filename when
        # yours is named something the heuristic cannot recognise — a pin is
        # honoured against the whole folder, which is the only reason it exists.
        'vae': '',
        # Target for the SHORT edge in pixels; the long edge follows the source
        # aspect. 1080 is the node's own default and a sane dataset target — LoRA
        # training buckets rarely exceed it, so going higher mostly costs VRAM.
        'resolution': 1080,
        # Hard cap on the LONG edge, 0 = none. The VRAM safety valve on a wide
        # panorama, where a 1080 short edge can mean 4000+ px across.
        'max_resolution': 0,
        # How the result is graded back onto the source's colours. 'lab' is the
        # node's default and the most conservative; 'wavelet' preserves broad
        # tone better on heavily degraded sources. Colour fidelity is the whole
        # reason this engine exists, so this is deliberately exposed.
        'color_correction': 'lab',
        # How the high-resolution (tiled) lane is chosen, when the TTP node pack
        # is installed. 'auto' (default) tiles when tiling helps — past the size
        # the model is comfortable at, or when the frame would not fit. Tiling
        # preserves high-frequency detail, not just VRAM (SurpassHR's
        # side-by-side, GitHub #32); the old VRAM-only rule meant the bigger
        # your card the less often you got the better picture, and it is gone.
        # 'always' tiles whenever there is more than one tile to make; 'never'
        # stays full-frame. Without the pack this has no effect.
        'tiling': 'auto',
        # Side of one tile, in pixels — THE VRAM lever of this engine. 1024 is
        # the contributed value and a good one on a big card; on 8 GB, 768 or
        # 512 is the difference between a 4K upscale and an out-of-memory, at
        # the cost of more seams. It also sizes the VAE's tiled encode/decode,
        # so it helps on the full-frame lane too, tiling pack or not.
        'tile_px': 1024,
        # Output short edge past which 'auto' tiles. 0 (default) = derive it from
        # the tile size (1.5x = the shipped 1536 at a 1024 tile) so the crossover
        # follows the tile. A positive value places it by hand.
        'tile_threshold': 0,
        # Transformer blocks offloaded to system RAM during inference. 0 = none
        # (fastest). Raise it to fit a bigger build on a smaller card; it trades
        # speed for VRAM headroom, it does not change the result.
        'blocks_to_swap': 0,
    },
    # Z-Image pipeline — the two loader refs the shipped Test Studio workflow used
    # to hardcode from the developer's own ComfyUI (reported by bobba84, GitHub #18).
    # BLANK = "find it yourself": services/zimage_model_resolver scans every
    # registered vae / text_encoders root, sub-folders included, case- and
    # separator-insensitively (z_ae, z ae, z-ae, ae.safetensors; qwen_3_4b in any
    # sub-folder). Set either to a filename to PIN it — a pinned value is used as-is
    # and is never second-guessed, which is also the escape hatch when a shared
    # ComfyUI carries several plausible files (a FLUX.1 `ae.safetensors`, say).
    'zimage': {'vae': '', 'text_encoder': ''},
    # Editable identity / quality prompts (feature request by @bbsorry / 雨田壹).
    # The identity "locks" that ride ahead of every generated variation used to be
    # hardcoded and invisible; these overrides expose them without touching the
    # reproducibility invariant. EACH string default is '' on purpose: blank means
    # "use the shipped default", so the no-override path stays byte-identical to
    # the historical hardcoded prompt (get_identity_prompt falls back to the
    # constant). A non-blank value wins. Keys:
    #   face_single  — API-engine identity guard, single reference (IDENTITY_GUARD)
    #   face_multi   — API-engine identity guard, multi reference (IDENTITY_GUARD_MULTI)
    #   klein_identity — Klein restage + face-identity block (wrap_variation_klein)
    #   klein_improve  — the fixed "Klein upscale & improve" instruction
    # klein_improve_enabled (default True): when False the manual "Klein upscale &
    # improve" applies NO prompt at all (pure upscale), instead of the default/override.
    # The four flat keys above are the HUMAN overrides and keep their historical
    # names/meaning (never renamed — they are in user config files since the feature
    # shipped). `by_subject` holds the non-human ones,
    # {animal|creature|object|other: {face_single|face_multi|klein_identity: text}},
    # each read with NO fallback to the flat key: an override written on an Animal
    # dataset must never ride on a human generation (reported by ashish.sinha).
    # Empty by default — a subject with no entry follows its shipped default.
    # The five OTHER prompt parts, hardcoded until this wave and shipped in every
    # local-edit prompt: markings_lock (the skin hold order), outfit_vary /
    # expression_neutral (the two directives baked into every human shot),
    # outfit_palette (the concrete garments, one per LINE), render_tail_sfw /
    # render_tail_nsfw (the photographic tail) and framing_face|bust|body|back
    # (the per-framing detail block). Same contract as the four above — '' means
    # the shipped default — and the same split: the tail and the framing blocks
    # are per subject (anime's tail asks for an illustration, not a photograph) so
    # they live under `by_subject` for non-human types; the rest are flat.
    'identity_prompts': {'face_single': '', 'face_multi': '', 'klein_identity': '',
                         'klein_improve': '', 'klein_improve_enabled': True,
                         'markings_lock': '', 'outfit_vary': '', 'expression_neutral': '',
                         'outfit_palette': '', 'render_tail_sfw': '', 'render_tail_nsfw': '',
                         'framing_face': '', 'framing_bust': '', 'framing_body': '',
                         'framing_back': '',
                         'by_subject': {}},
    # User shot catalogs imported from JSON, {subject_type: [{id,label,prompt,
    # framing,nsfw?}]} — idea by ashish.sinha (Discord): have an LLM write 40 shots
    # instead of typing them. Stored SERVER-side rather than in localStorage so a
    # catalog survives a browser wipe, shows up on the phone as well as the desktop
    # and rides along in the full backup. Written by the workspace's Import button
    # (validated client-side by shotImport.js) and re-checked on read by
    # face_variations.sanitize_custom_shots — this file is hand-editable, and a
    # label shadowing a built-in one would hijack prompt/aspect/NSFW resolution.
    'custom_shots': {},
    # User additions to the quick-generate component library (angle/
    # expression/pose/outfit/background), same storage spirit as
    # custom_shots above: optional, empty for the common case, hand-editable,
    # rides along in the full backup. Sanitized on read by
    # face_variations.sanitize_quick_gen_custom_components — never shadows a
    # shipped id.
    # User additions to the quick-generate NSFW pool, same storage spirit as
    # custom_components above but flatter ({subject_type: {framing: [entry]}}
    # — no axis nesting) since each NSFW entry is a whole self-contained
    # state+pose+décor prompt, never decomposed into axes. Sanitized on read
    # by face_variations.sanitize_quick_gen_custom_nsfw — never shadows a
    # shipped id, human-only.
    'quick_generate': {'custom_components': {}, 'custom_nsfw': {}},
    'updates': {'repo': 'makemayi/lora-dataset-studio-plus'},  # GitHub repo for the release feed
}

_lock = threading.Lock()
_cache = None


def defaults() -> dict:
    """A deep COPY of the shipped defaults, for callers that must show the user
    what a setting would be if they never touched it.

    Exposed over the API (`config_defaults` in the settings payload) so the
    Settings UI can offer a per-field "Reset to default" without ever holding a
    second copy of these numbers. A literal typed into the frontend would go
    stale the next time a default moves here, and the reset button would then
    quietly restore a value that is no longer the default — a lie the user
    cannot see. Derived, never duplicated: this is the SAME dict the merge in
    load_config() uses.

    A copy, not the live object: a caller mutating the returned tree (jsonify
    does not, but a future one might) must not rewrite the app's defaults."""
    return copy.deepcopy(DEFAULTS)

def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _number_is(value, expected):
    """Strict-enough numeric comparison for hand-editable JSON settings."""
    try:
        return float(value) == expected
    except (TypeError, ValueError):
        return False


def _migrate_krea_pose_profile(conf: dict, stored: dict, incoming: dict | None = None) -> dict:
    """Apply calibrated Krea pose defaults only to untouched old profiles.

    ``config.json`` overrides ``DEFAULTS``. The first Krea profile saved
    1024 / 4.0; calibration v2 then saved
    512 / 1.0 / 10.  Both are reference-dominated for diverse dataset poses. Only
    those exact shipped profiles are upgraded. Any other value is an intentional
    choice. An explicit save of a calibration knob is also intentional and gets
    the marker rather than being rewritten.
    """
    stored_krea = (stored or {}).get('krea')
    krea = conf.get('krea')
    if not isinstance(stored_krea, dict) or not isinstance(krea, dict):
        return conf
    try:
        stored_version = int(stored_krea.get('calibration_version', 0) or 0)
    except (TypeError, ValueError):
        stored_version = 0
    if stored_version >= KREA_CALIBRATION_VERSION:
        return conf

    incoming_krea = (incoming or {}).get('krea')
    explicit_calibration = isinstance(incoming_krea, dict) and any(
        key in incoming_krea for key in ('grounding_px', 'ref_boost', 'steps'))
    if explicit_calibration:
        krea['calibration_version'] = KREA_CALIBRATION_VERSION
        return conf

    is_v1_default = (
        stored_version < 2
        and _number_is(stored_krea.get('grounding_px'), _LEGACY_KREA_GROUNDING_PX)
        and _number_is(stored_krea.get('ref_boost'), _LEGACY_KREA_REF_BOOST)
        # Older config files did not record steps, so absence means their
        # shipped 10-step profile; a present non-10 value is a user choice.
        and _number_is(stored_krea.get('steps', _PREVIOUS_KREA_STEPS),
                       _PREVIOUS_KREA_STEPS))
    is_v2_default = (
        stored_version == 2
        and _number_is(stored_krea.get('grounding_px'), _PREVIOUS_KREA_GROUNDING_PX)
        and _number_is(stored_krea.get('ref_boost'), _PREVIOUS_KREA_REF_BOOST)
        and _number_is(stored_krea.get('steps', _PREVIOUS_KREA_STEPS),
                       _PREVIOUS_KREA_STEPS))
    # v3 only ever shipped ref_boost=0.25 (grounding_px/steps unchanged from
    # v2's already-migrated values) -- so only that one field marks an
    # untouched v3 profile, unlike v1/v2's full-triplet match above.
    is_v3_default = (
        stored_version == 3
        and _number_is(stored_krea.get('ref_boost'), _V3_KREA_REF_BOOST))
    if is_v1_default or is_v2_default:
        krea['grounding_px'] = DEFAULTS['krea']['grounding_px']
        krea['ref_boost'] = DEFAULTS['krea']['ref_boost']
        krea['steps'] = DEFAULTS['krea']['steps']
        krea['calibration_version'] = KREA_CALIBRATION_VERSION
    elif is_v3_default:
        # v3->v4 changed ONLY ref_boost's shipped default -- grounding_px and
        # steps are untouched by this bump, so a since-customised grounding_px
        # (a real value, unrelated to this migration) must survive it.
        krea['ref_boost'] = DEFAULTS['krea']['ref_boost']
        krea['calibration_version'] = KREA_CALIBRATION_VERSION
    return conf

# --- engines added by an update --------------------------------------------
# _deep_merge REPLACES lists (it only recurses into dicts), which is right for
# every other list we store — but engines.enabled doubles as "which engines
# exist", so a config saved before an engine shipped pinned its owner to the old
# catalogue forever: the more someone used the app, the fewer new engines they
# got, with no hint one existed. New SCALAR keys never had this problem, they
# fall back to their default; this is a list-only failure mode.
#
# The whole difficulty is telling "this engine didn't exist when I saved" from
# "I unchecked this on purpose" — blindly adding back what's missing would undo
# an explicit choice, which is worse than the bug. So a save records the
# catalogue the choice was made from (engines.known), and only engines absent
# from that ledger are merged in on read. Configs written before the ledger
# existed have no such record, but we know from the shipping history exactly
# which engines they could have been offered:
LEGACY_KNOWN_ENGINES = ('nanobanana', 'chatgpt', 'klein')
# ^ never extend this tuple. A new engine goes in DEFAULTS['engines']['enabled']
# and nowhere else; adding it here would mean "everyone has already seen it",
# i.e. exactly the bug this fixes.
#
# Only ONE key in DEFAULTS is a list of choices (engines.enabled) — the other
# list, klein.generation_lora_presets, is pure user data with an empty default
# and nothing to merge. Hence a named, tested helper rather than a framework;
# a second list-of-choices key should reuse the same known/enabled shape.


def _clean_engines(seq):
    return [e for e in (seq or []) if isinstance(e, str) and e]


def _engine_catalog(*groups):
    """Every engine this build knows about, in DEFAULTS order, plus any extra
    (older or hand-written) names the caller passes — nothing is ever dropped."""
    out = list(DEFAULTS['engines']['enabled'])
    for group in groups:
        for e in _clean_engines(group):
            if e not in out:
                out.append(e)
    return out


def _merge_new_engines(conf: dict, user: dict) -> dict:
    """Add engines that appeared since the user's saved selection (in place).

    Read-time only — the config file is never rewritten, so the fix applies to
    every existing install without a migration, and a downgrade still finds what
    it wrote. `user` is the raw file: an absent engines.enabled means the user
    never expressed a choice and already sits on the full default catalogue.

    Doubles as the shape guard for this section — config.json is hand-editable
    and a string where a list belongs would otherwise reach every consumer."""
    eng = conf.get('engines')
    if not isinstance(eng, dict):
        eng = conf['engines'] = copy.deepcopy(DEFAULTS['engines'])
    if not isinstance(eng.get('enabled'), list):
        eng['enabled'] = list(DEFAULTS['engines']['enabled'])
    saved = ((user or {}).get('engines') or {})
    saved = saved.get('enabled') if isinstance(saved, dict) else None
    if not isinstance(saved, list):
        return conf
    enabled = _clean_engines(eng.get('enabled'))
    if not enabled:
        # An empty list reads as "no restriction" downstream (face_dataset_service);
        # filling it in would turn that into a real, restrictive selection.
        eng['enabled'] = enabled
        return conf
    known = ((user or {}).get('engines') or {}).get('known')
    known = _clean_engines(known) if isinstance(known, list) else []
    known = known or list(LEGACY_KNOWN_ENGINES)
    eng['enabled'] = enabled + [e for e in DEFAULTS['engines']['enabled']
                                if e not in known and e not in enabled]
    eng['known'] = _engine_catalog(known, eng['enabled'])
    return conf


def _stamp_known_engines(merged: dict, partial: dict) -> dict:
    """Record the catalogue a selection was made from, on the saves that carry
    one. Only then: a save of some unrelated section must not certify that its
    author ever saw the engines they don't have enabled."""
    incoming = (partial or {}).get('engines')
    eng = merged.get('engines')
    if not isinstance(incoming, dict) or 'enabled' not in incoming or not isinstance(eng, dict):
        return merged
    eng['known'] = _engine_catalog(eng.get('known'), eng.get('enabled'))
    return merged


MIGRATED_LORA_PRESET_NAME = 'My LoRAs'

def _migrate_klein_loras(conf: dict, convert: bool = True) -> dict:
    """Two-stage soft migration of the pre-preset generation-LoRA formats into
    klein.generation_lora_presets (in place):
      (a) the very old single-slot keys ultra_real_lora / nsfw_lora become rows
          of the intermediate flat list (keeping their configured strengths);
      (b) a non-empty flat `generation_loras` list becomes ONE named preset
          ('My LoRAs'); the per-row nsfw_only flag is dropped — presets carry
          the intent now.
    Every legacy key is then removed so it can't shadow the presets. Idempotent
    (the preset is only created once, by name) and applied on EVERY load — a
    config.json written by any older version keeps working — and on save,
    which purges the legacy keys from the file.
    `convert=False` drops the legacy keys WITHOUT converting them: used when a
    save explicitly carries `generation_lora_presets` (the client already
    speaks the preset format, so the presets are authoritative — otherwise
    deleting the migrated preset in Settings would resurrect it from the
    file's legacy keys)."""
    k = conf.get('klein')
    if not isinstance(k, dict):
        return conf
    # (a) single-slot keys -> intermediate flat rows
    lst = k.pop('generation_loras', None)
    lst = [dict(e) for e in lst if isinstance(e, dict)] if isinstance(lst, list) else []
    for file_key, strength_key in (('ultra_real_lora', 'ultra_real_strength'),
                                   ('nsfw_lora', 'nsfw_strength')):
        f = (k.pop(file_key, '') or '')
        f = f.strip() if isinstance(f, str) else ''
        s = k.pop(strength_key, None)
        if convert and f and not any(e.get('file') == f for e in lst):
            lst.append({'file': f,
                        'strength': float(s) if isinstance(s, (int, float)) else 0.6})
    # (b) flat rows -> one named preset (nsfw_only dropped on purpose)
    presets = k.get('generation_lora_presets')
    presets = [dict(p) for p in presets if isinstance(p, dict)] if isinstance(presets, list) else []
    if convert:
        rows = []
        for e in lst:
            f = e.get('file')
            f = f.strip() if isinstance(f, str) else ''
            if not f:
                continue
            s = e.get('strength')
            rows.append({'file': f,
                         'strength': float(s) if isinstance(s, (int, float)) else 0.6})
        if rows and not any(p.get('name') == MIGRATED_LORA_PRESET_NAME for p in presets):
            presets.append({'name': MIGRATED_LORA_PRESET_NAME, 'loras': rows})
    k['generation_lora_presets'] = presets
    return conf


def _migrate_krea_character_lora(conf: dict, convert: bool = True) -> dict:
    """One-time soft migration of the pre-chain single character LoRA slot
    (krea.character_lora / krea.character_lora_strength) into row 0 of
    krea.character_loras, then drops the legacy keys. Same shape and same
    reasoning as _migrate_klein_loras: idempotent, applied on every load so a
    config.json written before the 5-slot chain existed keeps the value the
    user already set, and `convert=False` (a save that explicitly carries
    `character_loras`) drops the legacy keys WITHOUT reviving them — the
    client already speaks the list format, so a stale legacy key must not
    resurrect a row the user just cleared."""
    k = conf.get('krea')
    if not isinstance(k, dict):
        return conf
    f = (k.pop('character_lora', '') or '')
    f = f.strip() if isinstance(f, str) else ''
    s = k.pop('character_lora_strength', None)
    lst = k.get('character_loras')
    lst = [dict(e) for e in lst if isinstance(e, dict)] if isinstance(lst, list) else []
    if convert and f and not any((e.get('file') or '').strip() == f for e in lst):
        lst.insert(0, {'file': f, 'strength': float(s) if isinstance(s, (int, float)) else 0.8})
    k['character_loras'] = lst
    return conf

def load_config(force=False) -> dict:
    global _cache
    with _lock:
        if _cache is not None and not force:
            return copy.deepcopy(_cache)
        user = {}
        p = _config_path()
        if p.exists():
            try:
                user = json.loads(p.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                user = {}
        if not isinstance(user, dict):
            user = {}
        _cache = _merge_new_engines(
            _migrate_krea_character_lora(_migrate_klein_loras(
                _migrate_krea_pose_profile(_deep_merge(DEFAULTS, user), user))),
            user)
        return copy.deepcopy(_cache)

def save_config(partial: dict) -> dict:
    global _cache
    with _lock:
        p = _config_path()
        current = {}
        if p.exists():
            try:
                current = json.loads(p.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                current = {}
        # convert=False when this save explicitly carries the presets: the
        # client already speaks the preset format, so a legacy key left in the
        # file must not resurrect a preset the user just deleted — only purge.
        if not isinstance(current, dict):
            current = {}
        merged = _migrate_krea_character_lora(
            _stamp_known_engines(_migrate_klein_loras(
                _migrate_krea_pose_profile(_deep_merge(current, partial or {}), current,
                                           partial or {}),
                convert='generation_lora_presets' not in ((partial or {}).get('klein') or {})),
                partial),
            convert='character_loras' not in ((partial or {}).get('krea') or {}))
        tmp = p.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding='utf-8')
        tmp.replace(p)
        _cache = None
    return load_config()

def get(dotted: str, default=None):
    node = load_config()
    for part in dotted.split('.'):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node

def is_configured() -> bool:
    return _config_path().exists()

def secret(name: str):
    val = (os.environ.get(name) or '').strip()
    return val or None


# str.splitlines() recognizes more separators than CR/LF.  In particular, a
# vertical tab (and the Unicode line/paragraph separators) embedded in a value
# becomes a fresh NAME=value assignment the next time the file is rewritten.
# CR/LF remain valid delimiters *between* .env assignments; every other
# splitlines separator is rejected in an existing file before it can be
# normalized into a real newline.
_UNSAFE_ENV_FILE_SEPARATORS = frozenset(
    '\x00\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029'
)
_UNSAFE_SECRET_CATEGORIES = frozenset({'Cc', 'Cf', 'Zl', 'Zp'})


def _validated_secret_updates(d: dict) -> dict:
    updates = {}
    for name, value in (d or {}).items():
        if name not in SECRET_KEYS or not value:
            continue
        if (not isinstance(value, str)
                or any(unicodedata.category(ch) in _UNSAFE_SECRET_CATEGORIES
                       for ch in value)):
            raise ValueError(f"secret '{name}' must be a single line of text")
        updates[name] = value
    return updates


def _read_safe_env_lines() -> list[str]:
    if not ENV_PATH.exists():
        return []
    raw = ENV_PATH.read_text(encoding='utf-8')
    if any(ch in _UNSAFE_ENV_FILE_SEPARATORS for ch in raw):
        raise ValueError(
            'the existing .env contains an unsafe non-standard line separator'
        )
    return raw.splitlines()


def validate_secrets(d: dict) -> None:
    """Validate a prospective secret update without changing disk or process env.

    Routes call this before saving config.json so an invalid combined request is
    rejected atomically.  set_secrets repeats the checks as defence in depth for
    non-HTTP callers.
    """
    updates = _validated_secret_updates(d)
    if updates:
        _read_safe_env_lines()


def _quote_env_value(value: str) -> str:
    """Return a python-dotenv single-quoted value that round-trips exactly."""
    escaped = value.replace('\\', '\\\\').replace("'", "\\'")
    return f"'{escaped}'"


def set_secrets(d: dict) -> None:
    updates = _validated_secret_updates(d)
    if not updates:
        return
    lines = _read_safe_env_lines()
    for name, value in updates.items():
        lines = [l for l in lines if not l.startswith(f'{name}=')]
        lines.append(f'{name}={_quote_env_value(value)}')
    ENV_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Do not mutate the live process until persistence has succeeded.
    for name, value in updates.items():
        os.environ[name] = value
    load_dotenv(ENV_PATH, override=True)

def delete_secrets(names) -> None:
    """Remove saved secrets outright (clear a key). Separate from set_secrets,
    which SKIPS empty values on purpose so a blank field can't wipe a key by
    accident — deletion has to be an explicit action."""
    names = [n for n in (names or []) if n in SECRET_KEYS]
    if not names:
        return
    lines = _read_safe_env_lines()
    for name in names:
        lines = [l for l in lines if not l.startswith(f'{name}=')]
        os.environ.pop(name, None)   # load_dotenv won't unset a removed line, so drop it here
    ENV_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    load_dotenv(ENV_PATH, override=True)

_COMFY_DERIVED = {'output': ('output_dir', 'output'), 'input': ('input_dir', 'input'),
                  'models': ('models_dir', 'models'), 'loras': ('loras_dir', 'models/loras')}

# Stable display order for the four override fields (Settings, docs, API payload).
COMFY_DIR_KINDS = ('output', 'input', 'models', 'loras')


def resolve_comfyui_dir(kind: str, base_dir: str, explicit: str = ''):
    """Pure resolution of one ComfyUI folder: an explicit override wins, else it is
    derived from the install directory. Kept separate from `comfyui_dir` (which reads
    live config) so the Settings screen can PREVIEW the very same computation on
    unsaved field values — what the user is shown is then, by construction, what the
    app will use. Reported from Discord (vykas22): a ComfyUI launched with
    --input-directory/--output-directory looked like it was ignored, because the four
    override keys existed but had no field anywhere in the app.

    Whitespace-only is treated as empty: a stray space used to resolve to Path(' ')
    and silently shadow the derived folder."""
    _, sub = _COMFY_DERIVED[kind]
    explicit = (explicit or '').strip()
    if explicit:
        return Path(explicit)
    base = (base_dir or '').strip()
    return Path(base) / Path(sub) if base else None


def comfyui_dir(kind: str):
    key, _ = _COMFY_DERIVED[kind]
    return resolve_comfyui_dir(kind, get('comfyui.base_dir') or '',
                               get(f'comfyui.{key}') or '')

def aitoolkit_derived_python(root):
    """The interpreter an ai-toolkit checkout carries, ignoring any explicit
    `aitoolkit.python`. Both venv layouts exist: ai-toolkit's docs say `venv`,
    plenty of setups use `.venv`. Pick whichever actually exists; when neither
    does, return the historical default path so callers keep a concrete path to
    name in their "invalid" details (never None)."""
    root = Path(root)
    for env_dir in ('venv', '.venv'):
        p = (root / env_dir / 'Scripts' / 'python.exe' if os.name == 'nt'
             else root / env_dir / 'bin' / 'python')
        if p.exists():
            return p
    win = root / 'venv' / 'Scripts' / 'python.exe'
    return win if os.name == 'nt' else root / 'venv' / 'bin' / 'python'


def aitoolkit_path(kind: str):
    root = get('aitoolkit.dir') or ''
    if not root:
        return None
    root = Path(root)
    if kind == 'dir':
        return root
    if kind == 'datasets':
        return Path(get('aitoolkit.datasets_dir') or root / 'datasets')
    if kind == 'output':
        return Path(get('aitoolkit.output_dir') or root / 'output')
    if kind == 'hf_home':
        return Path(get('aitoolkit.hf_home') or root / 'hf-cache' / 'huggingface')
    if kind == 'venv_python':
        # An explicit interpreter wins — installs WITHOUT a venv folder exist
        # in the wild (conda, uv, system python; user-reported from Reddit).
        explicit = (get('aitoolkit.python') or '').strip()
        if explicit:
            return Path(explicit)
        return aitoolkit_derived_python(root)
    if kind == 'venv_python_derived':
        # What the app WOULD run without the explicit override. Only useful when
        # an explicit one is set and turns out to be broken: it is the working
        # interpreter we can then offer to switch to (GitHub #19, strouder —
        # a `aitoolkit.python` pointing at a torch-less Python silently beat a
        # perfectly good venv sitting right next to run.py).
        return aitoolkit_derived_python(root)
    if kind == 'jobs':
        return root / 'config' / 'generated'
    raise KeyError(kind)

def dataset_images_root() -> Path:
    p = get('paths.dataset_images_root') or ''
    root = Path(p) if p else _data_dir() / 'datasets'
    root.mkdir(parents=True, exist_ok=True)
    return root

def cloud_runs_root(create=True) -> Path:
    """Working area of cloud training runs (one ``run_<id>/`` per run: the
    exported dataset copy, the sample images and the mirrored training log).
    Relocatable — this is the directory that grows to tens of GB. It no longer
    holds the only copy of anything: checkpoints live in checkpoints_root()."""
    p = get('paths.cloud_runs_dir') or ''
    root = Path(p) if p else _data_dir() / 'cloud_runs'
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root

def checkpoints_root(create=True) -> Path:
    """The durable checkpoint store — one ``run_<id>/`` per cloud run holding the
    ``.safetensors`` it produced. Separate from cloud_runs_root() on purpose: the
    staging cleanup is allowed to throw its directory away, this one never is.

    ``create=False`` for the READ path: listing a run's saves happens on every
    hub poll, and an mkdir per run per poll buys nothing."""
    p = get('paths.checkpoints_dir') or ''
    root = Path(p) if p else _data_dir() / 'checkpoints'
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root

def backups_dir() -> Path:
    """Where 'Back up everything' writes its master archives (created on demand).
    Always under the app's data dir — never the (possibly relocated) datasets
    root — so a full backup never lands inside the very tree it is archiving."""
    d = _data_dir() / 'backups'
    d.mkdir(parents=True, exist_ok=True)
    return d

def bank_sources_root() -> Path:
    """Image folders CREATED by "Import to bank" — a copy of a dataset's kept
    images, so the new bank OWNS its files instead of pointing at the dataset's
    live folder (curating one would otherwise mutate the other). Deliberately
    NOT banks_root(): that one holds working data only and its contract is that
    it never contains source images."""
    root = _data_dir() / 'bank_sources'
    root.mkdir(parents=True, exist_ok=True)
    return root

def banks_root() -> Path:
    """Working data of the 🗃️ image banks (thumbnails + face-embedding cache),
    one subfolder per bank — never the source images, which stay in the user's
    folder untouched."""
    root = _data_dir() / 'banks'
    root.mkdir(parents=True, exist_ok=True)
    return root

def video_banks_root() -> Path:
    """Working data of the 🎬 video banks — THUMBNAILS AND NOTHING ELSE.

    The video bank stores bounds, not media: a clip is a pair of timestamps until
    the moment it is promoted. So unlike banks_root(), which also holds embedding
    caches, this tree only ever grows by one small .jpg per detected shot, and
    deleting it costs a thumbnail pass rather than a triage.

    Separate from banks_root() so the two lanes can be sized, moved and cleaned
    independently in Settings › Storage — a user with four hundred hours of rushes
    and a user with fifty thousand photos have very different problems."""
    root = _data_dir() / 'video_banks'
    root.mkdir(parents=True, exist_ok=True)
    return root

def video_datasets_root() -> Path:
    """Built video training sets: one flat ``<dataset id>/`` per set, holding the
    encoded ``clip_0001.mp4`` files and their homonym ``.txt`` captions.

    Relocatable, and it is the video lane's equivalent of dataset_images_root() —
    this is the directory that grows to tens of GB, because unlike the bank it
    holds real encoded media. NEVER the same tree as dataset_images_root(): the
    image lane's storage layout is one folder per dataset id too, and sharing the
    root would make two different tables claim the same folder name."""
    p = get('paths.video_datasets_dir') or ''
    root = Path(p) if p else _data_dir() / 'video_datasets'
    root.mkdir(parents=True, exist_ok=True)
    return root

def secret_key() -> str:
    d = _data_dir(); d.mkdir(parents=True, exist_ok=True)
    f = d / 'secret_key'
    if not f.exists():
        f.write_text(_secrets.token_hex(32), encoding='utf-8')
    return f.read_text(encoding='utf-8').strip()
