"""Private Hugging Face storage accounting for the cloud training lanes.

WHY THIS EXISTS
---------------
A dense (``full_transformer``) run delivers straight into a PRIVATE Hugging Face
model repo: every save is pushed from the pod, and one Krea 2 dense checkpoint
is ~26 GB. Run #146 died at step 2750/3000 on
``403 Forbidden: Private repository storage limit reached`` — hours of paid GPU
lost because the account's private allowance was already full of ``lds-base-*``
custom-base caches that nothing in the app listed, sized or cleaned. Every user
on a free plan hits the same wall, and always at the worst possible moment: at
the END of the run, after the money is spent.

So this module does two things: it MEASURES the private storage a namespace is
using before a pod is rented, and it inventories the ``lds-base-*`` caches so
they can be deleted from Settings.

WHAT THE HUB API ACTUALLY EXPOSES (checked against huggingface_hub 0.36)
-----------------------------------------------------------------------
* There is **no quota endpoint**. ``whoami-v2`` returns the identity, its
  organisations, the token's fine-grained scopes and ``isPro`` — no storage
  ceiling, no storage total.
* Per-repository **used** storage IS exposed: ``usedStorage`` is an expandable
  property of ``/api/models`` and ``/api/datasets``
  (``huggingface_hub.hf_api.ExpandModelProperty_T``). So
  ``list_models(author=<ns>, expand=['private', 'usedStorage'])`` returns the
  exact per-repo bytes the Hub bills — private repos included, for an
  authenticated owner. The SDK models no attribute for it, so the value lands
  on the dataclass verbatim as ``usedStorage``
  (``ModelInfo.__init__`` ends with ``self.__dict__.update(**kwargs)``).
  Summing that over the namespace's private repos is the only measurement of
  "private storage used" available, and it is exactly what was done by hand the
  day of the incident.
* The **ceiling is not knowable**. huggingface.co/docs/hub/storage-limits
  documents 100 GB of private storage for a free user/org and 1 TB for PRO, but
  the refusal in run #146 arrived at ~50 GB. This module therefore treats the
  limit as an ESTIMATE:
    - ``cloud.full_transformer.private_storage_limit_gb`` pins the real number
      for an account that knows it;
    - every refusal it produces is OVERRIDABLE (``HF_STORAGE_FULL:`` is a
      confirmable refusal, like the caption guards);
    - a probe that cannot measure NEVER blocks. Locking a user out of their own
      paid GPU over an unmeasurable number would be worse than the wall itself.

Storage is also counted over git history: a superseded revision keeps costing
until the repo history is squashed. The forecast below deliberately does not
model that (it would be a guess) — the shortfall message says so instead.
"""
import json
import logging
import re

from .. import config as cfg
from ..config import LOCAL_USER
from ..models import CloudTrainingRun
from .hf_publish import HfPublishError, _make_api

logger = logging.getLogger(__name__)

# The Hub reports and bills storage in decimal gigabytes; using 1024**3 here
# would silently under-report every number the user can read on huggingface.co.
GB = 1000 ** 3

# Documented private-storage allowances (huggingface.co/docs/hub/storage-limits,
# read 2026-08-03). Used only as a LAST-RESORT estimate — see the module note.
DOCUMENTED_PRIVATE_LIMIT_GB = {'free': 100, 'pro': 1000}

# Size of ONE dense checkpoint when no past run measured a real one.
# cloud_training documents the artifact as "~26 GB" and the deliveries observed
# so far were ~25 GB.
DENSE_CHECKPOINT_FALLBACK_BYTES = 26 * GB

# Default headroom on top of the raw checkpoint arithmetic: the repo also holds
# the model card / licence files, and a push that lands exactly on the ceiling
# fails just as hard as one that overshoots it.
DEFAULT_MARGIN_GB = 20

# The deterministic prefix hf_base_push gives every custom-base cache repo.
BASE_CACHE_PREFIX = 'lds-base-'

# Only ever delete a repo whose name matches this. The delete endpoints take a
# name from the client; without this a crafted value could target ANY repo the
# token can write, including the dense delivery of a live run.
_BASE_CACHE_NAME_RE = re.compile(r'^lds-base-[A-Za-z0-9._-]{1,64}$')

# Confirmable-refusal marker (frontend: utils/trainingRefusals.js). The window
# .confirm IS the answer and the retry carries `allow_hf_storage`.
STORAGE_REFUSAL_MARKER = 'HF_STORAGE_FULL: '


def _dense_cfg() -> dict:
    return ((cfg.get('cloud') or {}).get('full_transformer') or {})


def fmt_bytes(n) -> str:
    """Decimal GB with one decimal — the unit huggingface.co itself shows."""
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        return '0 GB'
    if n < GB:
        return f'{n / (1000 ** 2):.0f} MB'
    return f'{n / GB:.1f} GB'


# --- measurement ---------------------------------------------------------------

def _used_storage(info):
    """Bytes the Hub bills for one repo, or None when it did not report any.

    ``usedStorage`` is not a modelled SDK field: it arrives through
    ``ModelInfo.__dict__.update(**kwargs)``. Accept the snake_case spelling too
    so a future SDK that DOES model it keeps working.
    """
    for name in ('usedStorage', 'used_storage'):
        value = getattr(info, name, None)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)) and value >= 0:
            return int(value)
    return None


def _repo_row(info, kind) -> dict:
    repo_id = str(getattr(info, 'id', '') or '')
    last_modified = getattr(info, 'last_modified', None)
    return {
        'id': repo_id,
        'name': repo_id.split('/')[-1],
        'kind': kind,
        'private': bool(getattr(info, 'private', False)),
        'used_bytes': _used_storage(info),
        'last_modified': (last_modified.isoformat()
                          if hasattr(last_modified, 'isoformat') else None),
    }


_EXPAND = ['private', 'usedStorage', 'lastModified']


def _list_kind(api, namespace, kind):
    """One repo kind for one namespace, with the usedStorage expansion.

    Returns (rows, ok). ``ok=False`` means the listing itself failed — the sum
    is then incomplete and must never be presented as a measurement.
    """
    lister = getattr(api, 'list_models' if kind == 'model' else 'list_datasets', None)
    if lister is None:
        return [], False
    try:
        items = list(lister(author=namespace, expand=list(_EXPAND)))
    except TypeError:
        # An SDK (or a test double) that does not accept `expand`: the listing
        # still names the repos, it just cannot size them.
        try:
            items = list(lister(author=namespace))
        except Exception:
            return [], False
    except Exception:
        return [], False
    rows = []
    for info in items:
        try:
            rows.append(_repo_row(info, kind))
        except Exception:
            continue
    return rows, True


def private_storage_usage(namespace, token, _api=None) -> dict:
    """Measured private storage for ``namespace``.

    Never raises for an unreachable Hub: an unmeasurable account answers
    ``ok=False`` with a reason, and every caller treats that as "unknown", not
    as "full".
    """
    out = {'ok': False, 'namespace': namespace or None, 'used_bytes': None,
           'repos': [], 'private_repo_count': 0, 'unsized_repo_count': 0,
           'partial': False, 'reason': None}
    if not namespace:
        out['reason'] = 'no_namespace'
        return out
    if _api is None and not token:
        out['reason'] = 'no_token'
        return out
    try:
        api = _api or _make_api(token)
    except HfPublishError as e:
        out['reason'] = e.code
        return out
    rows = []
    any_ok = False
    for kind in ('model', 'dataset'):
        kind_rows, ok = _list_kind(api, namespace, kind)
        any_ok = any_ok or ok
        out['partial'] = out['partial'] or not ok
        rows.extend(kind_rows)
    if not any_ok:
        out['reason'] = 'listing_unavailable'
        return out
    private = [r for r in rows if r['private']]
    sized = [r for r in private if r['used_bytes'] is not None]
    out['repos'] = sorted(rows, key=lambda r: (-(r['used_bytes'] or 0), r['name']))
    out['private_repo_count'] = len(private)
    out['unsized_repo_count'] = len(private) - len(sized)
    out['used_bytes'] = sum(r['used_bytes'] for r in sized)
    # A namespace whose repos report no size at all was listed but not measured.
    out['ok'] = bool(sized) or not private
    if not out['ok']:
        out['reason'] = 'sizes_unavailable'
    return out


# --- what a dense run needs ----------------------------------------------------

def dense_checkpoint_bytes(_runs=None) -> tuple:
    """(bytes, source) for ONE dense checkpoint.

    Preference is a MEASUREMENT: past dense runs persist the delivered weight
    size in ``hf_artifact_proof.size_bytes`` (cloud_training's Hub-metadata
    integrity proof). The largest one observed is the honest planning figure;
    with no history, fall back to the documented ~26 GB.
    """
    override = _dense_cfg().get('checkpoint_size_gb')
    try:
        override = float(override or 0)
    except (TypeError, ValueError):
        override = 0
    if override > 0:
        return int(override * GB), 'configured'
    rows = _runs
    if rows is None:
        try:
            rows = (CloudTrainingRun.query
                    .order_by(CloudTrainingRun.id.desc()).limit(50).all())
        except Exception:
            rows = []
    best = 0
    for row in rows or []:
        try:
            params = json.loads(getattr(row, 'train_params', None) or '{}')
        except (TypeError, ValueError):
            continue
        if not isinstance(params, dict):
            continue
        if params.get('training_mode') != 'full_transformer':
            continue
        proof = params.get('hf_artifact_proof')
        size = (proof or {}).get('size_bytes') if isinstance(proof, dict) else None
        if isinstance(size, int) and not isinstance(size, bool) and size > best:
            best = size
    if best:
        return best, 'measured'
    return DENSE_CHECKPOINT_FALLBACK_BYTES, 'estimated'


def margin_bytes() -> int:
    try:
        gb = float(_dense_cfg().get('storage_margin_gb', DEFAULT_MARGIN_GB))
    except (TypeError, ValueError):
        gb = DEFAULT_MARGIN_GB
    return int(max(0, gb) * GB)


def private_limit_bytes(who=None) -> tuple:
    """(bytes, source) for the private allowance — an ESTIMATE by construction.

    ``configured`` is the only trustworthy source; the other two come from the
    published plan table, which the observed 403 proves is not what every
    account actually enforces.
    """
    try:
        configured = float(_dense_cfg().get('private_storage_limit_gb') or 0)
    except (TypeError, ValueError):
        configured = 0
    if configured > 0:
        return int(configured * GB), 'configured'
    if (who or {}).get('isPro'):
        return DOCUMENTED_PRIVATE_LIMIT_GB['pro'] * GB, 'plan_pro_documented'
    return DOCUMENTED_PRIVATE_LIMIT_GB['free'] * GB, 'plan_free_documented'


def dense_storage_forecast(namespace, token, keeps=1, who=None, _api=None,
                           _usage=None, fp8_export=False) -> dict:
    """Everything the launch guard and the Settings card need, in one shape.

    ``fits`` is deliberately tri-state: ``None`` = could not be measured, and
    that NEVER blocks a launch.

    ``fp8_export`` adds the ~10 GB ComfyUI twin the run also uploads. It is
    counted even when the bf16 master is going to be DROPPED afterwards: the
    master is pushed by the trainer first and the twin is uploaded before the
    master is deleted, so the peak — the only number a storage ceiling reacts
    to — always contains both.
    """
    keeps = max(1, int(keeps or 1))
    checkpoint, size_source = dense_checkpoint_bytes()
    from .fp8_export import estimate_fp8_bytes
    fp8 = estimate_fp8_bytes(checkpoint) if fp8_export else 0
    margin = margin_bytes()
    needed = checkpoint * keeps + fp8 + margin
    usage = _usage if _usage is not None else private_storage_usage(
        namespace, token, _api=_api)
    limit, limit_source = private_limit_bytes(who)
    out = {'needed_bytes': needed, 'checkpoint_bytes': checkpoint,
           'checkpoint_source': size_source, 'keeps': keeps,
           'fp8_bytes': fp8, 'fp8_export': bool(fp8_export),
           'margin_bytes': margin, 'limit_bytes': limit,
           'limit_source': limit_source, 'limit_is_estimate': limit_source != 'configured',
           'used_bytes': usage.get('used_bytes'), 'usage': usage,
           'free_bytes': None, 'shortfall_bytes': None, 'fits': None}
    if not usage.get('ok'):
        return out
    free = max(0, limit - int(usage['used_bytes'] or 0))
    out['free_bytes'] = free
    out['fits'] = needed <= free
    if not out['fits']:
        out['shortfall_bytes'] = needed - free
    return out


def _biggest_private(usage, limit=3) -> str:
    rows = [r for r in usage.get('repos') or []
            if r.get('private') and r.get('used_bytes')]
    if not rows:
        return ''
    top = rows[:limit]
    return ', '.join(f"{r['name']} ({fmt_bytes(r['used_bytes'])})" for r in top)


def storage_refusal_message(forecast) -> str:
    """The actionable sentence: how short, what fills it, where to click."""
    usage = forecast.get('usage') or {}
    caches = [r for r in usage.get('repos') or []
              if r.get('private') and str(r.get('name') or '').startswith(BASE_CACHE_PREFIX)]
    cache_bytes = sum(r.get('used_bytes') or 0 for r in caches)
    lines = [
        f"{STORAGE_REFUSAL_MARKER}This full-model run needs about "
        f"{fmt_bytes(forecast['needed_bytes'])} of PRIVATE Hugging Face storage "
        f"(one {fmt_bytes(forecast['checkpoint_bytes'])} checkpoint "
        f"× {forecast['keeps']} kept"
        + (f" + a {fmt_bytes(forecast.get('fp8_bytes'))} fp8 export"
           if forecast.get('fp8_bytes') else '')
        + f" + {fmt_bytes(forecast['margin_bytes'])} margin), "
        f"but {usage.get('namespace') or 'your account'} already uses "
        f"{fmt_bytes(forecast.get('used_bytes'))} of an estimated "
        f"{fmt_bytes(forecast['limit_bytes'])} private allowance — about "
        f"{fmt_bytes(forecast['shortfall_bytes'])} short.",
    ]
    biggest = _biggest_private(usage)
    if biggest:
        lines.append(f'Biggest private repos: {biggest}.')
    if caches:
        lines.append(
            f"{len(caches)} of them are {BASE_CACHE_PREFIX}* custom-base caches "
            f"({fmt_bytes(cache_bytes)}). They are re-pushable copies of local "
            f"files — delete them in Settings ▸ Training ▸ Hugging Face storage.")
    else:
        lines.append(
            'Free space from Settings ▸ Training ▸ Hugging Face storage, or from '
            'huggingface.co ▸ Settings ▸ Storage.')
    lines.append(
        'The allowance above is an ESTIMATE — Hugging Face publishes no quota '
        'endpoint, and superseded revisions keep counting until a repo history '
        'is squashed. Set cloud.full_transformer.private_storage_limit_gb if you '
        'know your real ceiling.')
    return '\n'.join(lines)


def assert_dense_storage_headroom(namespace, token, keeps=1, who=None,
                                  allow_override=False, _api=None,
                                  fp8_export=False) -> dict:
    """Pre-rent guard. Raises ValueError (confirmable) when it plainly will not
    fit; returns the forecast otherwise. An unmeasurable account passes."""
    forecast = dense_storage_forecast(namespace, token, keeps=keeps, who=who,
                                      _api=_api, fp8_export=fp8_export)
    if forecast.get('fits') is False and not allow_override:
        raise ValueError(storage_refusal_message(forecast))
    return forecast


# --- lds-base-* cache inventory ------------------------------------------------

def base_cache_index(user_id=LOCAL_USER) -> dict:
    """``repo_name -> {base_model, family, variant, last_run}``.

    The repo name carries a one-way sha1 of the local weights path, so the
    mapping cannot be inverted — it is REBUILT from the two places that recorded
    both halves:

    * cloud runs stamp ``base_repo_id`` next to ``base_model``/``train_type``
      (cloud_training.launch_cloud_training) — this also answers "which run last
      used this cache";
    * datasets keep their current custom selection, which covers a base pushed
      but never launched.
    """
    from . import face_dataset_service as fds
    from . import hf_base_push
    index = {}
    try:
        rows = (CloudTrainingRun.query
                .order_by(CloudTrainingRun.id.asc()).all())
    except Exception:
        rows = []
    for row in rows:
        try:
            params = json.loads(row.train_params or '{}')
        except (TypeError, ValueError):
            continue
        if not isinstance(params, dict):
            continue
        repo_id = params.get('base_repo_id')
        if not repo_id:
            continue
        name = str(repo_id).split('/')[-1]
        if not name.startswith(BASE_CACHE_PREFIX):
            continue
        # Ascending order: the LAST row wins, which is the most recent run.
        index[name] = {
            'base_model': params.get('base_model') or '',
            'family': params.get('train_type') or '',
            'variant': params.get('variant') or '',
            'last_run': {'id': row.id, 'name': row.run_name,
                         'status': row.status,
                         'created_at': (row.created_at.isoformat()
                                        if row.created_at else None)},
        }
    # Datasets fill the gaps only — a recorded run is the stronger evidence.
    try:
        datasets = fds.list_datasets(user_id)
    except Exception:
        datasets = []
    for ds in datasets or []:
        base_model = str(getattr(ds, 'train_base_model', None) or '').strip()
        if not base_model:
            continue
        family = fds.normalize_train_type(getattr(ds, 'train_type', None))
        try:
            name = hf_base_push.base_repo_name(ds, family, base_model)
        except Exception:
            continue
        index.setdefault(name, {
            'base_model': base_model, 'family': family,
            'variant': str(getattr(ds, 'train_variant', None) or ''),
            'last_run': None,
        })
    return index


def _local_base_state(family, base_model) -> dict:
    """Is the SOURCE of this cache still on disk? A cache whose local file is
    gone is the only copy of those weights — deleting it is not serene."""
    out = {'local_available': False, 'local_path': base_model or None,
           'local_size_bytes': None, 'local_reason': None}
    if not family or not base_model:
        out['local_reason'] = 'unknown_source'
        return out
    from . import hf_base_push
    try:
        payload = hf_base_push.local_base_payload(family, base_model)
    except HfPublishError as e:
        out['local_reason'] = e.code
        return out
    except Exception:
        out['local_reason'] = 'unknown_source'
        return out
    out.update(local_available=True, local_path=payload['path'],
               local_size_bytes=payload['size_bytes'])
    return out


def base_cache_inventory(namespace, token, user_id=LOCAL_USER, _api=None,
                         _usage=None) -> dict:
    """The Settings card's payload: every ``lds-base-*`` repo with its size, the
    local file it mirrors, and the run that last used it."""
    usage = _usage if _usage is not None else private_storage_usage(
        namespace, token, _api=_api)
    index = base_cache_index(user_id)
    caches = []
    for repo in usage.get('repos') or []:
        name = repo.get('name') or ''
        if not name.startswith(BASE_CACHE_PREFIX):
            continue
        meta = index.get(name) or {}
        entry = {**repo, **_local_base_state(meta.get('family'),
                                             meta.get('base_model')),
                 'family': meta.get('family') or None,
                 'variant': meta.get('variant') or None,
                 'last_run': meta.get('last_run')}
        caches.append(entry)
    return {'ok': usage.get('ok', False), 'namespace': usage.get('namespace'),
            'reason': usage.get('reason'), 'partial': usage.get('partial', False),
            'caches': caches,
            'cache_bytes': sum(c.get('used_bytes') or 0 for c in caches),
            'usage': usage}


def delete_base_cache(namespace, repo_name, token, _api=None) -> dict:
    """Delete ONE ``lds-base-*`` repo. The name is validated against the cache
    pattern first — this endpoint must never be able to address the dense
    delivery repo of a live run, or anything else the token can write."""
    name = str(repo_name or '').strip()
    if not _BASE_CACHE_NAME_RE.match(name):
        raise ValueError('only lds-base-* custom-base caches can be deleted here')
    if not namespace:
        raise ValueError('no Hugging Face namespace resolved for this token')
    api = _api or _make_api(token)
    repo_id = f'{namespace}/{name}'
    try:
        api.delete_repo(repo_id=repo_id, repo_type='model')
    except Exception as e:
        raise HfPublishError(
            'delete_failed',
            f'could not delete {repo_id}: {e}') from e
    return {'ok': True, 'repo_id': repo_id}


def delete_all_base_caches(namespace, token, user_id=LOCAL_USER, _api=None) -> dict:
    """Delete every ``lds-base-*`` repo of the namespace, reporting per repo.

    Failures do not abort the sweep: a single locked repo must not leave the
    other 40 GB behind.
    """
    inventory = base_cache_inventory(namespace, token, user_id=user_id, _api=_api)
    deleted, failed, freed = [], [], 0
    for cache in inventory['caches']:
        try:
            delete_base_cache(namespace, cache['name'], token, _api=_api)
        except (ValueError, HfPublishError) as e:
            failed.append({'name': cache['name'], 'error': str(e)})
            continue
        deleted.append(cache['name'])
        freed += cache.get('used_bytes') or 0
    return {'ok': not failed, 'deleted': deleted, 'failed': failed,
            'freed_bytes': freed}
