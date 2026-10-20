"""数据集总评分 + 低相似度标记（只读聚合 + 显式标记，不删任何东西）。

为什么是这两个放一起：总评分里的「一致性」维度就是靠 face_score，而用户对低分的
第一反应是"删掉" —— 所以标记必须从同一份数据出发，阈值和权重才不会各说各话。

设计要点（都是被坑出来的）：
  * 没测过的维度**不参与、不扣分**，权重重新归一，并如实报告 coverage。
    `face_score` 在库里只有一部分图有 —— 用 0 填充等于惩罚"没跑过那个 pass"。
  * 实测权重 < 50% → **不给总分**（NaN/None）。只测到 2 项时算出来的 100 分
    比没有分更危险。
  * 标记（mark）只改 `status='reject'`，**从不删文件**；删除复用已有的
    `purge_unused()`（一个 explicit 按钮），并且两边都跳过 `is_locked`。
"""
from __future__ import annotations

import math
import os
from collections import Counter

from ..models import FaceDatasetImage

# 各维度权重。合计 100。
WEIGHTS = {'consistency': 30, 'diversity': 20, 'caption': 20,
           'quality': 15, 'duplicates': 10, 'size': 5}
RECOMMENDED_IMAGES = 20          # 与 preflight 的 recommended 保持一致
CONFIDENCE_FLOOR = 0.2           # face_similarity.reference_agreement_floor()
DEFAULT_LOW_THRESHOLD = 0.55     # 实测分布：p50=0.74，<0.55 占 18%
STATUSES = ('keep', 'pending')


# ── 判定参数一律从 score_rule 表读（表里有就用表里的，没有就用下面的默认值）──
def _thr(key, default):
    try:
        from .score_rules import value_of
        return value_of(key, default)
    except Exception:
        return default


def weights():
    return {
        'consistency': _thr('weight_consistency', WEIGHTS['consistency']),
        'diversity': _thr('weight_diversity', WEIGHTS['diversity']),
        'caption': _thr('weight_caption', WEIGHTS['caption']),
        'quality': _thr('weight_quality', WEIGHTS['quality']),
        'duplicates': _thr('weight_duplicates', WEIGHTS['duplicates']),
        'size': _thr('weight_size', WEIGHTS['size']),
    }


def _entropy(counts):
    total = sum(counts) or 1
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def _pool(dataset_id):
    return (FaceDatasetImage.query
            .filter(FaceDatasetImage.dataset_id == dataset_id,
                    FaceDatasetImage.status.in_(STATUSES))
            .all())


def _dhash_file(path):
    """64 位感知哈希（9x8 灰度、按均值二值化）。只用于"同一场景连拍"，
    和身份 embedding 是两回事：夜景连拍的身份相似度只有 0.55-0.68，
    但视觉上几乎一样 —— 靠 embedding 判重复会全部漏掉。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            px = list(im.convert('L').resize((9, 8), Image.BILINEAR).getdata())
    except Exception:
        return None
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if px[row * 9 + col] > px[row * 9 + col + 1] else 0)
    return '%016x' % bits


def _dhash(img):
    """bank 分析快照里的 dhash（有才算）。"""
    import json
    raw = getattr(img, 'bank_analysis_snapshot', None)
    if not raw:
        return None
    try:
        return (json.loads(raw).get('analysis') or {}).get('dhash')
    except (ValueError, TypeError):
        return None


def _hamming(a, b):
    if not a or not b or len(a) != len(b):
        return 64
    try:
        return bin(int(a, 16) ^ int(b, 16)).count('1')
    except ValueError:
        return 64


def _score_consistency(rows, dataset_id=None):
    """一致性：face_score 的均值 + 达线比例。too_small/extreme_pose 也算（低权重）。"""
    weighted, weights = 0.0, 0.0
    good = total = 0
    seen = 0
    for img in rows:
        s = img.face_score
        if s is None:
            continue
        seen += 1
        w = {'scorable': 1.0, 'too_small': 0.85, 'extreme_pose': 0.7}.get(img.face_state, 0.6)
        weighted += float(s) * w
        weights += w
        total += 1
        good += 1 if float(s) >= _thr('face_floor', CONFIDENCE_FLOOR) else 0
    if not weights:
        # 没有参考分 → 看有没有「组内一致性」缓存（无参考图的数据集走这条）
        cache = cluster_cached(dataset_id, rows) if dataset_id is not None else None
        if not cache or cache.get('median') is None:
            return None, 0.0, {'mode': 'reference', 'scored': 0}
        med = float(cache['median'])
        return (round(100 * min(med / 0.75, 1.0)),
                len(cache.get('per_image') or {}) / max(len(rows), 1),
                {'mode': 'cluster', 'median': round(med, 3),
                 'p10': cache.get('p10'), 'n': cache.get('n')})
    detail = {'mode': 'reference', 'scored': seen,
              'scorable': sum(1 for r in rows if r.face_state == 'scorable'),
              'mean': round(weighted / weights, 3), 'above_floor': good}
    return round(100 * min((weighted / weights) / 0.75, 1.0)), seen / max(len(rows), 1), detail


SCENE_WORDS = ('room bed sofa chair window door wall floor street park beach forest garden '
               'kitchen bathroom office cafe bar car train stairs mirror curtain sheets '
               'pillow blanket towel table lamp light shadow sky water').split()


def _score_diversity(user_id, dataset_id, rows):
    """多样性：构图分布均匀度 + caption 里场景词的种类数（不需要读图）。"""
    framed = Counter(r.framing for r in rows if r.framing in ('face', 'bust', 'body', 'back'))
    if sum(framed.values()) < max(4, len(rows) // 4):
        return None, 0.0, {'buckets': dict(framed), 'reason': 'framing 未分类'}
    ent = min(_entropy(framed.values()) / 2.0, 1.0)          # 4 桶熵上限 2 bit
    caps = ' '.join((r.caption or '').lower() for r in rows)
    scene = sum(1 for w in SCENE_WORDS if w in caps)
    scene_div = min(scene / 10.0, 1.0)
    return round(100 * (0.75 * ent + 0.25 * scene_div)), 1.0, {
        'framing_entropy': round(ent, 3), 'buckets': dict(framed),
        'scene_words': scene}


def _score_caption(rows):
    caps = [(r.caption or '').strip() for r in rows if (r.caption or '').strip()]
    if not caps:
        return None, 0.0, {}
    med = sorted(len(c) for c in caps)[len(caps) // 2]
    leak = sum(1 for c in caps if any(w in c.lower() for w in
                                      ('hair', 'face', 'eyes', 'skin', 'nose', 'lips')))
    dup = 1 - len(set(caps)) / len(caps)
    length_ok = min(med / 120.0, 1.0)
    score = 100 * (0.40 * length_ok + 0.35 * (1 - min(leak / len(caps) * 2, 1))
                   + 0.25 * (1 - dup))
    return round(score), len(caps) / max(len(rows), 1), {
        'captioned': len(caps), 'median_len': med,
        'identity_leak': round(leak / len(caps), 2)}


def _score_quality(rows):
    import json
    blurs = []
    for img in rows:
        raw = getattr(img, 'bank_analysis_snapshot', None)
        if not raw:
            continue
        try:
            v = (json.loads(raw).get('analysis') or {}).get('blur_score')
        except (ValueError, TypeError):
            continue
        if isinstance(v, (int, float)):
            blurs.append(float(v))
    if not blurs:
        return None, 0.0, {'analysed': 0}
    med = sorted(blurs)[len(blurs) // 2]
    wm = sum(1 for r in rows if (r.watermark_state or 'none') not in ('', 'none'))
    score = 100 * (0.65 * min(med / 400.0, 1.0) + 0.35 * (1 - min(wm / max(len(rows), 1) * 4, 1)))
    return round(score), len(blurs) / max(len(rows), 1), {
        'analysed': len(blurs), 'median_blur': round(med, 1), 'watermarked': wm}


def _score_duplicates(rows):
    sigs = [r.content_sig for r in rows if r.content_sig]
    dh = [d for d in (_dhash(r) for r in rows) if d]
    if not sigs and not dh:
        return None, 0.0, {}
    exact = 1 - len(set(sigs)) / len(sigs) if sigs else 0.0
    pairs = 0
    for i in range(len(dh)):
        for j in range(i + 1, len(dh)):
            if _hamming(dh[i], dh[j]) <= 4:
                pairs += 1
    near = pairs / max(len(dh) * (len(dh) - 1) / 2, 1)
    score = 100 * ((1 - min(exact * 2, 1)) * 0.5 + (1 - min(near * 20, 1)) * 0.5)
    return round(score), (len(sigs) or len(dh)) / max(len(rows), 1), {
        'exact_dupes': round(exact, 3), 'near_pairs': pairs}


def score(user_id, dataset_id):
    """返回 {total, measured, dims, coverage, details, low_similarity}。只读。"""
    rows = _pool(dataset_id)
    n = len(rows)
    dims, coverage, details = {}, {}, {}

    for name, fn in (('consistency', lambda: _score_consistency(rows, dataset_id)),
                     ('diversity', lambda: _score_diversity(user_id, dataset_id, rows)),
                     ('caption', lambda: _score_caption(rows)),
                     ('quality', lambda: _score_quality(rows)),
                     ('duplicates', lambda: _score_duplicates(rows))):
        value, cov, detail = fn()
        dims[name] = value
        coverage[name] = round(cov, 2)
        details[name] = detail

    rec = _thr('recommended_images', RECOMMENDED_IMAGES)
    dims['size'] = round(100 * min(n / rec, 1)) if n else 0
    coverage['size'] = 1.0

    W = weights()
    got = {k: v for k, v in dims.items() if v is not None}
    weight_sum = sum(W[k] for k in got)
    measured = round(100 * weight_sum / sum(W.values()))
    total = round(sum(dims[k] * W[k] for k in got) / weight_sum) if weight_sum else None
    if measured < 50:
        total = None                                  # 测得太少，不给总分

    report = {
        'total': total,
        'measured': measured,
        'dims': dims,
        'coverage': coverage,
        'weights': WEIGHTS,
        'details': details,
        'images': n,
        'low_similarity': low_similarity_ids(rows, _thr('low_sim_threshold', DEFAULT_LOW_THRESHOLD), 'generated',
                                             dataset_id=dataset_id),
        'cluster': (cluster_cached(dataset_id, rows) or {}).get('median') and
                   {k: (cluster_cached(dataset_id, rows) or {}).get(k)
                    for k in ('median', 'p10', 'n', 'fingerprint')} or None,
        'top_fixes': _top_fixes(dims, coverage, details, n),
    }
    try:
        persist_score(dataset_id, report, fingerprint=_fingerprint(rows))
    except Exception:
        import logging
        logging.getLogger(__name__).exception('score could not be persisted')
    return report


def _top_fixes(dims, coverage, details, n):
    """按「补哪一项最划算」排序 —— 只说能操作的。"""
    fixes = []
    if dims['consistency'] is None or coverage['consistency'] < 0.8:
        fixes.append({'dim': 'consistency', 'action': 'analyze_faces',
                      'text': '一致性只覆盖了 %d%% 的图 —— 跑一次 🎯 人脸打分（CPU，'
                              '约 0.4 秒/张）' % round(100 * coverage['consistency'])})
    if dims['diversity'] is None:
        fixes.append({'dim': 'diversity', 'action': 'classify',
                      'text': '构图还没分类 —— 跑一次 📐 分类才能算多样性'})
    if dims['quality'] is None:
        fixes.append({'dim': 'quality', 'action': 'bank_analyse',
                      'text': '没有画质分析数据 —— 跑一次画质分析才能算质量分'})
    if dims['caption'] is None or coverage['caption'] < 1.0:
        fixes.append({'dim': 'caption', 'action': 'caption',
                      'text': '%d/%d 张还没有 caption' % (
                          n - round(coverage['caption'] * n), n)})
    leak = (details.get('caption') or {}).get('identity_leak')
    if isinstance(leak, float) and leak > 0.15:
        fixes.append({'dim': 'caption', 'action': 'recaption',
                      'text': '%.0f%% 的 caption 写了五官/发型（身份会绑到那些词上）' % (leak * 100)})
    if dims['size'] is not None and dims['size'] < 100:
        fixes.append({'dim': 'size', 'action': 'generate',
                      'text': '只有 %d 张（建议 %d+）' % (n, RECOMMENDED_IMAGES)})
    return fixes


def low_similarity_ids(rows, threshold=None, scope='generated', dataset_id=None):
    """similarity < threshold 的图 —— 只作候选，不改任何东西。

    有参考分 → 用 face_score；没有（无参考图的数据集）→ 用组内一致性缓存，
    门槛取 max(绝对下限, 组内中位数 - 0.15)：大集里 0.5 可能很正常，小集里 0.5 就很怪。"""
    if threshold is None:
        threshold = _thr('low_sim_threshold', DEFAULT_LOW_THRESHOLD)
    cluster = cluster_cached(dataset_id, rows) if dataset_id is not None else None
    if cluster and cluster.get('median') is not None:
        floor = max(_thr('cluster_min_floor', CLUSTER_MIN),
                    float(cluster['median']) - _thr('cluster_drop', CLUSTER_DROP))
        out = []
        by_id = {str(r.id): r for r in rows}
        for iid, detail in (cluster.get('per_image') or {}).items():
            sim = detail.get('sim') if isinstance(detail, dict) else detail
            if sim is None:
                continue
            img = by_id.get(str(iid))
            if img is None or img.is_locked or float(sim) >= floor:
                continue
            if scope == 'generated' and (getattr(img, 'source', None) or '') != 'generated':
                continue
            out.append({'id': img.id, 'filename': img.filename,
                        'score': round(float(sim), 3), 'state': 'cluster',
                        'source': getattr(img, 'source', None), 'floor': round(floor, 3)})
        return sorted(out, key=lambda r: r['score'])
    out = []
    for img in rows:
        if img.face_score is None or img.face_state in ('no_face', 'unreadable'):
            continue
        if scope == 'generated' and (getattr(img, 'source', None) or '') != 'generated':
            continue
        if img.is_locked:
            continue
        if float(img.face_score) < float(threshold):
            out.append({'id': img.id, 'filename': img.filename,
                        'score': round(float(img.face_score), 3),
                        'state': img.face_state,
                        'source': getattr(img, 'source', None)})
    return sorted(out, key=lambda r: r['score'])


# 护栏：离群超过这个比例就不是"个别图坏了"，而是整个集不一致 —— 那时打删除标识
# 等于把数据集清空一半，用户要的是"删掉几张烂图"，不是"删掉数据集"。
OUTLIER_GUARD = 0.30


def mark_low_similarity(user_id, dataset_id, threshold=DEFAULT_LOW_THRESHOLD,
                        scope='generated', dry_run=True, force=False):
    """把低相似度的图打上删除标识（status='reject' + fail_reason）。

    不删文件：删动作留给已有的「一键删除被拒图片」（purge_unused，同样跳过 is_locked）。
    dry_run=True 只返回候选，不写库。
    """
    from .face_dataset_service import get_dataset, set_image_status
    from ..extensions import db

    if not get_dataset(user_id, dataset_id):
        raise ValueError('dataset not found')
    rows = _pool(dataset_id)
    cands = low_similarity_ids(rows, threshold, scope, dataset_id=dataset_id)
    guard = None
    if len(rows) and len(cands) / len(rows) > _thr('outlier_guard', OUTLIER_GUARD) and not force:
        guard = ('%d/%d 张（%.0f%%）都在门槛下 — 这是整个数据集不一致，不是个别烂图。'
                 '请先看一致性分，确认后带 force=true 再来，或改阈值。'
                 % (len(cands), len(rows), 100 * len(cands) / len(rows)))
    if dry_run:
        return {'dry_run': True, 'threshold': float(threshold), 'scope': scope,
                'count': len(cands), 'guard': guard, 'candidates': cands[:200]}
    if guard:
        return {'dry_run': False, 'blocked': True, 'reason': guard,
                'count': len(cands), 'candidates': cands[:50]}
    by_id = {img.id: img for img in rows}
    changed = 0
    for c in cands:
        img = by_id.get(c['id'])
        if img is None or img.is_locked:
            continue
        # 走已有 service：它会校验 status、清掉水印元数据、处理派生图关系
        if not set_image_status(user_id, img.id, 'reject'):
            continue
        changed += 1
    # 理由单独写一次（set_image_status 不管 fail_reason）
    for c in cands:
        img = by_id.get(c['id'])
        if img is None:
            continue
        db.session.refresh(img)
        if img.status == 'reject':
            img.fail_reason = 'low face similarity %.3f < %.2f (%s)' % (
                c['score'], float(threshold), c['state'] or '?')
    db.session.commit()
    return {'dry_run': False, 'threshold': float(threshold), 'scope': scope,
            'marked': changed, 'count': len(cands),
            'undo': '把 status 改回 keep（fail_reason 以 low face similarity 开头的行）'}


# ── 无参考图模式（组内一致性）────────────────────────────────────────────────
# 17 个数据集没有参考图 → 参考模式打不了分。用「组内两两余弦中位数」代替，
# 结果缓存在 data/cluster_scores/<ds>.json（键是图片指纹，图片一变自动失效）。
CLUSTER_DIR = 'cluster_scores'
# 缓存格式版本：per_image 从 {sim: float} 升级成 {sim, dhash, det...} 时必须失效，
# 否则老缓存会让 classify_outliers 拿到 float 而 500。
CLUSTER_CACHE_V = 3
CLUSTER_MIN = 0.55          # 绝对下限：低于它基本是"不像其他人"
CLUSTER_DROP = 0.15         # 相对下限：比组内中位数低这么多的，算离群


def _cluster_path(dataset_id):
    from ..config import data_dir
    return os.path.join(str(data_dir()), CLUSTER_DIR, '%s.json' % dataset_id)


def _fingerprint(rows):
    import hashlib
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: x.filename or ''):
        h.update((r.filename or '').encode())
        h.update((r.content_sig or '').encode())
    return h.hexdigest()[:16]


def cluster_pass(user_id, dataset_id):
    """跑一次组内一致性，落缓存。CPU、不占 GPU。返回摘要 + 离群名单。"""
    import json
    import os as _os
    from .face_similarity import score_cluster
    from .face_dataset_service import dataset_path, get_dataset

    ds = get_dataset(user_id, dataset_id)
    if not ds:
        raise ValueError('dataset not found')
    rows = _pool(dataset_id)
    base = dataset_path(dataset_id)
    paths = {}
    for r in rows:
        p = _os.path.join(str(base), r.filename or '')
        if r.filename and _os.path.isfile(p):
            paths[p] = r
    if len(paths) < 3:
        raise ValueError('cluster scoring needs at least 3 images')
    data, err = score_cluster(list(paths))
    if err:
        raise ValueError(err.get('detail') or err.get('kind') or 'cluster scoring failed')
    hashes = {}
    for p in paths:
        h = _dhash_file(p)
        if h:
            hashes[str(paths[p].id)] = h
    per_image = {}
    for p, detail in (data.get('per_image') or {}).items():
        row = paths.get(p)
        if row is None or not isinstance(detail, dict) or detail.get('sim') is None:
            continue
        per_image[str(row.id)] = {
            'sim': round(float(detail['sim']), 4), 'sim_max': detail.get('sim_max'),
            'det': detail.get('det'), 'bbox_frac': detail.get('bbox_frac'),
            'yaw': detail.get('yaw'), 'state': detail.get('state'),
            'identity_group': detail.get('identity_group'),
            'dhash': hashes.get(str(row.id))}
    median = data.get('median')
    floor = (max(_thr('cluster_min_floor', CLUSTER_MIN),
                 float(median) - _thr('cluster_drop', CLUSTER_DROP))
             if median is not None else None)
    outliers = classify_outliers(per_image, rows, floor) if floor is not None else []
    payload = {'cache_v': CLUSTER_CACHE_V, 'fingerprint': _fingerprint(rows), 'median': median,
               'p10': data.get('p10'), 'n': data.get('n'),
               'identities': data.get('identities') or [],
               'multi_identity': bool(data.get('multi_identity')),
               'per_image': per_image, 'outliers': outliers}
    path = _cluster_path(dataset_id)
    _os.makedirs(_os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    try:
        persist_cluster(dataset_id, per_image, median, data.get('p10'),
                        data.get('identities') or [], bool(data.get('multi_identity')),
                        fingerprint=payload['fingerprint'], outliers=outliers)
    except Exception:                     # 落库失败不该让打分本身失败
        import logging
        logging.getLogger(__name__).exception('cluster result could not be persisted')
    return {'dataset_id': dataset_id, 'n': data.get('n'), 'median': median,
            'p10': data.get('p10'), 'floor': floor,
            'groups': _group_counts(outliers),
            'outliers': outliers[:200], 'outlier_count': len(outliers),
            'cached_to': path}


def cluster_cached(dataset_id, rows=None):
    """读上次的结果：**数据库优先**（跨重启、可 SQL 查），JSON 缓存其次。"""
    import json
    import os as _os
    try:
        stored = stored_cluster(dataset_id)
    except Exception:
        stored = None
    if stored and (rows is None or stored.get('per_image') is not None):
        return stored
    path = _cluster_path(dataset_id)
    if not _os.path.isfile(path):
        return None
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if data.get('cache_v') != CLUSTER_CACHE_V:
        return None
    if rows is not None and data.get('fingerprint') != _fingerprint(rows):
        return None
    return data


# ── 离群成因分类 ────────────────────────────────────────────────────────────
# 一张"低分图"可能是四种完全不同的东西，删之前必须分开：
#   duplicate  连拍/近重复   → 同一瞬间塞 4 张只会让模型记住那一个夜晚
#   profile    大侧脸        → 提供角度变化，训练有用，只是脸不好读
#   weak       检测勉强      → 遮挡/逆光/太小；可能有用也可能没用，要看图
#   different  脸读得很好却不像 → 这才是真的"混进了别人/崩图"
# 组内一致性衡量的是「脸可读性 + 相似度」，不是「这张图有没有用」——
# 侧脸、逆光、遮挡都会压低它，而它们恰恰是训练需要的变化。
DUP_SIM = 0.95          # 身份最近邻 ≥ 这个 → 基本是同一帧
DUP_HAMMING = 10        # 感知哈希汉明距离 ≤ 这个 → 同一场景连拍（不需要脸相像）
PROFILE_YAW = 50.0
WEAK_DET = 0.70
WEAK_BBOX = 0.05

CAUSE_ACTION = {
    'duplicate': '部分删除：每组只留最清晰的一张',
    'profile': '保留：这是角度变化，训练需要',
    'weak': '人工看一眼：遮挡/逆光/太小，有用就留',
    'drift': '保留但挑子集：同一人、外观跨太大（换装/光/年代/滤镜），训练会被平均掉',
    'other_identity': '优先删除：这属于另一个身份分量（多人的集）',
}


def _near_dup_partner(iid, per_image):
    """在全集里找它的"连拍伙伴"（感知哈希最近的那个，≤ DUP_HAMMING）。"""
    mine = (per_image.get(str(iid)) or {})
    mine = mine.get('dhash') if isinstance(mine, dict) else None
    if not mine:
        return None
    best, best_d = None, int(_thr('dup_hamming', DUP_HAMMING)) + 1
    for other, d in per_image.items():
        if other == str(iid):
            continue
        h = (d or {}).get('dhash') if isinstance(d, dict) else None
        if not h:
            continue
        try:
            dist = bin(int(mine, 16) ^ int(h, 16)).count('1')
        except ValueError:
            continue
        if dist < best_d:
            best, best_d = other, dist
    return (best, best_d) if best else None


def classify_outliers(per_image, rows, floor):
    """把低于 floor 的图分组成因。per_image[id] 是 cluster_pass 存下的详情。

    重复优先用感知哈希判：夜景连拍的身份相似度只有 0.55-0.68，但视觉上几乎一样，
    光靠 embedding 会把它们全部分到"不像"。"""
    by_id = {str(r.id): r for r in rows}
    groups = {}
    for iid, d in (per_image or {}).items():
        if isinstance(d, dict) and d.get('identity_group'):
            groups.setdefault(d['identity_group'], 0)
            groups[d['identity_group']] += 1
    big = [g for g, size in groups.items() if size >= 3]
    multi_identity = len(big) > 1
    main_group = max(groups, key=groups.get) if groups else None
    out = []
    for iid, d in (per_image or {}).items():
        sim = d.get('sim') if isinstance(d, dict) else d
        if sim is None or floor is None or float(sim) >= float(floor):
            continue
        det = d.get('det') if isinstance(d, dict) else None
        frac = d.get('bbox_frac') if isinstance(d, dict) else None
        yaw = d.get('yaw') if isinstance(d, dict) else None
        sim_max = d.get('sim_max') if isinstance(d, dict) else None
        partner = _near_dup_partner(iid, per_image)
        if partner or (isinstance(sim_max, (int, float)) and sim_max >= DUP_SIM):
            cause = 'duplicate'
        elif isinstance(yaw, (int, float)) and abs(yaw) > _thr('profile_yaw', PROFILE_YAW):
            cause = 'profile'
        elif (isinstance(det, (int, float)) and det < WEAK_DET) or \
                (isinstance(frac, (int, float)) and frac < WEAK_BBOX):
            cause = 'weak'
        elif multi_identity and d.get('identity_group') and                 d['identity_group'] != main_group and groups.get(d['identity_group'], 0) >= 3:
            cause = 'other_identity'
        else:
            cause = 'drift'
        row = by_id.get(str(iid))
        out.append({'id': int(iid), 'filename': row.filename if row else None,
                    'score': round(float(sim), 3),
                    'sim_max': sim_max, 'det': det, 'bbox_frac': frac, 'yaw': yaw,
                    'cause': cause, 'action': CAUSE_ACTION[cause],
                    'dup_with': int(partner[0]) if partner else None,
                    'dup_distance': partner[1] if partner else None,
                    'source': getattr(row, 'source', None) if row else None})
    order = {'other_identity': 0, 'duplicate': 1, 'weak': 2, 'profile': 3, 'drift': 4}
    return sorted(out, key=lambda o: (order[o['cause']], o['score']))


def _group_counts(outliers):
    counts = Counter(o['cause'] for o in outliers)
    return {k: {'count': counts.get(k, 0), 'action': CAUSE_ACTION[k],
                'ids': [o['id'] for o in outliers if o['cause'] == k]}
            for k in CAUSE_ACTION}


def outlier_report(user_id, dataset_id, rebuild=False):
    """带成因分组的离群报告。缓存过期或 rebuild=True 时重跑一次组内打分。"""
    from .face_dataset_service import get_dataset
    if not get_dataset(user_id, dataset_id):
        raise ValueError('dataset not found')
    rows = _pool(dataset_id)
    cache = None if rebuild else cluster_cached(dataset_id, rows)
    if cache is None:
        cluster_pass(user_id, dataset_id)
        cache = cluster_cached(dataset_id, rows)
    if not cache:
        raise ValueError('no cluster score for this dataset (needs at least 3 images)')
    floor = max(_thr('cluster_min_floor', CLUSTER_MIN),
                float(cache['median']) - _thr('cluster_drop', CLUSTER_DROP))
    outliers = classify_outliers(cache.get('per_image') or {}, rows, floor)
    plan = duplicate_plan(cache.get('per_image') or {}, outliers)
    return {'dataset_id': dataset_id, 'median': cache.get('median'),
            'identities': cache.get('identities') or [],
            'multi_identity': bool(cache.get('multi_identity')),
            'duplicate_plan': plan,
            'duplicate_delete': sum(len(p['delete']) for p in plan),
            'p10': cache.get('p10'), 'n': cache.get('n'), 'floor': round(floor, 3),
            'images': len(rows), 'outlier_count': len(outliers),
            'groups': _group_counts(outliers), 'outliers': outliers[:300]}


def duplicate_plan(per_image, outliers):
    """把「重复」分组成簇，每簇留一张最清晰的，其余列进删除候选。

    留哪张：det 高优先（检测更有把握），同分看 bbox_frac（脸更大），再看 sim 高。
    这样"连拍 4 张留 1 张"不需要人挑。
    """
    dups = [o for o in outliers if o['cause'] == 'duplicate']
    ids = {o['id'] for o in dups}
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    hashes = {int(i): (d or {}).get('dhash') for i, d in per_image.items()
              if isinstance(d, dict)}
    for a in ids:
        for b in ids:
            if a >= b:
                continue
            ha, hb = hashes.get(a), hashes.get(b)
            if not ha or not hb:
                continue
            try:
                if bin(int(ha, 16) ^ int(hb, 16)).count('1') <= DUP_HAMMING:
                    parent[find(a)] = find(b)
            except ValueError:
                continue
    clusters = {}
    for o in dups:
        clusters.setdefault(find(o['id']), []).append(o)
    plan = []
    for members in clusters.values():
        members.sort(key=lambda o: (-(o.get('det') or 0), -(o.get('bbox_frac') or 0),
                                    -(o.get('score') or 0)))
        plan.append({'keep': members[0]['id'], 'delete': [m['id'] for m in members[1:]],
                     'size': len(members),
                     'keep_det': members[0].get('det'),
                     'keep_score': members[0].get('score')})
    return sorted(plan, key=lambda p: -p['size'])


def mark_images(user_id, dataset_id, ids, reason='marked by user', dry_run=True):
    """给**指定的**图片打删除标识（人工挑出来的：连拍冗余、带字幕截图……）。

    只写 status='reject' + fail_reason，**不删文件**；删除走 purge（一键，跳过 is_locked）。
    锁定的、不属于该数据集的 id 一律跳过 —— 一个手滑的 id 不该改动别的数据集。
    """
    from .face_dataset_service import get_dataset, set_image_status
    from ..extensions import db

    if not get_dataset(user_id, dataset_id):
        raise ValueError('dataset not found')
    rows = _pool(dataset_id)
    by_id = {int(r.id): r for r in rows}
    wanted, skipped = [], []
    for raw in ids or []:
        try:
            iid = int(raw)
        except (TypeError, ValueError):
            skipped.append({'id': raw, 'why': 'not an id'})
            continue
        row = by_id.get(iid)
        if row is None:
            skipped.append({'id': iid, 'why': 'not in this dataset (or already not keep/pending)'})
            continue
        if row.is_locked:
            skipped.append({'id': iid, 'why': 'locked'})
            continue
        wanted.append(row)
    if dry_run:
        return {'dry_run': True, 'count': len(wanted), 'skipped': skipped,
                'ids': [r.id for r in wanted]}
    changed = 0
    for row in wanted:
        if set_image_status(user_id, row.id, 'reject'):
            changed += 1
    for row in wanted:
        db.session.refresh(row)
        if row.status == 'reject':
            row.fail_reason = reason
    db.session.commit()
    return {'dry_run': False, 'marked': changed, 'count': len(wanted),
            'skipped': skipped,
            'next': 'POST /api/dataset/%d/purge 一键删除（可逆：status 改回 keep）' % dataset_id}


# ── 画质 / 指纹分析（本地 PIL，无模型、无下载）────────────────────────────────
DETERMINISTIC_FIELDS = ('quality_state', 'blur_score', 'noise_score', 'uniformity_score',
                        'dhash', 'detail_ratio', 'bars_ratio', 'jpeg_quality', 'origin',
                        'origin_evidence')


def _snapshot_v2(metrics, dhash_hex, fingerprint):
    """bank_analysis_snapshot 的 v2 形状（bank_transfer_metadata._parse_v2 认这个）。"""
    import json
    analysis = {
        'quality_state': 'ok',
        'blur_score': float(metrics.get('blur_score') or 0.0),
        'noise_score': float(metrics.get('noise_score') or 0.0),
        'uniformity_score': float(metrics.get('uniformity_score') or 0.0),
        'dhash': dhash_hex,
        'detail_ratio': 1.0,
        'bars_ratio': 0.0,
        'jpeg_quality': None,
        'origin': 'unknown',
        'origin_evidence': None,
    }
    return json.dumps({'v': 2, 'fingerprint': fingerprint, 'analysis': analysis},
                      ensure_ascii=False)


def analyse_images(user_id, dataset_id, limit=None):
    """给数据集里的图算**本地**技术指标：清晰度/噪点/均匀度 + 感知哈希 + 内容指纹。

    这一步不需要任何模型（PIL + 直方图），因此不会下载任何东西。它补的是
    总评分里 quality（要 blur_score）和 duplicates（要 content_sig / dhash）两维 ——
    没有它这两个维度只能是 None。已分析过的图会被跳过（除非 force）。
    """
    import hashlib
    import os as _os
    from PIL import Image
    from .face_dataset_service import dataset_path, get_dataset
    from .image_quality import quality_metrics
    from ..extensions import db

    if not get_dataset(user_id, dataset_id):
        raise ValueError('dataset not found')
    rows = _pool(dataset_id)
    base = dataset_path(dataset_id)
    done = skipped = failed = 0
    for row in rows[:limit] if limit else rows:
        path = _os.path.join(str(base), row.filename or '')
        if not row.filename or not _os.path.isfile(path):
            failed += 1
            continue
        try:
            with Image.open(path) as im:
                metrics = quality_metrics(im.convert('RGB'))
            dhash_hex = _dhash_file(path)
            sig = hashlib.sha1(str(_os.path.getsize(path)).encode())
            with open(path, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b''):
                    sig.update(chunk)
            if not dhash_hex:
                failed += 1
                continue
            row.bank_analysis_snapshot = _snapshot_v2(
                metrics, dhash_hex, hashlib.sha256(
                    (row.filename + dhash_hex).encode()).hexdigest())
            row.content_sig = sig.hexdigest()[:16]
            st = _os.stat(path)
            row.content_sig_stat = '%d:%d' % (st.st_size, st.st_mtime_ns)
            done += 1
        except Exception:
            failed += 1
    db.session.commit()
    if _cluster_path(dataset_id) and _os.path.isfile(_cluster_path(dataset_id)):
        _os.remove(_cluster_path(dataset_id))     # 指纹变了，组内缓存必须失效
    return {'analysed': done, 'failed': failed, 'images': len(rows)}


# ── 落库：per-image 事实写进 face_dataset_image，汇总写进 dataset_score_report ──
def persist_cluster(dataset_id, per_image, median, p10, identities, multi_identity,
                    fingerprint=None, outliers=None):
    """把组内一致性结果写进数据库（不是只写 JSON 缓存）。

    per-image 的 cluster_sim / cluster_sim_max / identity_group 落在
    face_dataset_image 上，因此 UI 能直接 SQL 排序、筛选"这张离群不离群"；
    数据集级汇总量落在 dataset_score_report（同一 dataset 只有一行，覆盖写）。
    """
    import json
    from datetime import datetime
    from ..extensions import db
    from ..models import DatasetScoreReport

    by_id = {str(r.id): r for r in _pool(dataset_id)}
    for iid, detail in (per_image or {}).items():
        row = by_id.get(str(iid))
        if row is None or not isinstance(detail, dict):
            continue
        row.cluster_sim = detail.get('sim')
        row.cluster_sim_max = detail.get('sim_max')
        row.identity_group = detail.get('identity_group')
        if detail.get('det') is not None:
            row.face_det = detail.get('det')
        if detail.get('bbox_frac') is not None:
            row.face_bbox = detail.get('bbox_frac')
        if detail.get('dhash'):
            row.dhash = detail.get('dhash')
        if detail.get('yaw') is not None:
            row.face_yaw = detail.get('yaw')      # 组内模式也测了头姿，落库供成因判定
    for o in (outliers or []):
        row = by_id.get(str(o['id']))
        if row is not None:
            row.score_cause = o.get('cause')
    report = db.session.get(DatasetScoreReport, dataset_id)
    if report is None:
        report = DatasetScoreReport(dataset_id=dataset_id)
        db.session.add(report)
    report.fingerprint = fingerprint or _fingerprint(list(by_id.values()))
    report.median = median
    report.p10 = p10
    report.multi_identity = bool(multi_identity)
    report.identities_json = json.dumps(identities or [], ensure_ascii=False)
    report.groups_json = json.dumps(_group_counts(outliers or []), ensure_ascii=False)
    report.computed_at = datetime.utcnow()
    db.session.commit()
    return {'rows': len(per_image or {}), 'causes': len(outliers or [])}


def persist_score(dataset_id, report, fingerprint=None):
    """把总评分（total/measured/六维）落库，供 UI 列表排序与"上次算过"展示。"""
    import json
    from datetime import datetime
    from ..extensions import db
    from ..models import DatasetScoreReport

    row = db.session.get(DatasetScoreReport, dataset_id)
    if row is None:
        row = DatasetScoreReport(dataset_id=dataset_id)
        db.session.add(row)
    row.total = report.get('total')
    row.measured = report.get('measured')
    row.dims_json = json.dumps(report.get('dims') or {}, ensure_ascii=False)
    if fingerprint:
        row.fingerprint = fingerprint
    row.computed_at = datetime.utcnow()
    db.session.commit()
    return True


def stored_cluster(dataset_id):
    """从数据库读回上次落库的组内一致性（图片没变才认）。"""
    import json
    from ..models import DatasetScoreReport

    rep = DatasetScoreReport.query.get(dataset_id)
    if rep is None or rep.median is None:
        return None
    rows = _pool(dataset_id)
    if rep.fingerprint and rep.fingerprint != _fingerprint(rows):
        return None
    per_image = {}
    for r in rows:
        if r.cluster_sim is None:
            continue
        per_image[str(r.id)] = {'sim': r.cluster_sim, 'sim_max': r.cluster_sim_max,
                                'identity_group': r.identity_group,
                                'det': r.face_det, 'bbox_frac': r.face_bbox,
                                'yaw': r.face_yaw,
                                'dhash': r.dhash or _dhash(r)}
    try:
        identities = json.loads(rep.identities_json or '[]')
    except ValueError:
        identities = []
    return {'cache_v': CLUSTER_CACHE_V, 'median': rep.median, 'p10': rep.p10,
            'multi_identity': bool(rep.multi_identity), 'identities': identities,
            'per_image': per_image, 'from': 'db'}


def mark_causes(user_id, dataset_id, causes=('duplicate',), dry_run=True, force=False):
    """按**成因**打删除标识（只动指定的成因，drift/profile 默认不动）。

    依赖上次落库的 score_cause（cluster_pass 写进去的），所以这一步不需要重新推理，
    是"点击即执行"的那类动作。护栏同 mark_low_similarity：超过比例拒绝批量打标。
    """
    from .face_dataset_service import get_dataset, set_image_status
    from ..extensions import db

    if not get_dataset(user_id, dataset_id):
        raise ValueError('dataset not found')
    wanted_causes = {str(c) for c in (causes or []) if c}
    if not wanted_causes:
        raise ValueError('no causes given')
    rows = _pool(dataset_id)
    cands = [r for r in rows if (r.score_cause or '') in wanted_causes and not r.is_locked]
    guard = None
    if len(rows) and len(cands) / len(rows) > _thr('outlier_guard', OUTLIER_GUARD) and not force:
        guard = ('%d/%d 张（%.0f%%）命中成因 %s — 超过护栏比例，怀疑是整批问题；'
                 '确认后带 force=true' % (len(cands), len(rows),
                                        100 * len(cands) / len(rows), sorted(wanted_causes)))
    if dry_run:
        return {'dry_run': True, 'causes': sorted(wanted_causes), 'count': len(cands),
                'guard': guard,
                'ids': [r.id for r in cands][:300]}
    if guard:
        return {'dry_run': False, 'blocked': True, 'reason': guard, 'count': len(cands)}
    changed = 0
    for r in cands:
        if set_image_status(user_id, r.id, 'reject'):
            changed += 1
    for r in cands:
        db.session.refresh(r)
        if r.status == 'reject':
            r.fail_reason = 'cause %s (marked by rule)' % r.score_cause
    db.session.commit()
    return {'dry_run': False, 'causes': sorted(wanted_causes),
            'marked': changed, 'count': len(cands),
            'undo': 'status 改回 keep；删除用 purge'}


def purge_preview(dataset_id):
    """「一键删除」会删掉什么：reject/failed、未锁定、且不是小图救援派生对。"""
    from ..models import FaceDatasetImage
    from .face_dataset_service import _SMALL_IMAGE_DERIVATIONS
    q = (FaceDatasetImage.query
         .filter(FaceDatasetImage.dataset_id == dataset_id)
         .filter(FaceDatasetImage.status.in_(('reject', 'failed')))
         .filter(FaceDatasetImage.is_locked.isnot(True))
         .filter(FaceDatasetImage.derivation_kind.notin_(_SMALL_IMAGE_DERIVATIONS)
                 | FaceDatasetImage.derivation_kind.is_(None)))
    rows = q.all()
    return {'count': len(rows), 'ids': [r.id for r in rows][:300],
            'filenames': [r.filename for r in rows][:50]}
