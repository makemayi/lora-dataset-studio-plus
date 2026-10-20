"""判定规则表（score_rule）：阈值/权重/处置动作全部落库，可点击执行。

为什么不是常量：用户要的是「判定逻辑落库」——
  * 改阈值不用改代码、不用重启（每次执行都从表里读）；
  * 每条规则自带 action，前端把它渲染成按钮；
  * 同一份参数既被按钮用，也被脚本 / API 用，不会出现两套口径；
  * updated_at 让「上次是谁把 0.55 改成 0.45」有据可查。

表在首次访问时按 DEFAULTS 播种（幂等：只补缺的 key，不覆盖用户改过的值）。
"""
from __future__ import annotations

import json
import logging

from ..extensions import db
from ..models import ScoreRule

log = logging.getLogger(__name__)

# key -> (name, category, params, action, description, sort)
DEFAULTS = {
    # ── 阈值 ────────────────────────────────────────────────────────────────
    'face_floor': ('一致性可信下限', 'threshold', {'value': 0.2},
                   None, '余弦低于它视为不可信（face_similarity.reference_agreement_floor）', 10),
    'cluster_min_floor': ('组内一致性绝对下限', 'threshold', {'value': 0.55},
                          None, '无参考图时，组内中位余弦低于它即判为离群（绝对线）', 11),
    'cluster_drop': ('组内相对落差', 'threshold', {'value': 0.15},
                     None, '离群线 = max(绝对下限, 组内中位数 − 这个值)：大集里 0.5 正常，小集里就很怪', 12),
    'identity_threshold': ('身份分量阈值', 'threshold', {'value': 0.35},
                           None, '余弦 ≥ 它就连边；≥2 个各含 ≥3 张的分量才算「混了多个人」', 13),
    'dup_hamming': ('连拍哈希距离', 'threshold', {'value': 10},
                    None, '感知哈希汉明距离 ≤ 它 → 同一场景连拍（不用脸相像）', 14),
    'dup_sim': ('连拍身份相似度', 'threshold', {'value': 0.95},
                None, '身份最近邻 ≥ 它也算连拍（同一帧）', 15),
    'profile_yaw': ('侧脸角度上限', 'threshold', {'value': 50.0},
                    None, '|yaw| > 它 → 归 profile（角度变化，保留）', 16),
    'weak_det': ('检测置信下限', 'threshold', {'value': 0.70},
                 None, 'det < 它 → 归 weak（遮挡/低光，人工看一眼）', 17),
    'weak_bbox': ('人脸面积下限', 'threshold', {'value': 0.05},
                  None, '人脸面积占图比 < 它 → 归 weak（脸太小）', 18),
    'outlier_guard': ('离群比例护栏', 'threshold', {'value': 0.30},
                      None, '离群超过这个比例就拒绝批量打标：那是整个集不一致，不是个别烂图', 19),
    'recommended_images': ('推荐张数', 'threshold', {'value': 20},
                           None, '规模维度的满分线（preflight 的 recommended）', 20),
    'low_sim_threshold': ('低分打标阈值', 'threshold', {'value': 0.55},
                          None, '参考模式下 face_score < 它 → 低分候选', 21),
    # ── 权重 ────────────────────────────────────────────────────────────────
    'weight_consistency': ('权重·一致性', 'weight', {'value': 30}, None, '参考/组内模式的相似度分', 30),
    'weight_diversity': ('权重·多样性', 'weight', {'value': 20}, None, '构图熵 + 场景词种类', 31),
    'weight_caption': ('权重·标注', 'weight', {'value': 20}, None, '覆盖率 + 长度 + 身份泄漏', 32),
    'weight_quality': ('权重·质量', 'weight', {'value': 15}, None, '清晰度 + 分辨率 + 水印', 33),
    'weight_duplicates': ('权重·重复', 'weight', {'value': 10}, None, '完全重复 + 近重复对', 34),
    'weight_size': ('权重·规模', 'weight', {'value': 5}, None, '张数 / 推荐张数', 35),
    # ── 可点击执行的流程 ────────────────────────────────────────────────────
    'pass_analyse_images': ('① 分析画质与指纹', 'pass', {'force': False},
                            'analyse_images', '本地 PIL：清晰度/噪点/均匀度 + 感知哈希 + 内容指纹（无模型、不下载）', 40),
    'pass_analyze_faces': ('② 参考模式人脸打分', 'pass', {'threshold': 0.55},
                           'analyze_faces', 'InsightFace 对参考脸算相似度（CPU）；too_small/侧脸也给分', 41),
    'pass_score_cluster': ('③ 组内一致性打分', 'pass', {'identity_threshold': 0.35},
                           'score_cluster', '无参考图的集：每张与其它所有图的余弦中位 + 身份分量（CPU）', 42),
    'pass_mark_low': ('④ 低分打标（可撤销）', 'action', {'threshold': 0.55, 'scope': 'generated',
                                                     'dry_run': True, 'force': False},
                      'mark_low_similarity', '把低相似度的图标成 reject（不删文件）；超过护栏比例会拒绝', 43),
    'pass_mark_causes': ('⑤ 按成因打标', 'action', {'causes': ['duplicate', 'other_identity'],
                                              'dry_run': True},
                         'mark_causes', '只对指定成因（连拍冗余 / 另一个身份）打标；drift 与 profile 默认不动', 44),
    'pass_purge': ('⑥ 删除已标记图', 'action', {'dry_run': True},
                   'purge', '真正删掉 reject/failed 的文件（跳过锁定图）', 45),
}

APPLIERS = ('analyse_images', 'analyze_faces', 'score_cluster',
            'mark_low_similarity', 'mark_causes', 'purge')


def _seed():
    """幂等播种：只补缺失的 key，绝不覆盖用户改过的值。"""
    existing = {r.key for r in ScoreRule.query.all()}
    added = 0
    for key, (name, category, params, action, desc, sort) in DEFAULTS.items():
        if key in existing:
            continue
        db.session.add(ScoreRule(key=key, name=name, category=category,
                                 params=json.dumps(params, ensure_ascii=False),
                                 action=action, description=desc, sort=sort,
                                 enabled=True))
        added += 1
    if added:
        db.session.commit()
        log.info('score_rules: seeded %d rule(s)', added)
    return added


def all_rules(seed=True):
    if seed:
        _seed()
    rows = ScoreRule.query.order_by(ScoreRule.sort, ScoreRule.key).all()
    out = []
    for r in rows:
        try:
            params = json.loads(r.params or '{}')
        except ValueError:
            params = {}
        out.append({'key': r.key, 'name': r.name, 'category': r.category,
                    'params': params, 'action': r.action, 'description': r.description,
                    'enabled': bool(r.enabled), 'sort': r.sort,
                    'updated_at': r.updated_at.isoformat() if r.updated_at else None})
    return out


def params_of(key, fallback=None):
    """读一条规则的参数；表还没建/没播种/被停用 → 回退到 DEFAULTS。"""
    try:
        row = db.session.get(ScoreRule, key) or ScoreRule.query.get(key)
    except Exception:
        row = None
    if row is None:
        return (DEFAULTS.get(key) or (None, None, fallback))[2] if key in DEFAULTS else (fallback or {})
    try:
        return json.loads(row.params or '{}')
    except ValueError:
        return (DEFAULTS.get(key) or (None, None, fallback))[2]


def value_of(key, default):
    """阈值/权重的便捷读法：{"value": x} → x。"""
    try:
        v = (params_of(key) or {}).get('value')
    except Exception:
        return default
    return default if v is None else v


def update_rule(key, params=None, enabled=None):
    if key not in DEFAULTS:
        raise ValueError('unknown rule: %s' % key)
    _seed()
    row = db.session.get(ScoreRule, key)
    if row is None:
        raise ValueError('rule not found: %s' % key)
    if params is not None:
        if not isinstance(params, dict):
            raise ValueError('params must be an object')
        merged = json.loads(row.params or '{}')
        merged.update(params)
        if 'value' in merged:
            try:
                float(merged['value'])
            except (TypeError, ValueError):
                raise ValueError('params.value must be a number')
        row.params = json.dumps(merged, ensure_ascii=False)
    if enabled is not None:
        row.enabled = bool(enabled)
    from datetime import datetime
    row.updated_at = datetime.utcnow()
    db.session.commit()
    return {'key': key, 'params': json.loads(row.params or '{}'),
            'enabled': bool(row.enabled)}


def apply_rule(user_id, dataset_id, key, dry_run=None):
    """点击执行：按表里存的参数跑对应动作。"""
    if key not in DEFAULTS:
        raise ValueError('unknown rule: %s' % key)
    _seed()
    row = db.session.get(ScoreRule, key)
    if row is None or not row.enabled:
        raise ValueError('rule disabled or missing: %s' % key)
    action = row.action or DEFAULTS[key][3]
    if action not in APPLIERS:
        raise ValueError('rule %s has no runnable action' % key)
    params = json.loads(row.params or '{}')
    if dry_run is not None and 'dry_run' in params:
        params['dry_run'] = bool(dry_run)

    from . import dataset_score as ds
    if action == 'analyse_images':
        return {'action': action, **ds.analyse_images(user_id, dataset_id)}
    if action == 'analyze_faces':
        from .face_scoring_service import analyze_faces
        counts, err = analyze_faces(user_id, dataset_id)
        return {'action': action, 'states': counts, 'scoring_error': err,
                'analysed': sum((counts or {}).values())}
    if action == 'score_cluster':
        return {'action': action, **ds.cluster_pass(user_id, dataset_id)}
    if action == 'mark_low_similarity':
        return {'action': action, **ds.mark_low_similarity(
            user_id, dataset_id,
            threshold=float(params.get('threshold', 0.55)),
            scope=str(params.get('scope') or 'generated'),
            dry_run=bool(params.get('dry_run', True)),
            force=bool(params.get('force', False)))}
    if action == 'mark_causes':
        return {'action': action, **ds.mark_causes(
            user_id, dataset_id,
            causes=params.get('causes') or ['duplicate'],
            dry_run=bool(params.get('dry_run', True)))}
    if action == 'purge':
        from .face_dataset_service import purge_unused
        want_dry = params.get('dry_run', True) if dry_run is None else bool(dry_run)
        if want_dry:
            preview = ds.purge_preview(dataset_id)
            return {'action': action, 'dry_run': True,
                    'would_delete': preview['count'],
                    'ids': preview['ids'][:50],
                    'note': '确认后带 dry_run=false 再点一次才会真删'}
        return {'action': action, 'purged': purge_unused(user_id, dataset_id)}
    raise ValueError('unreachable')
