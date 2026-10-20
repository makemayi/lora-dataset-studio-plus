"""Dataset total score + low-similarity marking.

The two contracts worth pinning, because both were real traps:
  1. an UNMEASURED dimension must not be scored as 0 — otherwise not having run a
     pass reads as a bad dataset;
  2. a total computed from less than half the weight must not be shown at all —
     a 100 out of 100 built from two dimensions is worse than no number;
  3. marking only ever writes `status='reject'` (never deletes), and never touches
     a locked image.
"""
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, 'backend')

from app.services import dataset_score as ds   # noqa: E402


def img(id=1, score=None, state=None, source='generated', locked=False,
        caption='x', framing=None, status='keep'):
    return SimpleNamespace(id=id, face_score=score, face_state=state, source=source,
                           is_locked=locked, caption=caption, framing=framing,
                           status=status, content_sig=None, watermark_state=None,
                           bank_analysis_snapshot=None, filename='f%d.png' % id)


def test_unmeasured_dimensions_are_not_zero():
    """没有画质分析数据 → quality 是 None，不是 0。"""
    rows = [img(1, 0.9, 'scorable')]
    assert ds._score_quality(rows)[0] is None


def test_consistency_weights_low_confidence_states_lower():
    sure = ds._score_consistency([img(1, 0.60, 'scorable')])[0]
    small = ds._score_consistency([img(1, 0.60, 'too_small')])[0]
    assert sure is not None and small is not None
    assert small <= sure, 'too_small 的权重不能高于 scorable'


def test_consistency_none_without_scores():
    assert ds._score_consistency([img(1, None, None)])[0] is None


def test_total_is_withheld_below_half_measured(monkeypatch):
    """只测到 35% 权重 → total=None（实测覆盖率不足）。"""
    rows = [img(1, 0.9, 'scorable')]
    monkeypatch.setattr(ds, '_pool', lambda dataset_id: rows)
    monkeypatch.setattr(ds, '_score_consistency', lambda r, d=None: (100, 1.0, {}))
    monkeypatch.setattr(ds, '_score_diversity', lambda u, d, r: (None, 0.0, {}))
    for fn in ('_score_caption', '_score_quality', '_score_duplicates'):
        monkeypatch.setattr(ds, fn, lambda r, _fn=fn: (None, 0.0, {}))
    report = ds.score('u', 1)
    # consistency(30) + size(5) = 35%
    assert report['measured'] == 35
    assert report['total'] is None


def test_total_uses_available_weights(monkeypatch):
    rows = [img(1, 0.9, 'scorable')]
    monkeypatch.setattr(ds, '_pool', lambda dataset_id: rows)
    monkeypatch.setattr(ds, '_score_consistency', lambda r, d=None: (80, 1.0, {}))
    monkeypatch.setattr(ds, '_score_diversity', lambda u, d, r: (60, 1.0, {}))
    monkeypatch.setattr(ds, '_score_caption', lambda r: (70, 1.0, {}))
    monkeypatch.setattr(ds, '_score_quality', lambda r: (None, 0.0, {}))
    monkeypatch.setattr(ds, '_score_duplicates', lambda r: (100, 1.0, {}))
    report = ds.score('u', 1)
    assert report['measured'] == 85                       # 少 quality 的 15
    # size 维仍然参与（1 张图 → 5 分），所以分母是 85 而不是 80
    assert report['total'] == round((80 * 30 + 60 * 20 + 70 * 20 + 100 * 10 + 5 * 5) / 85)


def test_low_similarity_respects_threshold_scope_and_lock():
    rows = [img(1, 0.40, 'scorable', source='generated'),
            img(2, 0.40, 'scorable', source='import'),       # scope=generated 排除
            img(3, 0.40, 'scorable', locked=True),           # 锁定的不动
            img(4, 0.80, 'scorable'),                        # 分数够
            img(5, None, 'no_face'),                         # 没分
            img(6, 0.30, 'too_small')]                       # 低分但状态低置信，仍算
    ids = [c['id'] for c in ds.low_similarity_ids(rows, 0.55, 'generated')]
    assert ids == [6, 1], ids
    ids_all = [c['id'] for c in ds.low_similarity_ids(rows, 0.55, 'all')]
    assert ids_all == [6, 1, 2], ids_all


def test_mark_dry_run_writes_nothing(monkeypatch):
    rows = [img(1, 0.40, 'scorable')]
    monkeypatch.setattr(ds, '_pool', lambda dataset_id: rows)
    monkeypatch.setattr(ds, 'get_dataset' if hasattr(ds, 'get_dataset') else '_', lambda *a: True,
                        raising=False)
    called = []
    monkeypatch.setattr('app.services.face_dataset_service.get_dataset', lambda u, d: True)
    monkeypatch.setattr('app.services.face_dataset_service.set_image_status',
                        lambda *a, **k: called.append(a) or True)
    res = ds.mark_low_similarity('u', 1, threshold=0.55, scope='generated', dry_run=True)
    assert res['dry_run'] is True and res['count'] == 1
    assert called == [], 'dry_run 绝不能写库'


# ── 无参考图模式（组内一致性）───────────────────────────────────────────────
def _write_cluster(tmp_path, dataset_id, rows, median, per_image, fp=None):
    import json
    p = tmp_path / ('%d.json' % dataset_id)
    p.write_text(json.dumps({
        'cache_v': ds.CLUSTER_CACHE_V,
        'fingerprint': fp if fp is not None else ds._fingerprint(rows),
        'median': median, 'p10': median - 0.1, 'n': len(per_image),
        'identities': [{'group': 1, 'size': len(per_image)}], 'multi_identity': False,
        'per_image': per_image, 'outliers': []}), encoding='utf-8')
    return str(p)


def test_cluster_cache_invalidated_by_content(tmp_path, monkeypatch):
    rows = [img(1, caption='a'), img(2, caption='b')]
    path = _write_cluster(tmp_path, 7, rows, 0.6, {'1': _pi(0.6), '2': _pi(0.6)})
    monkeypatch.setattr(ds, '_cluster_path', lambda d: path)
    assert ds.cluster_cached(7, rows) is not None
    changed = [img(1, caption='a', status='keep'), img(2, caption='c')]
    changed[1].content_sig = 'different'
    assert ds.cluster_cached(7, changed) is None, '图变过就必须失效'


def test_consistency_falls_back_to_cluster(tmp_path, monkeypatch):
    rows = [img(1), img(2), img(3)]
    path = _write_cluster(tmp_path, 7, rows, 0.60,
                          {'1': _pi(0.6), '2': _pi(0.6), '3': _pi(0.6)})
    monkeypatch.setattr(ds, '_cluster_path', lambda d: path)
    value, coverage, detail = ds._score_consistency(rows, 7)
    assert detail['mode'] == 'cluster'
    assert value == round(100 * min(0.60 / 0.75, 1.0))
    assert coverage == 1.0


def test_cluster_outliers_respect_relative_floor(tmp_path, monkeypatch):
    rows = [img(1), img(2), img(3)]
    # 中位 0.70 → 相对下限 0.70-0.15=0.55；0.60 不算离群，0.54 才算
    path = _write_cluster(tmp_path, 7, rows, 0.70,
                          {'1': _pi(0.72), '2': _pi(0.60), '3': _pi(0.54)})
    monkeypatch.setattr(ds, '_cluster_path', lambda d: path)
    ids = [c['id'] for c in ds.low_similarity_ids(rows, 0.4, 'all', dataset_id=7)]
    assert ids == [3], ids


def test_reference_scores_win_over_cluster_cache(tmp_path, monkeypatch):
    """有参考分时不该用组内缓存（两者尺度不同）。"""
    rows = [img(1, 0.80, 'scorable'), img(2)]
    path = _write_cluster(tmp_path, 7, rows, 0.40, {'1': _pi(0.4)})
    monkeypatch.setattr(ds, '_cluster_path', lambda d: path)
    value, _cov, detail = ds._score_consistency(rows, 7)
    assert detail['mode'] == 'reference'
    assert value > 90


# ── 离群成因分类 + 重复簇计划 ────────────────────────────────────────────────
def _pi(sim, dhash=None, det=0.9, frac=0.09, yaw=0.0, sim_max=0.7, group=1):
    return {'sim': sim, 'sim_max': sim_max, 'det': det, 'bbox_frac': frac,
            'yaw': yaw, 'dhash': dhash, 'state': 'scorable', 'identity_group': group}


def test_outliers_are_grouped_by_cause_not_one_threshold():
    rows = [img(1), img(2), img(3), img(4), img(5)]
    per_image = {
        '1': _pi(0.40, dhash='0000000000000000', det=0.9, yaw=5),      # 和 2 连拍
        '2': _pi(0.42, dhash='0000000000000001', det=0.9, yaw=5),
        '3': _pi(0.40, dhash='ffffffffffffffff', det=0.9, yaw=65),     # 大侧脸
        '4': _pi(0.40, dhash='0f0f0f0f0f0f0f0f', det=0.55, yaw=10),    # 检测勉强
        '5': _pi(0.45, dhash='aaaaaaaaaaaaaaaa', det=0.9, yaw=5),      # 真不像
    }
    out = ds.classify_outliers(per_image, rows, floor=0.55)
    by_id = {o['id']: o['cause'] for o in out}
    assert by_id[1] == 'duplicate' and by_id[2] == 'duplicate'
    assert by_id[3] == 'profile'
    assert by_id[4] == 'weak'
    assert by_id[5] == 'drift'          # 单人 + 外观漂移，不是"别人"


def test_a_second_identity_component_is_reported_as_other_identity():
    """多人集的判据是连通分量：另一个 ≥3 张的分量才算"别人"。"""
    rows = [img(i) for i in range(1, 7)]
    per_image = {'1': _pi(0.40, dhash='1' * 16, group=1),
                 '2': _pi(0.41, dhash='2' * 16, group=1),
                 '3': _pi(0.42, dhash='3' * 16, group=1),
                 '4': _pi(0.40, dhash='4' * 16, group=2),
                 '5': _pi(0.41, dhash='5' * 16, group=2),
                 '6': _pi(0.42, dhash='6' * 16, group=2)}
    out = ds.classify_outliers(per_image, rows, floor=0.55)
    causes = {o['id']: o['cause'] for o in out}
    # 最大的分量是 1（`max` 在等大时取先出现者），另一组 ≥3 → other_identity
    assert 'other_identity' in causes.values()
    assert causes[1] == 'drift' and causes[4] == 'other_identity'


def test_duplicate_plan_keeps_the_most_confident_frame():
    rows = [img(1), img(2), img(3)]
    per_image = {
        '1': _pi(0.40, dhash='0000000000000000', det=0.72),
        '2': _pi(0.41, dhash='0000000000000002', det=0.88),   # 最可信 → 留
        '3': _pi(0.42, dhash='0000000000000004', det=0.80),
    }
    outliers = ds.classify_outliers(per_image, rows, floor=0.55)
    plan = ds.duplicate_plan(per_image, outliers)
    assert len(plan) == 1
    assert plan[0]['keep'] == 2
    assert sorted(plan[0]['delete']) == [1, 3]


def test_near_duplicate_uses_the_hash_not_the_face_embedding():
    """夜景连拍的身份相似度只有 0.55-0.68，但感知哈希几乎相同 → 必须判成重复。"""
    rows = [img(1), img(2)]
    per_image = {
        '1': _pi(0.39, dhash='1122334455667788', sim_max=0.55),
        '2': _pi(0.43, dhash='1122334455667789', sim_max=0.68),      # 距离 1 bit
    }
    out = ds.classify_outliers(per_image, rows, floor=0.55)
    assert all(o['cause'] == 'duplicate' for o in out), out


# ── 判定规则落库（score_rule）───────────────────────────────────────────────
def test_rules_seed_is_idempotent_and_keeps_user_values(app):
    from app.services import score_rules as rules
    from app.models import ScoreRule
    with app.app_context():
        rules.all_rules()
        first = ScoreRule.query.count()
        assert first >= 20
        rules.update_rule('cluster_min_floor', params={'value': 0.61})
        rules._seed()               # 再播种一次不能覆盖用户改的值
        assert ScoreRule.query.count() == first
        assert rules.params_of('cluster_min_floor')['value'] == 0.61
        rules.update_rule('cluster_min_floor', params={'value': 0.55})


def test_rule_update_rejects_unknown_and_non_numeric(app):
    from app.services import score_rules as rules
    with app.app_context():
        with pytest.raises(ValueError):
            rules.update_rule('no_such_rule', params={'value': 1})
        with pytest.raises(ValueError):
            rules.update_rule('cluster_min_floor', params={'value': 'abc'})


def test_thresholds_are_read_at_call_time(app):
    """改阈值不该需要重启或重算 —— 服务层每次调用都从表里读。"""
    from app.services import score_rules as rules
    with app.app_context():
        rules.all_rules()
        assert ds._thr('cluster_min_floor', 0.99) == 0.55
        rules.update_rule('cluster_min_floor', params={'value': 0.42})
        try:
            assert ds._thr('cluster_min_floor', 0.99) == 0.42
        finally:
            rules.update_rule('cluster_min_floor', params={'value': 0.55})


def test_mark_causes_only_touches_the_named_cause(monkeypatch):
    rows = [img(1), img(2)]
    rows[0].score_cause = 'duplicate'
    rows[1].score_cause = 'drift'
    monkeypatch.setattr(ds, '_pool', lambda dataset_id: rows)
    monkeypatch.setattr('app.services.face_dataset_service.get_dataset', lambda u, d: True)
    res = ds.mark_causes('u', 1, causes=['duplicate'], dry_run=True)
    assert res['count'] == 1 and res['ids'] == [1]
