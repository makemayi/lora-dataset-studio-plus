/**
 * Pure helpers for the 🩺 Score panel — kept out of the JSX so node --test can
 * check the decisions that are easy to get wrong:
 *
 *  • an UNMEASURED dimension is `null`, never `0` (a dataset that never ran a
 *    pass must not look bad, and must not look good either);
 *  • a total withheld by the server (`total: null`, measured < 50%) is shown as
 *    "不足", not as 0 and not as a round number;
 *  • the rule buttons come from the DATABASE rows, so what the user clicks is
 *    what the rule table says — including a disabled rule staying disabled.
 */

export const DIM_LABEL = {
  consistency: '一致性',
  diversity: '多样性',
  caption: '标注',
  quality: '质量',
  duplicates: '重复',
  size: '规模',
};

export const CAUSE_LABEL = {
  duplicate: '连拍重复',
  other_identity: '另一个身份',
  drift: '外观漂移',
  profile: '大侧脸',
  weak: '检测勉强',
};

const ORDER = ['consistency', 'diversity', 'caption', 'quality', 'duplicates', 'size'];

/** 六维按固定顺序排列；`null` 保留为 null（未测），不要变成 0。 */
export function dimensionRows(dims) {
  const d = dims || {};
  return ORDER.map((key) => ({
    key,
    label: DIM_LABEL[key],
    value: d[key] === null || d[key] === undefined ? null : Number(d[key]),
  }));
}

/** 总分展示：null → "不足"（服务端在实测权重 < 50% 时故意不给分）。 */
export function totalLabel(score) {
  const total = score?.total;
  if (total === null || total === undefined) return { text: '不足', measured: score?.measured ?? 0 };
  return { text: String(total), measured: score?.measured ?? 0 };
}

/** 成因分组 → 行（数量为 0 的也保留：让"这一类没有"成为一条可读的事实）。 */
export function causeRows(groups) {
  const g = groups || {};
  return Object.keys(CAUSE_LABEL)
    .filter((k) => g[k])
    .map((k) => ({ key: k, label: CAUSE_LABEL[k], count: g[k].count || 0,
      action: g[k].action || '', ids: g[k].ids || [] }));
}

/** 身份分量：多身份是硬伤（必删），单身份 + 外观漂移是软伤（挑子集）。 */
export function identitySummary(score) {
  const ids = score?.identities || [];
  const big = ids.filter((i) => (i.size || 0) >= 3);
  if (!ids.length) return { known: false, multi: false, text: '还没跑组内打分' };
  if (score?.multi_identity || big.length > 1) {
    return { known: true, multi: true,
      text: `检测到 ${big.length} 个身份分量（最大 ${big[0].size} 张）— 可能混了多个人` };
  }
  return { known: true, multi: false,
    text: `单身份（${big[0]?.size ?? 0} 张）— 分数低是外观漂移，不是别人` };
}

/** 规则 → 按钮。只保留表里 enabled 且有 action 的行；dry_run 缺省为真。 */
export function ruleButtons(rules) {
  return (rules || [])
    .filter((r) => r.enabled !== false && r.action)
    .map((r) => ({
      key: r.key,
      label: r.name,
      desc: r.description || '',
      action: r.action,
      dryRun: r.params?.dry_run !== false,
      danger: r.action === 'purge' || r.action === 'mark_low_similarity'
        || r.action === 'mark_causes',
    }));
}

/** 执行结果 → 一行人话（按钮下面那行）。 */
export function applyResultLine(key, res) {
  if (!res) return '';
  if (res.error) return `失败：${res.error}`;
  if (res.blocked) return `已拦截：${res.reason}`;
  if (res.dry_run) {
    if (key === 'pass_purge') return `预演：会删除 ${res.would_delete ?? 0} 张（未删）`;
    if (typeof res.count === 'number') return `预演：命中 ${res.count} 张（未改动）`;
    if (typeof res.analysed === 'number') return `已分析 ${res.analysed} 张`;
    return '预演完成（未改动）';
  }
  if (typeof res.marked === 'number') return `已标记 ${res.marked} 张为剔除（文件还在）`;
  if (typeof res.purged === 'number') return `已删除 ${res.purged} 张`;
  if (typeof res.analysed === 'number') return `已分析 ${res.analysed} 张`;
  if (typeof res.n === 'number') return `组内打分完成：${res.n} 张，中位 ${res.median ?? '-'}`;
  return '完成';
}
