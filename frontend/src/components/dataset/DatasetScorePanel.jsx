import { useCallback, useEffect, useState } from 'react';
import { apiFetch } from '../../api/fetchClient';
import {
  applyResultLine, causeRows, dimensionRows, identitySummary, ruleButtons, totalLabel,
} from './datasetScore.js';

/** 🩺 Score — one number, six dimensions, and the buttons that move them.
 *
 * The number is deliberately honest: a dimension the app never measured is shown
 * as "未测" (not 0), and the server withholds the total entirely when less than
 * half the weight was measured (a 100/100 built from two dimensions is worse
 * than no number). Every button here is a `score_rule` row from the database, so
 * changing a threshold is a data change, not a code change — and clicking runs
 * the same rule the API runs.
 */
const DIM_TONE = (v) => (v === null ? 'text-content-subtle'
  : v >= 80 ? 'bg-green-500' : v >= 60 ? 'bg-amber-400' : 'bg-rose-400');

function DimBar({ row }) {
  const v = row.value;
  return (
    <div className="flex items-center gap-2">
      <span className="w-14 shrink-0 text-[0.6875rem] text-content-muted">{row.label}</span>
      <span aria-hidden className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-raised">
        {v !== null && <span className={`block h-full ${DIM_TONE(v)}`} style={{ width: `${v}%` }} />}
      </span>
      <span className={`w-8 shrink-0 text-right text-[0.6875rem] tabular-nums ${DIM_TONE(v).replace('bg-', 'text-')}`}>
        {v === null ? '未测' : v}
      </span>
    </div>
  );
}

export default function DatasetScorePanel({ datasetId, refreshKey = 0 }) {
  const [open, setOpen] = useState(false);
  const [score, setScore] = useState(null);
  const [rules, setRules] = useState([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState('');
  const [lines, setLines] = useState({});

  const load = useCallback(async () => {
    if (!datasetId) return;
    setError('');
    try {
      const [s, r] = await Promise.all([
        apiFetch(`/api/dataset/${datasetId}/score`),
        apiFetch('/api/score-rules'),
      ]);
      setScore(s);
      setRules(r?.rules || []);
    } catch (e) {
      setError(e?.message || '无法读取评分');
    }
  }, [datasetId]);

  useEffect(() => { if (open) load(); }, [open, refreshKey, load]);

  const run = async (btn) => {
    if (busy) return;
    setBusy(btn.key);
    try {
      const res = await apiFetch(`/api/dataset/${datasetId}/apply-rule/${btn.key}`, {
        method: 'POST',
        body: JSON.stringify({ dry_run: btn.dryRun }),
      });
      setLines((prev) => ({ ...prev, [btn.key]: applyResultLine(btn.key, res) }));
      await load();
    } catch (e) {
      setLines((prev) => ({ ...prev, [btn.key]: `失败：${e?.message || '执行出错'}` }));
    } finally {
      setBusy('');
    }
  };

  const total = totalLabel(score);
  const identity = identitySummary(score);
  const rows = dimensionRows(score?.dims);
  const causes = causeRows(score?.low_similarity ? {} : {});
  const groups = score?.groups ? causeRows(score.groups) : causes;
  const buttons = ruleButtons(rules);

  return (
    <section id="ds-score" tabIndex={-1} className="scroll-mt-20 rounded-xl border border-border bg-surface-raised">
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
        className="flex w-full items-center gap-2 px-3 py-2 text-left">
        <span aria-hidden>🩺</span>
        <span className="text-sm font-medium text-content">Score</span>
        {score && (
          <span className="text-[0.6875rem] text-content-subtle">
            {total.text}{total.text !== '不足' ? ' / 100' : ''} · 实测 {total.measured}%
          </span>
        )}
        <span aria-hidden className="ml-auto text-content-subtle">{open ? '▾' : '▸'}</span>
      </button>

      {open && (
        <div className="flex flex-col gap-3 border-t border-border px-3 py-3">
          {error && <p className="text-[0.6875rem] text-rose-600">{error}</p>}

          {score && (
            <>
              <div className="flex flex-col gap-1">
                {rows.map((r) => <DimBar key={r.key} row={r} />)}
              </div>
              <p className="text-[0.6875rem] text-content-subtle">
                实测覆盖 {total.measured}% —— 没测过的维度按"未测"处理，不参与总分、也不扣分。
                {score.total === null && ' 覆盖不足一半，服务端故意不给总分。'}
              </p>
              <p className="text-[0.6875rem] text-content-muted">{identity.text}</p>

              {groups.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {groups.map((g) => (
                    <span key={g.key} title={g.action}
                      className="rounded-full border border-border px-2 py-0.5 text-[0.6875rem] text-content-muted">
                      {g.label} <span className="tabular-nums opacity-70">{g.count}</span>
                    </span>
                  ))}
                </div>
              )}

              {(score.top_fixes || []).length > 0 && (
                <ul className="flex flex-col gap-0.5 text-[0.6875rem] text-content-muted">
                  {score.top_fixes.slice(0, 3).map((f, i) => (
                    <li key={`${f.dim}-${i}`}>· {f.text}</li>
                  ))}
                </ul>
              )}
            </>
          )}

          <div className="flex flex-col gap-1">
            <span className="text-[0.6875rem] uppercase tracking-wide text-content-subtle">可执行规则</span>
            <div className="flex flex-wrap gap-1.5">
              {buttons.map((btn) => (
                <button key={btn.key} type="button" title={btn.desc}
                  disabled={!!busy} onClick={() => run(btn)}
                  className={`rounded-md border px-2 py-1 text-[0.6875rem] transition-colors disabled:opacity-50 ${
                    btn.danger
                      ? 'border-rose-400/50 text-rose-700 hover:bg-rose-50'
                      : 'border-border text-content-muted hover:bg-surface'}`}>
                  {busy === btn.key ? '…' : btn.label}
                </button>
              ))}
            </div>
            {Object.entries(lines).map(([k, v]) => v && (
              <p key={k} className="text-[0.6875rem] text-content-subtle">{v}</p>
            ))}
            <p className="text-[0.6875rem] text-content-subtle">
              按钮就是规则表里的一行：阈值改在 /api/score-rules，点同一个按钮即生效。
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
