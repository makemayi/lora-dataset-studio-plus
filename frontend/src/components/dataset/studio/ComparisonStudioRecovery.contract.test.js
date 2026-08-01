import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const source = readFileSync(new URL('./ComparisonStudio.jsx', import.meta.url), 'utf8');

test('comparison launch treats a returned error response as failure, never success', () => {
  const responseAt = source.indexOf("const dResp = await postJson('/api/studio/run', body)");
  const failureAt = source.indexOf('if (!dResp?.ok)', responseAt);
  const successAt = source.indexOf('toast.success(', responseAt);
  const runIdAt = source.indexOf('setRunId(dResp.run_id)', responseAt);

  assert.ok(responseAt >= 0 && failureAt > responseAt);
  assert.ok(successAt > failureAt && runIdAt > failureAt);

  const failureBranch = source.slice(failureAt, successAt);
  assert.match(failureBranch, /await dResp\.json\(\)/);
  assert.match(failureBranch, /setPreflight\(errorBody\?\.studio_missing \|\| null\)/);
  assert.match(failureBranch, /setArchMismatch\(errorBody\?\.studio_arch_mismatch \|\| null\)/);
  assert.match(failureBranch, /toast\.error\(errorBody\?\.error \|\| 'Error on launch'\)/);
  assert.match(failureBranch, /\breturn;/);
  assert.doesNotMatch(failureBranch, /setRunId\(/);
});
