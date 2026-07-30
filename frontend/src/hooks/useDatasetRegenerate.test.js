import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const hook = readFileSync(new URL('./useDataset.js', import.meta.url), 'utf8');

function regenerateSource() {
  const start = hook.indexOf('const regenerate = useCallback');
  const end = hook.indexOf('const purgeUnused = useCallback', start);
  return hook.slice(start, end);
}

test('ordinary regenerate reuses the image engine instead of the workspace picker', () => {
  const action = regenerateSource();
  const request = action.slice(action.indexOf('const d = await postJson'),
                               action.indexOf('if (d.ok)'));

  assert.ok(action.includes('`/api/dataset/image/${imageId}/regenerate`'));
  assert.match(request, /lora_strength: loraStrength/);
  assert.match(request, /\.\.\.\(prompt \? \{ prompt \} : \{\}\)/);
  assert.doesNotMatch(request, /\bengine\b|datasetGenerator|localStorage/);
});
