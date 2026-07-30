import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

test('unavailable checklist steps deep-link to the ComfyUI API setting', () => {
  const source = readFileSync(new URL('./GuidedChecklist.jsx', import.meta.url), 'utf8');

  assert.match(
    source,
    /<Link to="\/settings\/local-tools\?focus=comfyui-api-url" title=\{s\.hint\} className=\{cls\}>/,
  );
});