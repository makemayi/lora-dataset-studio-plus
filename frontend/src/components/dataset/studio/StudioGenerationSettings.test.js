import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const source = readFileSync(new URL('./StudioGenerationSettings.jsx', import.meta.url), 'utf8');

// Node's built-in test runner does not parse JSX. Evaluate only the deliberately
// pure helper block so these tests exercise the production migration logic.
const startMarker = '// KREA_WEIGHT_DTYPE_HELPERS_START';
const endMarker = '// KREA_WEIGHT_DTYPE_HELPERS_END';
const helperSource = source.slice(
  source.indexOf(startMarker) + startMarker.length,
  source.indexOf(endMarker),
).replaceAll('export ', '');
const context = {};
vm.runInNewContext(
  `${helperSource}\nglobalThis.resolve = resolveKreaWeightDtype;`,
  context,
);

const resolve = context.resolve;

test('missing precision storage defaults Krea Test Studio to fp8 e4m3fn', () => {
  assert.equal(resolve(null, null), 'fp8_e4m3fn');
});

test('legacy implicit default migrates to fp8 e4m3fn', () => {
  assert.equal(resolve(null, 'default'), 'fp8_e4m3fn');
});

test('legacy explicit fp8 e5m2 choice is preserved', () => {
  assert.equal(resolve(null, 'fp8_e5m2'), 'fp8_e5m2');
});

test('versioned explicit default choice is preserved', () => {
  assert.equal(resolve('default', 'fp8_e4m3fn'), 'default');
});

test('the component reads the versioned key first and setters only write v2', () => {
  assert.match(source, /resolveKreaWeightDtype\(\s*load\('wdt_v2', null\),\s*load\('wdt', null\)/);
  assert.match(source, /setWeightDtypeS\(v\); save\('wdt_v2', v\)/);
  assert.doesNotMatch(source, /setWeightDtypeS\(v\); save\('wdt', v\)/);
});
