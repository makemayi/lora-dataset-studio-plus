import test from 'node:test'
import assert from 'node:assert/strict'
import { parsePastedItems, PASTE_IMPORT_MAX } from './bankPasteParse.js'

const SIGNED = (n) =>
  `https://signed-cdn.example.com/tos-cn-i-abc/img${n}~noop.jpeg?x-expires=1&x-signature=s${n}`

test('the helper snippet shape is the happy path', () => {
  const { items, error, account } = parsePastedItems(JSON.stringify({
    source: 'douyin', account: 'Somebody',
    items: [{ url: SIGNED(1), title: 'Somebody 1' }, { url: SIGNED(2), title: 'Somebody 2' }],
  }))
  assert.equal(error, null)
  assert.equal(account, 'Somebody')
  assert.deepEqual(items.map((i) => i.title), ['Somebody 1', 'Somebody 2'])
})

test('a bare array works, and so does one URL per line', () => {
  const arr = parsePastedItems(JSON.stringify([SIGNED(1), SIGNED(2)]))
  assert.equal(arr.items.length, 2)
  const lines = parsePastedItems(`${SIGNED(3)}\n${SIGNED(4)}\n`)
  assert.equal(lines.items.length, 2)
  // Titles are generated when the paste carries none — the importer stores them.
  assert.match(lines.items[0].title, /^Pasted 1$/)
})

test('the signed query is preserved, because stripping it 403s', () => {
  const { items } = parsePastedItems(SIGNED(9))
  assert.equal(items[0].url, SIGNED(9))
  assert.match(items[0].url, /x-signature/)
})

test('the same image under two signatures is ONE item', () => {
  // The carousel re-renders each slide, so a post yields the same path several
  // times with a fresh query each time. De-duping on the full URL would keep
  // every copy and download the same picture repeatedly.
  const a = 'https://signed-cdn.example.com/tos/x~noop.jpeg?x-signature=aaa'
  const b = 'https://signed-cdn.example.com/tos/x~noop.jpeg?x-signature=bbb'
  const { items, dropped } = parsePastedItems(`${a}\n${b}`)
  assert.equal(items.length, 1)
  assert.equal(dropped, 1)
})

test('one bad line does not cost you the whole list', () => {
  const { items, dropped, error } = parsePastedItems(
    `${SIGNED(1)}\n\nnot a url\nftp://nope/x.jpg\n${SIGNED(2)},`)
  assert.equal(error, null)
  assert.equal(items.length, 2)
  assert.ok(dropped >= 2)
})

test('an empty or unusable paste says which of the two it is', () => {
  assert.match(parsePastedItems('').error, /Nothing pasted/)
  assert.match(parsePastedItems('   ').error, /Nothing pasted/)
  assert.match(parsePastedItems('hello there').error, /No http\(s\) image links/)
})

test('broken JSON is named as broken JSON, not as "no links"', () => {
  // Pasting a truncated buffer is the common mistake; "no links found" would
  // send someone looking at the wrong thing.
  const { error } = parsePastedItems('{"items":[{"url":"https://x/y.jpg"}')
  assert.match(error, /will not parse/)
})

test('JSON without a list says so rather than silently importing nothing', () => {
  const { error } = parsePastedItems('{"account":"x"}')
  assert.match(error, /no list of images/)
})

test('a runaway paste is capped', () => {
  const many = Array.from({ length: PASTE_IMPORT_MAX + 50 }, (_, i) => SIGNED(i)).join('\n')
  const { items } = parsePastedItems(many)
  assert.equal(items.length, PASTE_IMPORT_MAX)
})
