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

test('prose and non-http links around the URLs cost nothing', () => {
  // The links are EXTRACTED from whatever surrounds them rather than each line
  // having to be a bare URL, so junk is not "dropped" — it was never an entry.
  const { items, error } = parsePastedItems(
    `${SIGNED(1)}\n\nnot a url\nftp://nope/x.jpg\n${SIGNED(2)},`)
  assert.equal(error, null)
  assert.deepEqual(items.map((i) => i.url), [SIGNED(1), SIGNED(2)])
})

test('a Markdown export is a usable paste', () => {
  // What a collector that writes .md actually produces: embeds, links, and the
  // same image appearing as both. Requiring bare URLs per line rejected all of
  // it, which is the shape people arrive with.
  const md = [
    '# 小晓夏',
    '',
    `![](${SIGNED(1)})`,
    `[原图](${SIGNED(2)})`,
    `<img src="${SIGNED(3)}">`,
    `see ${SIGNED(1)} again`,     // duplicate, collapses
  ].join('\n')
  const { items, error, dropped } = parsePastedItems(md)
  assert.equal(error, null)
  assert.deepEqual(items.map((i) => i.url), [SIGNED(1), SIGNED(2), SIGNED(3)])
  assert.equal(dropped, 1, 'the repeated image counts once')
})

test('trailing markup never rides along into the URL', () => {
  // A URL that swallowed `)` or `",` 404s in a way that reads as an expired
  // link, which is the most expensive kind of wrong here.
  const { items } = parsePastedItems(`![](${SIGNED(4)}) and "${SIGNED(5)}",`)
  assert.deepEqual(items.map((i) => i.url), [SIGNED(4), SIGNED(5)])
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
