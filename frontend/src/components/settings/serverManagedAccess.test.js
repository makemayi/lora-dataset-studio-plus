import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

const source = readFileSync(new URL('./ServerSection.jsx', import.meta.url), 'utf8')
const helperBlock = source.match(
  /\/\* managed-access:start[\s\S]*?\*\/([\s\S]*?)\/\* managed-access:end \*\//,
)?.[1]

assert.ok(helperBlock, 'managed access helpers must stay available to the unit test')

const helpers = Function(
  `${helperBlock.replace(/\bexport\s+/g, '')}\nreturn { managedBrowserAccess, resolveServerAccess, shouldShowRemoteControls }`,
)()
const { managedBrowserAccess, resolveServerAccess, shouldShowRemoteControls } = helpers

test('managed access uses the browser origin instead of container addresses', () => {
  const access = resolveServerAccess({
    bindManaged: true,
    browserOrigin: 'http://studio-host.local:8080',
    configHost: '0.0.0.0',
    configPort: 5050,
    runtimeLanIp: '172.19.0.2',
    runtimePort: 5050,
  })

  assert.deepEqual(access, {
    origin: 'http://studio-host.local:8080',
    host: 'studio-host.local',
    port: 8080,
    lan: true,
    lanIp: null,
  })
})

test('managed loopback origins never claim LAN availability', () => {
  for (const origin of [
    'http://localhost:8080',
    'http://localhost.:8080',
    'http://127.0.0.1:8080',
    'http://127.42.0.7:8080',
    'http://[::1]:8080',
    'http://[::ffff:127.0.0.1]:8080',
    'http://[::ffff:127.42.3.4]:8080',
  ]) {
    const access = resolveServerAccess({
      bindManaged: true,
      browserOrigin: origin,
      configHost: '0.0.0.0',
      configPort: 5050,
      runtimeLanIp: '172.19.0.2',
    })

    assert.equal(access.lan, false, origin)
    assert.equal(access.port, 8080, origin)
    assert.equal(access.lanIp, null, origin)
  }
})

test('managed mapped non-loopback IPv4 origins remain network addresses', () => {
  const access = resolveServerAccess({
    bindManaged: true,
    browserOrigin: 'http://[::ffff:192.168.1.10]:8080',
    configHost: '0.0.0.0',
    configPort: 5050,
    runtimeLanIp: '172.19.0.2',
  })

  assert.equal(access.lan, true)
  assert.equal(access.origin, 'http://[::ffff:c0a8:10a]:8080')
})

test('loopback classification does not rewrite the browser origin used for links', () => {
  const access = managedBrowserAccess('http://localhost.:8080')
  assert.equal(access.lan, false)
  assert.equal(access.hostname, 'localhost')
  assert.equal(access.origin, 'http://localhost.:8080')
})

test('managed loopback keeps token controls available without offering a QR link', () => {
  const access = resolveServerAccess({
    bindManaged: true,
    browserOrigin: 'http://127.0.0.1:5050',
    configHost: '0.0.0.0',
    configPort: 5050,
    runtimeLanIp: '172.19.0.2',
  })

  assert.equal(access.lan, false)
  assert.equal(shouldShowRemoteControls(true, access.lan), true)
  assert.equal(shouldShowRemoteControls(false, access.lan), false)
  assert.match(source, /\{showRemoteControls && \([\s\S]*?Require an access token/)
  assert.match(source, /\{lan && \([\s\S]*?Open it on your phone/)
})

test('managed browser-origin fallback is safe when no HTTP origin exists', () => {
  for (const origin of [null, undefined, '', 'null', 'file:///tmp/index.html', 'not a URL']) {
    assert.equal(managedBrowserAccess(origin), null)
  }

  assert.deepEqual(resolveServerAccess({
    bindManaged: true,
    browserOrigin: null,
    configHost: '0.0.0.0',
    configPort: 5050,
    runtimeLanIp: '172.19.0.2',
  }), {
    origin: null,
    host: null,
    port: null,
    lan: false,
    lanIp: null,
  })
})

test('desktop access behavior still uses saved bind and detected host address', () => {
  assert.deepEqual(resolveServerAccess({
    bindManaged: false,
    browserOrigin: 'http://unrelated.example:8080',
    configHost: '0.0.0.0',
    configPort: 5050,
    runtimeLanIp: '192.168.1.25',
  }), {
    origin: null,
    host: '0.0.0.0',
    port: 5050,
    lan: true,
    lanIp: '192.168.1.25',
  })

  assert.equal(resolveServerAccess({
    bindManaged: false,
    browserOrigin: 'http://studio-host.local:8080',
    configHost: '127.0.0.1',
    configPort: 5050,
    runtimeLanIp: '192.168.1.25',
  }).lan, false)
})

test('managed UI wires URLs and status to browser access, never runtime LAN data', () => {
  assert.match(source, /browserOrigin: typeof window === 'undefined' \? null : window\.location\?\.origin/)
  assert.match(source, /const port = access\.port/)
  assert.match(source, /lanIp: null/)
  assert.match(source, /url: `\$\{access\.origin\}\/\$\{tokenQS\}`/)
  assert.match(source, /Opened at:[\s\S]{0,120}\{access\.origin\}/)
  assert.match(source, /Current browser address uses a network host/)
  assert.match(source, /That does not reveal whether Docker is exposed/)
  assert.match(source, /any device that can reach this Docker service/)
  assert.doesNotMatch(source, /Change the Docker host mapping to expose it/)
})
