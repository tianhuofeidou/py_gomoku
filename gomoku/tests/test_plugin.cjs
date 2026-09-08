// Local plugin regressions. Run: node --test gomoku/tests/test_plugin.cjs
const { test, before, after } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const os = require('node:os')
const vm = require('node:vm')
const { EventEmitter } = require('node:events')
const { pathToFileURL } = require('node:url')
const root = path.resolve(__dirname, '../..')
const plugin = path.join(root, 'lib/server')
const available = fs.existsSync(path.join(plugin, 'python-engine.js'))

function harness(live = false) {
  const children = []
  const timers = new Map()
  let nextTimer = 1
  let source = fs.readFileSync(path.join(plugin, 'python-engine.js'), 'utf8')
    .replace(/^import .*\r?\n/gm, '')
    .replace(/^const __dirname = .*\r?\n/m, '')
    .replace('export async function', 'async function')
  const context = {
    __dirname: plugin, join: path.join, dirname: path.dirname,
    process: { env: live ? { ...process.env, DSH_GOMOKU_GA: '0', DSH_GOMOKU_USE_NET: '0' }
      : { PYTHON: 'test-python', DSH_GOMOKU_USE_NET: '1' } },
    aiPlayerOf: () => 2,
    dataFile: () => live ? state.dataFile() : path.join(os.tmpdir(), 'test-gomoku', 'games.json'),
    setTimeout: live ? setTimeout : fn => { const id = nextTimer++; timers.set(id, fn); return id },
    clearTimeout: live ? clearTimeout : id => timers.delete(id),
    createInterface: live ? require('node:readline').createInterface : ({ input }) => input.lines,
    spawn: (cmd, args, options) => {
      if (live) {
        const child = require('node:child_process').spawn(cmd, args, options)
        children.push(child)
        return child
      }
      const child = new EventEmitter()
      child.stdout = new EventEmitter()
      child.stdout.lines = new EventEmitter()
      child.stderr = new EventEmitter()
      child.stdin = new EventEmitter()
      child.stdin.write = data => child.writes.push(data)
      child.stdin.end = () => {}
      child.kill = () => { child.killed = true }
      child.exitCode = null
      child.killed = false
      child.writes = []
      child.options = options
      children.push(child)
      return child
    },
  }
  vm.createContext(context)
  vm.runInContext(source + '\nglobalThis.api = {request, spawnOnce, computeAiMove};', context)
  const respond = (child, data = {}) => {
    const msg = JSON.parse(child.writes.at(-1))
    child.stdout.lines.emit('line', JSON.stringify({ id: msg.id, ok: true, row: 7, col: 8,
      ranked: [{ r: 7, c: 8, score: 123 }], net_used: true, ...data }))
  }
  return { api: context.api, children, timers, respond }
}

test('first persistent request resolves without fallback and preserves ranking', { skip: !available }, async () => {
  const h = harness()
  const result = h.api.computeAiMove({ moves: [] }, 1, 's')
  assert.equal(h.children.length, 1)
  h.respond(h.children[0])
  const move = await result
  assert.equal(move.ranked[0].score, 123)
  assert.equal(move.net_used, true)
  assert.equal(h.children.length, 1)
  assert.equal(h.timers.size, 0)
})

test('late exit from old process does not reject a new request', { skip: !available }, async () => {
  const h = harness()
  const first = h.api.request({ type: 'move' })
  const rejection = assert.rejects(first, /broken/)
  const old = h.children[0]
  old.emit('error', new Error('broken'))
  await rejection
  const second = h.api.request({ type: 'move' })
  old.emit('exit', 1)
  h.respond(h.children[1])
  assert.equal((await second).c, 8)
  assert.equal(h.timers.size, 0)
})

test('single process fallback has same configuration and complete output', { skip: !available }, async () => {
  const h = harness()
  const result = h.api.computeAiMove({ moves: [] }, 1, 's')
  h.children[0].emit('error', new Error('unavailable'))
  await Promise.resolve()
  const once = h.children[1]
  assert.deepEqual(once.options.env, h.children[0].options.env)
  assert.equal(once.options.env.GOMOKU_HOME, path.join(os.tmpdir(), 'test-gomoku'))
  once.stdout.emit('data', JSON.stringify({ row: 4, col: 5, ranked: [{ r: 4, c: 5, score: 7 }], net_used: true }))
  once.emit('close', 0)
  const move = await result
  assert.equal(move.ranked[0].score, 7)
  assert.equal(move.net_used, true)
  assert.equal(h.timers.size, 0)
})

test('both subprocess paths time out and kill the stalled process', { skip: !available }, async () => {
  for (const method of ['request', 'spawnOnce']) {
    const h = harness()
    const result = h.api[method]({ type: 'move' })
    const rejection = assert.rejects(result, /timeout/)
    h.timers.values().next().value()
    await rejection
    assert.equal(h.children[0].killed, true)
  }
})

let storage, state, memory
const oldHome = process.env.DSH_HOME
before(async () => {
  if (!available) return
  storage = fs.mkdtempSync(path.join(os.tmpdir(), 'gomoku-plugin-test-'))
  process.env.DSH_HOME = storage
  state = await import(pathToFileURL(path.join(plugin, 'state.js')).href)
  memory = await import(pathToFileURL(path.join(plugin, 'memory.js')).href)
})
after(() => {
  if (oldHome === undefined) delete process.env.DSH_HOME
  else process.env.DSH_HOME = oldHome
  if (storage) fs.rmSync(storage, { recursive: true, force: true })
})

test('plugin terminal undo reverses one result and preserves another identical game', { skip: !available }, () => {
  const gm = memory.globalMemory()
  Object.assign(gm, { totals: { wins: 0, losses: 0, draws: 0 }, goodLines: [], badLines: [], lossByType: {} })
  function finish(sid) {
    const g = state.gameOf(sid)
    for (const [r, c, p] of [[7,0,1],[0,0,2],[7,1,1],[0,2,2],[7,2,1],[0,4,2],[7,3,1],[0,6,2],[7,4,1]]) {
      assert.equal(state.applyMove(g, r, c, p).ok, true)
    }
    return g
  }
  finish('other')
  finish('one')
  state.games.clear()
  state.loadAll()
  const g = state.gameOf('one')
  state.undoLastMove(g)
  assert.equal(gm.totals.losses, 1)
  assert.equal(gm.badLines.length, 1)
  assert.equal(g.history.length, 0)
  assert.equal(g.archived, false)
  assert.equal(state.applyMove(g, 7, 4, 1).ok, true)
  assert.equal(gm.totals.losses, 2)
  assert.equal(g.history.length, 1)
})

test('real Node to Python first request uses persistent engine and blocks forced five', { skip: !available }, async () => {
  const h = harness(true)
  try {
    const moves = [[7,4,1],[7,3,2],[7,5,1],[1,1,2],[7,6,1],[3,1,2],[7,7,1]]
      .map(([r,c,player]) => ({r,c,player}))
    const move = await h.api.computeAiMove({ moves }, 2, 'live-test')
    assert.equal(h.children.length, 1)
    assert.equal(move.r, 7)
    assert.equal(move.c, 8)
    assert.equal(move.ranked.length, 1)
    assert.equal(move.ranked[0].score, 1e9)
  } finally {
    await Promise.all(h.children.map(child => new Promise(resolve => {
      if (child.exitCode !== null) return resolve()
      child.once('close', resolve)
      child.kill()
    })))
  }
})
