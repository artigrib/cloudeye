import { afterEach, describe, expect, it, vi } from 'vitest'
import { probeGlb } from './glbProbe'
import { resolveSceneLayers } from './layerAvailability'

/** Builds a GLB whose JSON chunk describes `meshes`, so the probe can be tested against
 * real bytes rather than a mocked parser. Only the header and chunk 0 matter here - the
 * probe never reads the binary chunk. */
function glb(doc: unknown): ArrayBuffer {
  const json = new TextEncoder().encode(JSON.stringify(doc))
  const pad = (4 - (json.byteLength % 4)) % 4
  const chunkLen = json.byteLength + pad
  const buf = new ArrayBuffer(20 + chunkLen)
  const dv = new DataView(buf)
  dv.setUint32(0, 0x46546c67, true) // 'glTF'
  dv.setUint32(4, 2, true)
  dv.setUint32(8, buf.byteLength, true)
  dv.setUint32(12, chunkLen, true)
  dv.setUint32(16, 0x4e4f534a, true) // 'JSON'
  new Uint8Array(buf, 20).set(json)
  for (let i = 0; i < pad; i++) new Uint8Array(buf)[20 + json.byteLength + i] = 0x20
  return buf
}

/** Serves `bodies[url]` for Range requests, and 404s for anything else. */
function serve(bodies: Record<string, ArrayBuffer | 'ok'>) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const body = bodies[url]
    if (body === undefined) return { ok: false, status: 404, arrayBuffer: async () => new ArrayBuffer(0) }
    if (body === 'ok') return { ok: true, status: 206, arrayBuffer: async () => new ArrayBuffer(1) }
    const header = String((init?.headers as Record<string, string>)?.Range ?? '')
    const [from, to] = header.replace('bytes=', '').split('-').map(Number)
    return { ok: true, status: 206, arrayBuffer: async () => body.slice(from, to + 1) }
  })
}

const TRIANGLES = glb({
  meshes: [{ primitives: [{ mode: 4, indices: 0, attributes: { POSITION: 1 } }] }],
  accessors: [{ count: 36 }, { count: 24 }],
})
const POINTS = glb({ meshes: [{ primitives: [{ mode: 0, attributes: { POSITION: 0 } }] }], accessors: [{ count: 900 }] })
const NO_MODE = glb({
  meshes: [{ primitives: [{ indices: 0, attributes: { POSITION: 1 } }] }],
  accessors: [{ count: 9 }, { count: 6 }],
})

afterEach(() => vi.unstubAllGlobals())

describe('probeGlb - what the file actually contains', () => {
  it('counts faces in a triangulated export', async () => {
    vi.stubGlobal('fetch', serve({ '/mesh': TRIANGLES }))
    const facts = await probeGlb('/mesh')
    expect(facts).toMatchObject({ ok: true, triangles: true, points: false, faces: 12 })
  })

  it('reports 0 faces for a points-primitive export - hero-74 is one', async () => {
    vi.stubGlobal('fetch', serve({ '/mesh': POINTS }))
    const facts = await probeGlb('/mesh')
    expect(facts).toMatchObject({ ok: true, triangles: false, points: true, faces: 0 })
    expect(facts.reason).toBe('points primitive, 0 faces')
  })

  it('treats an absent mode as TRIANGLES, per the glTF spec', async () => {
    vi.stubGlobal('fetch', serve({ '/mesh': NO_MODE }))
    expect(await probeGlb('/mesh')).toMatchObject({ triangles: true, faces: 3 })
  })

  it('answers "not fetched" for a 404 rather than throwing', async () => {
    vi.stubGlobal('fetch', serve({}))
    expect(await probeGlb('/mesh')).toMatchObject({ ok: false, triangles: false, faces: 0 })
  })

  it('rejects bytes that are not a glTF binary', async () => {
    vi.stubGlobal('fetch', serve({ '/mesh': new ArrayBuffer(64) }))
    expect(await probeGlb('/mesh')).toMatchObject({ ok: false, reason: 'not a glTF binary' })
  })
})

describe('resolveSceneLayers - one source for Points, never the same file twice', () => {
  it('reads /cloud when it answers, and leaves a triangulated /mesh to the Mesh layer', async () => {
    vi.stubGlobal('fetch', serve({ '/cloud': 'ok', '/mesh': TRIANGLES, '/msa': 'ok' }))
    const facts = await resolveSceneLayers({ cloud: '/cloud', mesh: '/mesh', msaGlb: '/msa' })
    expect(facts.pointsSource).toBe('cloud')
    expect(facts.mesh.triangles).toBe(true)
    expect(facts.layout).toBe(true)
    expect(facts.log).toContain('points <- cloud')
  })

  it('falls back to /mesh only when /mesh is points - and then Mesh stays disabled', async () => {
    // hero-74: /cloud 404s, /mesh is the points-primitive GLB.
    vi.stubGlobal('fetch', serve({ '/mesh': POINTS, '/msa': 'ok' }))
    const facts = await resolveSceneLayers({ cloud: '/cloud', mesh: '/mesh', msaGlb: '/msa' })
    expect(facts.pointsSource).toBe('mesh')
    expect(facts.mesh.triangles).toBe(false)
    expect(facts.log).toContain('/cloud 404')
    expect(facts.log).toContain('mesh toggle disabled')
  })

  it('does not hand a triangulated /mesh to Points when /cloud is missing', async () => {
    vi.stubGlobal('fetch', serve({ '/mesh': TRIANGLES }))
    const facts = await resolveSceneLayers({ cloud: '/cloud', mesh: '/mesh', msaGlb: '/msa' })
    expect(facts.pointsSource).toBe('none')
    expect(facts.mesh.triangles).toBe(true)
  })

  it('survives a 404 on /msa-glb and on /mesh - Map and Points are unaffected', async () => {
    vi.stubGlobal('fetch', serve({ '/cloud': 'ok' }))
    const facts = await resolveSceneLayers({ cloud: '/cloud', mesh: '/mesh', msaGlb: '/msa' })
    expect(facts.pointsSource).toBe('cloud')
    expect(facts.mesh.ok).toBe(false)
    expect(facts.layout).toBe(false)
  })

  it('never rejects, whatever the transport does', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('network down') }))
    const facts = await resolveSceneLayers({ cloud: '/cloud', mesh: '/mesh', msaGlb: '/msa' })
    expect(facts.pointsSource).toBe('none')
    expect(facts.mesh.reason).toContain('probe failed')
  })
})
