import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { dollhouseWallGeometry, fitKeyLight, viewPresets, WALL_NAME_RE } from './msaDollhouse'

/** A closed axis-aligned box (12 triangles, outward normals) like export_glb.py's
 * extruded walls, spanning [x0,x1] x [0,h] x [z0,z1]. */
function wallBox(x0: number, x1: number, z0: number, z1: number, h: number): THREE.BufferGeometry {
  const g = new THREE.BoxGeometry(x1 - x0, h, z1 - z0)
  g.translate((x0 + x1) / 2, h / 2, (z0 + z1) / 2)
  return g
}

function faceNormals(g: THREE.BufferGeometry): THREE.Vector3[] {
  const n = g.getAttribute('normal')
  const out: THREE.Vector3[] = []
  for (let i = 0; i < n.count; i += 3) out.push(new THREE.Vector3().fromBufferAttribute(n, i))
  return out
}

describe('dollhouseWallGeometry', () => {
  it('keeps only the room-facing side and the top cap of a wall box', () => {
    // Wall along +X side of a room centred at the origin: its inner face has normal -X.
    const g = dollhouseWallGeometry(wallBox(4, 4.2, -3, 3, 2.5), new THREE.Vector3(0, 0, 0))
    const normals = faceNormals(g)
    // 2 inner-face triangles + 2 top-cap triangles.
    expect(normals).toHaveLength(4)
    expect(normals.filter((n) => n.x < -0.99)).toHaveLength(2)
    expect(normals.filter((n) => n.y > 0.99)).toHaveLength(2)
    // Outer (+X), bottom (-Y) and the end caps (+/-Z, which don't face the centre) are gone.
    expect(normals.some((n) => n.x > 0.99 || n.y < -0.99)).toBe(false)
  })

  it('is non-indexed with flat normals and preserves UVs', () => {
    const src = wallBox(-2.2, -2, -1, 1, 2.5)
    const g = dollhouseWallGeometry(src, new THREE.Vector3(0, 0, 0))
    expect(g.index).toBeNull()
    expect(g.getAttribute('uv')).toBeDefined()
    expect(g.getAttribute('uv').count).toBe(g.getAttribute('position').count)
    // Inner face of a -X wall has normal +X.
    expect(faceNormals(g).filter((n) => n.x > 0.99)).toHaveLength(2)
  })

  it('works on an already non-indexed geometry without UVs', () => {
    const src = wallBox(0, 0.2, 2, 4, 2.5).toNonIndexed()
    src.deleteAttribute('uv')
    const g = dollhouseWallGeometry(src, new THREE.Vector3(0, 0, 0))
    expect(g.getAttribute('uv')).toBeUndefined()
    expect(faceNormals(g).filter((n) => n.z < -0.99)).toHaveLength(2)
  })
})

describe('viewPresets', () => {
  const box = new THREE.Box3(new THREE.Vector3(-2, 0, -4), new THREE.Vector3(6, 3, 2))

  it('puts the 3/4 view outside and above the box on the +X/+Z side, looking at the room', () => {
    const { threeQuarter } = viewPresets(box)
    expect(threeQuarter.position.x).toBeGreaterThan(box.max.x)
    expect(threeQuarter.position.z).toBeGreaterThan(box.max.z)
    expect(threeQuarter.position.y).toBeGreaterThan(box.max.y)
    expect(box.containsPoint(threeQuarter.target)).toBe(true)
  })

  it('puts the top view straight above the box centre', () => {
    const { top } = viewPresets(box)
    const center = box.getCenter(new THREE.Vector3())
    expect(top.position.x).toBeCloseTo(center.x)
    expect(top.position.z).toBeCloseTo(center.z, 2)
    expect(top.position.y).toBeGreaterThan(box.max.y + 2)
    expect(top.target.x).toBeCloseTo(center.x)
    expect(top.target.z).toBeCloseTo(center.z)
  })
})

describe('fitKeyLight', () => {
  it('sizes the shadow frustum to cover the box and aims at its centre', () => {
    const box = new THREE.Box3(new THREE.Vector3(-2, 0, -4), new THREE.Vector3(6, 3, 2))
    const light = new THREE.DirectionalLight()
    fitKeyLight(light, box)
    const center = box.getCenter(new THREE.Vector3())
    expect(light.target.position.distanceTo(center)).toBeLessThan(1e-6)
    expect(light.position.y).toBeGreaterThan(box.max.y)
    const halfDiag = box.getSize(new THREE.Vector3()).length() / 2
    expect(light.shadow.camera.right).toBeGreaterThanOrEqual(halfDiag)
    expect(light.shadow.camera.far).toBeGreaterThan(light.position.distanceTo(center))
  })
})

describe('WALL_NAME_RE', () => {
  it('matches wall nodes only', () => {
    expect(WALL_NAME_RE.test('wall_0')).toBe(true)
    expect(WALL_NAME_RE.test('wall_12')).toBe(true)
    expect(WALL_NAME_RE.test('wall_outline')).toBe(true) // T15g single wall band
    expect(WALL_NAME_RE.test('Planwall_0')).toBe(false)
    expect(WALL_NAME_RE.test('hallway_0visual')).toBe(false)
  })
})
