import { readFileSync } from 'node:fs'
import path from 'node:path'
import { describe, expect, it } from 'vitest'
import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import {
  applyMsaOverlayTransform,
  isNearRoomPolygonXZ,
  isTranslucentObjectNode,
  msaOverlayTransform,
  objectClassFromNodeName,
  overlayLocalToCloudFrame,
  roomPolygonToCloudFrame,
  TRANSLUCENT_OBJECT_CLASSES,
  TRANSLUCENT_OPACITY,
  type MsaSceneMeta,
} from './msaLayers'
import { applyDollhouseLook, classifyNode } from '../components/MsaSceneViewer'

// T15i: translucent soft-furnishing classes - mirrors scripts/msa/visibility.py
// (TRANSLUCENT_OBJECT_CLASSES / TRANSLUCENT_OPACITY) and render_perspective's still.
describe('objectClassFromNodeName', () => {
  it('strips the trailing _<index> of the object id prefix (slash-stripped glTF names)', () => {
    expect(objectClassFromNodeName('curtain_0visualpart_0')).toBe('curtain')
    expect(objectClassFromNodeName('curtain_12collisionhull')).toBe('curtain')
    expect(objectClassFromNodeName('bed_0')).toBe('bed')
    expect(objectClassFromNodeName('coffee maker_1visualpart_2')).toBe('coffee maker')
  })

  it('is null for walls, floor, Plan and cloud nodes', () => {
    for (const n of ['wall_0', 'wall_outline', 'floor', 'Planfloor', 'cloud', 'Path']) expect(objectClassFromNodeName(n)).toBeNull()
  })
})

describe('isTranslucentObjectNode', () => {
  it('matches exactly the shared class list, case-insensitively', () => {
    expect([...TRANSLUCENT_OBJECT_CLASSES].sort()).toEqual(['blind', 'curtain', 'drape', 'mirror'])
    expect(TRANSLUCENT_OPACITY).toBe(0.3)
    expect(isTranslucentObjectNode('curtain_0visualpart_0')).toBe(true)
    expect(isTranslucentObjectNode('Drape_3visualpart_1')).toBe(true)
    expect(isTranslucentObjectNode('bed_0visualpart_0')).toBe(false)
    expect(isTranslucentObjectNode('wall_outline')).toBe(false)
  })
})

describe('applyDollhouseLook translucency', () => {
  function mesh(name: string): THREE.Mesh {
    const m = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0xffffff }))
    m.name = name
    return m
  }

  it('makes curtain parts transparent at 0.3 without depth write, leaves the bed opaque', () => {
    const shared = new THREE.MeshStandardMaterial({ color: 0xeeeeee })
    const curtain = mesh('curtain_0visualpart_0')
    const bed = mesh('bed_0visualpart_0')
    curtain.material = shared
    bed.material = shared
    const root = new THREE.Group()
    root.add(curtain, bed)
    applyDollhouseLook(root, new THREE.Vector3())
    const cm = curtain.material as THREE.MeshStandardMaterial
    const bm = bed.material as THREE.MeshStandardMaterial
    expect(cm.transparent).toBe(true)
    expect(cm.opacity).toBeCloseTo(0.3)
    expect(cm.depthWrite).toBe(false)
    expect(cm.userData.msaTranslucent).toBe(true)
    expect(curtain.castShadow).toBe(false)
    // the shared source material was cloned, not mutated - the bed stays opaque
    expect(bm).toBe(shared)
    expect(bm.transparent).toBe(false)
    expect(bm.opacity).toBe(1)
    expect(bed.castShadow).toBe(true)
  })
})

// M7b addition (a)/(b): the real hero GLB nests its 2D plan curves under a "Plan"
// node ("Plan/wall_0", "Plan/floor" - see classifyNode's docstring for why every
// node's own glTF `name` already carries a slash-stripped ancestor prefix) while
// the actual 3D floor/wall MESHES are separate, top-level nodes ("floor",
// "wall_outline" or per-fragment "wall_N"). A regression here (export_glb.py
// nesting the real floor/wall meshes under "Plan" too, or losing their top-level
// names) would silently hide them behind the "plan" layer toggle in both viewers
// (SPEC #8) - this loads two real fixture GLBs through the actual GLTFLoader
// (not just hand-picked strings into classifyNode) and checks every geometry-
// bearing node's post-load name never classifies as 'plan' unless it also has no
// mesh of its own (i.e. is purely a 2D-curve/organizational node).
describe('classifyNode against real loaded GLBs (loader-level, not just string cases)', () => {
  const FIXTURES = [
    // Newer layout (T15g+): single wall_outline band.
    'var/scratch/run-20260906/m7_out/scene.glb',
    // Older layout: per-fragment wall_0/wall_1/... - both must stay un-hidden.
    path.resolve(__dirname, '../../public/msa-fixtures/02_modular_home.glb'),
  ]

  function loadGltfScene(glbPath: string): Promise<THREE.Group> {
    const buf = readFileSync(glbPath)
    // Rebuild the ArrayBuffer via this realm's own constructor rather than
    // slicing Node's Buffer#buffer directly - `instanceof ArrayBuffer` inside
    // GLTFLoader can otherwise fail across the Node/jsdom realm boundary and
    // silently fall through to the "not an ArrayBuffer" branch.
    const arrayBuffer = new ArrayBuffer(buf.byteLength)
    new Uint8Array(arrayBuffer).set(buf)
    const loader = new GLTFLoader()
    return new Promise((resolve, reject) => {
      loader.parse(arrayBuffer, '', (gltf) => resolve(gltf.scene), reject)
    })
  }

  for (const fixture of FIXTURES) {
    it(`floor/wall meshes in ${path.basename(fixture)} are never classified 'plan' after real glTF loading`, async () => {
      let scene: THREE.Group
      try {
        scene = await loadGltfScene(fixture)
      } catch (err) {
        // Large hero fixtures under var/scratch are only present on the
        // machine that generated them, not in a clean checkout/CI - skip
        // rather than fail if genuinely missing (the checked-in
        // msa-fixtures/02_modular_home.glb case must never hit this).
        if (fixture.startsWith('var/scratch') && (err as NodeJS.ErrnoException)?.code === 'ENOENT') return
        throw err
      }

      const meshNodesByLowerName = new Map<string, THREE.Object3D>()
      scene.traverse((obj) => {
        if ((obj as THREE.Mesh).isMesh) meshNodesByLowerName.set(obj.name.toLowerCase(), obj)
      })

      const floor = meshNodesByLowerName.get('floor')
      expect(floor, 'expected a top-level mesh node literally named "floor"').toBeDefined()
      expect(classifyNode(floor!.name)).toBe('mesh')

      const wallMeshNames = [...meshNodesByLowerName.keys()].filter((n) => /^wall_(\d+|outline)$/.test(n))
      expect(wallMeshNames.length, 'expected at least one wall_N/wall_outline mesh node').toBeGreaterThan(0)
      for (const n of wallMeshNames) {
        expect(classifyNode(meshNodesByLowerName.get(n)!.name)).toBe('mesh')
      }
    })
  }
})

// morning-2: the main viewer overlays the yaw-normalized MSA GLB on the UNROTATED
// point cloud by applying the inverse of bootstrap's yaw rotation to the layer group.
describe('msaOverlayTransform', () => {
  /** scripts/msa/geometry.rotate_point_xz, verbatim - bootstrap's rule. */
  function rotatePointXz(x: number, z: number, a: number, [cx, cz]: [number, number]): [number, number] {
    const dx = x - cx
    const dz = z - cz
    return [cx + dx * Math.cos(a) - dz * Math.sin(a), cz + dx * Math.sin(a) + dz * Math.cos(a)]
  }

  // The hero scene's actual T15i meta (yaw 56.76 deg about the room centroid).
  const meta: MsaSceneMeta = {
    yaw_applied: true,
    yaw_correction_rad: 0.9905827249253112,
    yaw_correction_deg: 56.756209396788904,
    yaw_rotation_center_xy: [1.5676628320393422, -3.2437318254474135],
    yaw_method: 'room_polygon_min_area_rect',
  }

  it('returns null when the export was not yaw-rotated', () => {
    expect(msaOverlayTransform(null)).toBeNull()
    expect(msaOverlayTransform({})).toBeNull()
    expect(msaOverlayTransform({ yaw_applied: false, yaw_correction_rad: 0.5 })).toBeNull()
    expect(msaOverlayTransform({ yaw_applied: true, yaw_correction_rad: 0 })).toBeNull()
  })

  it('undoes rotate_point_xz through a real three.js Object3D: bootstrap-rotated xz returns to the original', () => {
    const t = msaOverlayTransform(meta)
    expect(t).not.toBeNull()
    const group = new THREE.Group()
    applyMsaOverlayTransform(group, t)
    group.updateMatrixWorld(true)

    const originals: [number, number, number][] = [
      [0, 0, 0],
      [3.2, 1.1, -5.7],
      [-2.5, 2.9, 0.4],
      [1.5676628320393422, 0.5, -3.2437318254474135], // the pivot itself
    ]
    for (const [x, y, z] of originals) {
      const [rx, rz] = rotatePointXz(x, z, meta.yaw_correction_rad!, meta.yaw_rotation_center_xy!)
      const back = group.localToWorld(new THREE.Vector3(rx, y, rz))
      expect(back.x).toBeCloseTo(x, 9)
      expect(back.y).toBeCloseTo(y, 9)
      expect(back.z).toBeCloseTo(z, 9)
    }
  })

  it('is a pure rotation about Y through the pivot (pivot is a fixed point, y untouched)', () => {
    const group = new THREE.Group()
    applyMsaOverlayTransform(group, msaOverlayTransform(meta))
    group.updateMatrixWorld(true)
    const p = group.localToWorld(new THREE.Vector3(meta.yaw_rotation_center_xy![0], 0, meta.yaw_rotation_center_xy![1]))
    expect(p.x).toBeCloseTo(meta.yaw_rotation_center_xy![0], 9)
    expect(p.z).toBeCloseTo(meta.yaw_rotation_center_xy![1], 9)
    expect(group.rotation.x).toBe(0)
    expect(group.rotation.z).toBe(0)
    expect(group.position.y).toBe(0)
  })

  it('applying null resets a previously transformed group to identity', () => {
    const group = new THREE.Group()
    applyMsaOverlayTransform(group, msaOverlayTransform(meta))
    applyMsaOverlayTransform(group, null)
    expect(group.rotation.y).toBe(0)
    expect(group.position.length()).toBe(0)
  })

  // PM review of the take-1 dry run (2026-09-07): the point cloud spilled outside the
  // room outline on the west side once room_polygon became available - these pin down
  // overlayLocalToCloudFrame/roomPolygonToCloudFrame/isNearRoomPolygonXZ, the clip
  // SceneMap3D.tsx's point-cloud build effect applies.
  it('overlayLocalToCloudFrame agrees with the Object3D transform (same numbers as the round-trip test above)', () => {
    const t = msaOverlayTransform(meta)
    const group = new THREE.Group()
    applyMsaOverlayTransform(group, t)
    group.updateMatrixWorld(true)

    for (const [x, z] of [
      [0, 0],
      [3.2, -5.7],
      [-2.5, 0.4],
    ] as const) {
      const [cx, cz] = overlayLocalToCloudFrame(x, z, t)
      const viaObject3D = group.localToWorld(new THREE.Vector3(x, 0, z))
      expect(cx).toBeCloseTo(viaObject3D.x, 9)
      expect(cz).toBeCloseTo(viaObject3D.z, 9)
    }
  })

  it('overlayLocalToCloudFrame is identity when t is null', () => {
    expect(overlayLocalToCloudFrame(3.2, -5.7, null)).toEqual([3.2, -5.7])
  })

  it('roomPolygonToCloudFrame transforms every vertex, and is null for a degenerate/absent polygon', () => {
    const t = msaOverlayTransform(meta)
    const polygon: [number, number][] = [
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
    ]
    const transformed = roomPolygonToCloudFrame(polygon, t)
    expect(transformed).not.toBeNull()
    expect(transformed).toHaveLength(4)
    expect(transformed![0]).toEqual(overlayLocalToCloudFrame(0, 0, t))
    expect(roomPolygonToCloudFrame(undefined, t)).toBeNull()
    expect(roomPolygonToCloudFrame([[0, 0], [1, 1]], t)).toBeNull() // 2 vertices - degenerate
  })

  describe('isNearRoomPolygonXZ', () => {
    const square: [number, number][] = [
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2],
    ]

    it('is true for a point strictly inside the polygon', () => {
      expect(isNearRoomPolygonXZ(1, 1, square, 0.3)).toBe(true)
    })

    it('is true just outside the boundary, within the margin', () => {
      expect(isNearRoomPolygonXZ(2.2, 1, square, 0.3)).toBe(true) // 0.2m past the east edge
    })

    it('is false well outside the polygon, past the margin', () => {
      expect(isNearRoomPolygonXZ(3, 1, square, 0.3)).toBe(false) // 1m past the east edge
    })

    it('never clips when there is no polygon (absent MSA meta) - "don\'t clip" is the null/degenerate case', () => {
      expect(isNearRoomPolygonXZ(1000, 1000, null, 0.3)).toBe(true)
      expect(isNearRoomPolygonXZ(1000, 1000, [], 0.3)).toBe(true)
    })
  })
})
