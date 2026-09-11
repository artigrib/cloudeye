#!/usr/bin/env node
// One-off, offline conversion of the Unitree Go2's ROS meshes into a single scene-ready .glb.
// Not run at build time or runtime - see frontend/README.md "Regenerating the robot model"
// (Burger) for the general pattern this follows, and frontend/public/models/go2.LICENSE.txt
// for source/license details.
//
// Usage:
//   node scripts/build-go2-glb.mjs <path-to-unitree_ros-checkout>
//
// <path-to-unitree_ros-checkout> is a checkout of
//   https://github.com/unitreerobotics/unitree_ros (BSD-3-Clause), specifically
//   robots/go2_description/ (contains meshes/*.dae and urdf/go2_description.urdf).
//
// Unlike the Burger (STL, millimetres), the Go2's meshes are Collada (.dae), already in
// metres (<unit meter="1"/> in every file) and declared Z_UP - both confirmed by inspecting
// the .dae XML directly, not assumed. This script:
//   - loads each of the 7 unique part meshes (base/trunk, hip, thigh, thigh_mirror, calf,
//     calf_mirror, foot) once via ColladaLoader, in Node (jsdom supplies the DOMParser the
//     loader needs)
//   - deliberately does NOT use ColladaLoader's automatic Z-up -> Y-up scene rotation
//     (scene.rotation is zeroed right after parsing) - instead each part's own internal
//     per-node correction matrix (an artifact of how these particular .dae files were
//     exported; every part has one) is baked in via matrixWorld, leaving the geometry in the
//     same ROS/URDF Z-up metres frame the joint offsets below are written in
//   - positions all 17 body parts (trunk + 4 x [hip, thigh, calf, foot]) by the cumulative
//     joint offsets read from go2_description.urdf, at the zero joint angle (all joint
//     <origin rpy> are identity in this URDF - only translation - so this is a plain
//     translate-and-place, same as the Burger's wheels/LiDAR): legs hang straight down,
//     which is the URDF's own rest pose, not a crouched stance
//   - bakes the ROS (Z-up) -> three.js scene (Y-up) axis change once at the end, via
//     rotateX(-Math.PI/2) on every merged mesh - exactly the Burger script's convention (see
//     build-turtlebot-glb.mjs and src/lib/robot-heading.ts: ROS +X forward -> scene +X)
//   - keeps the trunk as its own mesh(es), separate from the 16 leg parts, specifically so
//     "corpus" (body-only, no legs) measurements can be taken off the *trunk* node in the
//     final decimated .glb - see frontend/README.md and ~/reports/M-go2.md for why a
//     quadruped's circular footprint is measured off the body, not the legs
//   - recenters the origin at the footprint centre (the mean of the 4 hip attachment points
//     in the URDF is exactly (0,0) in ROS X/Y by construction - asserted below, not assumed)
//     at floor level (the lowest point of the feet in this rest pose)
//
// After this script produces go2-raw.glb (in scripts/out/, gitignored), decimate + quantize it
// with gltf-transform (dev-only, not a project dependency - same tool/pipeline as
// build-turtlebot-glb.mjs, see frontend/README.md). The go2.glb committed to
// public/models/ was built with:
//   cd scripts/out
//   npx --yes @gltf-transform/cli@4 weld     go2-raw.glb  go2-w.glb
//   npx --yes @gltf-transform/cli@4 simplify go2-w.glb    go2-s.glb --ratio 0.05 --error 0.035
//   npx --yes @gltf-transform/cli@4 quantize go2-s.glb    go2-q.glb
//   npx --yes @gltf-transform/cli@4 dedup    go2-q.glb    ../../public/models/go2.glb
// A much higher --ratio/--error than the Burger's (0.10/0.005) was needed: the Go2's raw
// per-part .dae geometry is ~400k triangles pre-weld (vs. the Burger's ~154k), and two of the
// leg colour buckets (dense CAD housings) would not simplify past ~70% of their original
// triangle count at the Burger's tolerances no matter --ratio, --error, or --lock-border -
// see the NORMAL-dropping comment in loadColladaGroups() below for why (short version: the
// same class of problem as the Burger's flat STL normals, just from smooth-shaded CAD hard
// edges instead of flat STL facets). This produced an 8-primitive, ~360KB .glb (~26k
// triangles total) from the source's ~400k raw triangles, comfortably under the ~400KB target
// with the trunk ("corpus") bounding box unchanged to the millimetre against the
// pre-decimation reading.

import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// ColladaLoader needs a DOMParser (it parses the .dae XML with one) and GLTFExporter's binary
// path needs a browser-style FileReader - neither exists in plain Node. jsdom is already a
// project devDependency (used by the test setup), so borrow its DOMParser rather than adding
// a new dependency.
import { JSDOM } from 'jsdom'
const { window } = new JSDOM('')
globalThis.DOMParser = window.DOMParser
if (typeof globalThis.FileReader === 'undefined') {
  globalThis.FileReader = class FileReader {
    readAsArrayBuffer(blob) {
      blob
        .arrayBuffer()
        .then((buf) => {
          this.result = buf
          this.onloadend?.()
        })
        .catch((err) => this.onerror?.(err))
    }
  }
}

import * as THREE from 'three'
import { ColladaLoader } from 'three/examples/jsm/loaders/ColladaLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { mergeGeometries, mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const srcRoot = process.argv[2]
if (!srcRoot) {
  console.error('Usage: node scripts/build-go2-glb.mjs <path-to-unitree_ros-checkout>')
  process.exit(1)
}
const descDir = path.join(srcRoot, 'robots', 'go2_description')
const meshesDir = path.join(descDir, 'meshes')

// Joint offsets read directly from robots/go2_description/urdf/go2_description.urdf (the
// pre-xacro-expanded file - it matches the .dae files actually present in meshes/, unlike
// xacro/robot.xacro which references a nonexistent trunk.dae). Every joint's <origin rpy> is
// "0 0 0" - only translation - so the zero-angle assembly below is pure vector addition down
// each leg's kinematic chain (base -> hip -> thigh -> calf -> foot).
const HIP_XY = { FL: [0.1934, 0.0465], FR: [0.1934, -0.0465], RL: [-0.1934, 0.0465], RR: [-0.1934, -0.0465] }
const THIGH_Y_ADD = { FL: 0.0955, FR: -0.0955, RL: 0.0955, RR: -0.0955 } // <origin xyz="0 ${thigh_offset*mirror} 0">
const CALF_Z_ADD = -0.213 // hip_thigh -> calf, same on all 4 legs
const FOOT_Z_ADD = -0.213 // calf -> foot, same on all 4 legs

// Which source mesh each leg's thigh/calf visual uses (see xacro/leg.xacro's mirror_dae flag,
// confirmed against the actual per-leg visual <mesh filename> in the compiled urdf): FL/RL
// share thigh.dae + calf.dae, FR/RR share thigh_mirror.dae + calf_mirror.dae. The hip uses the
// same hip.dae for all 4, oriented per leg via the visual <origin rpy> read from the urdf
// (0; pi,0,0; 0,pi,0; pi,pi,0 for FL,FR,RL,RR respectively - all pure 180 degree axis flips,
// which are diagonal +-1 matrices and therefore commute, so application order doesn't matter).
const LEGS = {
  FL: { thighFile: 'thigh.dae', calfFile: 'calf.dae', hipRot: [] },
  FR: { thighFile: 'thigh_mirror.dae', calfFile: 'calf_mirror.dae', hipRot: ['x'] },
  RL: { thighFile: 'thigh.dae', calfFile: 'calf.dae', hipRot: ['y'] },
  RR: { thighFile: 'thigh_mirror.dae', calfFile: 'calf_mirror.dae', hipRot: ['x', 'y'] },
}

// Expected bounding box in SCENE axes (Y-up, metres) after the bake, for the WHOLE assembly
// (trunk + all 4 legs straight down - not a crouched stance). Measured directly off the
// assembled meshes on a known-good run. Guards against a future regeneration silently
// dropping a unit conversion or an axis convention.
const EXPECTED_BOUNDS = {
  x: [-0.2419, 0.332],
  y: [0, 0.5377],
  z: [-0.162, 0.162],
}
const BOUNDS_TOLERANCE_M = 0.003

const colladaCache = new Map()

/** Parse a .dae once, split its (possibly multi-material) mesh into per-material-group
 * geometries, and bake that part's own internal node correction matrix into them - but NOT
 * ColladaLoader's automatic Z-up->Y-up scene rotation (zeroed below), so the result stays in
 * the same ROS/URDF Z-up metres frame the joint offsets are written in. Returns
 * [{ geometry, color: THREE.Color }, ...], one per material group, cached by file. */
async function loadColladaGroups(file) {
  if (colladaCache.has(file)) return colladaCache.get(file)
  const full = path.join(meshesDir, file)
  const text = await readFile(full, 'utf-8')
  const loader = new ColladaLoader()
  const collada = loader.parse(text, full)
  const scene = collada.scene
  scene.rotation.set(0, 0, 0) // undo the loader's own Z-up->Y-up; we bake ROS->scene once, later
  scene.updateMatrix()
  scene.updateMatrixWorld(true)

  const meshes = []
  scene.traverse((obj) => {
    if (obj.isMesh) meshes.push(obj)
  })
  if (meshes.length !== 1) {
    throw new Error(`${file}: expected exactly 1 mesh in the Collada scene, found ${meshes.length}`)
  }
  const mesh = meshes[0]
  const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
  const srcGeo = mesh.geometry
  const groups = srcGeo.groups.length
    ? srcGeo.groups
    : [{ start: 0, count: srcGeo.attributes.position.count, materialIndex: 0 }]

  const pos = srcGeo.attributes.position
  const result = groups.map(({ start, count, materialIndex }) => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(pos.array.slice(start * 3, (start + count) * 3), 3))
    // No UV kept - none of these parts are textured (colour-only materials). NORMAL is also
    // dropped here rather than kept from the source: these CAD-exported .dae parts encode
    // plenty of genuine hard edges (chamfers, bearing housings) as per-vertex normal
    // discontinuities, which - exactly like the Burger's per-facet STL normals (see
    // build-turtlebot-glb.mjs) - hashes as "different vertex" in mergeVertices and blocks
    // welding, which in turn blocks gltf-transform's simplify step later (confirmed: with
    // normals kept, two of this model's material buckets - dense housings - wouldn't simplify
    // below ~70% of their original triangle count no matter the --error/--ratio/--lock-border
    // given, while position-only-welded buckets simplified to ~1-3% fine). Weld on position
    // alone, then computeVertexNormals() derives correct angle-averaged smooth normals from
    // the now-indexed geometry - acceptable since joint/housing surface detail isn't the goal
    // here (see ~/reports/M-go2.md), only a recognisable silhouette.
    geo.applyMatrix4(mesh.matrixWorld) // bake this part's own node-correction matrix (ROS frame)
    const welded = mergeVertices(geo)
    welded.computeVertexNormals()
    return { geometry: welded, color: materials[materialIndex].color.clone() }
  })
  colladaCache.set(file, result)
  return result
}

/** Clone a cached part's geometry groups, apply this instance's extra rotation (leg-specific
 * hip mirroring) then translate to its position in the kinematic chain - all still in ROS
 * Z-up metres. Returns [{ geometry, color }, ...]. */
async function instantiatePart(file, rotAxes, offset) {
  const groups = await loadColladaGroups(file)
  return groups.map(({ geometry, color }) => {
    const geo = geometry.clone()
    for (const axis of rotAxes) {
      if (axis === 'x') geo.rotateX(Math.PI)
      else if (axis === 'y') geo.rotateY(Math.PI)
    }
    geo.translate(offset[0], offset[1], offset[2])
    return { geometry: geo, color }
  })
}

async function main() {
  for (const f of ['base.dae', 'hip.dae', 'thigh.dae', 'thigh_mirror.dae', 'calf.dae', 'calf_mirror.dae', 'foot.dae']) {
    if (!existsSync(path.join(meshesDir, f))) {
      console.error(`Missing mesh: ${path.join(meshesDir, f)}\nIs ${srcRoot} a unitree_ros checkout?`)
      process.exit(1)
    }
  }

  // bucket key = `${tag}:${colorHex}` so same-coloured leg parts merge into one primitive,
  // while trunk stays entirely separate from legs regardless of colour (needed to measure the
  // "corpus" bbox off just the trunk in the final glb - see the block comment up top).
  const buckets = new Map() // key -> { tag, color, geos: BufferGeometry[] }
  function addToBucket(tag, color, geo) {
    const key = `${tag}:${color.getHexString()}`
    if (!buckets.has(key)) buckets.set(key, { tag, color, geos: [] })
    buckets.get(key).geos.push(geo)
  }

  // Trunk (== "base"/"corpus"): identity offset, identity rotation - its visual <origin> in
  // the urdf is xyz="0 0 0" rpy="0 0 0", and "base"/"trunk" IS the frame every other offset
  // below is relative to.
  for (const { geometry, color } of await loadColladaGroups('base.dae')) {
    addToBucket('trunk', color, geometry.clone())
  }

  // Assert the footprint symmetry the recentring step below relies on, instead of assuming it:
  // the 4 hip attachment points must average to (0,0) in ROS X/Y for "base" to already be the
  // footprint centre.
  const hipXs = Object.values(HIP_XY).map((p) => p[0])
  const hipYs = Object.values(HIP_XY).map((p) => p[1])
  const hipXMean = hipXs.reduce((a, b) => a + b, 0) / 4
  const hipYMean = hipYs.reduce((a, b) => a + b, 0) / 4
  if (Math.abs(hipXMean) > 1e-9 || Math.abs(hipYMean) > 1e-9) {
    console.error(`Hip attachment points are not centred on the origin: mean=(${hipXMean}, ${hipYMean})`)
    process.exit(1)
  }

  for (const legName of ['FL', 'FR', 'RL', 'RR']) {
    const { thighFile, calfFile, hipRot } = LEGS[legName]
    const [hx, hy] = HIP_XY[legName]
    const hipPos = [hx, hy, 0]
    const thighPos = [hx, hy + THIGH_Y_ADD[legName], 0]
    const calfPos = [thighPos[0], thighPos[1], CALF_Z_ADD]
    const footPos = [calfPos[0], calfPos[1], calfPos[2] + FOOT_Z_ADD]

    for (const { geometry, color } of await instantiatePart('hip.dae', hipRot, hipPos)) {
      addToBucket('legs', color, geometry)
    }
    for (const { geometry, color } of await instantiatePart(thighFile, [], thighPos)) {
      addToBucket('legs', color, geometry)
    }
    for (const { geometry, color } of await instantiatePart(calfFile, [], calfPos)) {
      addToBucket('legs', color, geometry)
    }
    for (const { geometry, color } of await instantiatePart('foot.dae', [], footPos)) {
      addToBucket('legs', color, geometry)
    }
  }

  // Merge each bucket, then bake ROS (Z-up) -> three.js scene (Y-up) once per merged mesh -
  // same convention as build-turtlebot-glb.mjs (rotateX(-PI/2); ROS +X forward -> scene +X,
  // see src/lib/robot-heading.ts).
  const root = new THREE.Group()
  root.name = 'go2'
  const meshesByTag = { trunk: [], legs: [] }
  for (const { tag, color, geos } of buckets.values()) {
    const merged = geos.length > 1 ? mergeGeometries(geos, false) : geos[0]
    merged.rotateX(-Math.PI / 2)
    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.85, metalness: 0.05 })
    const mesh = new THREE.Mesh(merged, material)
    mesh.name = `${tag}_${color.getHexString()}`
    meshesByTag[tag].push(mesh)
    root.add(mesh)
  }

  // Recentre: footprint centre is already at scene X=0/Z=0 (asserted above via the hip-mean
  // check - ROS X stays scene X, ROS Y becomes scene -Z under rotateX(-PI/2), and both means
  // are 0). Only the floor needs placing: shift everything up so the lowest point (the feet,
  // in this straight-legs rest pose) sits at scene Y=0.
  const preShiftBox = new THREE.Box3().setFromObject(root)
  const floorY = preShiftBox.min.y
  for (const mesh of root.children) mesh.geometry.translate(0, -floorY, 0)

  const box = new THREE.Box3().setFromObject(root)
  if (process.env.SKIP_BOUNDS_CHECK !== '1') {
    for (const [axis, [lo, hi]] of Object.entries(EXPECTED_BOUNDS)) {
      const gotLo = box.min[axis]
      const gotHi = box.max[axis]
      if (Math.abs(gotLo - lo) > BOUNDS_TOLERANCE_M || Math.abs(gotHi - hi) > BOUNDS_TOLERANCE_M) {
        console.error(
          `Bounding box mismatch on ${axis}: expected [${lo}, ${hi}], got [${gotLo.toFixed(4)}, ${gotHi.toFixed(4)}]`,
        )
        process.exit(1)
      }
    }
  }
  console.log(
    `Full-assembly bounds OK: x [${box.min.x.toFixed(4)}, ${box.max.x.toFixed(4)}] ` +
      `y [${box.min.y.toFixed(4)}, ${box.max.y.toFixed(4)}] ` +
      `z [${box.min.z.toFixed(4)}, ${box.max.z.toFixed(4)}]`,
  )

  // Trunk-only ("corpus") bbox, for the circular-footprint measurement - see
  // ~/reports/M-go2.md. This is a preliminary reading off the undecimated mesh; the report's
  // authoritative numbers come from re-loading the final, decimated public/models/go2.glb.
  const trunkBox = new THREE.Box3()
  for (const mesh of meshesByTag.trunk) trunkBox.expandByObject(mesh)
  const tx = trunkBox.max.x - trunkBox.min.x
  const tz = trunkBox.max.z - trunkBox.min.z
  console.log(
    `Trunk-only ("corpus") bounds (pre-decimation): x [${trunkBox.min.x.toFixed(4)}, ${trunkBox.max.x.toFixed(4)}] ` +
      `(${(tx * 1000).toFixed(1)}mm) z [${trunkBox.min.z.toFixed(4)}, ${trunkBox.max.z.toFixed(4)}] (${(tz * 1000).toFixed(1)}mm)`,
  )

  const exporter = new GLTFExporter()
  const glb = await new Promise((resolve, reject) => {
    exporter.parse(root, resolve, reject, { binary: true })
  })

  const outDir = path.join(__dirname, 'out')
  await mkdir(outDir, { recursive: true })
  const outPath = path.join(outDir, 'go2-raw.glb')
  await writeFile(outPath, Buffer.from(glb))
  console.log(`Wrote ${outPath} (${(glb.byteLength / 1e6).toFixed(2)} MB, undecimated)`)
  console.log('Next: decimate + quantize with gltf-transform, see frontend/README.md.')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
