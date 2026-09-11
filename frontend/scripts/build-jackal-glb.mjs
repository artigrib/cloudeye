#!/usr/bin/env node
// One-off, offline conversion of the Clearpath Jackal's ROS meshes into a single scene-ready
// .glb. Not run at build time or runtime - see frontend/README.md "Regenerating the robot
// model" (turtlebot3_burger's section; this script follows the same recipe) for why, and for
// how to re-run this after a mesh update upstream.
//
// Usage:
//   node scripts/build-jackal-glb.mjs <path-to-jackal-repo-root>
//
// <path-to-jackal-repo-root> is a checkout of https://github.com/jackal/jackal (BSD-3-Clause),
// specifically the parent directory that contains jackal_description/ (meshes/ and
// urdf/jackal.urdf.xacro).
//
// What this does that a runtime URDFLoader would otherwise do on every page load, baked in
// here instead - all offsets/rotations read straight from jackal_description/urdf/jackal.urdf.xacro:
//   - base (chassis), 4 wheels, front + rear fenders assembled per their <origin> elements
//     and joint offsets - no accessories (camera/lidar/GPS mounts), no articulation (wheels
//     fused static, not rotating joints)
//   - jackal-base.stl / jackal-wheel.stl / jackal-fender.stl are already authored in METRES in
//     the URDF's ROS frame (no <mesh scale="0.001 ..."> on these three, unlike some of the
//     optional accessory meshes elsewhere in the package) - so, unlike TurtleBot3 Burger, NO
//     mm->m scale is applied here. Verified directly off the STL bounding boxes before writing
//     this script.
//   - the chassis visual and each wheel visual carry a <origin rpy=...> (the STLs were
//     authored in their own CAD axes, not URDF/ROS axes) - reproduced here via sequential
//     rotateX/rotateY/rotateZ calls in roll,pitch,yaw order, which matches URDF's fixed-axis
//     R = Rz(yaw)*Ry(pitch)*Rx(roll) convention when applied as sequential three.js calls
//   - the ROS (Z-up) -> three.js scene (Y-up) axis change, via rotateX(-PI/2), applied once to
//     the whole assembled group - same as the Burger script
//   - flat materials matching the URDF's "dark_grey" (chassis) / "black" (wheels) / "yellow"
//     (fenders) colours
//
// After this script produces raw.glb, decimate + quantize it with gltf-transform (dev-only,
// not a project dependency - see frontend/README.md "Regenerating the robot model" for the
// exact command line, reused verbatim here):
//   npx --yes @gltf-transform/cli@4 weld     raw.glb  w.glb
//   npx --yes @gltf-transform/cli@4 simplify w.glb    s.glb --ratio 0.10 --error 0.005
//   npx --yes @gltf-transform/cli@4 quantize s.glb    q.glb
//   npx --yes @gltf-transform/cli@4 dedup    q.glb    ../public/models/jackal.glb
//
// Quantization is safe with no runtime change: three's GLTFLoader (three/examples/jsm) handles
// KHR_mesh_quantization natively. Do not Draco/meshopt-compress this model - neither decoder is
// wired into the app (see robotModelCache.ts).

import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// GLTFExporter's binary path uses the browser's FileReader to turn its merged Blob into an
// ArrayBuffer. Node has Blob but not FileReader - polyfill just enough of it (readAsArrayBuffer
// + onloadend + .result) since this script never runs in a browser.
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
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { mergeGeometries, mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const srcRoot = process.argv[2]
if (!srcRoot) {
  console.error('Usage: node scripts/build-jackal-glb.mjs <path-to-jackal-repo-root>')
  process.exit(1)
}
const meshesDir = path.join(srcRoot, 'jackal_description', 'meshes')

// Constants read straight from jackal_description/urdf/jackal.urdf.xacro.
const wheelbase = 0.262
const track = 0.37559
const wheel_vertical_offset = 0.0345
const footprint_vertical_offset = -0.0655

// Parts, in ROS/URDF metres. `rpy` (roll, pitch, yaw) is applied first, in that order, via
// sequential rotateX/Y/Z calls - matching the URDF fixed-axis R = Rz(yaw)*Ry(pitch)*Rx(roll)
// convention - then `offset` is applied as a plain translation. No mesh scale: these three
// STLs are already in metres (see header comment).
const PARTS = [
  {
    file: 'jackal-base.stl',
    rpy: [Math.PI / 2, 0, Math.PI / 2],
    offset: [0, 0, footprint_vertical_offset],
    material: 'chassis',
  },
  {
    file: 'jackal-wheel.stl',
    rpy: [Math.PI / 2, 0, 0],
    offset: [wheelbase / 2, track / 2, wheel_vertical_offset],
    material: 'wheel',
  },
  {
    file: 'jackal-wheel.stl',
    rpy: [Math.PI / 2, 0, 0],
    offset: [wheelbase / 2, -track / 2, wheel_vertical_offset],
    material: 'wheel',
  },
  {
    file: 'jackal-wheel.stl',
    rpy: [Math.PI / 2, 0, 0],
    offset: [-wheelbase / 2, track / 2, wheel_vertical_offset],
    material: 'wheel',
  },
  {
    file: 'jackal-wheel.stl',
    rpy: [Math.PI / 2, 0, 0],
    offset: [-wheelbase / 2, -track / 2, wheel_vertical_offset],
    material: 'wheel',
  },
  // Front fender: front_fender_joint (chassis -> front_fender_link) is identity, and the
  // default (non-accessory) visual has no <origin> (identity) either - mesh used as-is.
  { file: 'jackal-fender.stl', rpy: [0, 0, 0], offset: [0, 0, 0], material: 'fender' },
  // Rear fender: same mesh, but rear_fender_joint carries rpy="0 0 PI" (180 deg about Z) to
  // mirror it onto the back of the chassis; its own visual origin is identity.
  { file: 'jackal-fender.stl', rpy: [0, 0, Math.PI], offset: [0, 0, 0], material: 'fender' },
]

// URDF material rgba, read as sRGB display colours (three's Color.setHex/setRGB does the
// sRGB->linear conversion for us - do not hand-linearize these).
const MATERIAL_COLORS = {
  chassis: 0x333333, // "dark_grey" rgba 0.2 0.2 0.2
  wheel: 0x262626, // "black"     rgba 0.15 0.15 0.15
  fender: 0xcccc00, // "yellow"    rgba 0.8 0.8 0.0
}

async function loadPart({ file, rpy, offset }) {
  const buf = await readFile(path.join(meshesDir, file))
  const loader = new STLLoader()
  let geometry = loader.parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
  // No mm->m scale here - jackal-base.stl / jackal-wheel.stl / jackal-fender.stl are already in
  // metres (unlike TurtleBot3 Burger's STLs, and unlike some optional Jackal accessory meshes
  // elsewhere in the package that do carry a 0.001 <mesh scale>).
  //
  // STLLoader sets a per-vertex NORMAL attribute from the STL's raw facet normals (each
  // triangle's 3 vertex copies all carry that triangle's flat face normal). mergeVertices
  // hashes ALL attributes together, so if we welded now, position-coincident vertices on
  // adjacent, differently-angled faces would still count as distinct (different normal) and
  // stay unwelded. Drop NORMAL first so welding is purely position-based; computeVertexNormals()
  // below then derives correct per-shared-vertex (angle-averaged) normals on the now-indexed
  // geometry.
  geometry.deleteAttribute('normal')
  // STLLoader's output is also a triangle soup: every triangle gets its own 3 independent
  // vertex copies, none sharing an index, even along a shared edge - that's the STL format, not
  // a parsing bug. Weld coincident positions (0.1mm tolerance, reasonable for CAD-export noise)
  // into an indexed geometry. This step is also what makes decimation effective - meshoptimizer's
  // simplifier collapses shared edges, and an unwelded soup has none.
  geometry = mergeVertices(geometry)
  // Apply the part's <origin rpy="r p y"/> in fixed-axis roll,pitch,yaw order (matches URDF's
  // R = Rz(yaw)*Ry(pitch)*Rx(roll) convention when chained this way), then its translation.
  const [roll, pitch, yaw] = rpy
  if (roll) geometry.rotateX(roll)
  if (pitch) geometry.rotateY(pitch)
  if (yaw) geometry.rotateZ(yaw)
  geometry.translate(offset[0], offset[1], offset[2])
  geometry.deleteAttribute('uv') // STL has no UVs to begin with; mergeGeometries wants matching attrs
  return geometry
}

// Expected bounding box in SCENE axes (Y-up, metres) after the bake and floor-alignment shift,
// measured directly off the assembled meshes (base + 4 wheels + front/rear fenders) on a clean
// run of this script: 510.6 x 430.0 x 249.3 mm (LxWxH), floor at y=0. That's within 0.6% of the
// vendor spec (508 x 430 x 250 mm) on every axis - see ~/reports/M-jackal.md for the
// reconciliation. This assertion exists to catch a future regeneration silently dropping the
// axis convention, the floor-alignment shift, or introducing a stray mm/m scale, not to force
// an exact vendor-spec match.
const EXPECTED_BOUNDS = {
  x: [-0.2553, 0.2553],
  y: [0, 0.2493],
  z: [-0.215, 0.215],
}
const BOUNDS_TOLERANCE_M = 0.001

async function main() {
  const uniqueFiles = [...new Set(PARTS.map((p) => p.file))]
  for (const file of uniqueFiles) {
    const full = path.join(meshesDir, file)
    if (!existsSync(full)) {
      console.error(`Missing mesh: ${full}\nIs ${srcRoot} a jackal repo checkout?`)
      process.exit(1)
    }
  }

  const geosByMaterial = { chassis: [], wheel: [], fender: [] }
  for (const part of PARTS) {
    const geo = await loadPart(part)
    geosByMaterial[part.material].push(geo)
  }

  const root = new THREE.Group()
  root.name = 'jackal'

  const mergedGeometries = []
  for (const [key, geos] of Object.entries(geosByMaterial)) {
    if (geos.length === 0) continue
    const merged = geos.length > 1 ? mergeGeometries(geos, false) : geos[0]
    // Bake the ROS (Z-up) -> three.js scene (Y-up) axis change once, on the merged geometry,
    // not as a runtime transform on the loaded object - see robotModelCache.ts and
    // robot-heading.ts for why the model is authored directly in scene axes.
    merged.rotateX(-Math.PI / 2)
    mergedGeometries.push({ key, merged })
  }

  // Jackal's base_link (this assembly's origin) is NOT exactly at floor level: per
  // jackal.urdf.xacro, the wheel visual sits at wheel_vertical_offset (0.0345m) above
  // base_link, and the actual wheel mesh radius (~0.1m, off the STL) puts the wheel's lowest
  // point about 6.5cm *below* base_link - unlike TurtleBot3 Burger, whose base_link the
  // ROBOTIS URDF happens to already place exactly at floor level. Shift the whole assembly up
  // by that measured amount so the SCENE Y=0 plane is the floor under the wheels and the
  // origin sits at the footprint centre at floor level, matching turtlebot3_burger.glb's
  // convention (see build-turtlebot-glb.mjs's EXPECTED_BOUNDS, y starting at 0).
  const prelimBox = new THREE.Box3()
  for (const { merged } of mergedGeometries) {
    merged.computeBoundingBox()
    prelimBox.union(merged.boundingBox)
  }
  const floorShift = -prelimBox.min.y
  console.log(`Floor alignment: shifting scene Y by +${floorShift.toFixed(5)} m so wheel-bottom = 0`)

  for (const { key, merged } of mergedGeometries) {
    merged.translate(0, floorShift, 0)
    merged.computeVertexNormals()
    const material = new THREE.MeshStandardMaterial({
      color: MATERIAL_COLORS[key],
      roughness: 0.9,
      metalness: 0,
    })
    const mesh = new THREE.Mesh(merged, material)
    mesh.name = key
    root.add(mesh)
  }

  // Assert before writing: catches a forgotten axis convention or a stray scale immediately,
  // rather than as a "the robot is the wrong size" bug report later.
  const box = new THREE.Box3().setFromObject(root)
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
  console.log(
    `Bounds OK: x [${box.min.x.toFixed(4)}, ${box.max.x.toFixed(4)}] ` +
      `y [${box.min.y.toFixed(4)}, ${box.max.y.toFixed(4)}] ` +
      `z [${box.min.z.toFixed(4)}, ${box.max.z.toFixed(4)}]`,
  )

  const exporter = new GLTFExporter()
  const glb = await new Promise((resolve, reject) => {
    exporter.parse(root, resolve, reject, { binary: true })
  })

  const outDir = path.join(__dirname, 'out')
  await mkdir(outDir, { recursive: true })
  const outPath = path.join(outDir, 'jackal-raw.glb')
  await writeFile(outPath, Buffer.from(glb))
  console.log(`Wrote ${outPath} (${(glb.byteLength / 1e6).toFixed(2)} MB, undecimated)`)
  console.log('Next: decimate + quantize with gltf-transform, see the comment at the top of this script.')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
