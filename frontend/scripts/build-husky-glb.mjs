#!/usr/bin/env node
// One-off, offline conversion of the Clearpath Husky A200's ROS meshes into a single
// scene-ready .glb. Not run at build time or runtime - see frontend/README.md
// "Regenerating the robot model" for the general process (this script follows the same
// pattern as build-turtlebot-glb.mjs) and for how to re-run this after an upstream mesh
// update.
//
// Usage:
//   node scripts/build-husky-glb.mjs <path-to-husky_description>
//
// <path-to-husky_description> is a checkout of https://github.com/husky/husky.git
// (BSD-3-Clause, noetic-devel branch), specifically the husky_description/ directory
// (contains meshes/ and urdf/husky.urdf.xacro).
//
// IMPORTANT: this repo (husky/husky) describes the Husky A200 (classic), not the newer
// A300. Confirmed two ways:
//   - husky.urdf.xacro's base_x_size/base_y_size (0.9874 x 0.5709 m) plus track (0.555 m)
//     and wheel_length (0.1143 m) give an overall footprint of ~990 x ~669 mm, matching
//     Clearpath's published A200 dimensions (990 x 670 x 390 mm) almost exactly. The A300
//     is a visibly bigger/wider platform (990 x 698 x 381 mm per Clearpath's datasheet).
//   - The repo has no A300 xacro/mesh variant at all (single noetic-devel branch, ROS1
//     husky_description only); A300 lives in Clearpath's newer clearpath_common (ROS2).
//
// What this does that a runtime URDFLoader would otherwise do on every page load, baked in
// here instead:
//   - the part offsets from husky.urdf.xacro (base_link, top_chassis, front/rear bumper,
//     top_plate, 4 wheels) - no joints, no articulation, everything fused into a static
//     assembly. user_rail and the PACS mounting hardware are intentionally omitted (fine
//     mounting/bracket detail, not needed for silhouette recognition - see README).
//   - the ROS (Z-up) -> three.js scene (Y-up) axis change, via rotateX(-PI/2), applied once
//     to the whole assembly (matches build-turtlebot-glb.mjs)
//   - re-centring the origin at the footprint centre on the ground plane, using the
//     husky.urdf.xacro base_footprint_joint offset (0, 0, wheel_vertical_offset - wheel_radius)
//   - flat, unlit-friendly-but-lit-capable materials matching the URDF meshes' own collada
//     diffuse colours (chassis black, top deck safety yellow, wheels dark grey, bumpers black)
//
// bumper.dae has no .stl sibling in this repo (base_link/top_chassis/top_plate/wheel all
// do), so it's loaded via three's ColladaLoader instead of STLLoader. ColladaLoader applies
// its own Z-up -> Y-up bake while parsing, but (checked directly, see git history of this
// file) as a root Object3D rotation - not baked into the geometry's vertex data, and not
// even reflected in that root object's .matrix until something calls updateMatrixWorld().
// We never do that here, so bumper.geometry (read directly off the parsed mesh, no
// matrixWorld applied) is left in the file's raw, untransformed vertex data - which, cross-
// checked against the raw <float_array> numbers and against the physical bumper dimensions
// (a ~0.09m-deep, ~0.53m-wide, ~0.025m-thick bar), is already in the same ROS/Z-up frame
// (X forward, Y left, Z up) as the STL parts. So no undo-rotate is needed or applied -
// bumper geometry gets the URDF's rotZ/offset in that shared raw frame exactly like the
// STL parts, then the one shared rotateX(-PI/2) at the very end.
//
// After this script produces raw.glb, decimate + quantize it with gltf-transform (dev-only,
// not a project dependency - see frontend/README.md for the general command line; the
// ratio used for the committed Husky model is much more aggressive than Burger's because
// the source meshes are ~140k triangles for an object at the upper end of the robot size
// range in this app):
//   npx --yes @gltf-transform/cli@4 weld     raw.glb  w.glb
//   npx --yes @gltf-transform/cli@4 simplify w.glb    s.glb --ratio 0.03 --error 0.005
//   npx --yes @gltf-transform/cli@4 quantize s.glb    q.glb
//   npx --yes @gltf-transform/cli@4 dedup    q.glb    husky.glb
//
// Quantization is safe with no runtime change: three's GLTFLoader (three/examples/jsm)
// handles KHR_mesh_quantization natively. Do not Draco/meshopt-compress this model, no
// decoder is wired into the app (see robotModelCache.ts).

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

// ColladaLoader (used only for bumper.dae, see header comment) parses XML via the global
// DOMParser, which Node doesn't provide. jsdom is already a project devDependency
// (node_modules/jsdom) - reuse its DOMParser rather than adding anything new.
const { JSDOM } = await import('jsdom')
globalThis.DOMParser = new JSDOM().window.DOMParser

import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { ColladaLoader } from 'three/examples/jsm/loaders/ColladaLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { mergeGeometries, mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const srcRoot = process.argv[2]
if (!srcRoot) {
  console.error('Usage: node scripts/build-husky-glb.mjs <path-to-husky_description>')
  process.exit(1)
}
const meshesDir = path.join(srcRoot, 'meshes')

// husky.urdf.xacro property values (noetic-devel, commit 41e15d2), in metres.
const base_x_size = 0.9874
const base_y_size = 0.5709
const wheelbase = 0.512
const track = 0.555
const wheel_vertical_offset = 0.03282
const wheel_radius = 0.1651

// Part offsets, in ROS/URDF metres, read from husky.urdf.xacro's default configuration
// (HUSKY_FRONT_BUMPER=1, HUSKY_REAR_BUMPER=1, HUSKY_TOP_PLATE_ENABLED=true, all other
// accessory env vars unset/default-off). rotZ is applied to the geometry (about its own
// origin) before the translation, matching the URDF's <origin rpy="...">.
const STL_PARTS = [
  { file: 'base_link.stl', offset: [0, 0, 0], material: 'chassis' },
  { file: 'top_chassis.stl', offset: [0, 0, 0], material: 'deck' },
  { file: 'top_plate.stl', offset: [0.0812, 0, 0.245], material: 'deck_dark' },
  { file: 'wheel.stl', offset: [wheelbase / 2, track / 2, wheel_vertical_offset], material: 'wheel' },
  { file: 'wheel.stl', offset: [wheelbase / 2, -track / 2, wheel_vertical_offset], material: 'wheel' },
  { file: 'wheel.stl', offset: [-wheelbase / 2, track / 2, wheel_vertical_offset], material: 'wheel' },
  { file: 'wheel.stl', offset: [-wheelbase / 2, -track / 2, wheel_vertical_offset], material: 'wheel' },
]
const DAE_PARTS = [
  { file: 'bumper.dae', offset: [0.48, 0, 0.091], rotZ: 0, material: 'chassis' }, // front
  { file: 'bumper.dae', offset: [-0.48, 0, 0.091], rotZ: Math.PI, material: 'chassis' }, // rear
]

// base_footprint_joint's offset from base_link in husky.urdf.xacro: xyz="0 0
// ${wheel_vertical_offset - wheel_radius}". Wheelbase and track are symmetric about
// base_link's origin, so base_footprint's (x, y) already sits at the centre of the four
// wheels - only a Z shift is needed to move the authoring origin down onto the ground
// plane at the footprint centre (see build-turtlebot-glb.mjs for why this matters: the
// scene expects models authored with the origin at the footprint centre on the floor).
const FOOTPRINT_Z_SHIFT = -(wheel_vertical_offset - wheel_radius) // = +0.13228

// URDF/collada material colours, read as sRGB display colours (three's Color.setHex does
// the sRGB->linear conversion for us - do not hand-linearize these). Taken directly from
// the <color sid="diffuse"> values embedded in the source .dae files (base_link.dae,
// top_chassis.dae, top_plate.dae, wheel.dae, bumper.dae all share bumper<->base_link).
const MATERIAL_COLORS = {
  chassis: 0x1a1a1a, // base_link/bumper diffuse 0.1 0.1 0.1
  deck: 0xcccc00, // top_chassis diffuse 0.8 0.8 0 (Husky's safety-yellow top deck)
  deck_dark: 0x141414, // top_plate diffuse 0.08 0.08 0.08
  wheel: 0x333333, // wheel diffuse 0.2 0.2 0.2
}

// Expected bounding box in SCENE axes (Y-up, metres) after the bake, measured directly off
// the assembled meshes on a first successful run. Guards against a future regeneration
// silently dropping an offset or the axis convention.
// X (fwd/back): +-0.4927m, set by the front/rear bumpers (matches the ~990mm vendor length
//   incl. bumpers; base_link alone is narrower here).
// Y (up): -0.0127m (wheel tread bottom - see FOOTPRINT_Z_SHIFT comment for why the visual
//   tire mesh sits ~13mm below the nominal footprint/ground plane, which is defined by the
//   URDF's smaller collision-cylinder wheel_radius) to 0.3836m (top_plate's top face).
// Z (left/right, scene axis): +-0.3347m, set by the wheel outer faces (track/2 + wheel
//   half-width), which stick out further sideways than the 0.5709m-wide body.
const EXPECTED_BOUNDS = {
  x: [-0.4927, 0.4927],
  y: [-0.0127, 0.3836],
  z: [-0.3347, 0.3347],
}
const BOUNDS_TOLERANCE_M = 0.001

async function loadSTLPart({ file, offset }) {
  const buf = await readFile(path.join(meshesDir, file))
  const loader = new STLLoader()
  let geometry = loader.parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
  // See build-turtlebot-glb.mjs for why NORMAL must be dropped before welding: STLLoader
  // emits a flat per-triangle normal on every vertex copy, which would block welding of
  // position-coincident vertices on adjacent, differently-angled faces.
  geometry.deleteAttribute('normal')
  geometry = mergeVertices(geometry)
  geometry.translate(offset[0], offset[1], offset[2])
  if (geometry.getAttribute('uv')) geometry.deleteAttribute('uv')
  return geometry
}

async function loadDAEPart({ file, offset, rotZ }) {
  const text = await readFile(path.join(meshesDir, file), 'utf8')
  const loader = new ColladaLoader()
  const result = loader.parse(text, '')
  const geos = []
  result.scene.traverse((obj) => {
    if (obj.isMesh) geos.push(obj.geometry.clone())
  })
  let geometry = geos.length > 1 ? mergeGeometries(geos, false) : geos[0]
  geometry.deleteAttribute('normal')
  geometry = mergeVertices(geometry)
  // No axis fixup needed here - see header comment: this geometry is still in the file's
  // raw, untransformed vertex data, which is already the same ROS/Z-up frame as the STL
  // parts. Apply the URDF's rotation/offset in that shared raw frame directly.
  if (rotZ) geometry.rotateZ(rotZ)
  geometry.translate(offset[0], offset[1], offset[2])
  if (geometry.getAttribute('uv')) geometry.deleteAttribute('uv')
  return geometry
}

async function main() {
  for (const p of [...STL_PARTS, ...DAE_PARTS]) {
    const full = path.join(meshesDir, p.file)
    if (!existsSync(full)) {
      console.error(`Missing mesh: ${full}\nIs ${srcRoot} a husky_description checkout?`)
      process.exit(1)
    }
  }

  const geosByMaterial = { chassis: [], deck: [], deck_dark: [], wheel: [] }
  for (const part of STL_PARTS) {
    const geo = await loadSTLPart(part)
    geosByMaterial[part.material].push(geo)
  }
  for (const part of DAE_PARTS) {
    const geo = await loadDAEPart(part)
    geosByMaterial[part.material].push(geo)
  }

  const root = new THREE.Group()
  root.name = 'husky_a200'

  for (const [key, geos] of Object.entries(geosByMaterial)) {
    if (geos.length === 0) continue
    let merged = geos.length > 1 ? mergeGeometries(geos, false) : geos[0]
    // Re-centre the authoring origin at the footprint centre on the ground (see
    // FOOTPRINT_Z_SHIFT above), then bake the ROS (Z-up) -> three.js scene (Y-up) axis
    // change once, on the merged geometry - matches build-turtlebot-glb.mjs.
    merged.translate(0, 0, FOOTPRINT_Z_SHIFT)
    merged.rotateX(-Math.PI / 2)
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

  // Assert before writing: catches a forgotten offset or a wrong axis bake immediately,
  // rather than as a "the robot looks wrong" bug report later.
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
  const outPath = path.join(outDir, 'husky-raw.glb')
  await writeFile(outPath, Buffer.from(glb))
  console.log(`Wrote ${outPath} (${(glb.byteLength / 1e6).toFixed(2)} MB, undecimated)`)
  console.log('Next: decimate + quantize with gltf-transform, see the comment at the top of this script.')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
