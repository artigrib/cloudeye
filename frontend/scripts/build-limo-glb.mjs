#!/usr/bin/env node
// One-off, offline conversion of the AgileX LIMO's ROS meshes (four-wheel differential-drive
// variant, limo_four_diff.xacro) into a single scene-ready .glb. Not run at build time or
// runtime - see frontend/README.md "Regenerating the robot model" (turtlebot3 section; this
// script follows the same recipe) for why, and for how to re-run this after a mesh update
// upstream.
//
// Usage:
//   node scripts/build-limo-glb.mjs <path-to-limo_description>
//
// <path-to-limo_description> is a checkout of
//   https://github.com/agilexrobotics/limo_ros (BSD-3-Clause), specifically the
//   limo_description/ directory (contains meshes/ and urdf/limo_four_diff.xacro).
//
// LIMO ships four kinematic variants (mecanum / tracked / four-wheel-differential /
// Ackermann); limo_four_diff.xacro is the standard differential-drive base and is what this
// script bakes. limo_description has no separate mesh for the Ackermann steering hinges, so
// that variant isn't representable from this mesh set anyway.
//
// What this does that a runtime URDFLoader would otherwise do on every page load, baked in
// here instead:
//   - mm -> m (limo_base.stl / limo_wheel.stl are in millimetres; the sibling .dae files
//     encode the same geometry already in metres - this script uses limo_base.dae, not the
//     STL, for the base - see the note in loadBasePart() below on why)
//   - the 5 part offsets (base + 4 wheels) derived from limo_four_diff.xacro's properties
//     (wheelbase=0.2, track=0.13, wheel_vertical_offset=-0.10, base_footprint->base_link
//     offset=0.15) - no joints, no articulation, everything fused into a static assembly
//   - the base mesh's yaw=1.57rad (rounded here to the evident intent of exactly pi/2) visual
//     rotation from the xacro, which re-orients the CAD-authored mesh into the robot's
//     forward = +X convention; the visual origin's z-translation (-0.15) exactly cancels the
//     base_footprint->base_link joint's z-translation (+0.15), so the base needs no
//     translation, only that rotation
//   - a small (~2mm) uniform vertical shift so the lowest point of the assembled wheels sits
//     exactly at z=0 - the URDF's wheel_vertical_offset/wheel_radius properties place it
//     ~2mm above 0, because those properties approximate a wheel mesh whose real radius is
//     very slightly larger (48.1mm) than the property says (45mm); see the ground-contact
//     correction in main()
//   - the ROS (Z-up) -> three.js scene (Y-up) axis change, via rotateX(-Math.PI/2)
//   - flat materials approximating the two dominant colours actually authored in
//     limo_base.dae / limo_wheel.dae (see MATERIAL_COLORS below) - the small red tail-light
//     and green LED regions (<2% of the base mesh's vertices combined) are folded into the
//     dark material rather than kept as separate materials, since they're far too small a
//     silhouette detail to survive decimation to ~400KB anyway
//
// Two deliberate departures from the turtlebot3_burger recipe (build-turtlebot-glb.mjs),
// both because LIMO's source meshes are far higher-detail than Burger's:
//
//   1. limo_base.dae is authored as ~1600 separate <node> sub-meshes (screws, vents, small
//      brackets, decorative seams - "krepezh"), most with a bounding-box diagonal under a
//      centimetre. gltf-transform's simplify (meshoptimizer) cannot merge geometry across
//      disconnected components, so with all ~1600 kept, it hit a wall around 50k triangles
//      no matter how aggressive --ratio/--error were - those thousands of tiny islands were
//      each already at (or near) their own minimum representable triangle count. Since the
//      task only needs the base to read as a small rounded box from a few metres away (no
//      fastener detail), loadBasePart() below drops any sub-mesh whose bounding-box diagonal
//      is under HARDWARE_DETAIL_THRESHOLD_M before merging - this cuts ~1520 of ~1620 parts
//      (67% of triangles) while leaving the overall bounding box byte-for-byte unchanged
//      (verified: the dropped parts are all well inside the hull the large panels define).
//      That requires the DAE (which preserves the part boundaries) rather than the STL
//      (a single undifferentiated triangle soup with the same fragmentation baked in).
//   2. limo_wheel.dae's tyre tread is modelled as many small knobby lugs - the same
//      many-tiny-disconnected-islands problem, but as *one* node, so it can't be filtered the
//      same way. The URDF itself already stands in a plain cylinder for the wheel's collision
//      shape; loadWheelProxy() below does the same for the visual mesh, sized to the real
//      mesh's measured envelope (radius 48.1mm, width 74.7mm, off-centre by +6.3mm - see
//      WHEEL_* constants). A wheel is a small, mostly-hidden part of the silhouette; a plain
//      cylinder reads identically at 3D-viewer scale and costs ~120 triangles instead of
//      ~140,000.
//
// After this script produces raw.glb (~5MB even after the base-part filter and wheel proxies -
// still ~165k+88k triangles), decimate + quantize it with gltf-transform (dev-only, not a
// project dependency - see frontend/README.md for the general recipe). The committed model
// used a lower --ratio and a much looser --error than turtlebot3_burger needed, since even the
// filtered base mesh is far more tessellated than Burger's:
//   npx --yes @gltf-transform/cli@4 weld     raw.glb  w.glb
//   npx --yes @gltf-transform/cli@4 simplify w.glb    s.glb --ratio 0.1 --error 0.01
//   npx --yes @gltf-transform/cli@4 quantize s.glb    q.glb
//   npx --yes @gltf-transform/cli@4 dedup    q.glb    ../public/models/limo.glb
// This produced a ~28.9k-triangle, ~408KB .glb (comparable to Burger's own ~29k/~400KB),
// bounding box unchanged to within a millimetre of the pre-decimation assembly.
//
// Quantization is safe with no runtime change: three's GLTFLoader (three/examples/jsm) handles
// KHR_mesh_quantization natively. Do not Draco/meshopt-compress this model - for an asset this
// small the decoder costs more bytes than compression saves, and neither decoder is wired into
// the app (see robotModelCache.ts).

import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { JSDOM } from 'jsdom'

// ColladaLoader (used for the base mesh - see the header comment) needs a DOMParser, which
// Node doesn't have. jsdom is a devDependency already present for other tooling; this is the
// only place this script needs it.
const dom = new JSDOM('<!DOCTYPE html>')
globalThis.DOMParser = dom.window.DOMParser

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
import { ColladaLoader } from 'three/examples/jsm/loaders/ColladaLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { mergeGeometries, mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const srcRoot = process.argv[2]
if (!srcRoot) {
  console.error('Usage: node scripts/build-limo-glb.mjs <path-to-limo_description>')
  process.exit(1)
}
const meshesDir = path.join(srcRoot, 'meshes')

// limo_four_diff.xacro properties (metres, radians).
const WHEELBASE = 0.2
const TRACK = 0.13
const WHEEL_VERTICAL_OFFSET = -0.1
// base_footprint -> base_link fixed joint (<origin xyz="0 0 0.15">).
const BASE_LINK_HEIGHT = 0.15
// Net z offset of every wheel joint (given directly under base_link in the xacro) once
// re-expressed in the base_footprint frame this script assembles in.
const WHEEL_Z = WHEEL_VERTICAL_OFFSET + BASE_LINK_HEIGHT

// Sub-meshes of limo_base.dae with a bounding-box diagonal below this are dropped before
// merging - screws, vents, small brackets and other "krepezh" that don't affect the overall
// silhouette. Chosen by inspecting limo_base.dae's ~1619 <node> sub-meshes directly (their
// bbox diagonals, in ROS/native frame): at 15mm, 1519/1619 parts (94%) but only 67% of
// triangles are dropped, and the assembled bounding box is unchanged to the mm (the outer
// hull is defined by a handful of large panels, all comfortably above this threshold - the
// biggest is 371mm).
const HARDWARE_DETAIL_THRESHOLD_M = 0.015

// Measured directly off limo_wheel.stl/.dae (both agree - see build notes): the tyre's
// rolling surface is very close to a cylinder of this radius, off-centre along its own axis
// by WHEEL_OFFSET_M (the mesh's origin is at the wheel_link/joint, not the tyre's midpoint).
const WHEEL_RADIUS_M = 0.0481
const WHEEL_WIDTH_M = 0.0747
const WHEEL_OFFSET_M = 0.0063 // (-0.031 + 0.0437) / 2, from the mesh's native y bounding box
const WHEEL_SEGMENTS = 16

// Part offsets and rotations, in ROS/URDF metres and radians, derived from
// limo_four_diff.xacro. Each wheel joint origin is used directly (the wheel_link's own visual
// origin is identity, so the mesh sits exactly at the joint). The base's visual origin
// (xyz="0 0 -0.15" rpy="0 0 1.57") is combined with the base_joint above it
// (xyz="0 0 0.15") - the z-translations cancel exactly, leaving only the yaw rotation.
const BASE_PART = { material: 'body', rotate: [0, 0, Math.PI / 2], offset: [0, 0, 0] }
const WHEEL_PARTS = [
  // front_left: identity rotation (rpy 0 0 0)
  { material: 'dark', rotate: [0, 0, 0], offset: [WHEELBASE / 2, TRACK / 2, WHEEL_Z] },
  // front_right: rpy pi 0 0 - mirrors the wheel onto the right side
  { material: 'dark', rotate: [Math.PI, 0, 0], offset: [WHEELBASE / 2, -TRACK / 2, WHEEL_Z] },
  // rear_left: identity rotation
  { material: 'dark', rotate: [0, 0, 0], offset: [-WHEELBASE / 2, TRACK / 2, WHEEL_Z] },
  // rear_right: rpy pi 0 0
  { material: 'dark', rotate: [Math.PI, 0, 0], offset: [-WHEELBASE / 2, -TRACK / 2, WHEEL_Z] },
]

// Flat display colours, in sRGB, derived from the actual <diffuse> values authored in
// limo_base.dae / limo_wheel.dae (not the STL, which carries no colour - the DAE geometry was
// cross-checked against the STL and is identical up to the mm/m unit, see build notes). The
// DAE's raw diffuse floats are already sRGB (e.g. the dark value is exactly 25/255); three's
// Color.setHex/setRGB does the sRGB->linear conversion for us - do not hand-linearize these.
//   body ~0xfafafa: the dominant (~85% of base mesh vertices) white/cream chassis colour
//   dark ~0x262626: a flat stand-in for the base mesh's dark frame trim (~13%, authored
//     0x191919) and the wheel/tyre colour (100% of wheel mesh, authored 0x333333) - merged
//     into one tone rather than kept as 3+ materials, matching the turtlebot3 script's
//     one-flat-colour-per-part approach. The tiny red tail-light and green LED regions
//     (~1.6% of base mesh vertices combined) are also folded in here.
const MATERIAL_COLORS = {
  body: 0xfafafa,
  dark: 0x262626,
}

// Expected bounding box in SCENE axes (Y-up, metres) after the bake, measured directly off the
// assembled (pre-decimation) meshes. The vendor spec (322 x 220 x 251mm) describes the
// assembled robot including wheels; this mesh-derived box comes out within ~3mm of that on
// every axis (see ~/reports/M-limo.md) - the small residual is the wheel proxy's cylindrical
// approximation plus rounding in the xacro's own track/wheelbase properties, not a unit or
// axis bug. Guards against a future regeneration silently dropping the mm->m scale or an axis
// convention.
const EXPECTED_BOUNDS = {
  x: [-0.163, 0.1585],
  y: [0, 0.2517],
  z: [-0.1087, 0.1087],
}
const BOUNDS_TOLERANCE_M = 0.001

// Applies a part's URDF-derived rotation (fixed-axis rpy) then translation, in place - mirrors
// URDF's "rotate about the part's own origin, then translate into the parent frame" order.
function applyPartTransform(geometry, { offset, rotate }) {
  const [rx, ry, rz] = rotate
  if (rx) geometry.rotateX(rx)
  if (ry) geometry.rotateY(ry)
  if (rz) geometry.rotateZ(rz)
  geometry.translate(offset[0], offset[1], offset[2])
}

// Loads limo_base.dae via ColladaLoader rather than limo_base.stl (unlike
// build-turtlebot-glb.mjs, which uses STL throughout). Two reasons:
//  - the STL is one undifferentiated triangle soup, while the DAE preserves the ~1600
//    individual <node> sub-meshes CAD-exported it with - which is what lets us drop the small
//    hardware ones below (see HARDWARE_DETAIL_THRESHOLD_M).
//  - the DAE also carries each sub-mesh's authored diffuse colour (see MATERIAL_COLORS),
//    rather than requiring per-part colours to be pulled from a URDF <material> tag the way
//    turtlebot3_burger.urdf provides them (limo_four_diff.xacro doesn't colour the base at
//    all - the mesh is the only source of its colour).
async function loadBasePart() {
  const file = path.join(meshesDir, 'limo_base.dae')
  const text = await readFile(file, 'utf8')
  const loader = new ColladaLoader()
  const collada = loader.parse(text, path.dirname(file) + path.sep)
  const scene = collada.scene
  // Undo ColladaLoader's automatic Z-up -> Y-up scene rotation (it only rotates the Object3D,
  // it does not touch vertex data) so traversal below sees native ROS/URDF-frame coordinates,
  // matching the frame the wheel proxies and PARTS offsets are defined in. The single
  // Y-up bake happens once, at the very end, exactly as for the STL-based turtlebot3 script.
  scene.rotation.set(0, 0, 0)
  scene.updateMatrixWorld(true)

  const geosByMaterial = { body: [], dark: [] }
  let totalTri = 0
  let keptTri = 0
  scene.traverse((object) => {
    if (!object.isMesh) return
    let geometry = object.geometry.clone()
    // Bake this sub-mesh's node transform (translation from its parent <node>, if any) into
    // its own vertex data before measuring/merging - matrixWorld is relative to the
    // (now-de-rotated) scene root, i.e. already in the base_link-local frame the URDF uses.
    geometry.applyMatrix4(object.matrixWorld)
    geometry.computeBoundingBox()
    const size = new THREE.Vector3()
    geometry.boundingBox.getSize(size)
    const triCount = (geometry.index ? geometry.index.count : geometry.attributes.position.count) / 3
    totalTri += triCount
    if (size.length() < HARDWARE_DETAIL_THRESHOLD_M) return // drop: hardware/greeble detail
    keptTri += triCount

    geometry.deleteAttribute('normal') // re-derived after welding, see the STL-era comment this replaces
    geometry.deleteAttribute('uv')
    geometry = mergeVertices(geometry)

    // Colour bucket: three's ColladaLoader materials carry .color already converted
    // sRGB->linear; compare against the same linear values noted in MATERIAL_COLORS'
    // sRGB sources (0.098 authored dark, 0.956 authored light -> ~0.0097/0.956 linear).
    const c = object.material?.color
    const isDark = c && c.r < 0.5
    geosByMaterial[isDark ? 'dark' : 'body'].push(geometry)
  })
  console.log(
    `Base mesh: kept ${keptTri}/${totalTri} triangles ` +
      `(dropped sub-meshes under ${(HARDWARE_DETAIL_THRESHOLD_M * 1000).toFixed(0)}mm bbox diagonal)`,
  )

  for (const geos of Object.values(geosByMaterial)) {
    for (const geo of geos) applyPartTransform(geo, BASE_PART)
  }
  return geosByMaterial
}

// Stands in a plain cylinder for the wheel's visual mesh - see the header comment for why
// (the real mesh's tyre-tread detail resists decimation the same way the base's hardware
// detail does, but as a single node, so it can't be filtered the same way).
function loadWheelProxy(part) {
  // CylinderGeometry's default axis is local Y - which is exactly the ROS-frame wheel axis
  // (X-forward, Y-left, Z-up: a wheel's rotation axis points left-right), so no extra
  // rotation is needed to align it before applying the part's own rotate/offset.
  const geometry = new THREE.CylinderGeometry(WHEEL_RADIUS_M, WHEEL_RADIUS_M, WHEEL_WIDTH_M, WHEEL_SEGMENTS)
  geometry.translate(0, WHEEL_OFFSET_M, 0) // re-centre to match the real mesh's off-centre origin
  // Match the base part's attribute set exactly (mergeGeometries requires identical attributes
  // across all inputs in a material bucket) - normal is re-derived after merging, uv is unused.
  geometry.deleteAttribute('normal')
  geometry.deleteAttribute('uv')
  applyPartTransform(geometry, part)
  return geometry
}

async function main() {
  const baseFile = path.join(meshesDir, 'limo_base.dae')
  if (!existsSync(baseFile)) {
    console.error(`Missing mesh: ${baseFile}\nIs ${srcRoot} a limo_description checkout?`)
    process.exit(1)
  }

  const geosByMaterial = await loadBasePart()
  for (const part of WHEEL_PARTS) {
    geosByMaterial[part.material].push(loadWheelProxy(part))
  }

  // Ground-contact correction: find the lowest z across the whole assembly (still in ROS/
  // Z-up frame here) and shift everything down so it sits exactly at z=0, matching Burger's
  // footprint-at-ground-level convention. See the header comment for why this is a ~2mm
  // nudge, not a scale fix - the xacro's wheel_vertical_offset property is a round number
  // that only approximates the real wheel mesh's radius (WHEEL_RADIUS_M, measured off the
  // mesh directly).
  let minZ = Infinity
  for (const geos of Object.values(geosByMaterial)) {
    for (const geo of geos) {
      geo.computeBoundingBox()
      minZ = Math.min(minZ, geo.boundingBox.min.z)
    }
  }
  console.log(`Ground-contact correction: shifting assembly by ${(-minZ * 1000).toFixed(2)}mm`)
  for (const geos of Object.values(geosByMaterial)) {
    for (const geo of geos) geo.translate(0, 0, -minZ)
  }

  const root = new THREE.Group()
  root.name = 'limo'

  for (const [key, geos] of Object.entries(geosByMaterial)) {
    if (geos.length === 0) continue
    const merged = geos.length > 1 ? mergeGeometries(geos, false) : geos[0]
    // Bake the ROS (Z-up) -> three.js scene (Y-up) axis change once, on the merged geometry,
    // not as a runtime transform on the loaded object - see robotModelCache.ts and
    // robot-heading.ts for why the model is authored directly in scene axes.
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

  // Assert before writing: catches a forgotten mm->m scale or a wrong axis bake immediately,
  // rather than as a "the robot is 1000x too big" bug report later.
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
  const outPath = path.join(outDir, 'limo-raw.glb')
  await writeFile(outPath, Buffer.from(glb))
  console.log(`Wrote ${outPath} (${(glb.byteLength / 1e6).toFixed(2)} MB, undecimated)`)
  console.log('Next: decimate + quantize with gltf-transform, see the comment at the top of this script.')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
