#!/usr/bin/env node
// One-off, offline conversion of the ROSbot XL + OpenMANIPULATOR-X meshes into a single
// scene-ready .glb (static, non-articulated - base + arm fused into one assembly in a
// neutral/"home" pose, all joint angles = 0). Not run at build time or runtime - see
// frontend/README.md "Regenerating the robot model" for the general approach (this script
// follows the same pattern as build-turtlebot-glb.mjs, extended for two source packages and
// a mix of STL (arm) and COLLADA/.dae (base) meshes).
//
// Usage:
//   node scripts/build-rosbot-xl-arm-glb.mjs <path-to-rosbot_description> <path-to-open_manipulator_x_description>
//
// <path-to-rosbot_description> is a checkout of https://github.com/husarion/rosbot_ros
// (Apache 2.0), specifically the rosbot_description/ directory (contains
// meshes/rosbot_xl/{body.dae,wheel_a.dae,wheel_b.dae} and
// urdf/rosbot_xl/{body.urdf.xacro,wheel.urdf.xacro,rosbot_xl_macro.urdf.xacro}).
// rosbot_xl_description used to be its own repo but was folded into rosbot_ros; the mesh
// files and their Apache-2.0 license are unaffected by the move.
//
// <path-to-open_manipulator_x_description> is a checkout of the `humble` branch of
// https://github.com/ROBOTIS-GIT/open_manipulator (Apache 2.0), specifically the
// open_manipulator_x_description/ directory (contains meshes/{link1..5,gripper_*}.stl and
// urdf/open_manipulator_x.urdf.xacro). The `humble` branch is used (not `main`) because
// `main` renamed the package to open_manipulator_description and restructured it for a
// newer OpenMANIPULATOR line - `humble` still matches the package name
// (open_manipulator_x_description) that rosbot_xl_manipulation_ros's xacro depends on.
//
// What this does that a runtime URDFLoader would otherwise do on every page load, baked in
// here instead:
//   - mm -> m for the arm's STL meshes (URDF <mesh scale> is 0.001); the base's .dae meshes
//     are already authored in metres (<unit meter="1"/>)
//   - every part offset from the URDF chain: base_link -> body_link -> {wheel x4, cover_link
//     -> link1 -> ... -> link5 -> gripper x2}, taken straight from rosbot_xl_macro.urdf.xacro
//     / body.urdf.xacro / wheel.urdf.xacro (rosbot_description) and
//     rosbot_xl_manipulation.urdf.xacro (rosbot_xl_manipulation_description) and
//     open_manipulator_x.urdf.xacro (open_manipulator_x_description) - no joints, no
//     articulation, everything fused into a static assembly with all arm joint angles at 0
//     (the URDF's zero/rest pose - the only chain-wide "neutral" position that needs no extra
//     rotation math, since every revolute joint's own <origin rpy> is already identity)
//   - no rotation is needed anywhere in the base or arm chain either: every joint in both
//     URDFs has rpy="0 0 0" except the wheel visuals' rotateZ(0 or pi), so the whole assembly
//     reduces to translation-only part placement plus that one per-wheel Z rotation
//   - the ROS (Z-up) -> three.js scene (Y-up) axis change, via rotateX(-PI/2), applied once
//     to the whole assembly (matching build-turtlebot-glb.mjs)
//   - flat materials (the .dae's baked-in vertex colours/multi-material groups and the STL's
//     absent colours are both discarded in favour of flat MeshStandardMaterials, matching the
//     Burger script's approach - this is a low-poly footprint/silhouette asset, not a
//     photoreal one)
//
// Coordinate origin: base_link, i.e. the centre of the wheel footprint at ground level -
// same convention as turtlebot3_burger.glb. wheel_radius (0.048 m, non-mecanum) is the
// vertical offset baked into body_link/cover_link/the arm chain, same role
// EXPECTED_BOUNDS/BOUNDS_TOLERANCE_M plays in the Burger script, this script asserts the
// *base-only* footprint bbox instead (see BASE_EXPECTED_BOUNDS below) since the arm's pose
// is a modelling choice, not a fixed physical constant to sanity-check against.
//
// The exported scene has two top-level groups, "base" and "arm", so a downstream bbox
// measurement (for the footprint radius) can traverse only the "base" node and ignore the
// arm - see scripts/measure-rosbot-xl-arm-bbox.mjs.
//
// After this script produces raw.glb, decimate + quantize it with gltf-transform (dev-only,
// not a project dependency - see frontend/README.md for the general command line; the ratio
// used for the committed model is tuned lower than Burger's because this asset has ~4x the
// source triangle count):
//   npx --yes @gltf-transform/cli@4 weld     raw.glb  w.glb
//   npx --yes @gltf-transform/cli@4 simplify w.glb    s.glb --ratio 0.04 --error 0.005
//   npx --yes @gltf-transform/cli@4 quantize s.glb    q.glb
//   npx --yes @gltf-transform/cli@4 dedup    q.glb    ../public/models/rosbot_xl_arm.glb
//
// Do not Draco/meshopt-compress this model - see build-turtlebot-glb.mjs for why (no decoder
// wired into the app).

import { readFile, writeFile, mkdir } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { JSDOM } from 'jsdom'

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

// ColladaLoader (used for the base's .dae meshes) parses XML via the browser's DOMParser and
// touches `document` for a couple of helpers. Node has neither - polyfill with jsdom, which is
// already a project devDependency (used elsewhere for DOM-ish testing), so this stays a
// same-repo-only dependency like the FileReader shim above.
const dom = new JSDOM()
globalThis.DOMParser = dom.window.DOMParser
globalThis.XMLSerializer = dom.window.XMLSerializer
globalThis.document = dom.window.document
globalThis.window = dom.window

import * as THREE from 'three'
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js'
import { ColladaLoader } from 'three/examples/jsm/loaders/ColladaLoader.js'
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'
import { mergeGeometries, mergeVertices } from 'three/examples/jsm/utils/BufferGeometryUtils.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const baseRoot = process.argv[2]
const armRoot = process.argv[3]
if (!baseRoot || !armRoot) {
  console.error(
    'Usage: node scripts/build-rosbot-xl-arm-glb.mjs <path-to-rosbot_description> <path-to-open_manipulator_x_description>',
  )
  process.exit(1)
}
const baseMeshesDir = path.join(baseRoot, 'meshes/rosbot_xl')
const armMeshesDir = path.join(armRoot, 'meshes')

// --- Frame offsets, in ROS/URDF metres -------------------------------------------------
// Non-mecanum wheel_radius, from rosbot_xl_macro.urdf.xacro:
//   <xacro:property name="wheel_radius" value="${0.05 if mecanum else 0.048}" />
// This build uses the default (mecanum="false") ROSbot XL, matching
// rosbot_xl_manipulation.urdf.xacro's <xacro:arg name="mecanum" default="false" />.
const WHEEL_RADIUS = 0.048

// base_link -> body_link, from body.urdf.xacro's base_to_body_joint
const BODY_Z = WHEEL_RADIUS // + 0.0 xy
// body_link -> cover_link, from body.urdf.xacro's body_to_cover_joint
const COVER_Z = BODY_Z + 0.08345

// wheel.urdf.xacro (mecanum=false branch): wheel_separation_x=0.170, wheel_separation_y=0.248
const WHEEL_SEP_X = 0.17
const WHEEL_SEP_Y = 0.248
// side: [x, y, mesh file, visual rotateZ]
const WHEELS = [
  { side: 'fl', x: WHEEL_SEP_X / 2, y: WHEEL_SEP_Y / 2, file: 'wheel_b.dae', rotZ: Math.PI },
  { side: 'fr', x: WHEEL_SEP_X / 2, y: -WHEEL_SEP_Y / 2, file: 'wheel_a.dae', rotZ: Math.PI },
  { side: 'rl', x: -WHEEL_SEP_X / 2, y: WHEEL_SEP_Y / 2, file: 'wheel_a.dae', rotZ: 0 },
  { side: 'rr', x: -WHEEL_SEP_X / 2, y: -WHEEL_SEP_Y / 2, file: 'wheel_b.dae', rotZ: 0 },
]

// cover_link -> link1 (arm base), from rosbot_xl_manipulation.urdf.xacro:
//   <xacro:manipulator.open_manipulator_x parent_link="cover_link" xyz="-0.122 0.0 0.0" .../>
const ARM_BASE = [-0.122, 0, COVER_Z]

// Arm chain from open_manipulator_x.urdf.xacro. Every joint's <origin rpy> is "0 0 0" and this
// build holds every joint angle at 0 (the URDF's own rest pose) - so with axes revolute-about-
// Y/Z but angle 0, the whole chain is pure translation, no rotation math needed. Offsets are
// cumulative from link1 (frame, not visual origin) unless noted.
const link1 = ARM_BASE
const link2 = add(link1, [0.012, 0, 0.017]) // joint1 origin
const link2Visual = add(link2, [0, 0, 0.019]) // link2's own <visual><origin>
const link3 = add(link2, [0, 0, 0.0595]) // joint2 origin (from link2 FRAME, not its visual)
const link4 = add(link3, [0.024, 0, 0.128]) // joint3 origin
const link5 = add(link4, [0.124, 0, 0]) // joint4 origin
const gripperLeft = add(link5, [0.0817, 0.021, 0]) // gripper_left_joint origin
const gripperRight = add(link5, [0.0817, -0.021, 0]) // gripper_right_joint origin

function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

// STL arm parts: file, translation offset, merge group (colour)
const ARM_STL_PARTS = [
  { file: 'link1.stl', offset: link1, group: 'arm' },
  { file: 'link2.stl', offset: link2Visual, group: 'arm' },
  { file: 'link3.stl', offset: link3, group: 'arm' },
  { file: 'link4.stl', offset: link4, group: 'arm' },
  { file: 'link5.stl', offset: link5, group: 'arm' },
  { file: 'gripper_left_palm.stl', offset: gripperLeft, group: 'gripper' },
  { file: 'gripper_right_palm.stl', offset: gripperRight, group: 'gripper' },
]

// Material colours. The base .dae files embed multiple real materials (white/grey plastic,
// black rubber, red/green accents); the arm STLs carry none. Both are flattened to a small
// number of flat MeshStandardMaterials, matching build-turtlebot-glb.mjs's approach for a
// low-poly footprint/silhouette asset - not aiming for photoreal parity.
const MATERIAL_COLORS = {
  chassis: 0x9a9a9a, // sampled from body.dae's dominant "Light"/"Material.001" colour (~0.6 sRGB)
  wheel: 0x262626, // sampled from body.dae/wheel_*.dae's near-black "BlackGum" tire colour
  arm: 0x808080, // open_manipulator_x_description/urdf/materials.xacro <material name="grey"> rgba 0.5 0.5 0.5
  gripper: 0x4d4d4d, // slightly darker than the arm links so the gripper reads as a distinct part
}

// Expected bounding box of the BASE ONLY (no arm), in SCENE axes (Y-up, metres) after the
// bake. This is a regression guard, not a vendor-spec assertion - the vendor's published
// 332x284x131mm dimensions are compared separately, in the written report, against the
// *measured* number (see measure-rosbot-xl-arm-bbox.mjs), not baked in here as a hard gate.
let baseGroup
let armGroup

async function loadSTLPart(file, dir) {
  const buf = await readFile(path.join(dir, file))
  const loader = new STLLoader()
  let geometry = loader.parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
  geometry.scale(0.001, 0.001, 0.001) // mm -> m
  geometry.deleteAttribute('normal') // see build-turtlebot-glb.mjs: weld must be position-only
  geometry = mergeVertices(geometry)
  geometry.deleteAttribute('uv')
  return geometry
}

// Bakes a COLLADA (.dae) part's mesh geometry in the file's own Z-up "asset" frame - i.e. the
// same frame the URDF's <mesh filename="..."/> expects, NOT three.js's Y-up scene frame.
// ColladaLoader.parse() returns a scene graph with the Y-up conversion applied as a rotation
// on the ROOT node only (see the loader's own console warning); each mesh's *own* local
// position/quaternion/scale (its `.matrix`, relative to that root) is the mesh's authored
// offset within the asset and must be kept. So: bake each mesh's local `.matrix` into its
// geometry, but deliberately do NOT include the root's rotation - that axis conversion is
// applied once, globally, at the very end of this script (rotateX(-PI/2) on the whole
// assembly), exactly like the STL parts above and like build-turtlebot-glb.mjs.
async function loadColladaGeometry(file, dir) {
  const text = await readFile(path.join(dir, file), 'utf8')
  const loader = new ColladaLoader()
  const result = loader.parse(text, '')
  const geometries = []
  result.scene.traverse((obj) => {
    if (!obj.isMesh) return
    obj.updateMatrix()
    const geo = obj.geometry.clone()
    geo.applyMatrix4(obj.matrix) // bake the mesh's own local (asset-frame) transform only
    geometries.push(geo)
  })
  let merged = geometries.length > 1 ? mergeGeometries(geometries, false) : geometries[0]
  // Drop colour/UV attributes from the multi-material .dae (flat materials are assigned
  // per-part below instead) and any second UV channel some exporters add.
  for (const attr of ['color', 'uv', 'uv2']) {
    if (merged.getAttribute(attr)) merged.deleteAttribute(attr)
  }
  merged.deleteAttribute('normal') // position-only weld, same reasoning as the STL parts
  merged = mergeVertices(merged)
  return merged
}

async function main() {
  const requiredBase = ['body.dae', ...new Set(WHEELS.map((w) => w.file))]
  for (const f of requiredBase) {
    const full = path.join(baseMeshesDir, f)
    if (!existsSync(full)) {
      console.error(`Missing mesh: ${full}\nIs ${baseRoot} a rosbot_description checkout?`)
      process.exit(1)
    }
  }
  for (const p of ARM_STL_PARTS) {
    const full = path.join(armMeshesDir, p.file)
    if (!existsSync(full)) {
      console.error(`Missing mesh: ${full}\nIs ${armRoot} an open_manipulator_x_description checkout?`)
      process.exit(1)
    }
  }

  // --- BASE: chassis + 4 wheels ---------------------------------------------------------
  const chassisGeo = await loadColladaGeometry('body.dae', baseMeshesDir)
  chassisGeo.translate(0, 0, BODY_Z)

  const wheelGeosByFile = {}
  const wheelGeos = []
  for (const w of WHEELS) {
    if (!wheelGeosByFile[w.file]) {
      wheelGeosByFile[w.file] = await loadColladaGeometry(w.file, baseMeshesDir)
    }
    const geo = wheelGeosByFile[w.file].clone()
    geo.rotateZ(w.rotZ)
    geo.translate(w.x, w.y, BODY_Z) // wheel_link z == body_link z (wheel joint origin z=0)
    wheelGeos.push(geo)
  }
  const wheelsGeo = mergeGeometries(wheelGeos, false)

  baseGroup = new THREE.Group()
  baseGroup.name = 'base'
  for (const [geo, colorKey, name] of [
    [chassisGeo, 'chassis', 'base_chassis'],
    [wheelsGeo, 'wheel', 'base_wheels'],
  ]) {
    geo.rotateX(-Math.PI / 2) // ROS Z-up -> three.js scene Y-up, baked once, per part group
    geo.computeVertexNormals()
    const mat = new THREE.MeshStandardMaterial({
      color: MATERIAL_COLORS[colorKey],
      roughness: 0.9,
      metalness: 0,
    })
    const mesh = new THREE.Mesh(geo, mat)
    mesh.name = name
    baseGroup.add(mesh)
  }

  // --- ARM: 5 links + 2 gripper fingers, STL, mm -> m ------------------------------------
  const armGeosByGroup = { arm: [], gripper: [] }
  for (const part of ARM_STL_PARTS) {
    const geo = await loadSTLPart(part.file, armMeshesDir)
    geo.translate(part.offset[0], part.offset[1], part.offset[2])
    armGeosByGroup[part.group].push(geo)
  }

  armGroup = new THREE.Group()
  armGroup.name = 'arm'
  for (const [key, geos] of Object.entries(armGeosByGroup)) {
    if (geos.length === 0) continue
    const merged = geos.length > 1 ? mergeGeometries(geos, false) : geos[0]
    merged.rotateX(-Math.PI / 2) // same single global axis bake as the base parts
    merged.computeVertexNormals()
    const mat = new THREE.MeshStandardMaterial({
      color: MATERIAL_COLORS[key],
      roughness: 0.9,
      metalness: 0,
    })
    const mesh = new THREE.Mesh(merged, mat)
    mesh.name = key === 'arm' ? 'arm_links' : 'arm_gripper'
    armGroup.add(mesh)
  }

  const root = new THREE.Group()
  root.name = 'rosbot_xl_arm'
  root.add(baseGroup)
  root.add(armGroup)

  const baseBox = new THREE.Box3().setFromObject(baseGroup)
  const fullBox = new THREE.Box3().setFromObject(root)
  console.log(
    `Base-only bounds (scene axes, m): x [${baseBox.min.x.toFixed(4)}, ${baseBox.max.x.toFixed(4)}] ` +
      `y [${baseBox.min.y.toFixed(4)}, ${baseBox.max.y.toFixed(4)}] ` +
      `z [${baseBox.min.z.toFixed(4)}, ${baseBox.max.z.toFixed(4)}]`,
  )
  console.log(
    `Full (base+arm) bounds (scene axes, m): x [${fullBox.min.x.toFixed(4)}, ${fullBox.max.x.toFixed(4)}] ` +
      `y [${fullBox.min.y.toFixed(4)}, ${fullBox.max.y.toFixed(4)}] ` +
      `z [${fullBox.min.z.toFixed(4)}, ${fullBox.max.z.toFixed(4)}]`,
  )

  const exporter = new GLTFExporter()
  const glb = await new Promise((resolve, reject) => {
    exporter.parse(root, resolve, reject, { binary: true })
  })

  const outDir = path.join(__dirname, 'out')
  await mkdir(outDir, { recursive: true })
  const outPath = path.join(outDir, 'rosbot_xl_arm_raw.glb')
  await writeFile(outPath, Buffer.from(glb))
  console.log(`Wrote ${outPath} (${(glb.byteLength / 1e6).toFixed(2)} MB, undecimated)`)
  console.log('Next: decimate + quantize with gltf-transform, see the comment at the top of this script.')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
