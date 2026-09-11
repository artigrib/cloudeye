#!/usr/bin/env node
// One-off, offline conversion of the TurtleBot4 Standard's ROS meshes into a single scene-ready
// .glb. Not run at build time or runtime - see frontend/README.md "Regenerating the robot
// model" for the equivalent TurtleBot3 Burger process (this script follows the same shape).
//
// Usage:
//   node scripts/build-turtlebot4-glb.mjs <path-to-turtlebot4_description> <path-to-irobot_create_description>
//
// <path-to-turtlebot4_description> is a checkout of
//   https://github.com/turtlebot/turtlebot4 (Apache 2.0), specifically the
//   turtlebot4_description/ directory (contains meshes/ and urdf/standard/turtlebot4.urdf.xacro).
// <path-to-irobot_create_description> is a checkout of
//   https://github.com/iRobotEducation/create3_sim (BSD-3-Clause), specifically the
//   irobot_create_common/irobot_create_description/ directory - the Create3 base that the
//   TurtleBot4 sits on (turtlebot4_description depends on it via <xacro:include> of
//   create3.urdf.xacro and does not vendor its own base mesh).
//
// Unlike the Burger (STL, requires mm->m and per-part offset only), TurtleBot4's meshes are
// COLLADA (.dae), already authored in metres (<unit meter="1.0"/>), Z-up, and some of them
// (the two Create3 meshes) carry their own internal node transform hierarchy (baking a
// Blender-authored Y-up mesh + mm->m scale into a <matrix> on an ancestor <node>). We let
// three's ColladaLoader parse and resolve that hierarchy for us (scene graph of Object3D with
// position/quaternion/scale per node), then bake each mesh's matrixWorld into its geometry
// before applying the URDF-derived offset - see loadPart() below.
//
// What this does that a runtime URDFLoader would otherwise do on every page load, baked in
// here instead:
//   - the part offsets (translation + rpy) below, hand-derived from
//     turtlebot4_description/urdf/standard/turtlebot4.urdf.xacro and
//     irobot_create_description/urdf/{create3,bumper}.urdf.xacro - see the comment above PARTS
//   - no joints, no articulation, everything fused into a static assembly, all relative to
//     "base_link" (== "base_footprint": per turtlebot4.urdf.xacro's base_footprint_joint,
//     "Create3's base_link is already on the ground" - xyz 0 0 0 between them), i.e. our mesh
//     origin is already the footprint centre at floor level, no extra centring needed
//   - the ROS (Z-up) -> three.js scene (Y-up) axis change, via rotateX(-PI/2), applied once to
//     the whole assembly at the end
//   - flat, unlit-friendly-but-lit-capable materials, one flat colour per part bucket (sampled
//     from each source .dae's own <diffuse> colour - see MATERIAL_COLORS) rather than
//     preserving every mesh's original per-submesh material assignment, since we don't need
//     that fidelity at this size (see frontend/README.md's "detail not needed" note)
//
// Parts deliberately OMITTED (present in the URDF/meshes but not visually load-bearing for a
// silhouette at this scale, and this keeps triangle count down before decimation):
//   - the 4 weight_block.dae instances: fully enclosed inside shell_link, never visible
//   - Create3's bumper_collision.dae, shell_collision.dae: collision-only meshes, not visual
//   - buttons, cliff sensors, IR sensors, caster wheel, IMU, optical mouse: no distinct visual
//     mesh in the URDF (procedural or non-visual), and/or invisible under the shell
//   - dock/visual.dae: the charging dock is a separate object, not part of the robot
//
// After this script produces raw.glb, decimate + quantize it with gltf-transform (dev-only,
// not a project dependency - see frontend/README.md for the exact command line used for the
// committed turtlebot3_burger.glb; the same pipeline is used here):
//   npx --yes @gltf-transform/cli@4 weld     raw.glb  w.glb
//   npx --yes @gltf-transform/cli@4 simplify w.glb    s.glb --ratio 0.10 --error 0.005
//   npx --yes @gltf-transform/cli@4 quantize s.glb    q.glb
//   npx --yes @gltf-transform/cli@4 dedup    q.glb    ../public/models/turtlebot4.glb
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

// ColladaLoader's parser uses the browser's DOMParser to turn the .dae's XML text into a
// Document. Node has no global DOMParser - polyfill it from jsdom (a devDependency already in
// node_modules, only used here, offline, never at build/runtime).
if (typeof globalThis.DOMParser === 'undefined') {
  globalThis.DOMParser = new JSDOM().window.DOMParser
}

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

const tb4Root = process.argv[2]
const createRoot = process.argv[3]
if (!tb4Root || !createRoot) {
  console.error(
    'Usage: node scripts/build-turtlebot4-glb.mjs <path-to-turtlebot4_description> <path-to-irobot_create_description>',
  )
  process.exit(1)
}
const tb4MeshesDir = path.join(tb4Root, 'meshes')
const createMeshesDir = path.join(createRoot, 'meshes')

const cm2m = 0.01
// Height of the Create3's own "base_link" reference above the meshes' internal zero, per
// irobot_create_description/urdf/create3.urdf.xacro's base_link_z_offset property (6.42cm).
// This same xacro property is reused (via xacro's global property scope, since
// turtlebot4.urdf.xacro <xacro:include>s create3.urdf.xacro first) by every TurtleBot4-specific
// offset below that stacks something on top of the Create3 base.
const baseLinkZOffset = 6.42 * cm2m
// Height of "shell_link" above "base_link" (turtlebot4.urdf.xacro's shell_link_joint origin:
// shell_z_offset + base_link_z_offset). Every part below whose xacro macro parent_link defaults
// to shell_link (tower_standoff, tower_sensor_plate, rplidar, camera_bracket - all instantiated
// without overriding parent_link, per turtlebot4_description/urdf/standard/turtlebot4.urdf.xacro)
// is offset from THIS point, not from base_link directly.
const shellZ = 3 * cm2m + baseLinkZOffset

// Part offsets (translation + rpy), in ROS/URDF metres and radians, hand-derived from the URDF
// xacro files below and flattened to be relative to base_link == base_footprint (floor level,
// footprint centre - see the file header comment). Each entry's rpy is the sum of that part's
// <joint><origin rpy=...> (position in the chain) and its own <visual><origin rpy=...> (mesh
// orientation within its link) - most parts have identity on one or the other, noted per entry.
//
//   base (create3.urdf.xacro base_link):
//     body_visual: xyz (0,0, -2.5cm + base_link_z_offset), rpy (0,0,pi/2) [visual-level]
//     bumper_visual (xacro:bumper, instantiated in create3.urdf.xacro): xyz (0,0, -2.5cm +
//       base_link_z_offset), rpy identity (visual has no <origin> in bumper.urdf.xacro)
//   turtlebot4.urdf.xacro (parent shell_link unless noted):
//     shell.dae: joint xyz (0,0, shell_z_offset+base_link_z_offset) rpy 0; visual rpy
//       (pi/2,0,pi/2) - net: shellZ, rpy (pi/2,0,pi/2)
//     tower_standoff x4 (parent shell_link): joint xyz per front/rear-left/right offsets, z =
//       tower_standoff_z_offset; visual rpy (0,pi/2,0) (tower_standoff.urdf.xacro)
//     tower_sensor_plate (parent shell_link): joint xyz (0,0,tower_sensor_plate_z_offset);
//       visual identity
//     rplidar (parent shell_link): joint xyz (rplidar_x/y/z_offset), rpy (0,0,pi/2); visual
//       identity (rplidar.urdf.xacro)
//     camera_bracket "oakd_camera_bracket" (parent shell_link, macro default): joint xyz
//       (camera_mount_x/y/z_offset), rpy identity; visual identity
//     oakd "pro" (parent oakd_camera_bracket, i.e. chained onto the above): joint xyz
//       (oakd_pro_x/y/z_offset relative to oakd_camera_bracket), rpy identity; visual identity
const PARTS = [
  // --- Create3 base (irobot_create_description, BSD-3-Clause) ---
  {
    dir: createMeshesDir,
    file: 'body_visual.dae',
    xyz: [0, 0, -2.5 * cm2m + baseLinkZOffset],
    rpy: [0, 0, Math.PI / 2],
    material: 'base',
  },
  {
    dir: createMeshesDir,
    file: 'bumper_visual.dae',
    xyz: [0, 0, -2.5 * cm2m + baseLinkZOffset],
    rpy: [0, 0, 0],
    material: 'dark',
  },

  // --- TurtleBot4 Standard shell + tower (turtlebot4_description, Apache-2.0) ---
  { dir: tb4MeshesDir, file: 'shell.dae', xyz: [0, 0, shellZ], rpy: [Math.PI / 2, 0, Math.PI / 2], material: 'shell' },
  {
    dir: tb4MeshesDir,
    file: 'tower_standoff.dae',
    xyz: [3.063 * cm2m, 11.431 * cm2m, shellZ + 14.757 * cm2m],
    rpy: [0, Math.PI / 2, 0],
    material: 'dark',
  }, // front_left
  {
    dir: tb4MeshesDir,
    file: 'tower_standoff.dae',
    xyz: [3.063 * cm2m, -11.431 * cm2m, shellZ + 14.757 * cm2m],
    rpy: [0, Math.PI / 2, 0],
    material: 'dark',
  }, // front_right
  {
    dir: tb4MeshesDir,
    file: 'tower_standoff.dae',
    xyz: [-7.607 * cm2m, 9.066 * cm2m, shellZ + 14.757 * cm2m],
    rpy: [0, Math.PI / 2, 0],
    material: 'dark',
  }, // rear_left
  {
    dir: tb4MeshesDir,
    file: 'tower_standoff.dae',
    xyz: [-7.607 * cm2m, -9.066 * cm2m, shellZ + 14.757 * cm2m],
    rpy: [0, Math.PI / 2, 0],
    material: 'dark',
  }, // rear_right
  {
    dir: tb4MeshesDir,
    file: 'tower_sensor_plate.dae',
    xyz: [0, 0, shellZ + 25.257 * cm2m],
    rpy: [0, 0, 0],
    material: 'shell',
  },
  {
    dir: tb4MeshesDir,
    file: 'rplidar.dae',
    xyz: [-4 * cm2m, 0, shellZ + 9.8715 * cm2m],
    rpy: [0, 0, Math.PI / 2],
    material: 'shell',
  },
  {
    dir: tb4MeshesDir,
    file: 'camera_bracket.dae',
    xyz: [-11.8 * cm2m, 0, shellZ + 5.257 * cm2m],
    rpy: [0, 0, 0],
    material: 'dark',
  },
  {
    // oakd_camera_bracket position + oakd_pro's own offset relative to it (chained parent link)
    dir: tb4MeshesDir,
    file: 'oakd_pro.dae',
    xyz: [-11.8 * cm2m + 5.84 * cm2m, 0, shellZ + 5.257 * cm2m + 9.676 * cm2m],
    rpy: [0, 0, 0],
    material: 'dark',
  },
]

// Flat material colours, one per bucket, sampled from the dominant <diffuse> colour in the
// source .dae files (read directly from their <library_effects>, sRGB 0-1 floats * 255):
//   base  <- body_visual.dae "ABS (Top)"   0.2571 0.2571 0.2571 (Create3 body)
//   dark  <- bumper_visual.dae / tower_standoff.dae / camera_bracket.dae / oakd_pro.dae
//           0.1922 0.2039 0.2039 (bumper + tower skeleton + camera)
//   shell <- shell.dae 0.1020 / tower_sensor_plate.dae 0.1010 / rplidar.dae 0.1098 (all ~equal,
//           bucketed together as the dark shell/plate/lidar assembly)
const MATERIAL_COLORS = {
  base: 0x424242,
  dark: 0x313434,
  shell: 0x1a1a1a,
}

// Expected overall bounding box in SCENE axes (Y-up, metres), for a sanity printout only (not a
// hard assert like the Burger script's, since this bbox is hand-derived from ~11 chained URDF
// offsets rather than a single flat part list - see the file header). Compare the printed
// numbers against the vendor's published TurtleBot4 Standard footprint 341x339x351mm.
const VENDOR_DIMENSIONS_MM = { x: 341, y: 339, z: 351 }

const daeCache = new Map()
async function loadDae(dir, file) {
  const key = path.join(dir, file)
  if (daeCache.has(key)) return daeCache.get(key)
  const full = path.join(dir, file)
  if (!existsSync(full)) {
    console.error(`Missing mesh: ${full}`)
    process.exit(1)
  }
  const text = await readFile(full, 'utf-8')
  const loader = new ColladaLoader()
  const collada = loader.parse(text, dir + path.sep)
  const scene = collada.scene
  scene.updateMatrixWorld(true)

  const geos = []
  scene.traverse((obj) => {
    if (obj.isMesh && obj.geometry) {
      const geo = obj.geometry.clone()
      geo.applyMatrix4(obj.matrixWorld)
      geos.push(geo)
    }
  })
  if (geos.length === 0) {
    console.error(`No mesh geometry found in ${full}`)
    process.exit(1)
  }
  // Drop attributes that vary between our parts / that we don't need (see loadPart below for
  // why normal+uv are stripped) before merging multiple mesh nodes within one .dae into one.
  for (const geo of geos) {
    geo.deleteAttribute('normal')
    if (geo.getAttribute('uv')) geo.deleteAttribute('uv')
    if (geo.getAttribute('color')) geo.deleteAttribute('color')
  }
  const merged = geos.length > 1 ? mergeGeometries(geos, false) : geos[0]
  // Every asset here declares up_axis Z_UP (see the file header), which makes ColladaLoader
  // itself set scene.rotation = (-PI/2, 0, 0) to present a Y-up scene (its console.warn above
  // is this happening) - already baked into matrixWorld and hence into `merged` by the
  // applyMatrix4 above. Undo it here so `merged` is back in the .dae's native Z-up metres, i.e.
  // the same ROS/ URDF-native frame the PARTS offsets below (hand-derived from the xacro
  // <origin xyz rpy> values) are expressed in. The single shared Z-up -> Y-up conversion is
  // then re-applied exactly once, to the whole fused assembly, in main() - matching the Burger
  // script's approach instead of doing it piecemeal per source mesh.
  merged.rotateX(Math.PI / 2)
  daeCache.set(key, merged)
  return merged
}

async function loadPart({ dir, file, xyz, rpy }) {
  const base = await loadDae(dir, file)
  let geometry = base.clone()
  // Weld coincident positions (0.1mm tolerance) into an indexed geometry, purely position-based
  // (normal/uv/color already dropped in loadDae) - same rationale as the Burger script: this is
  // what makes decimation effective, and what lets computeVertexNormals() below produce correct
  // angle-averaged smooth normals instead of inheriting the source mesh's own normals verbatim.
  geometry = mergeVertices(geometry, 1e-4)
  // Part-level orientation then position, in the URDF's ROS frame. URDF's <origin rpy="r p y">
  // is fixed-axis (extrinsic) roll-X, pitch-Y, yaw-Z, i.e. R = Rz(yaw)*Ry(pitch)*Rx(roll) applied
  // to a column vector - verified against three's per-order Euler.makeRotationFromEuler output
  // (see scripts/dbg-tmp.mjs history in this change) to be three's 'ZYX' order, NOT 'XYZ' (three
  // and URDF both call it "roll, pitch, yaw" but three's default 'XYZ' order is a different,
  // non-equivalent matrix - confirmed by a non-degenerate test triple during development; this
  // bit the shell.dae part specifically, whose rpy has a non-trivial roll+yaw combination).
  const [roll, pitch, yaw] = rpy
  if (roll || pitch || yaw) {
    geometry.applyMatrix4(new THREE.Matrix4().makeRotationFromEuler(new THREE.Euler(roll, pitch, yaw, 'ZYX')))
  }
  geometry.translate(xyz[0], xyz[1], xyz[2])
  return geometry
}

async function main() {
  const geosByMaterial = { base: [], dark: [], shell: [] }
  for (const part of PARTS) {
    const geo = await loadPart(part)
    geosByMaterial[part.material].push(geo)
  }

  const root = new THREE.Group()
  root.name = 'turtlebot4'

  for (const [key, geos] of Object.entries(geosByMaterial)) {
    if (geos.length === 0) continue
    const merged = geos.length > 1 ? mergeGeometries(geos, false) : geos[0]
    // Bake the ROS (Z-up) -> three.js scene (Y-up) axis change once, on the merged geometry,
    // not as a runtime transform on the loaded object - see robotModelCache.ts and
    // robot-heading.ts for why the model is authored directly in scene axes. Origin stays the
    // footprint centre at floor level (base_link/base_footprint), unchanged by this rotation.
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

  const box = new THREE.Box3().setFromObject(root)
  const size = box.getSize(new THREE.Vector3())
  console.log(
    `Bounds: x [${box.min.x.toFixed(4)}, ${box.max.x.toFixed(4)}] ` +
      `y [${box.min.y.toFixed(4)}, ${box.max.y.toFixed(4)}] ` +
      `z [${box.min.z.toFixed(4)}, ${box.max.z.toFixed(4)}]`,
  )
  console.log(
    `Size (scene axes, m): x=${size.x.toFixed(4)} y(up)=${size.y.toFixed(4)} z=${size.z.toFixed(4)}\n` +
      `Vendor TurtleBot4 Standard (mm): x=${VENDOR_DIMENSIONS_MM.x} y=${VENDOR_DIMENSIONS_MM.y} (footprint) ` +
      `z(height)=${VENDOR_DIMENSIONS_MM.z} - compare footprint (scene x/z) and height (scene y) against these.`,
  )

  const exporter = new GLTFExporter()
  const glb = await new Promise((resolve, reject) => {
    exporter.parse(root, resolve, reject, { binary: true })
  })

  const outDir = path.join(__dirname, 'out')
  await mkdir(outDir, { recursive: true })
  const outPath = path.join(outDir, 'raw-turtlebot4.glb')
  await writeFile(outPath, Buffer.from(glb))
  console.log(`Wrote ${outPath} (${(glb.byteLength / 1e6).toFixed(2)} MB, undecimated)`)
  console.log('Next: decimate + quantize with gltf-transform, see the comment at the top of this script.')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
