#!/usr/bin/env node
// One-off, offline conversion of the TurtleBot3 Waffle Pi's ROS meshes into a single scene-ready
// .glb. Not run at build time or runtime - see frontend/README.md "Regenerating the robot
// model" for why, and for how to re-run this after a mesh update upstream.
//
// Usage:
//   node scripts/build-waffle-pi-glb.mjs <path-to-turtlebot3_description>
//
// <path-to-turtlebot3_description> is a checkout of
//   https://github.com/ROBOTIS-GIT/turtlebot3 (Apache 2.0), specifically the
//   turtlebot3_description/ directory (contains meshes/ and urdf/turtlebot3_waffle_pi.urdf).
//
// Waffle Pi, not Waffle: this uses waffle_pi_base.stl / turtlebot3_waffle_pi.urdf, the
// Raspberry Pi camera variant that ROBOTIS still supports. The plain "Waffle" (Intel Joule)
// is end-of-life upstream and is NOT what this script builds.
//
// What this does that a runtime URDFLoader would otherwise do on every page load, baked in
// here instead:
//   - mm -> m (the URDF's mesh <scale> is 0.001)
//   - the 4 part offsets from turtlebot3_waffle_pi.urdf (base, 2 wheels, LDS) - no joints, no
//     articulation, everything is fused into a static assembly
//   - the ROS (Z-up) -> three.js scene (Y-up) axis change, via rotateX(-PI/2)
//   - flat, unlit-friendly-but-lit-capable materials matching the URDF's "light_black" /
//     "dark" colours
//
// After this script produces raw.glb, decimate + quantize it with gltf-transform (dev-only,
// not a project dependency - see frontend/README.md for the exact command line used for the
// committed model). The Waffle Pi base is ~19x heavier in raw triangles than the Burger's, so
// it needs a much more aggressive simplify ratio to land near the same ~400KB budget:
//   npx --yes @gltf-transform/cli@4 weld     raw.glb  w.glb
//   npx --yes @gltf-transform/cli@4 simplify w.glb    s.glb --ratio 0.02 --error 0.005
//   npx --yes @gltf-transform/cli@4 quantize s.glb    q.glb
//   npx --yes @gltf-transform/cli@4 dedup    q.glb    ../public/models/waffle_pi.glb
//
// Quantization is safe with no runtime change: three's GLTFLoader (three/examples/jsm) handles
// KHR_mesh_quantization natively. Do not Draco/meshopt-compress this model - for an asset this
// small the decoder costs more bytes than compression saves, and neither decoder is wired into
// the app (see robotModelCache.ts).

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
  console.error('Usage: node scripts/build-waffle-pi-glb.mjs <path-to-turtlebot3_description>')
  process.exit(1)
}
const meshesDir = path.join(srcRoot, 'meshes')

// Part offsets, in ROS/URDF metres, read straight from turtlebot3_waffle_pi.urdf. Each offset
// below is base_joint's origin (base_footprint -> base_link, xyz="0 0 0.010") composed with
// that part's own joint/visual origin relative to base_link - the same "sum the chain down to
// base_footprint" rule the Burger script used, since base_footprint is where we want (0,0,0) to
// land. All net rotations in the chain down to each visual are identity (the mesh <visual>
// origins carry rpy="0 0 0", and the only non-zero rpy - wheel joints' -1.57 rad about X -
// exactly cancels the wheel STL visuals' own +1.57 rad, per the URDF), so only translation is
// needed per part - the shared Z-up -> Y-up rotation is applied once at the end, to the whole
// assembly.
//   base:  base_joint (0, 0, 0.010)      + base visual origin  (-0.064, 0, 0.0)   = (-0.064, 0, 0.010)
//   wheel: base_joint (0, 0, 0.010)      + wheel_*_joint       (0, ±0.144, 0.023) = (0, ±0.144, 0.033)
//   lds:   base_joint (0, 0, 0.010)      + scan_joint          (-0.064, 0, 0.122) = (-0.064, 0, 0.132)
const PARTS = [
  { file: 'bases/waffle_pi_base.stl', offset: [-0.064, 0, 0.01], material: 'base' },
  { file: 'wheels/left_tire.stl', offset: [0, 0.144, 0.033], material: 'dark' },
  { file: 'wheels/right_tire.stl', offset: [0, -0.144, 0.033], material: 'dark' },
  { file: 'sensors/lds.stl', offset: [-0.064, 0, 0.132], material: 'dark' },
]

// URDF material rgba, read as sRGB display colours (three's Color.setHex/setRGB does the
// sRGB->linear conversion for us - do not hand-linearize these). Same material names/values as
// the Burger's URDF ("light_black" / "dark" are shared across the turtlebot3 description pkg).
const MATERIAL_COLORS = {
  base: 0x666666, // "light_black" rgba 0.4 0.4 0.4
  dark: 0x4d4d4d, // "dark"        rgba 0.3 0.3 0.3
}

// Expected bounding box in SCENE axes (Y-up, metres) after the bake, measured directly off the
// assembled meshes. Height (y: 141.1mm) matches the ROBOTIS spec (281x306x141mm) almost
// exactly; footprint (x: 274.0mm, z: 306.2mm) matches the 306mm lateral dimension closely and
// is ~2.5% short of the 281mm front-back figure (mesh vs. vendor's marketing dimension, which
// likely includes the bumper/caster extremity - see ~/reports/M-waffle_pi.md). Guards against a
// future regeneration silently dropping the mm->m scale or an axis convention.
const EXPECTED_BOUNDS = {
  x: [-0.1968, 0.0773],
  y: [-0.0001, 0.141],
  z: [-0.1531, 0.1531],
}
const BOUNDS_TOLERANCE_M = 0.001

async function loadPart({ file, offset }) {
  const buf = await readFile(path.join(meshesDir, file))
  const loader = new STLLoader()
  let geometry = loader.parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
  geometry.scale(0.001, 0.001, 0.001) // mm -> m
  // STLLoader sets a per-vertex NORMAL attribute from the STL's raw facet normals (each
  // triangle's 3 vertex copies all carry that triangle's flat face normal). mergeVertices
  // hashes ALL attributes together, so if we welded now, position-coincident vertices on
  // adjacent, differently-angled faces would still count as distinct (different normal) and
  // stay unwelded. Drop NORMAL first so welding is purely position-based; computeVertexNormals()
  // below then derives correct per-shared-vertex (angle-averaged) normals on the now-indexed
  // geometry.
  geometry.deleteAttribute('normal')
  // STLLoader's output is also a triangle soup: every triangle gets its own 3 independent
  // vertex copies, none sharing an index, even along a shared edge - that's the STL format,
  // not a parsing bug. Weld coincident positions (0.1mm tolerance, reasonable for CAD-export
  // noise) into an indexed geometry. This step is also what makes decimation effective -
  // meshoptimizer's simplifier collapses shared edges, and an unwelded soup has none.
  geometry = mergeVertices(geometry)
  geometry.translate(offset[0], offset[1], offset[2]) // ROS-frame offset from the URDF
  geometry.deleteAttribute('uv') // STL has no UVs to begin with; mergeGeometries wants matching attrs
  return geometry
}

async function main() {
  for (const p of PARTS) {
    const full = path.join(meshesDir, p.file)
    if (!existsSync(full)) {
      console.error(`Missing mesh: ${full}\nIs ${srcRoot} a turtlebot3_description checkout?`)
      process.exit(1)
    }
  }

  const geosByMaterial = { base: [], dark: [] }
  for (const part of PARTS) {
    const geo = await loadPart(part)
    geosByMaterial[part.material].push(geo)
  }

  const root = new THREE.Group()
  root.name = 'turtlebot3_waffle_pi'

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
  const outPath = path.join(outDir, 'raw-waffle-pi.glb')
  await writeFile(outPath, Buffer.from(glb))
  console.log(`Wrote ${outPath} (${(glb.byteLength / 1e6).toFixed(2)} MB, undecimated)`)
  console.log('Next: decimate + quantize with gltf-transform, see the comment at the top of this script.')
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
