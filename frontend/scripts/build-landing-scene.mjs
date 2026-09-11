#!/usr/bin/env node
// Generates the landing page's synthetic room (spec doc 2, section 12) - CloudEye has no
// real captured scene checked into this repo (no hotel_room/basement_apartment/etc, no
// static point cloud), and the landing page needs something to show without a live
// backend. Run once, offline, output committed:
//
//   node scripts/build-landing-scene.mjs
//
// Writes public/landing/room.glb (a points-primitive GLB, same format
// gpu/stage_export_glb.py produces for a real scene - loadable by the exact same
// lib/scene3dPoints.ts buildDecimatedPointCloud() the app viewer uses) and
// public/landing/reachability.json (per-platform reachable-object counts for the
// interactive comparison block, spec 7.2 - computed for real from this room's own
// geometry via a distance transform, not hand-picked numbers).
//
// NOT generated here (needs actual WebGL rendering - a browser or headless-gl, neither
// available in an offline Node script): room.webp (the hero's pre-load/no-WebGL static
// frame) and gallery/*.webp (spec 7.5's scene gallery). The hero renders room.glb live
// in-browser instead, which covers the common case; the static-fallback and gallery
// pieces are a follow-up, not part of this pass.

import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// GLTFExporter's binary path uses the browser's FileReader to turn its merged Blob into
// an ArrayBuffer - same polyfill build-turtlebot-glb.mjs uses, see its comment.
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
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const outDir = path.join(__dirname, '../public/landing')

// Room footprint and ceiling, meters - unremarkable "one bedroom" proportions, nothing
// meant to resemble a specific real space.
const ROOM_X = 4.0
const ROOM_Z = 3.6
const CEILING_Y = 2.6
const GRID_RESOLUTION_M = 0.05 // matches the real pipeline's occupancy grid (gpu/stage_occupancy.py)

// --h-0..--h-4 (frontend/src/lib/tokens.ts) - duplicated as plain OKLCH triples rather
// than imported: this script is plain Node (no TS build step), and importing a `.ts`
// module here would need a transpiler this project doesn't otherwise carry. Keep these
// byte-for-byte identical to index.css's :root block, same rule tokens.test.ts enforces
// on the TS side.
const HEIGHT_RAMP = [
  { l: 0.32, c: 0.16, h: 300 },
  { l: 0.45, c: 0.17, h: 288 },
  { l: 0.58, c: 0.13, h: 240 },
  { l: 0.72, c: 0.13, h: 195 },
  { l: 0.85, c: 0.16, h: 155 },
]

function oklchToSrgb({ l, c, h }) {
  const hRad = (h * Math.PI) / 180
  const a = c * Math.cos(hRad)
  const b = c * Math.sin(hRad)
  const l_ = l + 0.3963377774 * a + 0.2158037573 * b
  const m_ = l - 0.1055613458 * a - 0.0638541728 * b
  const s_ = l - 0.0894841775 * a - 1.291485548 * b
  const l3 = l_ ** 3
  const m3 = m_ ** 3
  const s3 = s_ ** 3
  const rLin = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
  const gLin = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
  const bLin = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.707614701 * s3
  const transfer = (v) => {
    const clamped = Math.min(1, Math.max(0, v))
    return clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * clamped ** (1 / 2.4) - 0.055
  }
  return [transfer(rLin), transfer(gLin), transfer(bLin)]
}

function heightColor(y) {
  const t = Math.min(1, Math.max(0, y / CEILING_Y)) * 4
  const i0 = Math.min(3, Math.floor(t))
  const frac = t - i0
  const a = HEIGHT_RAMP[i0]
  const b = HEIGHT_RAMP[i0 + 1]
  const mixed = { l: a.l + (b.l - a.l) * frac, c: a.c + (b.c - a.c) * frac, h: a.h + (b.h - a.h) * frac }
  return oklchToSrgb(mixed)
}

// --- Point cloud construction ------------------------------------------------------

const positions = []
const colors = []

function addPoint(x, y, z) {
  positions.push(x, y, z)
  const [r, g, b] = heightColor(y)
  colors.push(r, g, b)
}

function jitter(amount) {
  return (Math.random() - 0.5) * amount
}

/** Scatters `count` points across a plane. `axis` is which world axis is held (near-)
 * constant - 'y' for floor/ceiling, 'x'/'z' for walls - jittered by +-2cm to avoid a
 * perfectly flat, obviously-synthetic sheet. */
function scatterPlane(count, axis, constant, uRange, vRange, uAxis, vAxis) {
  for (let i = 0; i < count; i++) {
    const u = uRange[0] + Math.random() * (uRange[1] - uRange[0])
    const v = vRange[0] + Math.random() * (vRange[1] - vRange[0])
    const p = { x: 0, y: 0, z: 0 }
    p[axis] = constant + jitter(0.02)
    p[uAxis] = u
    p[vAxis] = v
    addPoint(p.x, p.y, p.z)
  }
}

/** Scatters points over a box's 6 faces - a cheap stand-in for a piece of furniture,
 * enough to read as "an object" in the point cloud without modeling real geometry. */
function scatterBox(count, center, size) {
  const [cx, cy, cz] = center
  const [sx, sy, sz] = size
  for (let i = 0; i < count; i++) {
    const face = Math.floor(Math.random() * 6)
    const u = (Math.random() - 0.5) * sx
    const v = (Math.random() - 0.5) * sz
    const w = (Math.random() - 0.5) * sy
    let x, y, z
    if (face === 0) [x, y, z] = [cx + u, cy + sy / 2, cz + v] // top
    else if (face === 1) [x, y, z] = [cx + u, cy - sy / 2, cz + v] // bottom
    else if (face === 2) [x, y, z] = [cx + sx / 2, cy + w, cz + v] // +x
    else if (face === 3) [x, y, z] = [cx - sx / 2, cy + w, cz + v] // -x
    else if (face === 4) [x, y, z] = [cx + u, cy + w, cz + sz / 2] // +z
    else [x, y, z] = [cx + u, cy + w, cz - sz / 2] // -z
    addPoint(x, Math.max(0, y), z)
  }
}

// Floor and ceiling.
scatterPlane(60000, 'y', 0, [0, ROOM_X], [0, ROOM_Z], 'x', 'z')
scatterPlane(20000, 'y', CEILING_Y, [0, ROOM_X], [0, ROOM_Z], 'x', 'z')
// Four walls.
scatterPlane(20000, 'z', 0, [0, ROOM_X], [0, CEILING_Y], 'x', 'y')
scatterPlane(20000, 'z', ROOM_Z, [0, ROOM_X], [0, CEILING_Y], 'x', 'y')
scatterPlane(18000, 'x', 0, [0, ROOM_Z], [0, CEILING_Y], 'z', 'y')
scatterPlane(18000, 'x', ROOM_X, [0, ROOM_Z], [0, CEILING_Y], 'z', 'y')

// Furniture-shaped clusters - also doubles as the object list for reachability.json.
// Positions chosen so some objects sit deep in the open middle of the room (reachable by
// any platform) and some hug a wall or corner (only the smallest platforms fit).
const OBJECTS = [
  { name: 'bed', center: [0.9, 0.25, 0.8], size: [1.4, 0.5, 2.0], points: 14000 },
  { name: 'nightstand', center: [1.75, 0.25, 0.35], size: [0.4, 0.5, 0.4], points: 3000 },
  { name: 'desk', center: [3.55, 0.35, 0.5], size: [1.0, 0.7, 0.5], points: 6000 },
  { name: 'chair', center: [3.1, 0.22, 0.9], size: [0.45, 0.45, 0.45], points: 3000 },
  { name: 'wardrobe', center: [3.7, 0.9, 2.9], size: [0.5, 1.8, 0.9], points: 8000 },
  { name: 'rug', center: [1.6, 0.01, 2.2], size: [1.4, 0.02, 1.0], points: 5000 },
  { name: 'lamp', center: [0.25, 0.65, 2.9], size: [0.25, 1.3, 0.25], points: 2000 },
  { name: 'shelf', center: [0.15, 1.4, 1.6], size: [0.25, 1.6, 0.6], points: 4000 },
  { name: 'plant', center: [3.8, 0.35, 3.4], size: [0.3, 0.7, 0.3], points: 2000 },
  { name: 'stool', center: [2.2, 0.2, 3.35], size: [0.3, 0.4, 0.3], points: 2000 },
]
for (const o of OBJECTS) scatterBox(o.points, o.center, o.size)

console.log(`Generated ${positions.length / 3} points.`)

// --- Reachability (spec 7.2's static comparison data) ------------------------------
//
// A minimal, from-scratch Euclidean distance transform (Felzenszwalb & Huttenlocher,
// same algorithm as frontend/src/lib/clearanceField.ts - duplicated rather than
// imported for the same plain-Node reason as the OKLCH conversion above) over a real
// occupancy grid built from this room's own 4 walls. An object is "reachable" for a
// given platform if its own position has at least that platform's radius of clearance
// from every wall - correct for this room specifically because it has no internal
// obstacles, so clearance-from-walls and true path-connectivity coincide here.
const gridW = Math.ceil(ROOM_X / GRID_RESOLUTION_M)
const gridH = Math.ceil(ROOM_Z / GRID_RESOLUTION_M)
const obstacle = new Float64Array(gridW * gridH).fill(1e20)
for (let ix = 0; ix < gridW; ix++) {
  for (let iz = 0; iz < gridH; iz++) {
    if (ix === 0 || iz === 0 || ix === gridW - 1 || iz === gridH - 1) obstacle[ix * gridH + iz] = 0
  }
}

function distanceTransform1D(f) {
  const n = f.length
  const d = new Float64Array(n)
  const v = new Int32Array(n)
  const z = new Float64Array(n + 1)
  let k = 0
  v[0] = 0
  z[0] = -Infinity
  z[1] = Infinity
  for (let q = 1; q < n; q++) {
    let s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
    while (s <= z[k]) {
      k--
      s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k])
    }
    k++
    v[k] = q
    z[k] = s
    z[k + 1] = Infinity
  }
  k = 0
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++
    d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]
  }
  return d
}

const pass1 = new Float64Array(gridW * gridH)
for (let ix = 0; ix < gridW; ix++) {
  const col = obstacle.slice(ix * gridH, ix * gridH + gridH)
  const d = distanceTransform1D(col)
  for (let iz = 0; iz < gridH; iz++) pass1[ix * gridH + iz] = d[iz]
}
const clearance = new Float64Array(gridW * gridH)
for (let iz = 0; iz < gridH; iz++) {
  const row = new Float64Array(gridW)
  for (let ix = 0; ix < gridW; ix++) row[ix] = pass1[ix * gridH + iz]
  const d = distanceTransform1D(row)
  for (let ix = 0; ix < gridW; ix++) clearance[ix * gridH + iz] = Math.sqrt(d[ix]) * GRID_RESOLUTION_M
}

function clearanceAt(x, z) {
  const ix = Math.min(gridW - 1, Math.max(0, Math.round(x / GRID_RESOLUTION_M)))
  const iz = Math.min(gridH - 1, Math.max(0, Math.round(z / GRID_RESOLUTION_M)))
  return clearance[ix * gridH + iz]
}

// A representative subset of app/robots.py's registry (radius_m, display_name) -
// duplicated as literal numbers for the same reason as HEIGHT_RAMP above; keep these in
// sync by hand if the registry's own numbers change.
const PLATFORMS = [
  { id: 'burger', display_name: 'TurtleBot3 Burger', radius_m: 0.1 },
  { id: 'go2', display_name: 'Unitree Go2', radius_m: 0.2496 },
  { id: 'husky', display_name: 'Husky A200', radius_m: 0.5528 },
]

const objectReachability = OBJECTS.map((o) => {
  const d = clearanceAt(o.center[0], o.center[2])
  return {
    name: o.name,
    position: [o.center[0], o.center[1], o.center[2]],
    reachableBy: PLATFORMS.filter((p) => d >= p.radius_m).map((p) => p.id),
  }
})

const reachabilityJson = {
  room: { width_m: ROOM_X, depth_m: ROOM_Z, ceiling_m: CEILING_Y },
  platforms: PLATFORMS,
  objects: objectReachability,
  robotStart: [ROOM_X / 2, 0, ROOM_Z / 2],
}

// --- Write output --------------------------------------------------------------------

async function main() {
  await mkdir(outDir, { recursive: true })

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
  const points = new THREE.Points(geometry, new THREE.PointsMaterial())
  points.name = 'landing_room'
  const root = new THREE.Group()
  root.add(points)

  const exporter = new GLTFExporter()
  const glb = await new Promise((resolve, reject) => {
    exporter.parse(root, resolve, reject, { binary: true })
  })

  const glbPath = path.join(outDir, 'room.glb')
  await writeFile(glbPath, Buffer.from(glb))
  console.log(`Wrote ${glbPath} (${(glb.byteLength / 1e6).toFixed(2)} MB)`)

  const jsonPath = path.join(outDir, 'reachability.json')
  await writeFile(jsonPath, JSON.stringify(reachabilityJson, null, 2))
  console.log(`Wrote ${jsonPath}`)
  for (const p of PLATFORMS) {
    const n = objectReachability.filter((o) => o.reachableBy.includes(p.id)).length
    console.log(`  ${p.display_name} (${p.radius_m}m): ${n} / ${OBJECTS.length} reachable`)
  }
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
