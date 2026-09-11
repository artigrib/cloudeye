#!/usr/bin/env node
// One-off measurement helper (not part of the app, not a project dependency at runtime):
// loads the FINAL, decimated public/models/rosbot_xl_arm.glb, isolates the "base" node
// (chassis + wheels, added by build-rosbot-xl-arm-glb.mjs as a named top-level group under
// the scene root) and reports its bounding box - explicitly excluding the "arm" node's
// geometry, per the requirement that the footprint radius is computed from the base alone.
//
// Usage: node scripts/measure-rosbot-xl-arm-bbox.mjs <path-to-glb>

import { readFile } from 'node:fs/promises'
import { JSDOM } from 'jsdom'

const dom = new JSDOM()
globalThis.DOMParser = dom.window.DOMParser
globalThis.XMLSerializer = dom.window.XMLSerializer
globalThis.document = dom.window.document
globalThis.window = dom.window

import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

const file = process.argv[2]
if (!file) {
  console.error('Usage: node scripts/measure-rosbot-xl-arm-bbox.mjs <path-to-glb>')
  process.exit(1)
}

const buf = await readFile(file)
const loader = new GLTFLoader()
const gltf = await new Promise((resolve, reject) => {
  loader.parse(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength), '', resolve, reject)
})

function dump(o, depth = 0) {
  console.log('  '.repeat(depth) + `${o.type} "${o.name}"` + (o.isMesh ? ` [mesh]` : ''))
  o.children.forEach((c) => dump(c, depth + 1))
}
console.log('--- node hierarchy ---')
dump(gltf.scene)

const root = gltf.scene
let baseNode = null
root.traverse((o) => {
  if (o.name === 'base') baseNode = o
})
if (!baseNode) {
  console.error('Could not find a node named "base" in the scene graph.')
  process.exit(1)
}

const fullBox = new THREE.Box3().setFromObject(root)
const baseBox = new THREE.Box3().setFromObject(baseNode)

function report(label, box) {
  const size = new THREE.Vector3()
  box.getSize(size)
  console.log(
    `${label}: x [${box.min.x.toFixed(4)}, ${box.max.x.toFixed(4)}] (${(size.x * 1000).toFixed(1)}mm)  ` +
      `y(up) [${box.min.y.toFixed(4)}, ${box.max.y.toFixed(4)}] (${(size.y * 1000).toFixed(1)}mm)  ` +
      `z [${box.min.z.toFixed(4)}, ${box.max.z.toFixed(4)}] (${(size.z * 1000).toFixed(1)}mm)`,
  )
  return size
}

console.log('\n--- bounding boxes (scene axes: x=fwd, y=up, z=lateral) ---')
report('full (base+arm)', fullBox)
const baseSize = report('base only', baseBox)

// Horizontal footprint rectangle of the base: scene x (forward/back) and scene z (lateral).
const lengthX = baseSize.x
const widthZ = baseSize.z
const smallerSide = Math.min(lengthX, widthZ)
const diagonal = Math.sqrt(lengthX * lengthX + widthZ * widthZ)

console.log('\n--- footprint radius (base only) ---')
console.log(`horizontal rectangle: ${(lengthX * 1000).toFixed(1)}mm x ${(widthZ * 1000).toFixed(1)}mm`)
console.log(`half_width_m (half of smaller side)  = ${(smallerSide / 2).toFixed(6)} (${(smallerSide * 500).toFixed(1)}mm)`)
console.log(`half_diagonal_m (half of diagonal)    = ${(diagonal / 2).toFixed(6)} (${(diagonal * 500).toFixed(1)}mm)`)
console.log(`bbox_m: x=${lengthX.toFixed(6)} y=${baseSize.y.toFixed(6)} z=${widthZ.toFixed(6)}`)

// Vendor comparison: 332 x 284 x 131 mm (length x width x height)
const vendor = { x: 0.332, z: 0.284, y: 0.131 }
for (const [axis, v] of Object.entries(vendor)) {
  const measured = axis === 'x' ? lengthX : axis === 'z' ? widthZ : baseSize.y
  const pct = ((measured - v) / v) * 100
  console.log(`vendor ${axis}: ${(v * 1000).toFixed(0)}mm vs measured ${(measured * 1000).toFixed(1)}mm (${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%)`)
}
