import * as THREE from 'three/webgpu'
import { Line2NodeMaterial } from 'three/webgpu'
import { Line2 } from 'three/addons/lines/webgpu/Line2.js'
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js'
import type { Point } from './marchingSquares'

export interface ContourGridMeta {
  resolution: number
  origin_x: number
  origin_z: number
}

export interface ContourLayer {
  group: THREE.Group
  lines: Line2[]
  materials: Line2NodeMaterial[]
  fills: THREE.Mesh[]
}

function toWorld(p: Point, grid: ContourGridMeta, yOffset: number): [number, number, number] {
  return [grid.origin_x + p[0] * grid.resolution, yOffset, grid.origin_z + p[1] * grid.resolution]
}

/** Converts marching-squares polylines (grid-cell units - see marchingSquares.ts) into
 * world-space geometry on the floor: a screen-space-width Line2 stroke per polyline
 * (same technique as the trail/camera-path, see scene3dPoints.ts's neighbors), plus a
 *
 * The Line2/material pair MUST be the WebGPU one (`three/addons/lines/webgpu/Line2.js` +
 * `Line2NodeMaterial`), the same pair SceneMap3D's trail and camera-path already use -
 * NOT the classic `three/examples/jsm/lines/{Line2,LineMaterial}.js`. This module used
 * the classic pair, and under the app's WebGPURenderer (WebGL2 backend) three's
 * NodeBuilder rejected it with `Material "LineMaterial" is not compatible.` and fell
 * back to drawing LineGeometry's raw instanced quad with a default material - one solid
 * grey sheet standing in the middle of the room, growing as the contour set grew with
 * the selected platform's radius. Measured on the 15 fps hero: the sheet's region read
 * RGB [111.7, 108.9, 108.5] with the strokes present and [37.8, 30.0, 28.6] with every
 * Line2 hidden, while hiding the fill meshes changed nothing. Line2NodeMaterial also
 * reads the renderer's viewport itself (LineSegments2.onBeforeRender), so unlike
 * LineMaterial it needs no manual `resolution.set()` on resize.
 * translucent fill for each closed one - the same visual language OccupancyMap's SVG
 * version used (spec 5), just real scene objects instead of a DOM overlay. Fill
 * triangulation is done directly in grid-cell space via `ShapeUtils.triangulateShape`
 * (topology only - which vertex indices form each triangle) and then the SAME
 * `toWorld()` conversion used for the stroke is applied to the resulting vertices; this
 * deliberately avoids building the fill via a rotated/scaled `ShapeGeometry`, which
 * would require independently re-deriving the same axis mapping the stroke's `toWorld`
 * already gets right by construction - one conversion path, not two that have to agree. */
export function buildContours(
  contours: Point[][],
  grid: ContourGridMeta,
  opts: { yOffset: number; color: number; lineWidthPx: number; fillOpacity: number },
): ContourLayer {
  const group = new THREE.Group()
  const lines: Line2[] = []
  const materials: Line2NodeMaterial[] = []
  const fills: THREE.Mesh[] = []

  for (const line of contours) {
    if (line.length < 2) continue
    const closed =
      line.length > 2 && line[0][0] === line[line.length - 1][0] && line[0][1] === line[line.length - 1][1]

    const positions: number[] = []
    for (const p of line) positions.push(...toWorld(p, grid, opts.yOffset))
    const geometry = new LineGeometry()
    geometry.setPositions(positions)
    const material = new Line2NodeMaterial({ color: opts.color, linewidth: opts.lineWidthPx, transparent: true })
    const l2 = new Line2(geometry, material)
    l2.computeLineDistances()
    group.add(l2)
    lines.push(l2)
    materials.push(material)

    if (closed) {
      const shapePts = line.map((p) => new THREE.Vector2(p[0], p[1]))
      const triangles = THREE.ShapeUtils.triangulateShape(shapePts, [])
      if (triangles.length === 0) continue
      const worldPositions = new Float32Array(line.length * 3)
      for (let i = 0; i < line.length; i++) worldPositions.set(toWorld(line[i], grid, opts.yOffset - 0.001), i * 3)
      const fillGeom = new THREE.BufferGeometry()
      fillGeom.setAttribute('position', new THREE.BufferAttribute(worldPositions, 3))
      fillGeom.setIndex(triangles.flat())
      const fillMat = new THREE.MeshBasicMaterial({
        color: opts.color,
        transparent: true,
        opacity: opts.fillOpacity,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
      const fillMesh = new THREE.Mesh(fillGeom, fillMat)
      group.add(fillMesh)
      fills.push(fillMesh)
    }
  }

  return { group, lines, materials, fills }
}

export function disposeContours(layer: ContourLayer): void {
  for (const l2 of layer.lines) l2.geometry.dispose()
  for (const m of layer.materials) m.dispose()
  for (const mesh of layer.fills) {
    mesh.geometry.dispose()
    ;(mesh.material as THREE.Material).dispose()
  }
}
