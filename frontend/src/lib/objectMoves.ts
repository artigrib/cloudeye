import type { OccupancyGridResponse, SceneObjectResponse } from '../api/types'

/** One "what if that were somewhere else": an object, and how far to slide it, in
 * metres. `dz` is the world axis the 2D map draws vertically - the typed command calls
 * it `dy` because that is what someone looking at a top-down picture calls it. */
export interface ObjectMove {
  objectId: string
  dx: number
  dz: number
}

/** Occupancy values, from the grid response's own legend: {"0": "free", "1":
 * "obstacle", "2": "unknown"}. */
const FREE = 0
const OBSTACLE = 1

/** The `move` query parameters for GET /reachability - `<object id>:<dx>:<dz>`. The
 * SERVER's answer is the one that counts: it does this same cut and paste on its own
 * copy of the grid and re-runs reachability over the result. Everything else in this
 * module exists only so the picture agrees with that answer. */
export function moveParams(moves: readonly ObjectMove[]): string[] {
  return moves.map((m) => `${m.objectId}:${m.dx}:${m.dz}`)
}

/** The objects with each moved one slid by its offset - position and bbox together, so
 * its marker, its row and its hull all move as one. Objects with no move are returned
 * unchanged, by identity, so React sees no change for them.
 *
 * With no moves at all this returns the SAME ARRAY, not a copy. That identity is
 * load-bearing: SceneMap3D rebuilds its whole marker group whenever the `objects` prop
 * changes reference, so returning a fresh copy on every render would tear down and
 * recreate 17 meshes per frame. */
export function applyMovesToObjects(
  objects: readonly SceneObjectResponse[],
  moves: readonly ObjectMove[],
): readonly SceneObjectResponse[] {
  if (moves.length === 0) return objects
  const byId = new Map(moves.map((m) => [m.objectId, m]))
  return objects.map((o) => {
    const m = byId.get(o.id)
    if (!m) return o
    return {
      ...o,
      pos_x: o.pos_x + m.dx,
      pos_z: o.pos_z + m.dz,
      bbox_min_x: o.bbox_min_x + m.dx,
      bbox_max_x: o.bbox_max_x + m.dx,
      bbox_min_z: o.bbox_min_z + m.dz,
      bbox_max_z: o.bbox_max_z + m.dz,
    }
  })
}

/** The occupancy grid with each moved object's bbox cut out (set to free) and pasted
 * back at its offset (set to obstacle).
 *
 * A deliberate mirror of the backend's `cut_footprints_from_grid` +
 * `rasterize_footprints_as_obstacles`, cell-centre test included, so the floor the map
 * draws is the floor the answer was computed on. It decides nothing: reachability comes
 * from the server, which does this itself. If the two ever disagree the map is wrong and
 * the number is right, which is the correct way round for them to fail.
 *
 * Blunt in exactly the way the server's is: the cut clears whatever else the rectangle
 * contained (a bit of wall it overlapped, clutter under the scan band), so it can only
 * make the room look MORE open than it is. That is why the UI labels the result
 * approximate. */
export function applyMovesToGrid(
  grid: OccupancyGridResponse,
  objects: readonly SceneObjectResponse[],
  moves: readonly ObjectMove[],
): OccupancyGridResponse {
  if (moves.length === 0) return grid
  const byId = new Map(objects.map((o) => [o.id, o]))
  const boxes = moves
    .map((m) => ({ move: m, object: byId.get(m.objectId) }))
    .filter((b): b is { move: ObjectMove; object: SceneObjectResponse } => b.object !== undefined)
  if (boxes.length === 0) return grid

  const cells = grid.cells.map((column) => [...column])
  const { origin_x: ox, origin_z: oz, resolution: res } = grid

  for (const { object, move } of boxes) {
    for (let ix = 0; ix < grid.width; ix++) {
      const x = ox + (ix + 0.5) * res
      const inCutX = x >= object.bbox_min_x && x <= object.bbox_max_x
      const inPasteX = x >= object.bbox_min_x + move.dx && x <= object.bbox_max_x + move.dx
      if (!inCutX && !inPasteX) continue
      for (let iz = 0; iz < grid.height; iz++) {
        const z = oz + (iz + 0.5) * res
        // Cut first, then paste - an object slid by less than its own width overlaps
        // itself, and clearing after pasting would punch a hole in what was just drawn.
        if (inCutX && z >= object.bbox_min_z && z <= object.bbox_max_z) cells[ix][iz] = FREE
      }
    }
  }
  for (const { object, move } of boxes) {
    for (let ix = 0; ix < grid.width; ix++) {
      const x = ox + (ix + 0.5) * res
      if (x < object.bbox_min_x + move.dx || x > object.bbox_max_x + move.dx) continue
      for (let iz = 0; iz < grid.height; iz++) {
        const z = oz + (iz + 0.5) * res
        if (z >= object.bbox_min_z + move.dz && z <= object.bbox_max_z + move.dz) cells[ix][iz] = OBSTACLE
      }
    }
  }
  return { ...grid, cells }
}
