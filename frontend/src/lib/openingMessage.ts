import type { OccupancyGridResponse, ReachabilityResponse } from '../api/types'

/** The chat's opening line, from this scene's own numbers.
 *
 * Templated on purpose: no model call behind it and no artificial delay in front of it, so
 * it is on screen the moment the scene's data is. It says the three things the screen
 * already knows and the reader does not yet - how big the room is, how much was found in
 * it, and whether the selected robot can actually get around it.
 *
 * Every branch is a sentence about a state the screen can really be in, including the ones
 * that are nobody's fault: a scene with no navigation layer, a robot that fits nowhere, a
 * room with nothing detected in it. None of them is written as an error, because none of
 * them is one.
 */
export function openingMessage(input: {
  grid: OccupancyGridResponse | null
  objectCount: number
  reachableCount: number | null
  platformName: string | null
  reachability: ReachabilityResponse | null
}): string {
  const { grid, objectCount, reachableCount, platformName, reachability } = input
  const robot = platformName ?? 'This robot'

  const size =
    grid && grid.layer_available !== false && grid.width > 0
      ? `${(grid.width * grid.resolution).toFixed(1)} × ${(grid.height * grid.resolution).toFixed(1)} m`
      : null
  const room = size ? `This room is ${size}` : 'This room'
  const found =
    objectCount === 0
      ? 'nothing was detected in it'
      : `${objectCount} object${objectCount === 1 ? '' : 's'} ${objectCount === 1 ? 'was' : 'were'} detected in it`

  if (reachability && reachability.layer_available === false) {
    return `${room}. This scene has no navigation layer, so there is nothing to plan a route on yet.`
  }
  if (objectCount === 0) {
    return `${room}, and ${found}. There is nothing to drive to here.`
  }
  if (reachability?.start_status === 'none') {
    return `${room} and ${found}. ${robot} has nowhere to stand in it at its own clearance, so none of them is reachable.`
  }
  if (reachableCount === null) {
    return `${room} and ${found}. Working out what ${robot} can reach…`
  }
  if (reachableCount === objectCount) {
    return `${room} and ${found}. ${robot} can reach all of them - double-click one to drive there.`
  }
  return `${room} and ${found}. ${robot} can reach ${reachableCount} of ${objectCount} - double-click one to drive there.`
}
