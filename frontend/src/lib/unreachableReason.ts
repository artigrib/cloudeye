/** The backend's own word for why an object is not reachable, and a plain-English gloss
 * for it.
 *
 * The word itself is never rewritten: `ReachabilityResponse.unreachable_reasons` carries
 * one of a closed set of exact strings (app/services/pathfinding.py's
 * compute_reachability, documented in app/schemas.py) and the row prints that string
 * verbatim. Paraphrasing it would put a second vocabulary between the user and the log
 * line / API response they would have to search to follow it up - and inventing a
 * number for it ("blocked by a 0.42 m gap") would be worse still, because no such
 * measurement is taken on this path.
 *
 * The gloss is a tooltip. It explains the label; it never replaces it and never adds a
 * quantity the response did not carry.
 *
 * Note on `disconnected`: goal_resolution raises `no_connected_cell_within_2m`, and
 * compute_reachability renames it to `disconnected` before it reaches the wire - the
 * same label is also used for "the goal cell resolved fine, it is simply in another
 * component".
 *
 * **One exception to "never rewritten", added 2026-09-10 at the owner's request:** the
 * OBJECT LIST prints `disconnected` as `unreachable` (see `unreachableReasonLabel`). It
 * is the one label that reads as a network fault to anyone who has not read
 * compute_reachability, and on this screen it means only "the robot cannot get there".
 * The relabel is display-only and it is deliberately narrow - `data-object-reason` still
 * carries the backend's own word, the 2D map's hover label still prints it verbatim
 * (demo/probe_map2d.py asserts exactly that), and every other label is untouched. */
const GLOSS: Record<string, string> = {
  outside_grid:
    "This object's centre falls outside the occupancy grid, so no route to it can be planned. That is bad data for this one object, not a fact about the robot.",
  disconnected:
    'The cell beside this object is not connected to the part of the room the robot stands in. (The backend writes "disconnected" both for a goal in another component and for goal_resolution\'s no_connected_cell_within_2m.)',
  no_free_cell_within_2m:
    'No cell within 2 m of this object has enough clearance for this robot to stand in, so there is nowhere to drive to.',
  no_visible_cell_within_2m:
    'No cell within 2 m of this object is both clear enough for this robot and in line of sight of it.',
  robot_does_not_fit:
    'This robot has nowhere to stand anywhere in this room at this radius, so nothing in the scene is reachable for it.',
}

/** Display labels that differ from the backend's word. Exactly one entry, and adding a
 * second should need the same argument the first one needed: the API string has to be
 * actively misleading on this screen, not merely terse. */
const DISPLAY_LABEL: Record<string, string> = {
  disconnected: 'unreachable',
}

/** What the object list shows for a reason. The backend's own word for everything except
 * the one entry in DISPLAY_LABEL above; an unknown label passes through unchanged, so a
 * new backend reason shows rather than being swallowed. */
export function unreachableReasonLabel(reason: string): string {
  return DISPLAY_LABEL[reason] ?? reason
}

/** The tooltip for a reason label, or the label itself when it is one this build does
 * not have a gloss for - a new backend label must still show, unexplained rather than
 * swallowed. */
export function unreachableReasonTitle(reason: string): string {
  return GLOSS[reason] ?? reason
}

/** True when this build can explain the label. Only useful for tests and for deciding
 * whether the tooltip adds anything. */
export function hasGloss(reason: string): boolean {
  return reason in GLOSS
}
