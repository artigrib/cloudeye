/** The templated command language the scene screen's agent understands without any LLM.
 *
 * Four forms, parsed here in the browser:
 *
 *   go to <object>              plan a route to that object
 *   can <robot> reach <object>  answer from the reachability already on screen
 *   compare all                 open the platform x object matrix
 *   move <object> <dx> <dy>     an approximate what-if on the occupancy grid
 *
 * Anything else is `freeform` and goes to the chat LLM named in the workspace's job
 * spec. That split is the point: the four commands above are the ones this screen is
 * FOR, and they must keep working on a deployment with no model credentials at all -
 * which is every deployment in this worktree, by design.
 *
 * Object names are matched against the scene's own list rather than guessed from
 * whitespace, so a two-word object ("coffee table") parses exactly as a one-word one
 * does, and a name that is not in this room falls through to freeform rather than
 * becoming a command that cannot be executed.
 */

export interface GotoCommand {
  kind: 'goto'
  /** The object name as it appears in the scene's list, not as the user typed it. */
  object: string
}
export interface CanReachCommand {
  kind: 'canReach'
  /** The robot name as it appears in the platform list. */
  robot: string
  object: string
}
export interface CompareAllCommand {
  kind: 'compareAll'
}
export interface MoveCommand {
  kind: 'move'
  object: string
  /** Metres along +x and +z respectively. `dy` is the screen-vertical axis on the 2D
   * map, which is the world's z - the command says "dy" because that is what someone
   * looking at a top-down map calls it. */
  dx: number
  dy: number
}
export interface FreeformCommand {
  kind: 'freeform'
  text: string
}

export type ParsedCommand =
  | GotoCommand
  | CanReachCommand
  | CompareAllCommand
  | MoveCommand
  | FreeformCommand

export interface Vocabulary {
  /** Every object name in this scene, as the object list spells them. */
  objects: readonly string[]
  /** Every platform name the picker or the comparison can name. */
  robots: readonly string[]
}

/** The longest vocabulary entry that `text` starts with, case-insensitively, or null.
 * Longest-first so "night stand" never wins over "night stand lamp" when both exist. */
function matchAtStart(text: string, vocabulary: readonly string[]): { match: string; rest: string } | null {
  const lower = text.toLowerCase()
  let best: string | null = null
  for (const entry of vocabulary) {
    const e = entry.toLowerCase()
    if (!e) continue
    if (!lower.startsWith(e)) continue
    // A vocabulary entry may only match a WHOLE word: "bed" must not match "bedside".
    const next = lower[e.length]
    if (next !== undefined && /[a-z0-9]/.test(next)) continue
    if (best === null || e.length > best.length) best = entry
  }
  return best === null ? null : { match: best, rest: text.slice(best.length).trim() }
}

/** The whole of `text` as a vocabulary entry, or null. */
function matchWhole(text: string, vocabulary: readonly string[]): string | null {
  const found = matchAtStart(text, vocabulary)
  return found && found.rest === '' ? found.match : null
}

/** A signed distance in metres, with an optional unit: "0.5", "-1.25", "0.5m", "0.5 m". */
function parseMetres(token: string): number | null {
  const m = /^([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*m?$/i.exec(token.trim())
  if (!m) return null
  const value = Number(m[1])
  return Number.isFinite(value) ? value : null
}

export function parseCommand(input: string, vocabulary: Vocabulary): ParsedCommand {
  const text = input.trim().replace(/\s+/g, ' ')
  if (!text) return { kind: 'freeform', text }
  const lower = text.toLowerCase()

  if (lower === 'compare all') return { kind: 'compareAll' }

  // "go to <object>" / "goto <object>". "go" alone is accepted too - the verb is
  // unambiguous here and a typist who omits "to" means the same thing.
  const gotoPrefix = /^go(?:\s*to|to)?\s+/i.exec(text)
  if (gotoPrefix) {
    const name = matchWhole(text.slice(gotoPrefix[0].length), vocabulary.objects)
    if (name) return { kind: 'goto', object: name }
  }

  // "can <robot> reach <object>", with an optional "?" and an optional "the".
  const canPrefix = /^can\s+/i.exec(text)
  if (canPrefix) {
    const afterCan = text.slice(canPrefix[0].length).replace(/\?+$/, '').trim()
    const robot = matchAtStart(afterCan, vocabulary.robots)
    if (robot) {
      const reach = /^reach\s+/i.exec(robot.rest) ?? /^get\s+to\s+/i.exec(robot.rest)
      if (reach) {
        const object = matchWhole(robot.rest.slice(reach[0].length), vocabulary.objects)
        if (object) return { kind: 'canReach', robot: robot.match, object }
      }
    }
  }

  // "move <object> <dx> <dy>". The two numbers are read off the END, so an object whose
  // own name contains a number still parses.
  const movePrefix = /^move\s+/i.exec(text)
  if (movePrefix) {
    const rest = text.slice(movePrefix[0].length)
    const found = matchAtStart(rest, vocabulary.objects)
    if (found) {
      const tail = found.rest.split(' ').filter(Boolean)
      if (tail.length === 2) {
        const dx = parseMetres(tail[0])
        const dy = parseMetres(tail[1])
        if (dx !== null && dy !== null) return { kind: 'move', object: found.match, dx, dy }
      }
    }
  }

  return { kind: 'freeform', text }
}

/** What the "not understood" line offers instead. Kept beside the grammar so the two
 * cannot drift: a help string that lists a form the parser does not accept is worse
 * than no help string. */
export const TEMPLATED_FORMS = [
  'go to <object>',
  'can <robot> reach <object>',
  'compare all',
  'move <object> <dx> <dy>',
] as const
