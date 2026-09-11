// Object-name completion for the command input - pure logic, no DOM, colocated test, the
// same convention as pointSizing.ts/edlSettings.ts. ChatPanel owns the popup, the keyboard
// handling and the focus; everything decidable without a DOM lives here.
//
// It completes the LAST whitespace-delimited token of the input, so "go to the si"
// suggests "sink" and replacing it yields "go to the sink". That is the shape a command
// actually has ("go to the sink", "take the mirror" - see ChatPanel's placeholder), and
// completing the whole field instead would throw away the verb the user just typed.

export const MAX_SUGGESTIONS = 6

/** Deduplicated, non-fragment object names in list order. Fragments are excluded for the
 * same reason ObjectList hides them by default: they are reconstruction debris, not
 * things anyone commands a robot to visit. */
export function completionNames(objects: readonly { name: string; is_fragment: boolean }[]): string[] {
  const seen = new Set<string>()
  const names: string[] = []
  for (const o of objects) {
    const key = o.name.toLowerCase()
    if (o.is_fragment || seen.has(key)) continue
    seen.add(key)
    names.push(o.name)
  }
  return names
}

/** The token being completed: everything after the last whitespace. Empty when the input
 * is empty or ends in a space - a trailing space means "I finished that word", and
 * offering the whole object list at that moment is noise, not help. */
export function currentToken(input: string): string {
  const match = input.match(/(\S*)$/)
  return match ? match[1] : ''
}

/** Names matching the current token, prefix first and then substring, case-insensitively.
 *
 * Prefix before substring because a prefix match is what the typist is almost always
 * after ("la" -> "lamp"), while a substring match is the useful fallback for a name whose
 * distinguishing part is not at the front ("stand" -> "nightstand"). A token that exactly
 * equals its only match returns nothing: there is nothing left to complete, and leaving
 * the popup open there would swallow the Enter that was meant to send the command. */
export function suggestObjectNames(input: string, names: string[], limit = MAX_SUGGESTIONS): string[] {
  const token = currentToken(input).toLowerCase()
  if (!token) return []

  const prefix: string[] = []
  const substring: string[] = []
  for (const name of names) {
    const lower = name.toLowerCase()
    if (lower === token) continue
    if (lower.startsWith(token)) prefix.push(name)
    else if (lower.includes(token)) substring.push(name)
  }
  return [...prefix, ...substring].slice(0, limit)
}

/** The input with its last token replaced by `name`, plus a trailing space so the next
 * word can be typed straight away (and so the popup closes, `currentToken` being empty). */
export function applyCompletion(input: string, name: string): string {
  const token = currentToken(input)
  const head = token ? input.slice(0, input.length - token.length) : input
  return `${head}${name} `
}

/** Wraps an index into [0, length) - Arrow navigation cycles rather than sticking at the
 * ends. Returns -1 for an empty list, i.e. "nothing is highlighted". */
export function wrapIndex(index: number, length: number): number {
  if (length <= 0) return -1
  return ((index % length) + length) % length
}
