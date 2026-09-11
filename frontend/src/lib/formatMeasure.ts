/** Null-safe formatting for the numeric fields the robot/reachability panel shows.
 *
 * Why this exists: `ReachabilityCard` called `fitProb.p5CorridorWidthM.toFixed(2)` on a
 * field the API can legitimately return as `null`. `scripts/audit/fit_prob.py` reports
 * `corridor_width_p5_m: null` for any platform whose `fits_count` is 0 - there is no
 * successful trial to take a percentile over. Husky A200 (radius 0.5528 m) is exactly
 * that case on the hero scene (0/100 fits, 100 goal_unresolved trials), so selecting it
 * threw `Cannot read properties of null (reading 'toFixed')` inside render; React then
 * unmounted the whole panel subtree - counter, platform name and command input all left
 * the DOM - and the three.js canvas below it followed with a WebGL device loss.
 *
 * The rule for every measured quantity in that panel: a missing measurement renders as
 * an em dash, it never reaches `toFixed`. "0.00 m" would be a lie (it claims a measured
 * zero); hiding the row would silently drop a field that is present for other platforms.
 */

/** What a missing measurement renders as. */
export const NO_VALUE = '—'

/** True for the three ways a number can fail to be one: null, undefined, NaN (JSON
 * cannot carry NaN, but a client-side computation can produce one). Infinity is left
 * alone - it is a meaningful answer for a distance search that found nothing. */
function isMissing(value: number | null | undefined): value is null | undefined {
  return value == null || Number.isNaN(value)
}

/** Strip a sign that only survives rounding.
 *
 * `(-0.001).toFixed(2)` is `"-0.00"`, and so is `(-0).toFixed(2)`. Both print a minus in
 * front of a zero, which reads as a measured negative - the start position at the scene
 * origin showed `(0.00, -0.00)`, one coordinate apparently signed and the other not, for
 * a point that is simply 0. A number that rounds to zero IS zero at the precision being
 * shown; the sign is noise from a digit the reader was not given. */
function unsignZero(text: string): string {
  return /^-0(\.0*)?$/.test(text) ? text.slice(1) : text
}

/** A length in metres: `0.55 m`, or `—` when the measurement is missing. */
export function formatMetres(value: number | null | undefined, digits = 2): string {
  return isMissing(value) ? NO_VALUE : `${unsignZero(value.toFixed(digits))} m`
}

/** A duration in seconds: `4.7s`, or `—` when missing. */
export function formatSeconds(value: number | null | undefined, digits = 1): string {
  return isMissing(value) ? NO_VALUE : `${unsignZero(value.toFixed(digits))}s`
}

/** A bare number with no unit - for the places that add their own suffix. */
export function formatNumber(value: number | null | undefined, digits = 2): string {
  return isMissing(value) ? NO_VALUE : unsignZero(value.toFixed(digits))
}

/** A signed offset, for a delta the reader needs the direction of: `+0.80`, `-0.25`,
 * `0.00`. The sign is explicit where it means something and absent where it does not -
 * a delta that rounds to zero is not "negative zero", it is no movement. */
export function formatSigned(value: number | null | undefined, digits = 2): string {
  if (isMissing(value)) return NO_VALUE
  const text = unsignZero(value.toFixed(digits))
  return text.startsWith('-') || Number(text) === 0 ? text : `+${text}`
}
