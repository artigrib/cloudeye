import { formatMetres, formatNumber } from '../lib/formatMeasure'
import Tooltip from './ui/Tooltip'

export interface StartPosition {
  status: 'original' | 'moved' | 'none'
  x: number | null
  z: number | null
  movedM: number
}

interface Props {
  /** ReachabilityResponse's start_status/start_x/start_z/start_moved_m for the
   * SELECTED platform, or null before the first response for its radius. */
  start: StartPosition | null
  /** The selected platform's radius, for the "nothing has this much clearance"
   * sentence - the whole point of the "none" case is naming the number that failed. */
  radiusM: number | null
}

/** Where the selected robot actually stands, as ONE LINE under the robot picker.
 *
 * This was a section of its own - a header, a coordinate in mono, and a line of prose -
 * which is three rows of the left column spent on a number nobody types in. The
 * coordinate and the reason moved into the (i); what stays on screen is the only part
 * that changes what you would do next: whether the robot starts where the scene says,
 * somewhere else, or nowhere at all.
 *
 * This is what "fits" means on this screen: the start EXISTS for this radius. It is not
 * a fit probability - the Monte-Carlo `Fits: N/100` line this replaced was a different
 * question (how often a jittered geometry admits a route) shown in the same word, in the
 * same place, as the headline. "0 of 17 reachable" reads identically whether the room is
 * impassable or the robot merely has nowhere to start, so the "none" case says so in
 * words instead of showing a zero.
 *
 * `data-testid="start-position"` stays on this element: it is inventory #36, which keeps
 * its number across the re-layout and moves its selector here (demo/probe_layout_inventory.py). */
export default function StartPositionCaption({ start, radiusM }: Props) {
  if (start === null) {
    return (
      <p className="ui-caption text-muted" data-testid="start-position">
        Start: checking…
      </p>
    )
  }

  if (start.status === 'none') {
    return (
      <p
        className="ui-caption flex items-center gap-1.5 text-data-unreachable"
        data-testid="start-position"
      >
        <span data-testid="start-none">Start: none</span>
        <Why label="Why there is no start">
          No cell in this room has {formatMetres(radiusM)} of clearance, so this robot has
          nowhere to stand.
        </Why>
      </p>
    )
  }

  const where = `(${formatNumber(start.x)}, ${formatNumber(start.z)})`
  return (
    <p className="ui-caption flex items-center gap-1.5 text-muted" data-testid="start-position">
      {start.status === 'moved' ? (
        <span data-testid="start-moved">Start: auto · moved {formatMetres(start.movedM)}</span>
      ) : (
        <span data-testid="start-original">Start: auto</span>
      )}
      <Why label="Where the robot starts">
        {where} ·{' '}
        {start.status === 'moved'
          ? "the scene's own start point is too tight for this robot, so it begins at the nearest cell that fits."
          : "the scene's own start point fits this robot."}
      </Why>
    </p>
  )
}

/** A focusable span, not a button, and deliberately: it opens nothing and does nothing on
 * click, and demo/probe_visual.py requires every real control to be at least 32px tall - a
 * 16px <button> would either fail that or force a 32px row for a hint. Same shape
 * SceneVerdict uses for "tightest gap". */
function Why({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Tooltip content={children} maxWidthPx={240}>
      <span
        tabIndex={0}
        role="note"
        aria-label={label}
        data-testid="start-why"
        className="flex h-4 w-4 shrink-0 cursor-default items-center justify-center rounded-full border-hair border-border text-[9px] leading-none text-muted transition-colors duration-100 hover:bg-fg/10 hover:text-fg"
      >
        i
      </span>
    </Tooltip>
  )
}
