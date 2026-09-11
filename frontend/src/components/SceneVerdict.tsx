import { formatMetres } from '../lib/formatMeasure'
import Tooltip from './ui/Tooltip'

interface Props {
  /** The selected platform's display name - the verdict is about one robot, and the
   * sentence is unreadable without saying which. */
  platformName: string | null
  /** null until the first reachability response for the selected platform's radius. */
  reachableCount: number | null
  totalCount: number
  /** True when this platform has nowhere to stand in this room at all (start_status
   * "none"). "0 of 17 reachable" would be arithmetically right and completely
   * misleading - the robot never got as far as failing to reach anything. */
  noStart: boolean
  /** The selected platform's own clearance radius. Only used for the second half of
   * "does not fit": a robot wider than the tightest corridor the fit-probability audit
   * ever got through does not fit this room, whatever its start position says. */
  robotRadiusM?: number | null
  /** True when this scene has no navigation layer at all, so there is nothing to give a
   * verdict FROM. Distinct from `noStart` on purpose: both are all-zero, but one is an
   * answer about the robot and the other is the absence of anything to answer with. */
  layerAvailable?: boolean
  /** DISTINCT cells the routes behind the reachable count cross that nobody ever
   * observed. Unknown is traversable at 1.5x rather than blocked - blocking it measures
   * 0 of 17 for every robot on this scene - so the count above can lean on ground the
   * scan never covered, and this says by how much. 0 shows nothing: a footnote that is
   * always on stops being read.
   *
   * SHOWN IN METRES, not in cells. "139 cells" is a number about the grid, and nobody
   * reading a verdict about a robot knows what a cell is worth; `139 x 0.05 m` is 7.0 m
   * of floor, which is a thing you can picture standing in the room. The API keeps
   * `unknown_cells_on_route` - the grid is where the count is exact, and a metre figure
   * is a presentation of it. */
  unknownCellsOnRoute?: number
  /** The grid's cell size in metres, from GET /map. Required to turn the cell count into
   * the distance the bar prints; undefined hides the bar rather than guessing a scale. */
  gridResolutionM?: number | null
  /** The 5th-percentile corridor width from this scene's fit-probability audit
   * (`corridor_width_p5_m`), for the selected platform. Three states, all different:
   *   undefined - no audit on disk for this scene, so nothing is shown at all;
   *   null      - an audit exists but has no successful trial to take a percentile over
   *               (Husky A200 on the hero scene, 0/100 fits), which the shared formatter
   *               renders as an em dash rather than a zero;
   *   a number  - the measured tightest place along the routes that did succeed.
   * Never computed here: this screen has no gap measurement of its own to offer. */
  tightestGapM?: number | null
}

/** The screen's headline, in the toolbar: how many of this room's objects this robot
 * can actually get to, and how tight it gets on the way. */
export default function SceneVerdict({
  platformName,
  reachableCount,
  totalCount,
  noStart,
  robotRadiusM = null,
  tightestGapM,
  layerAvailable = true,
  unknownCellsOnRoute = 0,
  gridResolutionM = null,
}: Props) {
  // Nothing to give a verdict from. Said in the bar itself rather than left as a blank or
  // an em dash, because a scene with no navigation layer looks identical to one where the
  // robot fits nowhere, and the two need different actions from whoever is reading.
  if (!layerAvailable) {
    return (
      <p className="truncate text-[15px] text-muted" data-testid="verdict">
        <span data-testid="verdict-fit">layer not available</span>{' '}
        <span className="text-[13px] text-subtle">· this scene has no navigation layer</span>
      </p>
    )
  }

  // ONE RULE, THREE STATES, all under the one `verdict-fit` testid:
  //
  //   "does not fit"        no start at this radius, OR the robot is wider than the
  //                         tightest corridor the audit ever got through
  //   "fits · N unreachable" it has somewhere to stand and cannot reach everything
  //   "fits"                 it has somewhere to stand and reaches every listed object
  //
  // The middle state is the one that was missing, and it was the screen's other
  // contradiction: "8 of 24 reachable" sat next to a flat green "fits", which is true
  // about the start position and reads as true about the room. The badge now carries the
  // same fact as the count instead of quietly disagreeing with it.
  //
  // "Fits" still means the start position EXISTS at this radius - it is NOT the
  // Monte-Carlo `fits_count` from the fit-probability audit, which answers a different
  // question and which this screen does not read.
  //
  // The second half of "does not fit" is new: a platform whose own radius exceeds
  // `tightestGapM` cannot make the journeys the audit measured, whatever the start says.
  const tooWideForTheRoom =
    tightestGapM != null && robotRadiusM != null && robotRadiusM > tightestGapM
  if (noStart || tooWideForTheRoom) {
    return (
      <p className="truncate text-[15px] text-data-unreachable" data-testid="verdict">
        <span className="font-medium">{platformName ?? 'This robot'}</span>{' '}
        <span data-testid="verdict-fit">does not fit in this room</span>
      </p>
    )
  }

  // Only alongside routes that exist. A tightest gap with nothing reachable would be a
  // number about journeys nobody can make, and the audit it comes from measures exactly
  // the trials that succeeded.
  const showGap = tightestGapM !== undefined && reachableCount !== null && reachableCount > 0
  const unreached = reachableCount === null ? null : totalCount - reachableCount
  const allReached = unreached === 0

  return (
    <p className="flex min-w-0 items-baseline gap-2 truncate" data-testid="verdict">
      <span className="font-mono text-[17px] font-medium tabular-nums text-fg" data-testid="verdict-count">
        {reachableCount === null ? '…' : `${reachableCount} of ${totalCount}`}
      </span>
      <span className="text-[13px] text-muted">reachable</span>
      {platformName && <span className="truncate text-[13px] text-subtle">· {platformName}</span>}
      <Tooltip
        content={
          allReached
            ? 'This robot has somewhere to stand in this room and reaches every object on the list. Not a fit probability - see the start position.'
            : 'This robot has somewhere to stand in this room, but cannot reach every object on the list. Each unreachable object carries the reason on its own chip.'
        }
      >
        <span
          tabIndex={0}
          data-testid="verdict-fit"
          className={`shrink-0 cursor-default rounded-full border-hair px-1.5 text-[11px] ${
            allReached
              ? 'border-accent/40 text-accent'
              : 'border-status-partial/40 text-status-partial'
          }`}
        >
          {allReached || unreached === null ? 'fits' : `fits · ${unreached} unreachable`}
        </span>
      </Tooltip>
      {unknownCellsOnRoute > 0 && gridResolutionM != null && (
        <Tooltip content="How much of those routes runs over floor the scan never covered. It is crossed at a 1.5x cost rather than blocked - blocking unknown outright measures 0 reachable objects for every robot on this scene, because most of a room's floor is never directly observed.">
          <span
            tabIndex={0}
            data-testid="unobserved-cells"
            className="shrink-0 cursor-default font-mono text-[11px] tabular-nums text-muted"
          >
            via {formatMetres(unknownCellsOnRoute * gridResolutionM, 1)} unobserved
          </span>
        </Tooltip>
      )}
      {showGap && (
        <Tooltip content="5th-percentile corridor width along the routes that planned successfully, from this scene's fit-probability audit. An em dash means the audit has no successful trial to take a percentile over.">
          <span
            tabIndex={0}
            data-testid="tightest-gap"
            className="shrink-0 cursor-default font-mono text-[11px] tabular-nums text-muted"
          >
            tightest gap {formatMetres(tightestGapM)}
          </span>
        </Tooltip>
      )}
    </p>
  )
}
