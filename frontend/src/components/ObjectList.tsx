import { Fragment, memo, useEffect, useMemo, useRef, useState } from 'react'
import type { SceneObjectResponse } from '../api/types'
import { createDeferredClick } from '../lib/deferredClick'
import { formatMetres } from '../lib/formatMeasure'
import { groupObjects, visibleObjects } from '../lib/objectGroups'
import { unreachableReasonLabel, unreachableReasonTitle } from '../lib/unreachableReason'
import Badge from './ui/Badge'
import PanelSection from './ui/PanelSection'
import Tooltip from './ui/Tooltip'

interface Props {
  objects: readonly SceneObjectResponse[]
  selectedObjectId: string | null
  onSelectObject: (id: string | null) => void
  /** The object the robot is heading for RIGHT NOW - null the moment it arrives or the
   * command ends. Distinct from `activeObjectId`, which is the last command's target and
   * stays set after arrival: this is "where is it going", that one is "what was it last
   * asked about", and only the first deserves a live highlight. */
  targetObjectId?: string | null
  activeObjectId?: string | null
  pickedIds?: Set<string>
  hoveredObjectId: string | null
  onHoverObject: (id: string | null) => void
  /** Single click, selecting (not deselecting) a row: frames the 3D camera on that
   * object and highlights its hull - see SceneMap3D's focusObjectId/focusNonce props.
   * That, plus the selection highlight, is ALL a single click does. It used to also
   * push the object's name into the command input (an `onPick` prop, now gone), which
   * meant every click littered the box - and a double-click, which fires two clicks
   * first, littered it twice: the shot that showed `desk desk` in the input.
   *
   * Deferred by DOUBLE_CLICK_GRACE_MS and cancelled by a double-click (see
   * lib/deferredClick.ts), because the DOM fires two clicks before every dblclick and
   * this one moves the camera - which is how a double-click meant to plan a route
   * used to fly the camera inside the desk. */
  onFocusObject: (id: string) => void
  /** Double-click, or the row's Go button: send the robot straight there, no chat
   * round-trip. Picks the object (selects it, highlights its hull) and plans the
   * route - and moves no camera at all. */
  onGoto: (id: string, name: string) => void
  /** null = reachability not loaded yet, so nothing is dimmed. */
  reachableObjectIds: Set<string> | null
  /** Object id -> the backend's own word for why it is not reachable, printed verbatim
   * on the row (see lib/unreachableReason.ts). Absent for a reachable object. */
  unreachableReasons: Record<string, string>
  /** Object id -> the route length in metres from this platform's start, keyed on
   * exactly the reachable ids. The row prints this where an unreachable row prints its
   * reason: every row says either how far, or why not. */
  pathLengthM: Record<string, number>
  onUnreachableClick: (name: string) => void
  /** Lifted to ScenePage: the toggle lives on the screen's toolbar now, because it
   * changes what the platform comparison enumerates as well as this list. */
  showFragments: boolean
}

/** Memoised, and it matters: ScenePage re-renders on every animation frame of a move (see
 * `useCommandAnimation`), and this list is 17 chips with a tooltip and a status mark each.
 * Nothing here depends on the robot's position, so re-rendering it 60 times a second while
 * the robot drives is pure cost. Measured on hero-74, production build, application JS per
 * frame during a goto: 9.65 ms median before the panels were memoised. */
function ObjectList({
  objects,
  selectedObjectId,
  onSelectObject,
  targetObjectId,
  activeObjectId,
  pickedIds,
  hoveredObjectId,
  onHoverObject,
  onFocusObject,
  onGoto,
  reachableObjectIds,
  unreachableReasons,
  pathLengthM,
  onUnreachableClick,
  showFragments,
}: Props) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  // One arbiter for the whole list: only one row can be mid-click at a time, and a
  // pending focus must not survive the component going away.
  const deferredRef = useRef(createDeferredClick())
  useEffect(() => {
    const deferred = deferredRef.current
    return () => deferred.cancel()
  }, [])
  const visible = useMemo(() => visibleObjects(objects, showFragments), [objects, showFragments])
  // ONE COUNT. The header, the verdict's "N of M" and the badge all count the same set:
  // the non-fragment SceneObject rows this list is showing (`visibleObjects`, shared with
  // ScenePage and the platform comparison).
  //
  // This used to sum the canonical MSA export's per-class counts instead, and that was
  // the screen's one arithmetic contradiction: the header read "OBJECTS (15)" beside a
  // verdict of "13 of 17" on the same room. Two reasons it had to be this set and not
  // that one. The MSA list uses its own id namespace ("bed_0") with no 1:1 mapping to
  // these rows - its own prop doc said so - and reachability, path_length_m, the map
  // markers and every goto are keyed on THESE ids, so a count from a list the planner
  // cannot address is a second source of truth by construction. And it did not track the
  // fragments toggle at all: hiding fragments changed the chips and left the header
  // still counting them.
  const headerCount = visible.length

  const groups = useMemo(() => groupObjects(visible), [visible])

  function toggleGroup(key: string) {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  /** What a single click does, once it is known to BE a single click. */
  function pick(o: SceneObjectResponse) {
    if (reachableObjectIds && !reachableObjectIds.has(o.id)) {
      onUnreachableClick(o.name)
      return
    }
    const isSelected = o.id === selectedObjectId
    onSelectObject(isSelected ? null : o.id)
    // Only on a genuine select (not the second click that deselects) - deselecting has
    // nothing to focus the camera on.
    if (!isSelected) onFocusObject(o.id)
  }

  function clickRow(o: SceneObjectResponse) {
    deferredRef.current.click(() => pick(o))
  }

  /** Double-click, and the Go button: pick the object and plan the route. It selects
   * (so the object is highlighted and the list agrees with the map) and it does NOT
   * call onFocusObject - no camera move of any kind. The pending single-click focus is
   * cancelled first, since the DOM has already fired two clicks by the time we get here. */
  function goto(o: SceneObjectResponse) {
    deferredRef.current.cancel()
    if (reachableObjectIds && !reachableObjectIds.has(o.id)) {
      onUnreachableClick(o.name)
      return
    }
    onSelectObject(o.id)
    onGoto(o.id, o.name)
  }

  return (
    <PanelSection title="Objects" count={headerCount}>
      {/* Chips that wrap, not one row per object. Seventeen full-width rows filled the
         column and pushed the chat off the screen; the same seventeen objects fit in a
         few lines here, and the two things a chip has to say - can the robot get there,
         and which object is it - are a tick or a strikethrough and a name. How far, or
         exactly why not, is one hover away rather than competing for width on every
         line. `data-object-length` and `data-object-reason` still carry the numbers, so
         demo/probe_hero_numbers.py checks them against the API exactly as before. */}
      <div className="flex flex-wrap content-start gap-s1 px-s3 pb-s2 pt-s1">
        {groups.length === 0 && <p className="py-s2 text-bodyux text-muted">No objects detected.</p>}
        {(() => {
          // Running position across every rendered row (group headers and, once
          // expanded, their children) - purely for the reachability-change stagger
          // below, so the recolor visibly cascades down the list top to bottom.
          let rowIndex = 0
          return groups.map((g) => {
            if (g.items.length === 1) {
              const o = g.items[0]
              return (
                <ObjectChip
                  key={o.id}
                  o={o}
                  {...rowState(o)}
                  staggerIndex={rowIndex++}
                  onClick={() => clickRow(o)}
                  onDoubleClick={() => goto(o)}
                />
              )
            }

            const groupStagger = rowIndex++
            const isOpen = expanded.has(g.key)
            const containsFocused =
              g.items.some((o) => o.id === hoveredObjectId) || g.items.some((o) => o.id === selectedObjectId)
            const anyActive = g.items.some((o) => o.id === activeObjectId)
            const containsTarget = g.items.some((o) => o.id === targetObjectId)
            const allUnreachable =
              reachableObjectIds !== null && g.items.every((o) => !reachableObjectIds.has(o.id))
            // The nearest reachable instance's own route length - the honest one-number
            // answer for a collapsed group of four lamps, and the only one that is a
            // real measurement rather than an average of routes nobody would drive.
            const groupLengths = g.items
              .map((o) => pathLengthM[o.id])
              .filter((v): v is number => typeof v === 'number')
            const nearestM = groupLengths.length ? Math.min(...groupLengths) : null

            // The instances this chip actually expands into, and nothing else - a badge
            // saying "x3" has to open into three rows.
            const count = g.items.length
            return (
              // A Fragment, not a wrapper div: the expanded members have to join the same
              // wrapping flex line as every other chip, and a div would stack them in a
              // column of their own.
              <Fragment key={g.key}>
                <Tooltip
                  content={
                    allUnreachable ? 'unreachable' : nearestM !== null ? `${formatMetres(nearestM)} to the nearest` : `${count} objects`
                  }
                >
                  <button
                    type="button"
                    onClick={() => toggleGroup(g.key)}
                    data-object-group={g.key}
                    data-object-nearest={nearestM ?? undefined}
                    aria-expanded={isOpen}
                    title={
                      allUnreachable
                        ? `Robot can't reach any of these ${count} at the current radius`
                        : nearestM !== null
                          ? groupLengths.length === g.items.length
                            ? `nearest of ${g.items.length} · click to open`
                            : `nearest of the ${groupLengths.length} reachable (of ${g.items.length}) · click to open`
                          : 'click to open'
                    }
                    className={`${CHIP_BASE} ${
                      allUnreachable
                        ? CHIP_UNREACHABLE
                        : isOpen || containsFocused
                          ? CHIP_OPEN
                          : CHIP_REACHABLE
                    }`}
                  >
                    <StatusMark
                      unreachable={allUnreachable}
                      active={anyActive}
                      staggerIndex={groupStagger}
                    />
                    <span className={`truncate ${allUnreachable ? 'line-through' : ''}`}>{g.name}</span>
                    <span className="shrink-0 font-mono text-xs text-muted">×{count}</span>
                    <span
                      className={`shrink-0 text-[10px] text-faint transition-transform duration-100 ${isOpen ? 'rotate-90' : ''}`}
                    >
                      ›
                    </span>
                    {!isOpen && containsTarget && <Badge variant="live">driving here</Badge>}
                  </button>
                </Tooltip>

                {isOpen &&
                  g.items.map((o, i) => (
                    <ObjectChip
                      key={o.id}
                      o={o}
                      {...rowState(o)}
                      staggerIndex={rowIndex++}
                      onClick={() => clickRow(o)}
                      onDoubleClick={() => goto(o)}
                      index={i + 1}
                    />
                  ))}
              </Fragment>
            )
          })
        })()}
      </div>
    </PanelSection>
  )

  function rowState(o: SceneObjectResponse) {
    return {
      isSelected: o.id === selectedObjectId,
      isActive: o.id === activeObjectId,
      isHovered: o.id === hoveredObjectId,
      isTarget: o.id === targetObjectId,
      isPicked: pickedIds?.has(o.id) ?? false,
      isUnreachable: reachableObjectIds !== null && !reachableObjectIds.has(o.id),
      reason: unreachableReasons[o.id],
      lengthM: pathLengthM[o.id],
      onMouseEnter: () => onHoverObject(o.id),
      onMouseLeave: () => onHoverObject(null),
    }
  }
}

/** Shared chip shape. `h-row` is 32px, which is also demo/probe_visual.py's minimum hit
 * area - a chip is a real target, not a tag. `max-w-full` plus `min-w-0` keep a long
 * object name inside the column instead of widening the flex line past it. */
const CHIP_BASE =
  'inline-flex h-row max-w-full min-w-0 items-center gap-1.5 rounded-full border-hair px-s2 ' +
  'text-bodyux transition-colors duration-100'
const CHIP_REACHABLE = 'border-border text-subtle hover:bg-fg/10 hover:text-fg'
const CHIP_UNREACHABLE = 'border-data-unreachable/30 text-data-unreachable hover:bg-data-unreachable/10'
const CHIP_OPEN = 'border-accent/40 bg-accent-dim text-fg'
const CHIP_TARGET = 'border-accent bg-accent-dim text-fg ring-1 ring-inset ring-accent'

/** Reachable or not, in one glyph. A tick in the reachable green, a dash in the
 * unreachable red - paired with the name's strikethrough so the state survives being read
 * by someone who cannot tell the two colours apart. The 40ms-per-chip stagger is kept from
 * the old rows (spec 6.5): switching platform visibly cascades through the list rather
 * than repainting all at once. */
function StatusMark({
  unreachable,
  active,
  staggerIndex,
}: {
  unreachable: boolean
  active?: boolean
  staggerIndex: number
}) {
  return (
    <span
      aria-hidden="true"
      className={`shrink-0 text-[11px] leading-none transition-colors duration-200 ${
        unreachable ? 'text-data-unreachable' : active ? 'text-accent' : 'text-accent'
      }`}
      style={{ transitionDelay: `${staggerIndex * 40}ms` }}
    >
      {unreachable ? '–' : '✓'}
    </span>
  )
}

function ObjectChip({
  o,
  index,
  staggerIndex = 0,
  isSelected,
  isActive,
  isHovered,
  isTarget,
  isPicked,
  isUnreachable,
  reason,
  lengthM,
  onMouseEnter,
  onMouseLeave,
  onClick,
  onDoubleClick,
}: {
  o: SceneObjectResponse
  /** Position within an expanded group. Rendered as `lamp #2`, because a bare `#2` on a
   * chip that has wrapped away from its group header names nothing at all - the old
   * full-width rows sat directly under their header and could get away with it. */
  index?: number
  /** Position among all visible chips - staggers the tick's recolor 40ms per chip. */
  staggerIndex?: number
  isSelected: boolean
  isActive: boolean
  isHovered: boolean
  /** The robot is driving here right now - see the `targetObjectId` prop. */
  isTarget: boolean
  isPicked: boolean
  isUnreachable: boolean
  /** The backend's own word for why this object is unreachable. Carried verbatim in
   * `data-object-reason`; DISPLAYED through `unreachableReasonLabel`, which rewrites
   * exactly one of them (`disconnected` -> `unreachable`) and passes the rest through.
   * Never turned into a gap measurement - see lib/unreachableReason.ts. */
  reason?: string
  /** Route length in metres from this platform's start. undefined when unreachable. */
  lengthM?: number
  onMouseEnter: () => void
  onMouseLeave: () => void
  onClick: () => void
  onDoubleClick: () => void
}) {
  // How far, or why not - the question this screen exists to answer, now one hover away
  // instead of one column of every row. The Go button that used to sit beside the row is
  // gone with the rows: double-click is the goto path (demo/record_take2.py's
  // goto_object() and every shot in take 2 already use it), and a chip cannot carry a
  // second button without becoming a row again.
  // Short enough for the tooltip, which is `whitespace-nowrap` and positioned from the
  // trigger's own box - a full gloss sentence in there would run off the screen. The
  // sentence goes in `title`, exactly where the old row's reason span kept it.
  const tip = isUnreachable
    ? reason
      ? unreachableReasonLabel(reason)
      : 'unreachable'
    : `${formatMetres(lengthM)} away`
  const title = isUnreachable
    ? reason
      ? unreachableReasonTitle(reason)
      : "Robot can't reach this at the current radius"
    : 'Click to select and focus · double-click to drive here'

  // The hover handlers sit on a wrapper, NOT on the button: Tooltip clones its child and
  // overrides onMouseEnter/onMouseLeave to drive itself, so a handler passed to the button
  // would be silently replaced - and `onHoverObject` is what lights the matching marker and
  // its reason label on the 2D map (demo/probe_map2d.py asserts that hover). The old
  // full-width row kept them on its flex wrapper for the same reason.
  return (
    <span className="inline-flex min-w-0 max-w-full" onMouseEnter={onMouseEnter} onMouseLeave={onMouseLeave}>
    <Tooltip content={tip}>
      <button
        type="button"
        data-object-id={o.id}
        data-object-target={isTarget ? 'true' : undefined}
        data-object-reason={isUnreachable ? reason : undefined}
        data-object-length={isUnreachable ? undefined : (lengthM ?? undefined)}
        title={title}
        onClick={onClick}
        onDoubleClick={onDoubleClick}
        className={`${CHIP_BASE} ${
          // The live target outranks selection and hover: while the robot is moving, where
          // it is going is the most important thing this list can say.
          isTarget
            ? CHIP_TARGET
            : isUnreachable
              ? CHIP_UNREACHABLE
              : isSelected || isHovered
                ? CHIP_OPEN
                : CHIP_REACHABLE
        }`}
      >
        <StatusMark unreachable={isUnreachable} active={isActive} staggerIndex={staggerIndex} />
        <span className={`truncate ${isUnreachable ? 'line-through' : ''} ${o.is_fragment ? 'text-muted' : ''}`}>
          {index !== undefined ? `${o.name} #${index}` : o.name}
        </span>
        {o.is_fragment && <Badge variant="neutral">fragment</Badge>}
        {isTarget && <Badge variant="live">driving here</Badge>}
        {isPicked && <Badge variant="live">picked</Badge>}
      </button>
    </Tooltip>
    </span>
  )
}

export default memo(ObjectList)
