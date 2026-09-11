import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { DOUBLE_CLICK_GRACE_MS } from '../lib/deferredClick'
import { unreachableReasonTitle } from '../lib/unreachableReason'
import ObjectList from './ObjectList'
import type { SceneObjectResponse } from '../api/types'

function makeObject(overrides: Partial<SceneObjectResponse> & { id: string; name: string }): SceneObjectResponse {
  return {
    scene_id: 'scene-1',
    description: null,
    pos_x: 0,
    pos_y: 0,
    pos_z: 0,
    bbox_min_x: 0,
    bbox_min_y: 0,
    bbox_min_z: 0,
    bbox_max_x: 1,
    bbox_max_y: 1,
    bbox_max_z: 1,
    num_views: 1,
    num_points: 100,
    is_fragment: false,
    mesh_path: null,
    created_at: '2026-01-01T00:00:00Z',
    ...overrides,
  }
}

const noop = () => {}

function baseProps() {
  return {
    selectedObjectId: null,
    onSelectObject: noop,
    hoveredObjectId: null,
    onHoverObject: noop,
    showFragments: false,
    onFocusObject: noop,
    onGoto: noop,
    reachableObjectIds: null,
    unreachableReasons: {},
    pathLengthM: {},
    onUnreachableClick: noop,
  }
}

// M7b addition (c): the DB has 2 raw "television" detections (pre-de-clutter) but
// the canonical MSA export only kept 1 after M7's merge/class-cap stages - the
// ONE COUNT. The header total and every group badge count the rows this list is actually
// showing - the same set the verdict's "N of M" and the fit badge use. The MSA canonical
// list used to override both; it addressed a different id namespace than the rows, so the
// screen could read "OBJECTS (15)" beside "13 of 17 reachable" for one room.
describe('ObjectList counts one set', () => {
  const objects = [
    makeObject({ id: 'a', name: 'television' }),
    makeObject({ id: 'b', name: 'television' }),
    makeObject({ id: 'c', name: 'lamp' }),
  ]

  it('counts the rows it is showing, in the header and on the badge', () => {
    render(<ObjectList objects={objects} {...baseProps()} />)
    expect(screen.getByText('(3)')).toBeInTheDocument() // header total
    expect(screen.getByText('×2')).toBeInTheDocument() // television group chip
  })

  it('a group badge counts exactly the instances it expands into', () => {
    // "x2" that opens into three rows is worse than no badge: the number is the promise.
    const { container } = render(<ObjectList objects={objects} {...baseProps()} />)
    fireEvent.click(container.querySelector('[data-object-group="television"]')!)
    expect(container.querySelectorAll('[data-object-id="a"], [data-object-id="b"]')).toHaveLength(2)
  })

  it('hiding fragments moves the header total with the chips', () => {
    // The old header summed a list that did not know about the toggle, so hiding
    // fragments changed the chips and left the total counting them.
    const withFragment = [...objects, makeObject({ id: 'd', name: 'shard', is_fragment: true })]
    const { rerender } = render(
      <ObjectList objects={withFragment} {...baseProps()} showFragments />,
    )
    expect(screen.getByText('(4)')).toBeInTheDocument()
    rerender(<ObjectList objects={withFragment} {...baseProps()} showFragments={false} />)
    expect(screen.getByText('(3)')).toBeInTheDocument()
  })
})

// Step 2: every row answers the screen's question for its own object - how far, or why
// not, in the backend's own words. The reason string is printed verbatim: the row must
// never paraphrase it, and must never turn it into a gap measurement, because no gap is
// measured on this path.
describe('ObjectList per-row verdict', () => {
  const objects = [
    makeObject({ id: 'reach', name: 'desk' }),
    makeObject({ id: 'blocked', name: 'pillow' }),
  ]

  function withReachability(extra: Record<string, unknown> = {}) {
    return render(
      <ObjectList
        objects={objects}
        {...baseProps()}
        reachableObjectIds={new Set(['reach'])}
        unreachableReasons={{ blocked: 'no_free_cell_within_2m' }}
        pathLengthM={{ reach: 3.922 }}
        {...extra}
      />,
    )
  }

  // The chip carries the number in `data-object-length` and shows it on hover, rather
  // than spending a column of every line on it. demo/probe_hero_numbers.py reads the same
  // attribute and checks it against the API to 1e-6, so this is the load-bearing form.
  it('carries the route length for a reachable object, rounded by the shared formatter', () => {
    const { container } = withReachability()
    const chip = container.querySelector('[data-object-id="reach"]')!
    expect(chip.getAttribute('data-object-length')).toBe('3.922')
    expect(chip.getAttribute('title')).toContain('double-click to drive here')
  })

  it("carries the backend's own reason, verbatim, for an unreachable one", () => {
    const { container } = withReachability()
    const chip = container.querySelector('[data-object-id="blocked"]')!
    expect(chip.getAttribute('data-object-reason')).toBe('no_free_cell_within_2m')
    // The gloss, not a paraphrase of the label and not an invented measurement.
    expect(chip.getAttribute('title')).toBe(unreachableReasonTitle('no_free_cell_within_2m'))
  })

  it('strikes through an unreachable name and ticks a reachable one', () => {
    const { container } = withReachability()
    const blocked = container.querySelector('[data-object-id="blocked"]')!
    const reach = container.querySelector('[data-object-id="reach"]')!
    expect(blocked.querySelector('.line-through')).not.toBeNull()
    expect(reach.querySelector('.line-through')).toBeNull()
    expect(reach.textContent).toContain('✓')
  })

  it('shows no gap number anywhere on an unreachable chip', () => {
    const { container } = withReachability()
    const chip = container.querySelector('[data-object-id="blocked"]')!
    expect(chip.textContent).not.toMatch(/\d+\.\d+\s*m/)
    expect(chip.getAttribute('title')).not.toMatch(/\d+\.\d+\s*m/)
    expect(chip.hasAttribute('data-object-length')).toBe(false)
  })

  it('leaves the length attribute off, rather than writing 0, when there is no length yet', () => {
    const { container } = withReachability({ pathLengthM: {} })
    const chip = container.querySelector('[data-object-id="reach"]')!
    expect(chip.hasAttribute('data-object-length')).toBe(false)
    expect(chip.textContent).not.toContain('0.00 m')
  })

  it('gives a collapsed group the nearest reachable instance, not a sum or an average', () => {
    render(
      <ObjectList
        objects={[
          makeObject({ id: 'l1', name: 'lamp', num_views: 9 }),
          makeObject({ id: 'l2', name: 'lamp', num_views: 3 }),
        ]}
        {...baseProps()}
        reachableObjectIds={new Set(['l1', 'l2'])}
        pathLengthM={{ l1: 4.25, l2: 2.69 }}
      />,
    )
    const group = document.querySelector('[data-object-group="lamp"]')!
    expect(group.getAttribute('data-object-nearest')).toBe('2.69')
    expect(group.textContent).not.toContain('6.94')
    expect(group.textContent).not.toContain('3.47')
  })
})

// Step 3, the §7 error: the DOM fires two `click`s before every `dblclick`, and this
// list's click handler frames the 3D camera on the object. Double-clicking a row to
// send the robot somewhere therefore flew the camera into that object - a whole demo
// take was thrown away over it (HANDOFF 5b, trap 2: background mode 57.2% -> 28.7%,
// the camera inside the desk, no route visible).
describe('ObjectList double-click is a goto, not a camera move', () => {
  const desk = makeObject({ id: 'desk', name: 'desk' })

  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  function setup() {
    const onFocusObject = vi.fn()
    const onGoto = vi.fn()
    const onSelectObject = vi.fn()
    const { container } = render(
      <ObjectList
        objects={[desk]}
        {...baseProps()}
        onFocusObject={onFocusObject}
        onGoto={onGoto}
        onSelectObject={onSelectObject}
      />,
    )
    return { onFocusObject, onGoto, onSelectObject, row: container.querySelector('[data-object-id="desk"]')! }
  }

  it('never frames the camera when the click pair turns out to be a double-click', () => {
    const { onFocusObject, onGoto, onSelectObject, row } = setup()
    // Exactly what a browser delivers for one double-click.
    fireEvent.click(row)
    fireEvent.click(row)
    fireEvent.doubleClick(row)
    act(() => vi.advanceTimersByTime(10_000))

    expect(onFocusObject).not.toHaveBeenCalled()
    expect(onGoto).toHaveBeenCalledExactlyOnceWith('desk', 'desk')
    // "pick only": the object ends up selected, not toggled back off.
    expect(onSelectObject).toHaveBeenCalledWith('desk')
  })

  it('still frames the camera for a genuine single click, after the grace period', () => {
    const { onFocusObject, onGoto, row } = setup()
    fireEvent.click(row)
    expect(onFocusObject).not.toHaveBeenCalled()
    act(() => vi.advanceTimersByTime(DOUBLE_CLICK_GRACE_MS))

    expect(onFocusObject).toHaveBeenCalledExactlyOnceWith('desk')
    expect(onGoto).not.toHaveBeenCalled()
  })

  // The per-row Go button is gone with the rows - a chip cannot carry a second button
  // without becoming a row again. Double-click is the goto path, which is what
  // demo/record_take2.py's goto_object() has always used.
  it('has no per-object Go button any more', () => {
    setup()
    expect(screen.queryByRole('button', { name: 'Go to desk' })).toBeNull()
    expect(document.querySelector('[data-goto-object-id]')).toBeNull()
  })

  it('a double-click on an unreachable row says so and plans nothing', () => {
    const onFocusObject = vi.fn()
    const onGoto = vi.fn()
    const onUnreachableClick = vi.fn()
    const { container } = render(
      <ObjectList
        objects={[desk]}
        {...baseProps()}
        reachableObjectIds={new Set<string>()}
        unreachableReasons={{ desk: 'disconnected' }}
        onFocusObject={onFocusObject}
        onGoto={onGoto}
        onUnreachableClick={onUnreachableClick}
      />,
    )
    const row = container.querySelector('[data-object-id="desk"]')!
    fireEvent.click(row)
    fireEvent.click(row)
    fireEvent.doubleClick(row)
    act(() => vi.advanceTimersByTime(10_000))

    expect(onGoto).not.toHaveBeenCalled()
    expect(onFocusObject).not.toHaveBeenCalled()
    expect(onUnreachableClick).toHaveBeenCalledExactlyOnceWith('desk')
  })
})
